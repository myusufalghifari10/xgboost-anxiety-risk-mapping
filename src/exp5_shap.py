"""Eksperimen 5 — PRODUK SHAP SUBGRUP: "fitur apa yang berpengaruh pada SISWA STRES".

Ini deliverable inti proyek: bukan sekadar MAE, tetapi penjelasan fitur untuk siswa
dengan kecemasan tinggi (y >= AMBANG_EKOR = 2,5; n=43) DIBANDINGKAN siswa lain (n=263).
Model E5 ("Perang Warp") dipakai karena warp sentralnya diluruskan di sisi training,
sehingga SHAP wilayah ekor tidak lagi ter-attenuasi.

Alur:
  1. Muat artefak E5: outputs/exp5/exp5_full_{fp}.json (fingerprint terbaru atau --fingerprint)
     + CSV per-siswa + seluruh booster fold di outputs/exp5/models/{fp}/ronde_r{rep}_f{fold}.ubj
       (di-namespace per fingerprint: --fingerprint menunjuk run lama tidak bisa tertukar).
  2. Rekonstruksi partisi outer KFold (seed = C.RANDOM_SEED + rep) PERSIS seperti exp5_train
     dan VERIFIKASI terhadap CSV per-siswa: set(row_id uji) harus sama; fail-fast bila beda.
  3. Tiap booster -> shap.TreeExplainer -> nilai SHAP untuk baris uji fold itu saja.
  4. Agregasi: rata-rata 10 ulangan per row_id; kolom one-hot dijumlahkan BERTANDA ke
     fitur sumbernya (agregasi sah untuk fitur kategorikal), baru diambil |.| per siswa.
  5. Keluaran:
       outputs/exp5/shap_lokal_{{fp}}.csv    : SHAP bertanda per siswa x fitur + y + grup
       outputs/exp5/shap_subgrup_{{fp}}.csv  : statistik ekor vs badan, rank, CI bootstrap, arah
       outputs/exp5/shap_kategori_{{fp}}.csv : kontribusi tingkat kategori (kolom one-hot)
       outputs/exp5/shap_kontras_{{fp}}.md   : tabel + paragraf faktual + catatan jujur
     (nama file memuat fingerprint: keluaran run berbeda tidak saling menimpa)
  6. Bootstrap 500 resample SISWA EKOR (seed tetap) -> CI 95% rank 10 fitur teratas.
  7. Arah berbasis model: PDP-slope (fitur diset persentil 25 vs 75, fitur lain tetap)
     pada siswa ekor, dirata-rata atas seluruh booster fold (repeats x folds). Hanya untuk fitur numerik/item.

CATATAN JUJUR (otomatis ditulis ke shap_kontras.md):
  * SHAP menjelaskan kontribusi terhadap PREDIKSI model, bukan hubungan kausal.
  * Subgrup ekor hanya n=43 -> setiap peringkat dilaporkan dengan CI bootstrap.
  * Kualitas penjelasan dibatasi kualitas model: verdict_prareg run E5 ikut dilaporkan.

Pemakaian (setelah `src/exp5_train.py` selesai):
    .venv/bin/python src/exp5_shap.py                 # fingerprint terbaru
    .venv/bin/python src/exp5_shap.py --fingerprint XXXX
    .venv/bin/python src/exp5_shap.py --self-test     # data sintetis, tanpa artefak
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from exp2_features import FEATURES_EXP2, ITEM_COLS  # noqa: E402
from exp2_train import _encode  # noqa: E402
from exp3_train import AMBANG_EKOR  # noqa: E402

EXP5_DIR = C.ROOT / "outputs" / "exp5"
MODELS_BASE = EXP5_DIR / "models"  # booster di-namespace per fingerprint: MODELS_BASE/<fp>/
MODEL_DIR = MODELS_BASE  # di-set ulang per-fingerprint di main() (temuan reviewer: jangan
#                       ambil booster run lain saat --fingerprint menunjuk run lama)
TOP_K = 10
N_BOOT = 500


# ---------- fungsi murni (diuji --self-test, tanpa data proyek) ----------

def _peta_kolom(nama_num: list[str], kategori: list[list[str]],
                nama_kat: list[str]) -> tuple[list[str], list[str]]:
    """Label + fitur-sumber untuk setiap kolom hasil _encode (numerik dulu, lalu one-hot).

    _encode() menumpuk [C.NUMERIC_FEATURES + ITEM_COLS] lebih dulu, baru one-hot
    C.CATEGORICAL_FEATURES (urutan kategori mengikuti encoder).
    """
    assert len(kategori) == len(nama_kat), "jumlah kategori != jumlah nama kategorikal"
    label = list(nama_num)
    sumber = list(nama_num)
    for nm, kats in zip(nama_kat, kategori):
        for k in kats:
            label.append(f"{nm}={k}")
            sumber.append(nm)
    assert len(label) == len(sumber), "label & sumber harus sepanjang"
    return label, sumber


def _agregat_shap(mat: np.ndarray, sumber: list[str]) -> pd.DataFrame:
    """Agregasi kolom one-hot ke fitur sumbernya secara BERTANDA (jumlah), lalu |.| per siswa.

    mat = (n_siswa, n_kolom). Untuk fitur kategorikal, jumlah bertanda atas dummy-nya
    adalah kontribusi total fitur tsb (dummy yang tidak aktif ikut menyumbang baseline).
    """
    m = np.asarray(mat, float)
    assert m.ndim == 2 and m.shape[1] == len(sumber), \
        f"dimensi SHAP {m.shape} != jumlah sumber {len(sumber)}"
    assert np.isfinite(m).all(), "ada nilai SHAP non-finite"
    kolom: dict[str, list[int]] = {}
    for i, s in enumerate(sumber):
        kolom.setdefault(s, []).append(i)
    return pd.DataFrame({s: m[:, idx].sum(axis=1) for s, idx in kolom.items()})


def _tabel_subgrup(lokal: pd.DataFrame, fitur: list[str]) -> pd.DataFrame:
    """Statistik ekor vs badan per fitur + kontras + rank. lokal = SHAP bertanda per siswa."""
    assert "grup" in lokal.columns and "y" in lokal.columns, "lokal harus punya kolom y & grup"
    ekor = (lokal["grup"] == "ekor").to_numpy()
    assert ekor.any() and (~ekor).any(), "kedua grup harus tidak kosong"
    baris = []
    for f in fitur:
        v = lokal[f].to_numpy(float)
        assert np.isfinite(v).all(), f"SHAP non-finite pada fitur {f}"
        baris.append({"fitur": f,
                      "mean_abs_ekor": float(np.abs(v[ekor]).mean()),
                      "mean_abs_badan": float(np.abs(v[~ekor]).mean()),
                      "mean_shap_ekor": float(v[ekor].mean()),
                      "mean_shap_badan": float(v[~ekor].mean()),
                      "n_ekor": int(ekor.sum()), "n_badan": int((~ekor).sum())})
    t = pd.DataFrame(baris)
    t["kontras_abs"] = t["mean_abs_ekor"] - t["mean_abs_badan"]
    t["rank_ekor"] = t["mean_abs_ekor"].rank(ascending=False, method="min").astype(int)
    t["rank_badan"] = t["mean_abs_badan"].rank(ascending=False, method="min").astype(int)
    return t.sort_values("mean_abs_ekor", ascending=False).reset_index(drop=True)


def _bootstrap_rank(abs_ekor: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """CI 95% rank tiap fitur (1 = paling penting) lewat resample SISWA ekor.
    abs_ekor = matriks |SHAP| (n_siswa_ekor, n_fitur). Deterministik pada seed tetap."""
    A = np.asarray(abs_ekor, float)
    assert A.ndim == 2 and A.shape[0] >= 2 and A.shape[1] >= 2, "matriks |SHAP| ekor tidak sah"
    assert n_boot > 0, "n_boot harus positif"
    n, _ = A.shape
    rng = np.random.default_rng(seed)
    ranks = np.empty((n_boot, A.shape[1]), dtype=int)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rerata = A[idx].mean(axis=0)
        ranks[b] = (-rerata).argsort().argsort() + 1  # rank 1 = mean|SHAP| terbesar
    lo = np.floor(np.percentile(ranks, 2.5, axis=0)).astype(int)
    hi = np.ceil(np.percentile(ranks, 97.5, axis=0)).astype(int)
    # floor/ceil (bukan truncation) -> CI TIDAK menyempit diam-diam (temuan reviewer)
    return (np.clip(lo, 1, A.shape[1]), np.clip(hi, 1, A.shape[1]))


def _shap_nilai(expl, X: np.ndarray) -> np.ndarray:
    """Wrapper SHAP: menerima ndarray maupun Explanation; assert bentuk + finite."""
    v = expl.shap_values(X)
    a = np.asarray(getattr(v, "values", v), float)
    assert a.ndim == 2 and a.shape == X.shape, f"bentuk SHAP tak terduga: {a.shape} (X={X.shape})"
    assert np.isfinite(a).all(), "ada nilai SHAP non-finite"
    return a


def _pdp_slope(booster: xgb.Booster, X_ekor: np.ndarray, pos: int, lo: float, hi: float) -> float:
    """PDP-slope: fitur `pos` diset `lo` vs `hi` (fitur lain tetap), rata-rata atas siswa ekor."""
    assert X_ekor.ndim == 2 and 0 <= pos < X_ekor.shape[1], "posisi fitur di luar rentang"
    assert hi > lo, "kuantil hi harus > lo"
    xa, xb = X_ekor.copy(), X_ekor.copy()
    xa[:, pos], xb[:, pos] = lo, hi
    pa = booster.predict(xgb.DMatrix(xa))
    pb = booster.predict(xgb.DMatrix(xb))
    return float(np.mean(pb - pa) / (hi - lo))


# ---------- pemuatan artefak ----------

def _sumber_run(fp: str | None) -> tuple[str, Path]:
    if fp:
        p = EXP5_DIR / f"exp5_full_{fp}.json"
        assert p.exists(), (f"artefak E5 tidak ditemukan: {p} — jalankan dulu "
                            f"`.venv/bin/python src/exp5_train.py`")
        return fp, p
    js = sorted(EXP5_DIR.glob("exp5_full_*.json"), key=lambda q: q.stat().st_mtime)
    assert js, (f"artefak E5 tidak ada di {EXP5_DIR} — jalankan dulu: "
                f".venv/bin/python src/exp5_train.py")
    return js[-1].stem.replace("exp5_full_", ""), js[-1]


def _muat(fp_pil: str | None):
    fp, run_json = _sumber_run(fp_pil)
    meta = json.loads(run_json.read_text())
    csv = EXP5_DIR / f"exp5_full_{fp}_per_siswa.csv"
    assert csv.exists(), f"CSV per-siswa E5 hilang: {csv} — jalankan ulang exp5_train.py"
    df = pd.read_csv(csv)
    for k in ("repeats", "folds", "verdict_prareg", "monotone"):
        assert k in meta, f"field '{k}' tidak ada di {run_json}"
    for k in ("row_id", "rep", "fold", "y", "pred_proc"):
        assert k in df.columns, f"kolom '{k}' tidak ada di {csv}"
    feats = pd.read_parquet(FEATURES_EXP2)
    assert len(feats) == 306 and C.TARGET_CONT in feats.columns, "features_exp2 tak terduga"
    y_csv = df.groupby("row_id")["y"].first().to_numpy(float)
    y_feats = feats[C.TARGET_CONT].to_numpy(float)
    assert np.allclose(y_csv, y_feats, atol=1e-12), (
        "y di CSV per-siswa != y di features_exp2 — baris tidak sejajar (SHAP akan salah pasang)")
    return fp, meta, df, feats


def _shap_per_siswa(meta: dict, df: pd.DataFrame, feats: pd.DataFrame,
                    num_cols: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """SHAP rata-rata 10 ulangan per siswa; verifikasi partisi fold vs CSV. (mat, label, sumber)."""
    ref = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ref.fit(feats[C.CATEGORICAL_FEATURES])
    label, sumber = _peta_kolom(list(num_cols), [list(c) for c in ref.categories_],
                                list(C.CATEGORICAL_FEATURES))
    n_kolom = len(label)
    n = len(feats)
    total = np.zeros((n, n_kolom), float)
    hitung = np.zeros(n, dtype=int)
    for rep in range(int(meta["repeats"])):
        outer = KFold(int(meta["folds"]), shuffle=True, random_state=C.RANDOM_SEED + rep)
        for fold, (itr, ite) in enumerate(outer.split(feats)):
            sub = df[(df["rep"] == rep) & (df["fold"] == fold)]
            assert set(sub["row_id"]) == set(int(i) for i in ite), (
                f"partisi fold E5 != CSV per-siswa di rep {rep} fold {fold}")
            mp = MODEL_DIR / f"ronde_r{rep}_f{fold}.ubj"
            assert mp.exists(), (f"booster hilang: {mp} — jalankan ulang "
                                 f"`.venv/bin/python src/exp5_train.py`")
            b = xgb.Booster()
            b.load_model(str(mp))
            _, xte = _encode(feats.iloc[itr], feats.iloc[ite], "76")
            assert xte.shape[1] == n_kolom, (
                f"lebar encode {xte.shape[1]} != peta kolom {n_kolom} — kategori berubah?")
            expl = shap.TreeExplainer(b, feature_perturbation="tree_path_dependent")
            sv = _shap_nilai(expl, xte)
            total[np.array([int(i) for i in ite])] += sv
            hitung[np.array([int(i) for i in ite])] += 1
    assert (hitung == int(meta["repeats"])).all(), \
        f"ada siswa tanpa SHAP lengkap: min {hitung.min()} dari {meta['repeats']}"
    return total / hitung[:, None], label, sumber


def _arah_pdp(meta: dict, feats: pd.DataFrame, num_cols: list[str],
              top_fitur: list[str]) -> dict[str, float]:
    """PDP-slope rata-rata atas seluruh booster fold untuk fitur numerik/item teratas (siswa ekor)."""
    pos = {f: i for i, f in enumerate(num_cols)}
    y_all = feats[C.TARGET_CONT].to_numpy(float)
    ekor_all = y_all >= AMBANG_EKOR
    assert ekor_all.sum() > 0, "tidak ada siswa ekor"
    fitur_pdp = [f for f in top_fitur if f in pos]
    hasil: dict[str, list[float]] = {f: [] for f in fitur_pdp}
    for rep in range(int(meta["repeats"])):
        outer = KFold(int(meta["folds"]), shuffle=True, random_state=C.RANDOM_SEED + rep)
        for fold, (itr, ite) in enumerate(outer.split(feats)):
            ekor_tr = ekor_all[np.array([int(i) for i in itr])]
            if not ekor_tr.any():
                continue
            q = {}
            for f in fitur_pdp:  # kuantil dari baris TRAIN fold ini (fold-safe, temuan reviewer)
                lo = float(feats.loc[ekor_tr, f].quantile(0.25))
                hi = float(feats.loc[ekor_tr, f].quantile(0.75))
                if hi > lo:  # fitur konstan di fold ini -> dilewati (tak terdefinisi)
                    q[f] = (lo, hi)
            _, xte = _encode(feats.iloc[itr], feats.iloc[ite], "76")
            ekor_fold = ekor_all[np.array([int(i) for i in ite])]
            if not ekor_fold.any():
                continue
            b = xgb.Booster()
            b.load_model(str(MODEL_DIR / f"ronde_r{rep}_f{fold}.ubj"))
            Xe = xte[ekor_fold]
            for f, (lo, hi) in q.items():
                hasil[f].append(_pdp_slope(b, Xe, pos[f], lo, hi))
    return {f: float(np.mean(v)) for f, v in hasil.items() if v}


def _tulis_teks(path: Path, teks: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(teks)
    os.replace(tmp, path)


def _tulis_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _angka(x: float) -> str:
    return f"{x:+.3f}" if x < 0 else f"{x:.3f}"


def _narasi(meta: dict, tabel: pd.DataFrame, n_ekor: int, n_badan: int,
            fp: str) -> str:
    """Laporan faktual bahasa Indonesia — semua angka diambil dari tabel/artefak."""
    top = tabel.head(15)
    kontras = tabel.reindex(tabel["kontras_abs"].abs().sort_values(ascending=False).index).head(10)
    vp = meta["verdict_prareg"]
    baris_top = "\n".join(
        f"| {i + 1} | {r.fitur} | {r.mean_abs_ekor:.3f} | {r.mean_abs_badan:.3f} | "
        f"{r.rank_ekor} | {int(r.rank_ci_lo)}-{int(r.rank_ci_hi)} | {_angka(r.pdp_slope) if r.pdp_slope == r.pdp_slope else '-'} |"
        for i, r in enumerate(top.itertuples()))
    baris_kontras = "\n".join(
        f"| {r.fitur} | {_angka(r.kontras_abs)} | {r.mean_abs_ekor:.3f} | {r.mean_abs_badan:.3f} |"
        for r in kontras.itertuples())
    tiga = ", ".join(f"{r.fitur} (mean|SHAP| = {r.mean_abs_ekor:.3f}, "
                     f"CI rank {int(r.rank_ci_lo)}-{int(r.rank_ci_hi)})"
                     for r in tabel.head(3).itertuples())
    return f"""# SHAP Subgrup — "Fitur apa yang berpengaruh pada SISWA STRES"

Sumber: `outputs/exp5/exp5_full_{fp}.json` (+ {int(meta['repeats']) * int(meta['folds'])} booster
fold dari namespace `outputs/exp5/models/{fp}/`). Model: Eksperimen-5
("Perang Warp": loss asimetris anti-warp + monotone constraints).

- Kelompok **ekor (siswa stres)**: y >= {AMBANG_EKOR} — **n={n_ekor}**.
- Kelompok **badan**: y < {AMBANG_EKOR} — **n={n_badan}**.
- SHAP = kontribusi terhadap **prediksi model** (bukan kausalitas). Kolom kategorikal
  dijumlahkan bertanda atas dummy-nya, lalu |.| diambil per siswa.

## Untuk siswa stres (n={n_ekor}), fitur paling berpengaruh

{tiga} — peringkat lengkap: `shap_subgrup.csv`.

| # | Fitur | mean\\|SHAP\\| ekor | mean\\|SHAP\\| badan | rank ekor | CI 95% rank | PDP-slope |
|---|---|---|---|---|---|---|
{baris_top}

## Fitur yang paling BERBEDA antara siswa stres dan siswa lain

| Fitur | Kontras (ekor − badan) | mean\\|SHAP\\| ekor | mean\\|SHAP\\| badan |
|---|---|---|---|
{baris_kontras}

## Catatan jujur

1. **n={n_ekor} siswa ekor** — peringkat tidak presisi; CI 95% rank (kolom di atas) adalah
   lebar ketidakpastiannya. Jangan mengklaim urutan halus (mis. rank 1 vs rank 2) bila CI-nya tumpang tindih.
2. **SHAP bukan kausalitas** — ini penjelasan perilaku model, sejalan dengan kerangka
   "association graph", bukan DAG kausal.
3. **Kualitas penjelasan dibatasi kualitas model**: gate pra-registrasi E5 →
   `lolos = {vp['lolos']}` (slope warp {vp['nilai']['slope_ensemble']:.3f}; MAE ensemble {vp['nilai']['mae_ensemble']:.4f};
   MAE ekor {vp['nilai']['mae_ensemble_ekor']:.4f}). Bila gate tidak lolos, sebutkan
   keterbatasan ini saat melaporkan.
4. **PDP-slope** = rata-rata perubahan prediksi saat fitur digeser dari persentil 25 ke 75
   (fitur lain tetap) pada siswa ekor, dirata-rata atas {int(meta['repeats']) * int(meta['folds'])} booster; kuantil 25/75 dihitung
   **dari baris latih fold itu** (fold-safe) — hanya untuk fitur numerik/item; fitur
   kategorikal dilaporkan "-".
5. **Skala penjelasan vs skala metrik**: SHAP menjelaskan output **mentah** booster
   (raw margin), sedangkan MAE dan gerbang pra-registrasi dihitung pada `pred_proc`
   (hasil decode: shrink badan + snap, tanpa shrink di ekor). Untuk siswa ekor keduanya
   koheren (pred_proc = snap(raw) saat raw >= 2,5); untuk siswa badan, skala penjelasan
   tidak sama dengan skala prediksi terdecode — jangan membandingkan magnitudo SHAP
   badan langsung dengan magnitudo error.
"""


# ---------- self-test (data sintetis, tanpa artefak proyek) ----------

def _self_test() -> None:
    # (a) peta kolom encode -> sumber (numerik dulu, lalu one-hot)
    label, sumber = _peta_kolom(["n1", "n2"], [["a", "b"], ["x", "y", "z"]], ["k1", "k2"])
    assert label == ["n1", "n2", "k1=a", "k1=b", "k2=x", "k2=y", "k2=z"], label
    assert sumber == ["n1", "n2", "k1", "k1", "k2", "k2", "k2"], sumber

    # (b) agregasi one-hot bertanda
    agg = _agregat_shap(np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]]), ["n1", "k", "k"])
    assert list(agg.columns) == ["n1", "k"], list(agg.columns)
    assert np.allclose(agg["k"].to_numpy(), [5.0, 2.5]), agg["k"].to_numpy()
    try:
        _agregat_shap(np.zeros((2, 3)), ["a", "b"])
        raise AssertionError("fail-fast dimensi SHAP vs sumber TIDAK bekerja")
    except AssertionError as e:
        assert "dimensi" in str(e), f"pesan fail-fast tak terduga: {e}"

    # (c) end-to-end SHAP pada booster sintetis: fitur informatif harus rank 1
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4))
    yv = 2.0 + 1.5 * X[:, 0] + 0.1 * X[:, 1] + rng.normal(0, 0.05, 90)
    b = xgb.train({**C.BASE_XGB, "max_depth": 3, "learning_rate": 0.3,
                   "objective": "reg:squarederror"},
                  xgb.DMatrix(X, label=yv), num_boost_round=40)
    expl = shap.TreeExplainer(b, feature_perturbation="tree_path_dependent")
    sv = _shap_nilai(expl, X)
    assert sv.shape == X.shape and np.isfinite(sv).all(), "SHAP sintetis tak sah"
    agg2 = _agregat_shap(sv, ["f0", "f1", "f2", "f3"])
    assert agg2.abs().mean(axis=0).idxmax() == "f0", "fitur informatif harus rank 1"

    # (d) tabel subgrup: mean|SHAP|, rank, kontras, n grup
    lokal = pd.DataFrame({"row_id": [0, 1, 2, 3], "y": [3.0, 2.6, 1.5, 1.2],
                          "grup": ["ekor", "ekor", "badan", "badan"],
                          "a": [1.0, 3.0, 0.5, 0.5], "b": [0.2, 0.2, 2.0, 2.0]})
    t = _tabel_subgrup(lokal, ["a", "b"])
    assert t.loc[t.fitur == "a", "mean_abs_ekor"].iloc[0] == 2.0
    assert t.loc[t.fitur == "b", "mean_abs_badan"].iloc[0] == 2.0
    assert t.loc[t.fitur == "a", "rank_ekor"].iloc[0] == 1
    assert t.loc[t.fitur == "a", "n_ekor"].iloc[0] == 2
    assert t.loc[t.fitur == "a", "kontras_abs"].iloc[0] == 1.5

    # (e) bootstrap rank: deterministik pada seed, rank dalam rentang, CI sah
    A = rng.random((43, 6)) + np.linspace(1.0, 0.1, 6)  # fitur 0 paling besar
    lo1, hi1 = _bootstrap_rank(A, 200, 42)
    lo2, hi2 = _bootstrap_rank(A, 200, 42)
    assert np.array_equal(lo1, lo2) and np.array_equal(hi1, hi2), "bootstrap tidak deterministik"
    assert lo1[0] == 1 and hi1[0] == 1, f"fitur dominan harus rank 1, dapat {lo1[0]}-{hi1[0]}"
    assert (lo1 >= 1).all() and (hi1 <= 6).all() and (lo1 <= hi1).all(), "CI rank tidak sah"

    # (f) PDP-slope: tanda sesuai arah fitur pada model monotone
    bm = xgb.train({**C.BASE_XGB, "max_depth": 2, "learning_rate": 0.3,
                    "objective": "reg:squarederror", "monotone_constraints": "(1,0)"},
                   xgb.DMatrix(np.column_stack([np.linspace(0, 1, 60), rng.normal(size=60)]),
                               label=np.linspace(0, 1, 60) * 2.0),
                   num_boost_round=40)
    Xe = np.column_stack([rng.uniform(0.1, 0.9, 10), rng.normal(size=10)])
    s_naik = _pdp_slope(bm, Xe, 0, 0.2, 0.8)
    assert s_naik > 0, f"fitur monotone +1 harus slope positif, dapat {s_naik}"

    # (g) fail-fast artefak: fingerprint tak ada -> pesan menyebut exp5_train
    try:
        _sumber_run("tidakada")
        raise AssertionError("fail-fast artefak hilang TIDAK bekerja")
    except AssertionError as e:
        assert "exp5_train" in str(e), f"pesan fail-fast tak terduga: {e}"

    print("self-test OK: peta kolom encode, agregasi one-hot bertanda, SHAP end-to-end "
          "(fitur informatif rank 1), tabel subgrup+rank, bootstrap deterministik+CI sah, "
          "PDP-slope tanda benar, fail-fast artefak hilang")


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Eksperimen 5 — SHAP subgrup siswa stres (y >= 2,5)")
    ap.add_argument("--fingerprint", default=None, help="fingerprint run exp5 (default: terbaru)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return

    fp, meta, df, feats = _muat(a.fingerprint)
    global MODEL_DIR  # booster WAJIB dari run yang sama dengan artefak JSON (anti-tertukar)
    MODEL_DIR = MODELS_BASE / fp
    assert meta.get("fingerprint", fp) == fp, (
        f"fingerprint di JSON ({meta.get('fingerprint')}) != nama artefak ({fp})")
    n_boost = int(meta["repeats"]) * int(meta["folds"])
    ada_boost = sorted(MODEL_DIR.glob("ronde_r*_f*.ubj"))
    assert len(ada_boost) == n_boost, (
        f"booster di {MODEL_DIR}: {len(ada_boost)} != {n_boost} (repeats x folds) — "
        f"jalankan ulang `.venv/bin/python src/exp5_train.py` dengan fingerprint ini")
    num_cols = list(C.NUMERIC_FEATURES) + list(ITEM_COLS)
    y = feats[C.TARGET_CONT].to_numpy(float)
    ekor = y >= AMBANG_EKOR
    print(f"[exp5-shap] fp={fp} | ekor={int(ekor.sum())} badan={int((~ekor).sum())} "
          f"| ulangan={meta['repeats']} x {meta['folds']} fold", flush=True)

    mat, label, sumber = _shap_per_siswa(meta, df, feats, num_cols)
    lokal = pd.DataFrame({"row_id": np.arange(len(feats)), "y": y,
                          "grup": np.where(ekor, "ekor", "badan")})
    agg = _agregat_shap(mat, sumber)
    fitur_asli = list(agg.columns)
    for f in fitur_asli:
        lokal[f] = agg[f].to_numpy()

    tabel = _tabel_subgrup(lokal, fitur_asli)
    abs_ekor = np.abs(lokal.loc[ekor, fitur_asli].to_numpy(float))
    lo, hi = _bootstrap_rank(abs_ekor, N_BOOT, C.RANDOM_SEED)
    urut = {f: i for i, f in enumerate(fitur_asli)}
    assert len(lo) == len(hi) == len(fitur_asli), "CI bootstrap tidak sepanjang fitur"
    tabel["rank_ci_lo"] = [int(lo[urut[f]]) for f in tabel["fitur"]]
    tabel["rank_ci_hi"] = [int(hi[urut[f]]) for f in tabel["fitur"]]

    top_fitur = list(tabel["fitur"].head(TOP_K))
    pdp = _arah_pdp(meta, feats, num_cols, top_fitur)
    tabel["pdp_slope"] = [pdp.get(f, float("nan")) for f in tabel["fitur"]]
    tabel["arah_pdp"] = ["naik" if v > 0 else ("turun" if v < 0 else "-")
                         if v == v else "-" for v in tabel["pdp_slope"]]

    kat = pd.DataFrame({
        "kolom": label[len(num_cols):],
        "fitur": sumber[len(num_cols):],
        "mean_abs_ekor": np.abs(mat[ekor][:, len(num_cols):]).mean(axis=0),
        "mean_abs_badan": np.abs(mat[~ekor][:, len(num_cols):]).mean(axis=0),
    })
    kat["kontras_abs"] = kat["mean_abs_ekor"] - kat["mean_abs_badan"]
    kat = kat.sort_values("mean_abs_ekor", ascending=False).reset_index(drop=True)

    _tulis_csv(EXP5_DIR / f"shap_lokal_{fp}.csv", lokal)
    _tulis_csv(EXP5_DIR / f"shap_subgrup_{fp}.csv", tabel)
    _tulis_csv(EXP5_DIR / f"shap_kategori_{fp}.csv", kat)
    _tulis_teks(EXP5_DIR / f"shap_kontras_{fp}.md",
                _narasi(meta, tabel, int(ekor.sum()), int((~ekor).sum()), fp))

    print(f"[exp5-shap] selesai -> {EXP5_DIR}/shap_lokal_{fp}.csv, shap_subgrup_{fp}.csv, "
          f"shap_kategori_{fp}.csv, shap_kontras_{fp}.md")
    print(tabel.head(TOP_K)[["fitur", "mean_abs_ekor", "mean_abs_badan", "rank_ekor",
                             "rank_ci_lo", "rank_ci_hi", "arah_pdp"]].to_string(index=False))


if __name__ == "__main__":
    main()
