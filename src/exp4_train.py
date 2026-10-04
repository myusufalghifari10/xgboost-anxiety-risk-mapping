"""Eksperimen 4 — "Soft-Label Empiris-Bayes + Tail-Preserving Post-Processing".

Temuan terverifikasi yang menjadi alasan file ini ada (eksperimen-3 = model
saat ini, MAE pooled 0,2338 / ekor 0,3774;alpha decode 0,97):
  * LANGKAH SHRINK alpha MENJATUHKAN MAE EKOR: pred_raw ekor 0,3696 ->
    pred_proc ekor 0,3774, sementara pooled hanya untung 0,00096.
  * Jarak ke lantai noise (±0,12) ~ seluruhnya variansi estimasi akibat label
    berisik: y = mean 20 item skala 1-4, jadi tiap titik y membawa galat
    pengukuran sigma_i/sqrt(20).

Jadi E4 menyerang PENYEBAB (label berisik), bukan loss/bobot baru:
  1. SOFT-LABEL Empires-Bayes (inti): y_tilde_i = mu + alpha_i (y_i - mu),
     alpha_i = var_true / (var_true + se_i^2), var_true = Var(y_train) -
     mean(se_i^2). Semua statistik HANYA dari baris train fold.
  2. DEKODE yang mempertahankan ekor: shrink hanya untuk pred_raw < 2,5;
     di atas ambang -> pakai pred_snap apa adanya (tanpa shrink).
  3. Empat kolom prediksi keluar sekaligus supaya efeknya bisa didekomposisi:
     raw -> snap -> proc (VARIAN UTAMA) -> proc_old (shrink untuk semua = ATURAN
     dekode E3, tapi dengan alpha per-fold; BUKAN salinan pred_proc E3 yang
     memakai alpha terpool 0,97).

Hiperparameter: TANPA Optuna. Tiap outer fold (rep r, fold f) mewarisi params
dari checkpoint eksperimen-3, jadi perbandingan vs E3 berpasangan maksimal —
satu-satunya perubahan = label + decode.

Pra-daftar (kemenangan E4): MAE pooled < 0,2338 ATAU MAE ekor turun nyata
(ambang noise ±0,003), tanpa menaikkan sisi lain > 0,003.

CATATAN PENYIMPANGAN dari rancangan (semua disengaja, dicatat di sini):
  a) Sumber params E3 TIDAK di-hardcode `exp3_91945185763d`, melainkan
     file `outputs/exp3/exp3_full_*.json` TERBARU -> field `fingerprint` -> folder
     checkpoint (hari ini folder itu = exp3_91945185763d). Konvensi yang sama
     dipakai eksperimen-4 versi lama di repo; fail-fast bila artefak E3 hilang.
  b) alpha DEKODE dihitung PER FOLD (dari inner-OOF fold itu), bukan terpool
     seperti alpha E3 — mengikuti rancangan yang menyebut "alpha decode per fold";
     alpha terpool tetap direkam sebagai pembanding diagnostik. alpha yang sama
     dipakai pred_proc DAN pred_proc_old, jadi selisih keduanya murni efek aturan
     ekor (bukan efek alpha).
  c) FLOOR_VAR = 1e-4 untuk var_true. Numerik: Var(y)=0,2182, mean(se²)=0,0284
     -> var_true=0,1898 (alpha label 0,70..1,00, mean 0,873). Jadi FLOOR hanya
     mencegah keruntuhan fold degenerate (y_train konstan -> var_true tetap > 0,
     bukan alpha=0 yang membuat semua prediksi jatuh ke mu).
  d) Penulis JSON lokal `_tulis` (sort_keys + ensure_ascii + os.replace) alih-alih
     exp2_train._simpan, yang tidak memakai sort_keys/ensure_ascii.
  e) Bobot inner-OOF mengikuti konvensi E3: bobot(ytr) dihitung sekali lalu
     diiris [jtr], bukan dihitung ulang per inner-train (lihat exp3_train.py:163,172).

Semua metrik dihitung terhadap y ASLI (tanpa modifikasi). Metric per varian:
pooled / ekor (y>=2,5) / non-ekor / ensemble (rata-rata prediksi 10 ulangan per
siswa) / per-repeat.

Pemakaian (EKSEKUSI = YUSUF):
  .venv/bin/python src/exp4_train.py --self-test     # data sintetis, tanpa data proyek
  .venv/bin/python src/exp4_train.py                 # full run 10 ronde (±menit)
  .venv/bin/python src/exp4_train.py --repeats 3     # gerbang cepat
  .venv/bin/python src/exp4_train.py --fresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from exp2_features import FEATURES_EXP2, ITEM_COLS  # noqa: E402
from exp2_train import GRID, _decode, _encode, _fit, _fit_alpha, _mae, _rmse  # noqa: E402
from exp3_train import AMBANG_EKOR, BIN_EDGES, WEIGHT_CAP, bobot  # noqa: E402

EXP3_DIR = C.ROOT / "outputs" / "exp3"
EXP4_DIR = C.ROOT / "outputs" / "exp4"
CKPT_DIR = EXP4_DIR / "checkpoints"
OBJECTIVE = "reg:absoluteerror"
FLOOR_VAR = 1e-4  # lantai var_true (lihat catatan c di docstring)
VARIAN = ("pred_raw", "pred_snap", "pred_proc", "pred_proc_old")
N_FITUR = len(C.NUMERIC_FEATURES) + len(ITEM_COLS) + len(C.CATEGORICAL_FEATURES)  # 76


# ---------- data ----------

def _muat_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """X (featur E2, 76), y_kontinu, matriks 20 item kecemasan (urutan baris sama)."""
    feats = pd.read_parquet(FEATURES_EXP2)
    dc = pd.read_parquet(C.DATA_CLEAN)
    items = [c for c in C.ANXIETY_ITEMS_ALL if c in dc.columns]
    y = feats[C.TARGET_CONT].to_numpy(float)
    itm = dc[items].to_numpy(float)

    assert len(feats) == 306, f"jumlah siswa tak terduga: {len(feats)} (harus 306)"
    assert N_FITUR == 76, f"jumlah fitur tak terduga: {N_FITUR} (harus 76)"
    assert len(items) == 20, f"item kecemasan valid harus 20, dapat {len(items)}: {items}"
    assert itm.shape == (306, 20), f"matriks item tak terduga: {itm.shape}"
    assert not np.isnan(itm).any(), "ada NaN pada matriks item kecemasan"
    assert not feats[C.NUMERIC_FEATURES + ITEM_COLS].isna().any().any(), "ada NaN pada fitur"
    selisih = float(np.abs(itm.mean(axis=1) - y).max())
    assert selisih < 1e-9, f"y != mean(20 item) (selisih {selisih}) — urutan baris tidak sama"
    lo, hi = C.ANXIETY_VALID_RANGE
    assert y.min() >= lo and y.max() <= hi, f"y di luar [{lo},{hi}]: [{y.min()},{y.max()}]"
    assert np.allclose(y / GRID, np.round(y / GRID)), "y off-grid — periksa granularitas 0,05"
    return feats, y, itm, items


# ---------- soft-label (fold-safe: hanya menerima array train) ----------

def _se_item(items: np.ndarray) -> np.ndarray:
    """se_i = SD antar-item (ddof=1) / sqrt(n_item) = galat baku mean item siswa i.
    Dihitung dari item saja (bukan dari y) -> aman dipakai pada baris test."""
    it = np.asarray(items, float)
    assert it.shape[1] >= 2, "butuh minimal 2 item untuk menghitung SD"
    return it.std(axis=1, ddof=1) / np.sqrt(it.shape[1])


def _var_true(y_train: np.ndarray, se_train: np.ndarray) -> float:
    """Var(y) - mean(se^2): varian 'benar' di luar noise pengukuran item."""
    return max(float(np.var(y_train, ddof=1)) - float(np.mean(se_train**2)), FLOOR_VAR)


def _alpha_label(var_true: float, se: np.ndarray) -> np.ndarray:
    """alpha_i = var_true / (var_true + se_i^2) — faktor kepercayaan label siswa i."""
    return np.clip(var_true / (var_true + np.asarray(se, float) ** 2), 0.0, 1.0)


def _soft_label(y_train: np.ndarray, se_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """y_tilde_i = mu + alpha_i (y_i - mu). Signature SENGAJA tidak punya parameter
    val/test — tidak ada jalur data dari baris luar train ke fungsi ini."""
    y = np.asarray(y_train, float)
    mu = float(y.mean())
    vt = _var_true(y, se_train)
    a = _alpha_label(vt, se_train)
    return mu + a * (y - mu), a, vt, mu


# ---------- decode ----------

def _decode_varian(pred_raw: np.ndarray, mu: float, lo: float, hi: float,
                   alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """(pred_snap, pred_proc). snap = clip -> grid 0,05 (tanpa shrink).
    proc = snap untuk ekor (pred_raw >= 2,5), shrink alpha untuk badan."""
    p = np.asarray(pred_raw, float)
    snap = _decode(p, mu, lo, hi, 1.0)
    # gate memakai pred_raw sesuai rancangan (nilai yang keluar tetap hasil clip+snap);
    # pred_proc bisa tidak kontinu di 2,5 — didokumentasikan, diterima.
    proc = np.where(p >= AMBANG_EKOR, snap, _decode(p, mu, lo, hi, alpha))
    return snap, proc


def _baris_fold(rec: dict) -> list[dict]:
    """Satu rekaman checkpoint fold -> baris per siswa (dipakai run_cv & self-test)."""
    p = np.array(rec["pred_raw"], float)
    snap, proc = _decode_varian(p, rec["mu"], rec["lo"], rec["hi"], rec["alpha_decode"])
    old = _decode(p, rec["mu"], rec["lo"], rec["hi"], rec["alpha_decode"])
    return [{"row_id": rec["idx"][i], "rep": rec["rep"], "fold": rec["fold"],
             "y": rec["y_true"][i], "pred_raw": float(p[i]), "pred_snap": float(snap[i]),
             "pred_proc": float(proc[i]), "pred_proc_old": float(old[i]),
             "se_i": rec["se_test"][i], "alpha_label": rec["alpha_label_test"][i]}
            for i in range(len(p))]


# ---------- sumber params E3 + fingerprint ----------

def _sumber_e3() -> tuple[str, Path]:
    """Fingerprint + folder checkpoint E3 (pemilik hiperparameter warisan)."""
    js = sorted(EXP3_DIR.glob("exp3_full_*.json"), key=lambda p: p.stat().st_mtime)
    assert js, (f"artefak eksperimen-3 tidak ditemukan di {EXP3_DIR} — "
                f"jalankan dulu: .venv/bin/python src/exp3_train.py")
    fp = json.loads(js[-1].read_text())["fingerprint"]
    ck = EXP3_DIR / "checkpoints" / f"exp3_{fp}"
    assert ck.is_dir(), f"checkpoint E3 {ck} tidak ada — jalankan exp3_train.py sampai tuntas"
    return fp, ck


def _rekaman_e3(ck: Path, repeats: int, folds: int) -> dict[tuple[int, int], dict]:
    """Muat seluruh checkpoint E3 (params + idx + inner_y) sekali, fail-fast bila kurang."""
    out: dict[tuple[int, int], dict] = {}
    for rep in range(repeats):
        for fold in range(folds):
            f_json = ck / f"ronde_r{rep}_f{fold}.json"
            assert f_json.exists(), (
                f"checkpoint E3 hilang: {f_json} — E4 mewarisi params per ronde dari E3; "
                f"jalankan ulang `.venv/bin/python src/exp3_train.py` (--repeats {repeats})")
            rec = json.loads(f_json.read_text())
            assert "params" in rec, f"field 'params' tidak ada di {f_json}"
            out[(rep, fold)] = rec
    return out


def _fingerprint(repeats: int, folds: int, data_sha: str, e3_fp: str,
                 params: dict[tuple[int, int], dict], items: list[str]) -> str:
    """Kunci run: hash cfg + hash file item/y + hash daftar params E3 yang dipakai."""
    cfg_sha = hashlib.sha256(json.dumps(
        {"e3": e3_fp, "items": items, "obj": OBJECTIVE, "base": C.BASE_XGB, "grid": GRID,
         "bins": BIN_EDGES, "cap": WEIGHT_CAP, "floor_var": FLOOR_VAR,
         "seed": C.RANDOM_SEED, "inner_folds": C.INNER_FOLDS,
         "ambang_ekor": AMBANG_EKOR, "varian": list(VARIAN),
         "params_e3": {f"r{r}_f{f}": p["params"] for (r, f), p in sorted(params.items())}},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    return hashlib.sha256(
        f"exp4|{repeats}|{folds}|{data_sha}|{cfg_sha}".encode()).hexdigest()[:12]


def _tulis(path: Path, obj: dict) -> None:
    """JSON atomik (tmp -> os.replace) + kunci urut; lihat catatan d di docstring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                             default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


# ---------- satu outer fold ----------

def _jalankan_fold(xtr: np.ndarray, xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
                   item_tr: np.ndarray, item_te: np.ndarray, idx_te: np.ndarray,
                   params: dict, rep: int, fold: int) -> dict:
    """Train -> pred_raw -> alpha dekode (inner-OOF) -> rekaman checkpoint.
    x/y sudah ter-encode; item_* = matriks 20 item baris train/test, idx_te = indeks asli test."""
    assert xtr.shape[1] == xte.shape[1], f"lebar fitur tidak sama: {xtr.shape[1]} vs {xte.shape[1]}"
    assert not np.isnan(xtr).any() and not np.isnan(xte).any(), "NaN masuk ke matriks fitur"

    se_tr, se_te = _se_item(item_tr), _se_item(item_te)
    y_tilde, a_tr, var_true, mu = _soft_label(ytr, se_tr)
    w = bobot(ytr)  # bobot dari y RAW (satu-prinsip-perubahan: bukan y_tilde)
    pred_raw = _fit(params, xtr, y_tilde, OBJECTIVE, w).predict(xte)
    lo, hi = float(ytr.min()), float(ytr.max())

    # alpha dekode: inner-OOF train-side, seed konvensi E3, target latih = y_tilde inner-train
    inner = KFold(C.INNER_FOLDS, shuffle=True,
                  random_state=C.RANDOM_SEED + 1000 * rep + fold)
    splits = list(inner.split(xtr))
    assert len(splits) == C.INNER_FOLDS, f"inner split != {C.INNER_FOLDS}"
    ip, iy = [], []
    for jtr, jva in splits:
        yj_t, _, _, _ = _soft_label(ytr[jtr], se_tr[jtr])
        # bobot outer-train diiris [jtr] = konvensi E3 (exp3_train.py:163,172)
        mj = _fit(params, xtr[jtr], yj_t, OBJECTIVE, w[jtr])
        ip.extend(mj.predict(xtr[jva]).tolist())
        iy.extend(ytr[jva].tolist())

    return {
        "rep": rep, "fold": fold, "idx": [int(i) for i in idx_te],
        "y_true": [float(v) for v in yte], "pred_raw": [float(v) for v in pred_raw],
        "mu": mu, "lo": lo, "hi": hi,
        "se_test": [float(v) for v in se_te],
        "alpha_label_test": [float(v) for v in _alpha_label(var_true, se_te)],
        "var_true": var_true,
        "alpha_label_train": [float(v) for v in a_tr],
        "alpha_decode": _fit_alpha([(np.array(ip), np.array(iy), mu)]),
        "label_diag": {
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "mean_alpha_label_train": float(a_tr.mean()), "median_alpha_label_train": float(np.median(a_tr)),
            "mean_se_train": float(se_tr.mean()), "mean_se_test": float(se_te.mean()),
            "mean_y_tilde_train": float(y_tilde.mean()),
        },
        "params": params, "inner_p": ip, "inner_y": iy,
    }


# ---------- CV utama ----------

def run_cv(repeats: int, folds: int, fresh: bool) -> dict:
    e3_fp, e3_ck = _sumber_e3()
    e3 = _rekaman_e3(e3_ck, repeats, folds)
    feats, y, itm, items = _muat_data()

    data_sha = hashlib.sha256(FEATURES_EXP2.read_bytes() + C.DATA_CLEAN.read_bytes()).hexdigest()[:16]
    fp = _fingerprint(repeats, folds, data_sha, e3_fp, e3, items)
    ckpt = CKPT_DIR / f"exp4_{fp}"
    if fresh and ckpt.exists():
        # termasuk .tmp yatim dari tulis terputus
        for f in ckpt.glob("ronde_*"):  # hanya checkpoint run INI
            f.unlink()
    ckpt.mkdir(parents=True, exist_ok=True)

    rounds: list[dict] = []
    t0 = time.time()
    for rep in range(repeats):
        f_json = ckpt / f"ronde_r{rep}.json"
        if f_json.exists():  # resume: lompati ronde yang sudah jadi
            rounds.extend(json.loads(f_json.read_text())["folds"])
            continue
        outer = KFold(folds, shuffle=True, random_state=C.RANDOM_SEED + rep)  # identik E3
        recs_rep = []
        for fold, (itr, ite) in enumerate(outer.split(feats)):
            assert len(itr) + len(ite) == len(feats), "partisi outer fold tidak menutup semua baris"
            e3_rec = e3[(rep, fold)]
            assert list(e3_rec["idx"]) == [int(i) for i in ite], (
                f"partisi fold E4 != E3 di ronde {rep} fold {fold} — fold harus identik E3")
            tr, te = feats.iloc[itr], feats.iloc[ite]
            xtr, xte = _encode(tr, te, "76")
            rec = _jalankan_fold(xtr, xte, y[itr], y[ite], itm[itr], itm[ite], ite,
                                 e3_rec["params"], rep, fold)
            assert np.allclose(rec["inner_y"], e3_rec["inner_y"]), (
                f"inner-OOF y != checkpoint E3 di ronde {rep} fold {fold} — seed inner berubah")
            recs_rep.append(rec)
            print(f"[exp4] rep{rep + 1}/{repeats} fold{fold + 1}/{folds} "
                  f"MAE={_mae(y[ite], np.array(rec['pred_raw'])):.4f} "
                  f"alpha_dec={rec['alpha_decode']:.3f} var_true={rec['var_true']:.4f} "
                  f"alpha_lab={rec['label_diag']['mean_alpha_label_train']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        _tulis(f_json, {"rep": rep, "folds": recs_rep})
        rounds.extend(recs_rep)

    assert len(rounds) == repeats * folds, f"jumlah fold tak terduga: {len(rounds)}"
    assert len({r["fold"] for r in rounds}) == folds, "ada fold yang tidak terisi"

    df = pd.DataFrame([baris for r in rounds for baris in _baris_fold(r)])
    assert df.shape[0] == repeats * len(y), f"baris per-siswa tak terduga: {df.shape[0]}"
    assert df["row_id"].nunique() == len(y), "row_id tidak unik per siswa"
    ekor = df["y"] >= AMBANG_EKOR
    alpha_pooled = _fit_alpha([(np.array(r["inner_p"]), np.array(r["inner_y"]), r["mu"])
                               for r in rounds])

    hasil = {
        "eksperimen": 4, "mode": "exp4_full", "fingerprint": fp, "basis_e3": e3_fp,
        "sumber_params_e3": str(e3_ck), "repeats": repeats, "folds": folds,
        "objective": OBJECTIVE, "n_fitur": N_FITUR, "grid": GRID,
        "floor_var": FLOOR_VAR, "ambang_ekor": AMBANG_EKOR,
        "skema_berat": f"min({WEIGHT_CAP}, sqrt(ref/n_bin)) per bin {BIN_EDGES}, dari y RAW, mean-1",
        "label": {
            "var_true_per_fold": [r["var_true"] for r in rounds],
            "alpha_label_mean_per_fold": [r["label_diag"]["mean_alpha_label_train"] for r in rounds],
            "alpha_label_median_per_fold": [r["label_diag"]["median_alpha_label_train"] for r in rounds],
            "alpha_label_test_mean_per_fold": [
                float(np.mean(r["alpha_label_test"])) for r in rounds],
            "alpha_decode_per_fold": [r["alpha_decode"] for r in rounds],
            "alpha_decode_pooled": alpha_pooled,
            "mean_se_train": float(np.mean([r["label_diag"]["mean_se_train"] for r in rounds])),
            "catatan": "alpha label dipakai di target latih (y_tilde); alpha decode dipakai di post-processing",
        },
        "catatan": ("params warisan E3 per ronde (tanpa Optuna); pred_proc = snap untuk ekor "
                    "pred_raw>=2,5, shrink alpha untuk badan; pred_proc_old = shrink semua "
                    "(aturan dekode E3 dengan alpha per-fold — bukan salinan pred_proc E3 "
                    "yang alpha-nya terpool 0,97)"),
    }
    for nama in VARIAN:
        kol = df[nama].to_numpy(float)
        err = (df["y"] - kol).abs()
        per_siswa = df.assign(e=err).groupby("row_id")["e"].mean()
        ens = df.groupby("row_id").agg(y=("y", "first"), pred=(nama, "mean"))
        ens_err = (ens["y"] - ens["pred"]).abs()
        hasil[nama] = {
            "mae_pooled": _mae(df["y"].to_numpy(float), kol),
            "rmse_pooled": _rmse(df["y"].to_numpy(float), kol),
            "mae_ekor": float(err[ekor].mean()),
            "mae_non_ekor": float(err[~ekor].mean()),
            "mae_per_siswa_rata2": float(per_siswa.mean()),
            "mae_ensemble": float(ens_err.mean()),
            "mae_ensemble_ekor": float(ens_err[ens["y"] >= AMBANG_EKOR].mean()),
            "mae_per_repeat": [float(v) for v in df.assign(e=err).groupby("rep")["e"].mean()],
        }
    # verdict pra-registrasi: kriteria dikunci SEBELUM run, angka referensi dari artefak E3
    e3_js = json.loads((EXP3_DIR / f"exp3_full_{e3_fp}.json").read_text())["pred_proc"]
    ref_p, ref_e = float(e3_js["mae_pooled"]), float(e3_js["mae_ekor"])
    p4 = hasil["pred_proc"]
    lolos = ((p4["mae_pooled"] < ref_p) or (p4["mae_ekor"] < ref_e - 0.003)) \
        and (p4["mae_pooled"] <= ref_p + 0.003) and (p4["mae_ekor"] <= ref_e + 0.003)
    hasil["verdict_prareg"] = {
        "ref_e3": {"mae_pooled": ref_p, "mae_ekor": ref_e,
                   "sumber": f"exp3_full_{e3_fp}.json -> pred_proc"},
        "kriteria": "mae_pooled < ref ATAU mae_ekor < ref-0,003; dan tidak menaikkan sisi lain > 0,003",
        "delta_pooled": float(p4["mae_pooled"] - ref_p),
        "delta_ekor": float(p4["mae_ekor"] - ref_e),
        "lolos": bool(lolos),
    }
    # peta ringkas kepala hasil (mesin-baca cepat; angkanya sama dengan blok di atas)
    for k, met in (("mae_pooled", "mae_pooled"), ("mae_ekor", "mae_ekor"),
                   ("mae_non_ekor", "mae_non_ekor"), ("mae_ensemble", "mae_ensemble")):
        hasil[k] = {v: hasil[v][met] for v in VARIAN}

    _tulis(EXP4_DIR / f"exp4_full_{fp}.json", hasil)
    df.to_csv(EXP4_DIR / f"exp4_full_{fp}_per_siswa.csv", index=False)
    print(f"\nSELESAI [exp4] -> {EXP4_DIR / f'exp4_full_{fp}.json'}")
    print(json.dumps({k: hasil[k] for k in
                      ("mae_pooled", "mae_ekor", "mae_non_ekor", "mae_ensemble")}, indent=2))
    return hasil


# ---------- self-test (data sintetis, tanpa data proyek) ----------

def _self_test() -> None:
    rng = np.random.default_rng(0)

    # (a) se_i = 0 -> alpha_i = 1 dan y_tilde = y
    it = rng.integers(1, 5, size=(30, 20)).astype(float)
    it[3] = 4.0                       # jawaban seragam: SD = 0 -> se = 0
    y = it.mean(axis=1)
    yt, a, vt, mu = _soft_label(y, _se_item(it))
    assert _se_item(it)[3] == 0.0, "baris konstan harus se=0"
    assert a[3] == 1.0, f"se=0 harus alpha=1, dapat {a[3]}"
    assert yt[3] == y[3], "se=0 harus y_tilde = y"

    # (b) regime non-degenerate: var_true > FLOOR, alpha di (0,1), shrink ke arah mu
    lvl = np.linspace(1.0, 4.0, 40)
    it2 = np.clip(lvl[:, None] + rng.normal(0.0, 0.6, size=(40, 20)), 1.0, 4.0)
    it2[0] = 3.0                        # baris konstan: se = 0 -> alpha = 1
    y2 = it2.mean(axis=1)
    yt2, a2, vt2, mu2 = _soft_label(y2, _se_item(it2))
    assert vt2 > FLOOR_VAR, "fixture (b) harus di regime var_true > FLOOR"
    assert a2[0] == 1.0, "se=0 harus alpha=1"
    k = int(np.argmax(_se_item(it2)))
    assert 0.0 < a2[k] < 1.0, f"alpha siswa paling berisik harus di (0,1), dapat {a2[k]}"
    assert abs(yt2[k] - mu2) < abs(y2[k] - mu2), "shrink harus mendekatkan y ke mu"
    assert (min(mu2, y2[k]) <= yt2[k]) and (yt2[k] <= max(mu2, y2[k])), "y_tilde harus di antara mu dan y"

    # (c) y tanpa varian -> var_true = FLOOR (bukan 0, supaya tidak kolaps ke alpha=0)
    yk = np.full(25, 2.0)
    assert _var_true(yk, _se_item(it[:25])) == FLOOR_VAR, "var_y < mean(se^2) harus kena FLOOR"
    assert _var_true(np.full(25, 2.0), np.zeros(25)) == FLOOR_VAR, "y konstan harus kena FLOOR"

    # (d) fold-safety: mengubah baris val tidak mengubah output pada baris train
    itr, iva = np.arange(0, 18), np.arange(18, 30)
    a_awal = _soft_label(y[itr], _se_item(it[itr]))
    it_nanti = it.copy()
    it_nanti[iva] = 1.0                      # fold test dibalik total -> y ikut berubah
    a_nanti = _soft_label(it_nanti.mean(axis=1)[itr], _se_item(it_nanti[itr]))
    for u, v in zip(a_awal, a_nanti):
        assert np.allclose(u, v), "fungsi soft-label bocor data fold val"

    # (e) decode: ekor tanpa shrink, alpha=1 identik dengan E3, badan ikut shrink
    p = np.array([1.2, 1.6, 2.49, 2.5, 3.4, 9.0])
    lo, hi = 1.0, 3.5
    snap, proc = _decode_varian(p, mu=2.0, lo=lo, hi=hi, alpha=0.5)
    assert np.allclose(snap[p >= AMBANG_EKOR], proc[p >= AMBANG_EKOR]), "ekor harus tanpa shrink"
    assert np.all(snap >= lo) and np.all(snap <= hi), "clip rentang gagal"
    assert np.allclose(snap / GRID, np.round(snap / GRID)), "snap grid gagal"
    assert snap[-1] == hi, "clip atas harus menahan prediksi 9.0"
    _, proc1 = _decode_varian(p, mu=2.0, lo=lo, hi=hi, alpha=1.0)
    old1 = _decode(p, mu=2.0, lo=lo, hi=hi, alpha=1.0)
    assert np.allclose(proc1, old1), "alpha=1 harus identik dengan perilaku E3"
    assert abs(proc[1] - 1.8) < 1e-12, f"badan: 2 + 0,5*(1,6-2) harus 1,8, dapat {proc[1]}"
    assert abs(proc[1] - 2.0) < abs(p[1] - 2.0), "shrink alpha<1 harus mendekatkan ke mu"

    # (f) bobot bin: jumlah sama -> bobot sama (null-invariant terhadap index bin)
    yb = np.array([1.1] * 4 + [1.6] * 4 + [2.1] * 4 + [3.4] * 4)
    w = bobot(yb)
    assert abs(w.mean() - 1.0) < 1e-9, "bobot harus mean 1"
    assert abs(w[0] - w[5]) < 1e-12 and abs(w[0] - w[10]) < 1e-12, "bin sama jumlah harus bobot sama"
    assert abs(float(w[yb == 3.4].mean()) - float(w[yb == 1.1].mean())) < 1e-12, "bin n=4 semua sama"

    # (g) end-to-end: fit XGB + sample_weight pada data sintetis, 1 rep x 2 fold
    params = {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.2, "min_child_weight": 1.0,
              "subsample": 1.0, "colsample_bytree": 1.0, "gamma": 0.0,
              "reg_alpha": 0.0, "reg_lambda": 1.0}
    x = rng.normal(size=(40, 6))
    itm_s = rng.integers(1, 5, size=(40, 20)).astype(float)
    ys = itm_s.mean(axis=1)
    recs = []
    for fold, (tr, te) in enumerate(KFold(2, shuffle=True, random_state=C.RANDOM_SEED).split(x)):
        rec = _jalankan_fold(x[tr], x[te], ys[tr], ys[te], itm_s[tr], itm_s[te], te,
                             params, 0, fold)
        assert len(rec["idx"]) == len(te) and len(rec["pred_raw"]) == len(te)
        assert not np.isnan(rec["pred_raw"]).any(), "NaN pada pred_raw"
        recs.append(rec)
    rows = [b for r in recs for b in _baris_fold(r)]
    assert len(rows) == 40, f"baris gabungan harus 40, dapat {len(rows)}"
    assert set(rows[0]) == {"row_id", "rep", "fold", "y", "pred_raw", "pred_snap", "pred_proc",
                            "pred_proc_old", "se_i", "alpha_label"}, "skema kolom per-siswa berubah"
    assert not any(np.isnan(b["pred_proc"]) or np.isnan(b["pred_snap"]) for b in rows), "NaN pada decode"
    with tempfile.TemporaryDirectory() as tmpd:  # checkpoint JSON, tanpa menyentuh repo
        f_json = Path(tmpd) / "ronde_r0.json"
        _tulis(f_json, {"rep": 0, "folds": recs})
        assert f_json.exists() and not f_json.with_suffix(".tmp").exists(), "tulis atomik gagal"
        muat = json.loads(f_json.read_text())
        assert len(muat["folds"]) == 2 and muat["folds"][0]["alpha_decode"] >= 0.0

    print("self-test OK: soft-label (se=0/alpha-di-(0,1)/FLOOR/fold-safe), decode ekor, "
          "bobot, end-to-end XGB 1rep x 2fold + checkpoint JSON")


def main() -> None:
    ap = argparse.ArgumentParser(description="Eksperimen 4 — soft-label Empires-Bayes + decode ekor")
    ap.add_argument("--repeats", type=int, default=C.OUTER_REPEATS)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    EXP4_DIR.mkdir(parents=True, exist_ok=True)
    run_cv(a.repeats, C.OUTER_FOLDS, a.fresh)


if __name__ == "__main__":
    main()