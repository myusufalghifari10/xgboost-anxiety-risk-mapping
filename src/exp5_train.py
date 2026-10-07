"""Eksperimen 5 — "PERANG WARP": loss asimetris anti-warp + monotone constraints.

Temuan terverifikasi yang menjadi alasan file ini ada (diagnosis dari artefak E1-E4):
  * WARP SENTRAL: regresi pred_ensemble ~ y pada model champion (E3) punya slope
    hanya 0,609 (ideal 1,0). Angka ini DIUKUR (bukan dari ingatan) dari
    outputs/exp3/exp3_full_91945185763d_per_siswa.csv -> rata-rata pred_proc per row_id,
    slope = polyfit(y, pred, 1)[0]; nilai yang sama kini disimpan sebagai
    ref_e3.slope_ensemble di artefak hasil E5. Prediksi siswa cemas ditarik ke tengah.
  * Bias per pita y (E3): [1,0-1,5) +0,219 | [2,0-2,5) -0,077 | [2,5-3,0) -0,307
    | [3,0-3,5) -0,539 (n=9; MAE = |bias| -> error ekor MURNI bias, bukan noise).
  * Dekomposisi APROKSIMATIF (berbasis MAE, bukan RMSE): SD(error) ~ 0,283 =
    MAE pooled 0,2257 / 0,7979 (asumsi error normal); kontribusi warp ~ (1-0,609)^2 x
    Var(y)=0,218 -> 0,033 dari total ~0,080 varian. Angka dari RMSE^2 pooled E3 (0,0928)
    BERBEDA karena distribusi error tidak normal (dan berbias) — dipakai hanya sebagai
    indikasi kualitatif, bukan identitas. Mematikan warp adalah SATU-SATUNYA jalan ke MAE 0,1x
    dan sekaligus menyelamatkan SHAP wilayah siswa stres (yang kini ter-attenuasi ~40%).

E5 menyerang warp di sisi TRAINING (bukan post-hoc, supaya fungsi model ikut lurus):
  1. CUSTOM OBJECTIVE ASIMETRIS (inti): L_i = tau_i*max(e,0) + (1-tau_i)*max(-e,0)
     dengan e = y - pred dan tau_i = clip(0,5 + gamma*(y_i - y_mid); 0,20; 0,80),
     y_mid = mean label FOLD LATIH (fold-safe). Di atas y_mid: underprediksi
     dipenalti lebih berat (mendorong prediksi naik); di bawah y_mid: overprediksi
     dipenalti lebih berat (menarik prediksi turun). Kedua ujung diluruskan.
     grad = -tau_i (e>0) / +(1-tau_i) (e<0) / 0 (e=0); hess = 1 (gaya L1);
     bobot bin E3 (w) mengalikan grad & hess secara EKSPLISIT (lihat catatan b).
  2. MONOTONE CONSTRAINTS pada 3 fitur berarah dari temuan: f_coping_negatif (+1),
     f_digital (+1), f_duk_keluarga (-1) — respons model tidak boleh melipat di
     wilayah data jarang. Tidak menyentuh f_perilaku_sehat (kontribusi nol terbukti).
  3. GATE PRA-REGISTRASI berbasis warp-index (slope) + MAE ekor + MAE total (di bawah).

Fondasi TIDAK berubah (satu-prinsip-perubahan): 76 fitur, bobot bin fold-safe dari
y raw (bobot(ytr), diiris [jtr] di inner — konvensi E3), hyperparameter diwarisi
per-fold dari checkpoint E3 (Tanpa Optuna), fold identik E3 (assert idx + inner_y),
decode keluarga E3 (clip -> shrink badan -> snap grid; ekor tanpa shrink).

PRA-REGISTRASI (mesin: verdict_prareg, dikunci sebelum run) — lolos HANYA jika SEMUA:
  (1) slope_warp (koef. regresi linier pred_ensemble_proc ~ y) >= 0,78   [sekarang 0,609]
  (2) MAE ensemble proc <= 0,2000                                        [E3: 0,22572]
  (3) MAE ensemble ekor proc <= 0,3200                                   [E3: 0,37070]
  (4) MAE non-ekor pooled proc <= ref_badan_E3 + 0,003                    [E3 pooled: 0,21030]
Referensi dibaca dari artefak outputs/exp3/exp3_full_*.json (bukan dari ingatan).
Proyeksi jujur: total 0,17-0,20, ekor 0,29-0,33, slope 0,78-0,85.

CATATAN PENYIMPANGAN (semua disengaja & dicatat):
  a) API: XGBoost sklearn TIDAK menerima objective callable + sample_weight
     (ValueError terverifikasi di xgboost 3.4.1) -> pakai API FUNKSIONAL
     xgb.train(params, DMatrix, obj=...). Paritas terverifikasi: fit reg:absoluteerror
     identik bit (max|diff| = 0,0) antara kedua API pada seed/params sama.
  b) Bobot diterapkan MANUAL di grad/hess (DMatrix dibuat TANPA weight) supaya
     perilaku bobot terdefinisi penuh oleh kode kita, tidak bergantung pada
     penerapan bobot internal xgboost untuk objective Python. Self-test (e)
     membuktikan bobot memang mengubah hasil fit.
  c) gamma dipilih per outer fold dari MAE pooled inner-OOF (grid {0,15; 0,25; 0,35},
     tie -> gamma terkecil, deterministik). TANPA seleksi berbasis ekor
     (anti winner's-curse — pelajaran D3).
  d) Decode badan = clip -> shrink alpha -> snap (urutan _decode, konsisten E4,
     hasil tetap di lattice 0,05); alpha di-fit per fold dari inner-OOF fold itu
     dengan mu = mean(ytr) (mu yang sama dipakai saat decode test).
  e) ref_badan pada gate adalah MAE non-ekor POOLED milik E3 yang terbaca dari
     JSON-nya (0,21030), bukan angka 0,2020 (angka itu basis ensemble per-siswa
     dari analisis, tidak ada di artefak E3) — semua referensi dari artefak.
  f) Fingerprint mencakup cfg (seed, inner_folds, repeats, folds, grid gamma,
     batas tau, nama+arah 3 fitur mono), bytes data, fingerprint E3, params E3 per fold,
     hash kode sumber exp5_train.py, dan C.BASE_XGB (edit kode/params dasar = fingerprint
     baru, sehingga resume tak pernah mencampur kode lama dan baru). Booster disimpan
     di models/<fingerprint>/ (tidak pernah tertukar antar run); artefak JSON ditulis
     dengan allow_nan=False (NaN/Inf = error, bukan diam-diam masuk artefak); CSV
     per-siswa ditulis atomik; dan SEMUA ronde (termasuk hasil resume) divalidasi ulang
     terhadap checkpoint E3 (idx + inner_y) sebelum metrik dihitung.
  g) base_score di-set EKSPLISIT ke mean(y fold latih) di _fit_custom: dengan obj= custom
     XGBoost TIDAK mengestimasi base_score dari label dan jatuh ke default 0,5
     (terverifikasi via save_config) -> tanpa ini E5 start dari intercept 0,5, bukan ~1,97.
  h) Skala gradien dinormalisasi (1/mean(tau), |grad| rata-rata ~ 1) agar ukuran langkah
     per ronde sebanding dengan reg:absoluteerror E3 (|grad| = 1); yang diuji adalah
     ASIMETRI loss, bukan penskalaan learning-rate efektif. Catatan jujur: implementasi
     Newton-L1 custom ini TIDAK identik-bit dengan native reg:absoluteerror (terukur
     berbeda di data sintetis) -> selisih E5 vs E3 = efek asimetri + efek parameterisasi
     objective custom, dan itu dilaporkan apa adanya (bukan diklaim murni asimetri).

Pemakaian (EKSEKUSI = YUSUF):
  .venv/bin/python src/exp5_train.py --self-test       # data sintetis
  .venv/bin/python src/exp5_train.py --repeats 3       # gerbang cepat
  .venv/bin/python src/exp5_train.py                   # full 10 ronde
  .venv/bin/python src/exp5_shap.py                    # produk SHAP subgrup (setelah run)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from exp2_features import FEATURES_EXP2, ITEM_COLS  # noqa: E402
from exp2_train import GRID, _decode, _encode, _fit_alpha, _mae, _rmse  # noqa: E402
from exp3_train import AMBANG_EKOR, BIN_EDGES, WEIGHT_CAP, bobot  # noqa: E402

EXP3_DIR = C.ROOT / "outputs" / "exp3"
EXP5_DIR = C.ROOT / "outputs" / "exp5"
CKPT_DIR = EXP5_DIR / "checkpoints"
MODELS_BASE = EXP5_DIR / "models"  # booster di-namespace per fingerprint: MODELS_BASE/<fp>/

GAMMA_GRID = (0.15, 0.25, 0.35)   # kekuatan kemiringan tau (grid pre-registrasi)
TAU_MIN, TAU_MAX = 0.20, 0.80     # batas clip tau_i
MONO = {"f_coping_negatif": 1, "f_digital": 1, "f_duk_keluarga": -1}
VARIAN = ("pred_raw", "pred_snap", "pred_proc")
N_FITUR = len(C.NUMERIC_FEATURES) + len(ITEM_COLS) + len(C.CATEGORICAL_FEATURES)  # 76
GATE = {"slope": 0.78, "total": 0.2000, "ekor": 0.3200, "badan_slack": 0.003}


# ---------- loss asimetris (fungsi murni, diuji --self-test) ----------

def _tau(y: np.ndarray, y_mid: float, gamma: float) -> np.ndarray:
    """tau_i = clip(0,5 + gamma*(y_i - y_mid); 0,20; 0,80) — kemiringan per baris."""
    return np.clip(0.5 + gamma * (np.asarray(y, float) - y_mid), TAU_MIN, TAU_MAX)


def _grad_hess(pred: np.ndarray, y: np.ndarray, tau: np.ndarray, w: np.ndarray,
               skala: float = 1.0):
    """grad = dL/dpred; hess = 1 (gaya L1); keduanya dikalikan bobot bin w.

    L = tau*max(e,0) + (1-tau)*max(-e,0), e = y - pred
      e > 0 (underprediksi) -> grad = -tau   (negatif -> update booster menaikkan pred)
      e < 0 (overprediksi)  -> grad = +(1-tau)
    skala mengalikan grad SAJA (normalisasi rata-rata |grad| ~ 1; catatan h)
    supaya ukuran langkah per ronde sebanding dengan reg:absoluteerror E3.
    """
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    assert pred.ndim == 1 and y.ndim == 1, (
        f"pred/y harus 1D (hindari broadcast senyap), dapat {pred.shape}/{y.shape}")
    assert len(pred) == len(y) == len(tau) == len(w), (
        f"panjang tidak sejajar: pred={len(pred)} y={len(y)} tau={len(tau)} w={len(w)}")
    e = y - pred
    grad = np.where(e > 0, -tau, np.where(e < 0, 1.0 - tau, 0.0)) * (w * skala)
    hess = np.ones_like(pred) * w
    return grad, hess


def _make_obj(tau: np.ndarray, w: np.ndarray, skala: float):
    """Closure objective untuk xgb.train. tau/w/skala = urutan BARIS LATIH (dibuat per fit;
    xgb.train hanya memanggil objective dengan DMatrix latih, tanpa evals)."""
    assert len(tau) == len(w), f"panjang tau ({len(tau)}) != w ({len(w)})"
    assert np.isfinite(skala) and skala > 0, f"skala tidak valid: {skala}"

    def obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        assert len(y) == len(tau), f"panjang label ({len(y)}) != tau ({len(tau)}) — fitur campur"
        return _grad_hess(preds, y, tau, w, skala)
    return obj


def _fit_custom(params_e3: dict, x: np.ndarray, y: np.ndarray, gamma: float,
                w: np.ndarray, mono_vec: np.ndarray | None = None) -> xgb.Booster:
    """Fit booster dengan objective asimetris E5. params_e3 = hyperparameter warisan E3
    (field 'params' checkpoint; n_estimators dipisah jadi num_boost_round)."""
    assert int(params_e3["n_estimators"]) > 0, "n_estimators tidak valid"
    y_mid = float(np.mean(y))
    tau = _tau(y, y_mid, gamma)
    # Normalisasi skala gradien (catatan h): rata-rata |grad| ~ 1 setara parameterisasi
    # reg:absoluteerror E3 (|grad| = 1, hess = 1) -> yang diuji ASIMETRI loss, bukan
    # penskalaan learning-rate efektif. hess TETAP ones*w.
    skala = 1.0 / float(np.mean(tau))
    assert 1.2 < skala < 3.0, (
        f"skala gradien {skala:.3f} di luar rentang wajar (mean tau {np.mean(tau):.3f})")
    bp = {**C.BASE_XGB,
          **{k: v for k, v in params_e3.items() if k != "n_estimators"},
          "objective": "reg:squarederror",  # placeholder INERT (diuji self-test e5)
          # obj= custom -> XGBoost TIDAK mengestimasi base_score dari label (jatuh ke 0,5);
          # E3 (reg:absoluteerror) memakai auto ~ mean(y) -> set eksplisit (catatan g).
          "base_score": float(np.mean(y))}
    if mono_vec is not None:
        assert len(mono_vec) == x.shape[1], (
            f"vektor monotone sepanjang {len(mono_vec)} != lebar fitur {x.shape[1]}")
        bp["monotone_constraints"] = "(" + ",".join(str(int(v)) for v in mono_vec) + ")"
    dm = xgb.DMatrix(x, label=y)  # TANPA weight: bobot sudah di grad/hess (catatan b)
    return xgb.train(bp, dm, num_boost_round=int(params_e3["n_estimators"]),
                     obj=_make_obj(tau, w, skala))


def _vektor_monotone(n_cols: int, mono: dict, num_cols: list[str]) -> np.ndarray:
    """Vektor constraint sepanjang kolom hasil encode: +1/-1 tepat pada nama fitur
    numerik (posisi = indeks di num_cols; kolom encode = numerik dulu, baru one-hot),
    0 di tempat lain. Fail-fast bila nama tidak ada."""
    vec = np.zeros(n_cols, dtype=int)
    for nama, arah in mono.items():
        assert nama in num_cols, f"fitur monotone '{nama}' tidak ada di daftar fitur numerik"
        i = num_cols.index(nama)
        assert i < n_cols, f"posisi '{nama}' ({i}) di luar lebar fitur ({n_cols})"
        vec[i] = int(arah)
    assert int(np.count_nonzero(vec)) == len(mono), "vektor monotone harus tepat 3 nonzero"
    assert set(np.unique(vec)) <= {-1, 0, 1}, "arah constraint harus -1/0/1"
    return vec


def _pilih_gamma(caches: dict[float, tuple[np.ndarray, np.ndarray]]) -> tuple[float, list[float]]:
    """gamma dengan MAE pooled inner-OOF terkecil; tie -> gamma terkecil (deterministik).
    caches[g] = (ip, iy) dari inner-OOF gamma g. Tanpa seleksi berbasis ekor."""
    maes = []
    for g in GAMMA_GRID:
        ip, iy = caches[g]
        assert len(ip) == len(iy) > 0, f"pasangan inner-OOF kosong untuk gamma={g}"
        maes.append(_mae(iy, ip))
    best = min(GAMMA_GRID, key=lambda g: (maes[GAMMA_GRID.index(g)], g))
    return float(best), [float(m) for m in maes]


# ---------- decode (varian prediksi; pola E4) ----------

def _decode_varian(pred_raw: np.ndarray, mu: float, lo: float, hi: float,
                   alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """(pred_snap, pred_proc). snap = clip + grid (tanpa shrink).
    proc: ekor (pred_raw >= 2,5) tanpa shrink; badan shrink alpha (urutan
    clip -> shrink -> snap; hasil tetap di lattice 0,05 — konsisten E4, catatan d)."""
    p = np.asarray(pred_raw, float)
    snap = _decode(p, mu, lo, hi, 1.0)
    proc = np.where(p >= AMBANG_EKOR, snap, _decode(p, mu, lo, hi, alpha))
    return snap, proc


def _slope_warp(y: np.ndarray, pred: np.ndarray) -> float:
    """Warp-index: koefisien kemiringan regresi linier pred ~ y (ideal 1,0)."""
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    assert len(y) == len(pred) >= 2, "slope butuh >= 2 titik"
    assert np.std(y) > 0, "y konstan — slope tidak terdefinisi"
    return float(np.polyfit(y, pred, 1)[0])


def _se_item(items: np.ndarray) -> np.ndarray:
    """se_i = SD antar-item (ddof=1)/sqrt(n_item) — galat baku mean item (konvensi E4).
    Dipakai HANYA sebagai kolom diagnostik di CSV per-siswa (tidak masuk model)."""
    it = np.asarray(items, float)
    assert it.ndim == 2 and it.shape[1] > 1, "matriks item harus 2D dengan >= 2 item"
    return it.std(axis=1, ddof=1) / np.sqrt(it.shape[1])


def _verdict_prareg(m: dict, ref: dict) -> dict:
    """Gate mesin pra-registrasi. m = metrik pred_proc E5, ref = referensi E3 (artefak).
    Kriteria dikunci di docstring; fungsi ini murni (diuji self-test h)."""
    c1 = m["slope_ensemble"] >= GATE["slope"]
    c2 = m["mae_ensemble"] <= GATE["total"]
    c3 = m["mae_ensemble_ekor"] <= GATE["ekor"]
    c4 = m["mae_non_ekor"] <= ref["mae_non_ekor"] + GATE["badan_slack"]
    return {
        "ref_e3": ref,
        "kriteria": ("lolos HANYA jika SEMUA: slope>=0,78 DAN mae_ensemble<=0,2000 DAN "
                     "mae_ensemble_ekor<=0,3200 DAN mae_non_ekor<=ref_badan+0,003"),
        "nilai": {"slope_ensemble": m["slope_ensemble"],
                  "mae_ensemble": m["mae_ensemble"],
                  "mae_ensemble_ekor": m["mae_ensemble_ekor"],
                  "mae_non_ekor": m["mae_non_ekor"]},
        "delta_vs_ambang": {"slope": m["slope_ensemble"] - GATE["slope"],
                            "total": m["mae_ensemble"] - GATE["total"],
                            "ekor": m["mae_ensemble_ekor"] - GATE["ekor"],
                            "badan": m["mae_non_ekor"] - (ref["mae_non_ekor"] + GATE["badan_slack"])},
        "delta_vs_e3": {"mae_ensemble": m["mae_ensemble"] - ref["mae_ensemble"],
                        "mae_ensemble_ekor": m["mae_ensemble_ekor"] - ref["mae_ensemble_ekor"],
                        "slope_ensemble": m["slope_ensemble"] - ref["slope_ensemble"]},
        "syarat": {"c1_slope": bool(c1), "c2_total": bool(c2),
                   "c3_ekor": bool(c3), "c4_badan": bool(c4)},
        "lolos": bool(c1 and c2 and c3 and c4),
    }


# ---------- data (identik E4) ----------

def _muat_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
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
    for nama in MONO:
        assert nama in feats.columns, f"fitur monotone '{nama}' tidak ada di features_exp2"
        assert nama not in C.CATEGORICAL_FEATURES, f"'{nama}' kategorikal — arah mono tak sah"
    return feats, y, itm, items


# ---------- sumber E3 + fingerprint (pola E4) ----------

def _sumber_e3() -> tuple[str, Path]:
    js = sorted(EXP3_DIR.glob("exp3_full_*.json"), key=lambda p: p.stat().st_mtime)
    assert js, (f"artefak eksperimen-3 tidak ditemukan di {EXP3_DIR} — "
                f"jalankan dulu: .venv/bin/python src/exp3_train.py")
    fp = json.loads(js[-1].read_text())["fingerprint"]
    ck = EXP3_DIR / "checkpoints" / f"exp3_{fp}"
    assert ck.is_dir(), f"checkpoint E3 {ck} tidak ada — jalankan exp3_train.py sampai tuntas"
    return fp, ck


def _rekaman_e3(ck: Path, repeats: int, folds: int) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for rep in range(repeats):
        for fold in range(folds):
            f_json = ck / f"ronde_r{rep}_f{fold}.json"
            assert f_json.exists(), (
                f"checkpoint E3 hilang: {f_json} — E5 mewarisi params per fold dari E3; "
                f"jalankan ulang `.venv/bin/python src/exp3_train.py` (--repeats {repeats})")
            rec = json.loads(f_json.read_text())
            for k in ("params", "idx", "inner_y"):
                assert k in rec, f"field '{k}' tidak ada di {f_json}"
            out[(rep, fold)] = rec
    return out


def _fingerprint(repeats: int, folds: int, data_sha: str, e3_fp: str,
                 params: dict[tuple[int, int], dict], items: list[str]) -> str:
    cfg = {"e3": e3_fp, "items": items, "seed": C.RANDOM_SEED,
           "inner_folds": C.INNER_FOLDS, "gamma_grid": list(GAMMA_GRID),
           "tau_bounds": [TAU_MIN, TAU_MAX], "mono": MONO, "bins": BIN_EDGES,
           "cap": WEIGHT_CAP, "grid": GRID, "gate": GATE,
           "base_xgb": {k: str(v) for k, v in sorted(C.BASE_XGB.items())},
           # hash kode sumber: edit pada objective/decode mengubah fingerprint -> tidak
           # mungkin resume diam-diam dengan separuh ronde dari kode lama (temuan reviewer)
           "kode": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
           "params_e3": {f"r{r}_f{f}": p["params"] for (r, f), p in sorted(params.items())}}
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    return hashlib.sha256(
        f"exp5|{repeats}|{folds}|{data_sha}|{cfg_sha}".encode()).hexdigest()[:16]


def _tulis(path: Path, obj: dict) -> None:
    """Tulis JSON atomik. allow_nan=False -> NaN/Inf MELEMPAR error (tidak ditelan)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False,
                             default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def _tulis_csv(path: Path, df: pd.DataFrame) -> None:
    """Tulis CSV atomik (tmp + os.replace), pola sama dengan _tulis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _slope_e3(e3_fp: str) -> float:
    """Baseline slope warp E3 DIUKUR dari CSV per-siswa E3 (mean pred_proc per row_id),
    bukan dari ingatan: nilai ini masuk ref_e3.slope_ensemble pada artefak hasil E5."""
    csv = EXP3_DIR / f"exp3_full_{e3_fp}_per_siswa.csv"
    assert csv.exists(), (
        f"CSV per-siswa E3 tidak ada: {csv} — dibutuhkan untuk ref slope warp E3")
    d = pd.read_csv(csv)
    for k in ("row_id", "y", "pred_proc"):
        assert k in d.columns, f"kolom '{k}' tidak ada di {csv}"
    ens = d.groupby("row_id").agg(y=("y", "first"), pred=("pred_proc", "mean"))
    return _slope_warp(ens["y"].to_numpy(float), ens["pred"].to_numpy(float))


def _validasi_ronde(rounds: list[dict], repeats: int, folds: int, n_samples: int,
                    e3: dict[tuple[int, int], dict]) -> None:
    """Validasi menyeluruh SEMUA ronde (termasuk hasil resume dari disk): tiap (rep,fold)
    ada tepat sekali, partisi outer identik E3, inner_y identik E3 -> fail-fast."""
    assert len(rounds) == repeats * folds, (
        f"jumlah fold tak terduga: {len(rounds)} (harus {repeats * folds})")
    kunci = [(int(r["rep"]), int(r["fold"])) for r in rounds]
    assert len(set(kunci)) == repeats * folds, (
        f"ada (rep,fold) duplikat/hilang: {len(set(kunci))} unik dari {repeats * folds}")
    for r in rounds:
        rep, fold = int(r["rep"]), int(r["fold"])
        assert (rep, fold) in e3, f"checkpoint E3 hilang untuk ronde {rep} fold {fold}"
        outer = KFold(folds, shuffle=True, random_state=C.RANDOM_SEED + rep)
        ite = list(outer.split(np.arange(n_samples)))[fold][1]
        assert list(r["idx"]) == [int(i) for i in ite], (
            f"partisi fold != E3 di ronde {rep} fold {fold} (validasi resume)")
        assert np.allclose(r["inner_y"], e3[(rep, fold)]["inner_y"]), (
            f"inner-OOF y != checkpoint E3 di ronde {rep} fold {fold} (validasi resume)")


def _muat_ronde(f_json: Path) -> list[dict]:
    """Resume: rekaman ronde sudah ada di disk -> dipakai ulang apa adanya."""
    return json.loads(f_json.read_text())["folds"]


# ---------- satu outer fold ----------

def _jalankan_fold(xtr: np.ndarray, xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray,
                   idx_te: np.ndarray, params: dict, mono_vec: np.ndarray,
                   rep: int, fold: int, model_path: Path | None = None) -> dict:
    """Gamma-selection (inner-OOF) -> fit final -> prediksi -> alpha decode.
    Semua statistik (y_mid tau, bobot, mu, lo/hi) HANYA dari baris train fold ini."""
    assert xtr.shape[1] == xte.shape[1], f"lebar fitur tidak sama: {xtr.shape[1]} vs {xte.shape[1]}"
    assert not np.isnan(xtr).any() and not np.isnan(xte).any(), "NaN masuk ke matriks fitur"
    w = bobot(ytr)  # bobot dari y RAW (konvensi E3), diiris [jtr] di inner
    mu, lo, hi = float(ytr.mean()), float(ytr.min()), float(ytr.max())

    inner = KFold(C.INNER_FOLDS, shuffle=True, random_state=C.RANDOM_SEED + 1000 * rep + fold)
    splits = list(inner.split(xtr))
    assert len(splits) == C.INNER_FOLDS, f"inner split != {C.INNER_FOLDS}"
    caches: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for g in GAMMA_GRID:  # inner-OOF per gamma (cache; dipakai untuk pilih + alpha)
        ip, iy = [], []
        for jtr, jva in splits:
            mj = _fit_custom(params, xtr[jtr], ytr[jtr], g, w[jtr], mono_vec)
            ip.extend(mj.predict(xgb.DMatrix(xtr[jva])).tolist())
            iy.extend(ytr[jva].tolist())
        caches[g] = (np.array(ip, float), np.array(iy, float))
    gamma, gamma_maes = _pilih_gamma(caches)
    ip, iy = caches[gamma]
    alpha = _fit_alpha([(ip, iy, mu)])

    booster = _fit_custom(params, xtr, ytr, gamma, w, mono_vec)
    pred_raw = booster.predict(xgb.DMatrix(xte))
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_path))

    return {
        "rep": rep, "fold": fold, "idx": [int(i) for i in idx_te],
        "y_true": [float(v) for v in yte], "pred_raw": [float(v) for v in pred_raw],
        "mu": mu, "lo": lo, "hi": hi,
        "params": params, "gamma": gamma, "gamma_maes": gamma_maes,
        "y_mid_tau": float(np.mean(ytr)), "alpha_decode": alpha,
        "inner_p": ip.tolist(), "inner_y": iy.tolist(),
        "mono": [int(v) for v in mono_vec],
        "diag": {"n_train": int(len(ytr)), "n_test": int(len(yte)),
                 "mean_w": float(w.mean()),
                 "tau_min": float(_tau(ytr, float(np.mean(ytr)), gamma).min()),
                 "tau_max": float(_tau(ytr, float(np.mean(ytr)), gamma).max())},
    }


def _baris_fold(rec: dict) -> list[dict]:
    p = np.array(rec["pred_raw"], float)
    snap, proc = _decode_varian(p, rec["mu"], rec["lo"], rec["hi"], rec["alpha_decode"])
    return [{"row_id": rec["idx"][i], "rep": rec["rep"], "fold": rec["fold"],
             "y": rec["y_true"][i], "pred_raw": float(p[i]), "pred_snap": float(snap[i]),
             "pred_proc": float(proc[i]), "gamma_fold": rec["gamma"],
             "alpha_fold": rec["alpha_decode"]}
            for i in range(len(p))]


# ---------- CV utama ----------

def run_cv(repeats: int, folds: int, fresh: bool) -> dict:
    e3_fp, e3_ck = _sumber_e3()
    e3 = _rekaman_e3(e3_ck, repeats, folds)
    feats, y, itm, items = _muat_data()
    num_cols = list(C.NUMERIC_FEATURES) + list(ITEM_COLS)

    data_sha = hashlib.sha256(FEATURES_EXP2.read_bytes() + C.DATA_CLEAN.read_bytes()).hexdigest()[:16]
    fp = _fingerprint(repeats, folds, data_sha, e3_fp, e3, items)
    ckpt = CKPT_DIR / f"exp5_{fp}"
    if fresh and ckpt.exists():
        for f in ckpt.glob("ronde_*"):  # hanya run INI (termasuk .tmp yatim)
            f.unlink()
    ckpt.mkdir(parents=True, exist_ok=True)
    MODEL_DIR = MODELS_BASE / fp  # booster di-namespace per fingerprint (anti tertukar run)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rounds: list[dict] = []
    t0 = time.time()
    for rep in range(repeats):
        f_json = ckpt / f"ronde_r{rep}.json"
        if f_json.exists():  # resume: lompati ronde yang sudah jadi
            rounds.extend(_muat_ronde(f_json))
            continue
        outer = KFold(folds, shuffle=True, random_state=C.RANDOM_SEED + rep)  # identik E3
        recs_rep = []
        for fold, (itr, ite) in enumerate(outer.split(feats)):
            assert len(itr) + len(ite) == len(feats), "partisi outer fold tidak menutup semua baris"
            e3_rec = e3[(rep, fold)]
            assert list(e3_rec["idx"]) == [int(i) for i in ite], (
                f"partisi fold E5 != E3 di ronde {rep} fold {fold} — fold harus identik E3")
            tr, te = feats.iloc[itr], feats.iloc[ite]
            xtr, xte = _encode(tr, te, "76")
            mono_vec = _vektor_monotone(xtr.shape[1], MONO, num_cols)
            rec = _jalankan_fold(xtr, xte, y[itr], y[ite], ite, e3_rec["params"],
                                 mono_vec, rep, fold,
                                 model_path=MODEL_DIR / f"ronde_r{rep}_f{fold}.ubj")
            assert np.allclose(rec["inner_y"], e3_rec["inner_y"]), (
                f"inner-OOF y != checkpoint E3 di ronde {rep} fold {fold} — seed inner berubah")
            recs_rep.append(rec)
            print(f"[exp5] rep{rep + 1}/{repeats} fold{fold + 1}/{folds} "
                  f"MAE={_mae(y[ite], np.array(rec['pred_raw'])):.4f} "
                  f"gamma={rec['gamma']:.2f} alpha_dec={rec['alpha_decode']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        _tulis(f_json, {"rep": rep, "folds": recs_rep})
        rounds.extend(recs_rep)

    # validasi menyeluruh (termasuk ronde hasil RESUME yang tadinya melewati guard fold)
    _validasi_ronde(rounds, repeats, folds, len(feats), e3)
    for r in rounds:
        mp = MODEL_DIR / f"ronde_r{int(r['rep'])}_f{int(r['fold'])}.ubj"
        assert mp.exists(), (
            f"booster hilang: {mp} — ronde ini ter-resume tanpa booster; "
            f"jalankan ulang dengan --fresh")

    df = pd.DataFrame([baris for r in rounds for baris in _baris_fold(r)])
    assert df.shape[0] == repeats * len(y), f"baris per-siswa tak terduga: {df.shape[0]}"
    assert df["row_id"].nunique() == len(y), "row_id tidak unik per siswa"
    assert np.array_equal(np.sort(df["row_id"].unique()), np.arange(len(y))), (
        "row_id bukan indeks posisional 0..N-1 — pemetaan se_i tidak sah")
    df["se_i"] = _se_item(itm)[df["row_id"].to_numpy(int)]  # diagnostik, bukan fitur model
    assert df["se_i"].notna().all(), "ada se_i NaN"
    ekor = df["y"] >= AMBANG_EKOR

    hasil: dict = {
        "eksperimen": 5, "mode": "exp5_full", "fingerprint": fp, "basis_e3": e3_fp,
        "sumber_params_e3": str(e3_ck), "repeats": repeats, "folds": folds,
        "objective": "custom_asimetris (pinball tau_i per baris, hess=1, bobot manual)",
        "api": "xgb.train fungsional (sklearn menolak callable+sample_weight)",
        "gamma_grid": list(GAMMA_GRID), "tau_bounds": [TAU_MIN, TAU_MAX],
        "monotone": MONO, "n_fitur": N_FITUR, "ambang_ekor": AMBANG_EKOR,
        "skema_bobot": f"min({WEIGHT_CAP}, sqrt(ref/n_bin)) per bin {BIN_EDGES}, dari y RAW, mean-1",
        "gamma_per_fold": [r["gamma"] for r in rounds],
        "alpha_decode_per_fold": [r["alpha_decode"] for r in rounds],
        "n_ekor_siswa": int((df.groupby("row_id")["y"].first() >= AMBANG_EKOR).sum()),
        "n_badan_siswa": int((df.groupby("row_id")["y"].first() < AMBANG_EKOR).sum()),
        "n_ekor_baris": int(ekor.sum()),
        "n_badan_baris": int((~ekor).sum()),
        "catatan": [
            "proc = snap ekor (pred_raw>=2,5) tanpa shrink, badan shrink alpha per fold",
            "params warisan E3 (tanpa Optuna); fold identik E3 (validasi idx+inner_y di semua ronde)",
            "loss sengaja BIAS-INFLATING: tau_i bergantung label y_i -> model BUKAN "
            "estimator konsisten E[y|x] (desain anti-warp, bukan regresi biasa)",
            "hyperparameter diwarisi dari E3 yang dituning untuk reg:absoluteerror TANPA "
            "monotone constraints (perbandingan berpasangan, bukan tuning ulang)",
            "skala gradien dinormalisasi 1/mean(tau) (|grad| rata-rata ~ 1) supaya langkah "
            "per ronde sebanding E3; yang diuji asimetri loss, bukan penskalaan langkah",
            "leaf value custom memakai Newton (-sum g/(sum h+lambda)) sedangkan native "
            "reg:absoluteerror memakai pembaruan leaf khusus -> sebanding, TIDAK identik bit",
            "mae_ekor/mae_non_ekor pooled dihitung atas baris-ulangan (43/263 siswa x repeats) "
            "sehingga presisinya tampak lebih tinggi daripada n siswa sebenarnya",
        ],
    }
    for nama in VARIAN:
        kol = df[nama].to_numpy(float)
        err = (df["y"] - kol).abs()
        ens = df.groupby("row_id").agg(y=("y", "first"), pred=(nama, "mean"))
        ens_err = (ens["y"] - ens["pred"]).abs()
        hasil[nama] = {
            "mae_pooled": _mae(df["y"].to_numpy(float), kol),
            "rmse_pooled": _rmse(df["y"].to_numpy(float), kol),
            "mae_ekor": float(err[ekor].mean()),
            "mae_non_ekor": float(err[~ekor].mean()),
            "mae_ensemble": float(ens_err.mean()),
            "mae_ensemble_ekor": float(ens_err[ens["y"] >= AMBANG_EKOR].mean()),
            "slope_ensemble": _slope_warp(ens["y"].to_numpy(float), ens["pred"].to_numpy(float)),
            "slope_per_repeat": [
                _slope_warp(g["y"].to_numpy(float), g[nama].to_numpy(float))
                for _, g in df.groupby("rep")],
            "mae_per_repeat": [
                float(v) for v in df.assign(e=err).groupby("rep")["e"].mean()],
        }
    # head-map ringkas (mesin-baca cepat; nilai sama dengan blok varian di atas)
    for k in ("mae_pooled", "mae_ekor", "mae_non_ekor", "mae_ensemble", "slope_ensemble"):
        hasil[k] = {v: hasil[v][k] for v in VARIAN}

    # verdict pra-registrasi: kriteria dikunci, referensi dari artefak E3
    e3p = json.loads((EXP3_DIR / f"exp3_full_{e3_fp}.json").read_text())["pred_proc"]
    ref = {"mae_ensemble": float(e3p["mae_ensemble"]),
           "mae_ensemble_ekor": float(e3p["mae_ensemble_ekor"]),
           "mae_non_ekor": float(e3p["mae_non_ekor"]),  # pooled (catatan e)
           "slope_ensemble": _slope_e3(e3_fp),  # DIUKUR dari CSV per-siswa E3 (bukan NaN)
           "sumber": f"exp3_full_{e3_fp}.json -> pred_proc + CSV per-siswa (slope)"}
    hasil["verdict_prareg"] = _verdict_prareg(hasil["pred_proc"], ref)

    _tulis(EXP5_DIR / f"exp5_full_{fp}.json", hasil)
    _tulis_csv(EXP5_DIR / f"exp5_full_{fp}_per_siswa.csv", df)
    print(f"\nSELESAI [exp5] -> {EXP5_DIR / f'exp5_full_{fp}.json'}")
    print(json.dumps({k: hasil[k] for k in
                      ("mae_pooled", "mae_ekor", "mae_non_ekor", "mae_ensemble",
                       "slope_ensemble", "verdict_prareg")}, indent=2, default=str))
    return hasil


# ---------- self-test (data sintetis, tanpa data proyek) ----------

def _self_test() -> None:
    rng = np.random.default_rng(0)

    # (a) grad/hess: tanda & magnitudo pada e>0, e<0, e=0; tau clip; tau<0.5 saat y<y_mid
    pred = np.array([1.0, 3.0, 2.0])
    y_a = np.array([3.0, 1.0, 2.0])  # e = +2, -2, 0
    tau_a = np.array([0.7, 0.7, 0.7])
    w_a = np.array([2.0, 2.0, 2.0])
    g, h = _grad_hess(pred, y_a, tau_a, w_a)
    assert np.isclose(g[0], -0.7 * 2.0) and np.isclose(g[1], 0.3 * 2.0) and g[2] == 0.0, \
        f"grad salah: {g} (harus -tau*w, +(1-tau)*w, 0)"
    assert np.allclose(h, 2.0), f"hess harus w (gaya L1), dapat {h}"
    t = _tau(np.array([1.0, 4.0]), y_mid=2.0, gamma=0.5)
    assert np.isclose(t[0], 0.2), f"tau y=1: clip bawah harus 0,20, dapat {t[0]}"
    assert np.isclose(t[1], 0.8), f"tau y=4: clip atas harus 0,80, dapat {t[1]}"
    assert np.all(_tau(np.array([1.5, 1.9]), 2.0, 0.3) < 0.5), "tau harus <0,5 di bawah y_mid"
    assert np.all(_tau(np.array([2.1, 3.0]), 2.0, 0.3) > 0.5), "tau harus >0,5 di atas y_mid"

    # (b) pemilihan gamma: deterministik, tie -> gamma terkecil
    iy = np.array([1.5, 2.0, 2.5])
    caches = {0.15: (iy + 0.10, iy), 0.25: (iy + 0.01, iy), 0.35: (iy + 0.50, iy)}
    g_best, maes = _pilih_gamma(caches)
    assert g_best == 0.25, f"gamma terbaik harus 0,25 (MAE terkecil), dapat {g_best}"
    assert len(maes) == 3 and abs(maes[1] - 0.01) < 1e-9, f"MAE gamma salah: {maes}"
    tie = {0.15: (iy + 0.05, iy), 0.25: (iy + 0.05, iy), 0.35: (iy + 0.9, iy)}
    assert _pilih_gamma(tie)[0] == 0.15, "tie harus memilih gamma terkecil"

    # (c) vektor monotone: 3 nonzero, tanda benar; fail-fast saat nama hilang
    num_cols = list(C.NUMERIC_FEATURES) + list(ITEM_COLS)
    vec = _vektor_monotone(72 + 6, MONO, num_cols)
    assert int((vec != 0).sum()) == 3, "vektor harus tepat 3 nonzero"
    for nama, arah in MONO.items():
        assert vec[num_cols.index(nama)] == arah, f"arah salah untuk {nama}"
    try:
        _vektor_monotone(10, {"f_tidak_ada": 1}, num_cols)
        raise AssertionError("fail-fast nama fitur hilang TIDAK bekerja")
    except AssertionError as e:
        assert "tidak ada" in str(e), f"pesan fail-fast tak terduga: {e}"

    # (d) constraint aktif: prediksi monotone naik pada fitur +1, turun pada -1
    n = 120
    x_t = rng.normal(size=(n, 4))
    y_t = 2.0 + 1.5 * x_t[:, 0] - 1.2 * x_t[:, 1] + rng.normal(0, 0.05, n)
    params_t = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.3,
                "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
                "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}
    vec_t = np.array([1, -1, 0, 0])
    w_t = np.ones(n)
    bt = _fit_custom(params_t, x_t, y_t, 0.25, w_t, vec_t)
    base = x_t[:1].repeat(9, axis=0)
    grid0 = np.linspace(-2, 2, 9)
    xa = base.copy(); xa[:, 0] = grid0
    pa = bt.predict(xgb.DMatrix(xa))
    assert np.all(np.diff(pa) >= -1e-6), f"fitur +1 harus monotone naik: {np.diff(pa)}"
    xb = base.copy(); xb[:, 1] = grid0
    pb = bt.predict(xgb.DMatrix(xb))
    assert np.all(np.diff(pb) <= 1e-6), f"fitur -1 harus monotone turun: {np.diff(pb)}"

    # (e) objective custom benar-benar terpakai + bobot mempengaruhi fit
    dm_t = xgb.DMatrix(x_t, label=y_t)
    b_custom = _fit_custom(params_t, x_t, y_t, 0.5, w_t, vec_t)
    b_ref = xgb.train({**C.BASE_XGB, **params_t, "objective": "reg:absoluteerror"},
                      dm_t, num_boost_round=params_t["n_estimators"])
    p_custom = b_custom.predict(dm_t)
    p_ref = b_ref.predict(dm_t)
    assert np.max(np.abs(p_custom - p_ref)) > 1e-6, \
        "prediksi custom identik dengan reg:absoluteerror — objective custom TIDAK terpakai"
    b_nomega = _fit_custom(params_t, x_t, y_t, 0.5, rng.uniform(0.1, 3.0, n), vec_t)
    assert np.max(np.abs(p_custom - b_nomega.predict(dm_t))) > 1e-6, \
        "bobot acak tidak mempengaruhi fit — bobot tidak terpakai (harus terpasang manual)"
    # bukti tersimpan di config booster: monotone_constraints ada
    assert "monotone_constraints" in b_custom.save_config(), "config booster tanpa constraint"

    # (e2) constraint benar-benar MENGIKAT (bukan sekadar tertulis di config):
    # data anti-monotone pada fitur ber-constraint +1 -> fit terikat != fit bebas
    y_anti = 2.0 - 1.5 * x_t[:, 0] + rng.normal(0, 0.05, n)
    v1 = np.array([1, 0, 0, 0])
    b_k = _fit_custom(params_t, x_t, y_anti, 0.5, w_t, v1)
    b_nk = _fit_custom(params_t, x_t, y_anti, 0.5, w_t, None)
    assert np.max(np.abs(b_k.predict(dm_t) - b_nk.predict(dm_t))) > 1e-6, \
        "constraint aktif tidak mengubah fit — constraint tidak terpasang"
    xa2 = x_t[:1].repeat(9, axis=0)
    xa2[:, 0] = np.linspace(-2, 2, 9)
    pk = b_k.predict(xgb.DMatrix(xa2))
    assert np.all(np.diff(pk) >= -1e-6), f"constraint +1 harus memaksa naik: {np.diff(pk)}"

    # (e3) normalisasi skala gradien: mean|grad| ~ 1 saat mean(tau) = 0,5 (catatan h);
    # assert batas skala benar-benar aktif (fixture ekor-berat, gamma besar)
    tau_b = np.full(n, 0.5)
    y_b = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    g_b, _ = _grad_hess(np.zeros(n), y_b, tau_b, np.ones(n), 1.0 / float(np.mean(tau_b)))
    assert abs(float(np.mean(np.abs(g_b))) - 1.0) < 1e-9, \
        f"mean|grad| harus ~1 setelah normalisasi, dapat {np.mean(np.abs(g_b))}"
    y_skew = np.where(np.arange(n) < 108, 1.0, 3.5)  # mean 1,25 -> mean(tau) ~ 0,26
    try:
        _fit_custom({**params_t, "n_estimators": 3}, x_t, y_skew, 3.0, w_t, None)
        raise AssertionError("assert batas skala gradien tidak bekerja")
    except AssertionError as e:
        assert "skala" in str(e), f"pesan skala tak terduga: {e}"

    # (e4) base_score eksplisit = mean(y latih) di config booster (catatan g)
    bs_raw = json.loads(b_custom.save_config())["learner"]["learner_model_param"]["base_score"]
    bs = float(str(bs_raw).strip("[]"))  # xgboost menulisnya sebagai string "[1.9237304E0]"
    assert abs(bs - float(y_t.mean())) < 1e-6, \
        f"base_score {bs} != mean(y) {float(y_t.mean())} — obj= custom jatuh ke 0,5"

    # (e5) placeholder objective INERT: dua fit beda placeholder, obj sama -> identik bit
    dm_ph = xgb.DMatrix(x_t, label=y_t)
    tau_ph = np.full(n, 0.5)
    w_ph = np.ones(n)

    def obj_ph(p_, dm_):
        return _grad_hess(p_, dm_.get_label(), tau_ph, w_ph, 2.0)

    def _fit_ph_ph(ph: str):
        return xgb.train({**C.BASE_XGB,
                          **{k: v for k, v in params_t.items() if k != "n_estimators"},
                          "objective": ph, "base_score": float(y_t.mean())},
                         dm_ph, num_boost_round=params_t["n_estimators"], obj=obj_ph)

    assert np.array_equal(_fit_ph_ph("reg:squarederror").predict(dm_ph),
                          _fit_ph_ph("reg:absoluteerror").predict(dm_ph)), \
        "placeholder objective TIDAK inert — prediksi berbeda"

    # (f) end-to-end 1 rep x 2 fold + checkpoint atomik + resume loader
    params_s = dict(params_t)
    recs = []
    with tempfile_dir() as tmpd:
        tmp = Path(tmpd)
        for fold, (tr, te) in enumerate(KFold(2, shuffle=True, random_state=C.RANDOM_SEED).split(x_t)):
            mv = _vektor_monotone(4, {"c0": 1, "c1": -1}, ["c0", "c1", "c2", "c3"])
            rec = _jalankan_fold(x_t[tr], x_t[te], y_t[tr], y_t[te], te, params_s, mv,
                                 0, fold, model_path=tmp / f"m_f{fold}.ubj")
            assert len(rec["idx"]) == len(te) and len(rec["pred_raw"]) == len(te)
            assert not np.isnan(rec["pred_raw"]).any(), "NaN pada pred_raw"
            assert rec["gamma"] in GAMMA_GRID and 0.0 <= rec["alpha_decode"] <= 1.0
            recs.append(rec)
            assert (tmp / f"m_f{fold}.ubj").exists(), "booster tidak tersimpan"
        rows = [b for r in recs for b in _baris_fold(r)]
        assert len(rows) == len(x_t), f"baris gabungan harus {len(x_t)}, dapat {len(rows)}"
        assert set(rows[0]) == {"row_id", "rep", "fold", "y", "pred_raw", "pred_snap",
                                "pred_proc", "gamma_fold", "alpha_fold"}, \
            "skema kolom per-siswa berubah"
        f_json = tmp / "ronde_r0.json"
        _tulis(f_json, {"rep": 0, "folds": recs})
        assert f_json.exists() and not f_json.with_suffix(".tmp").exists(), "tulis atomik gagal"
        assert len(_muat_ronde(f_json)) == 2, "resume loader gagal"

    # (g) slope_warp mengembalikan kemiringan yang ditanam
    y_g = np.linspace(1.0, 3.5, 100)
    pred_g = 0.6 * y_g + 0.3 + rng.normal(0, 0.001, 100)
    s = _slope_warp(y_g, pred_g)
    assert abs(s - 0.6) < 0.01, f"slope harus ~0,60, dapat {s}"

    # (h) verdict_prareg: boolean pada angka toy
    ref_t = {"mae_ensemble": 0.2257, "mae_ensemble_ekor": 0.3707, "mae_non_ekor": 0.2103,
             "slope_ensemble": 0.609, "sumber": "toy"}
    baik = {"slope_ensemble": 0.82, "mae_ensemble": 0.19, "mae_ensemble_ekor": 0.30,
            "mae_non_ekor": 0.2050}
    assert _verdict_prareg(baik, ref_t)["lolos"] is True, "kasus baik harus lolos"
    m = dict(baik); m["slope_ensemble"] = 0.70
    assert _verdict_prareg(m, ref_t)["lolos"] is False, "slope 0,70 harus gagal"
    m = dict(baik); m["mae_ensemble_ekor"] = 0.33
    assert _verdict_prareg(m, ref_t)["lolos"] is False, "ekor 0,33 harus gagal"
    m = dict(baik); m["mae_non_ekor"] = 0.2134
    assert _verdict_prareg(m, ref_t)["lolos"] is False, "badan > ref+0,003 harus gagal"
    m = dict(baik); m["mae_ensemble"] = 0.2001
    assert _verdict_prareg(m, ref_t)["lolos"] is False, "total 0,2001 harus gagal"

    # (i) se_item: baris konstan -> 0; nilai cocok rumus (kolom diagnostik CSV)
    it_t = np.array([[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]])
    se_t = _se_item(it_t)
    assert se_t.shape == (2,) and se_t[0] == 0.0, f"se baris konstan harus 0: {se_t}"
    assert abs(se_t[1] - it_t[1].std(ddof=1) / np.sqrt(3)) < 1e-12, "rumus se salah"
    assert np.isnan(se_t).sum() == 0, "se tidak boleh NaN"

    # (j) validasi ronde: lolos pada data sah, MENOLAK idx salah / inner_y salah / duplikat
    n_s = 40
    e3_fake: dict[tuple[int, int], dict] = {}
    rounds_fake: list[dict] = []
    for rep in range(2):
        outer = KFold(2, shuffle=True, random_state=C.RANDOM_SEED + rep)
        for fold, (_, ite) in enumerate(outer.split(np.arange(n_s))):
            e3_fake[(rep, fold)] = {"idx": [int(i) for i in ite], "inner_y": [0.0] * len(ite)}
            rounds_fake.append({"rep": rep, "fold": fold, "idx": [int(i) for i in ite],
                                "inner_y": [0.0] * len(ite)})
    _validasi_ronde(rounds_fake, 2, 2, n_s, e3_fake)  # data sah harus lolos
    bad = [dict(r) for r in rounds_fake]
    bad[0]["idx"] = [0] * len(bad[0]["idx"])
    try:
        _validasi_ronde(bad, 2, 2, n_s, e3_fake)
        raise AssertionError("validasi ronde tidak menolak idx yang salah")
    except AssertionError as e:
        assert "partisi fold" in str(e), f"pesan validasi idx tak terduga: {e}"
    bad2 = [dict(r) for r in rounds_fake]
    bad2[1]["inner_y"] = [9.9] * len(bad2[1]["inner_y"])
    try:
        _validasi_ronde(bad2, 2, 2, n_s, e3_fake)
        raise AssertionError("validasi ronde tidak menolak inner_y yang salah")
    except AssertionError as e:
        assert "inner-OOF" in str(e), f"pesan validasi inner_y tak terduga: {e}"
    dup = [dict(r) for r in rounds_fake]
    dup[1] = dict(rounds_fake[0])  # (0,0) dua kali, (0,1) hilang
    try:
        _validasi_ronde(dup, 2, 2, n_s, e3_fake)
        raise AssertionError("validasi ronde tidak menolak duplikat")
    except AssertionError as e:
        assert "duplikat" in str(e), f"pesan validasi duplikat tak terduga: {e}"

    print("self-test OK: grad/hess+tau clip+skala (mean|grad|=1), gamma deterministik "
          "(tie->kecil), monotone 3-nonzero+fail-fast+constraint MENGIKAT, "
          "objective custom terpakai+bobot aktif+base_score=mean(y)+placeholder inert, "
          "end-to-end 1rep x 2fold + booster .ubj + checkpoint atomik + resume loader, "
          "slope 0,60, verdict_prareg 5 kasus, se_item, validasi ronde (idx/inner_y/duplikat)")


import contextlib  # noqa: E402


@contextlib.contextmanager
def tempfile_dir():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        yield td


def main() -> None:
    ap = argparse.ArgumentParser(description="Eksperimen 5 — Perang Warp (loss asimetris + mono constraints)")
    ap.add_argument("--repeats", type=int, default=C.OUTER_REPEATS)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    EXP5_DIR.mkdir(parents=True, exist_ok=True)
    run_cv(a.repeats, C.OUTER_FOLDS, a.fresh)


if __name__ == "__main__":
    main()
