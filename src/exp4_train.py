"""Eksperimen 4 — Dekomposisi Target Item-Level (multi-target, hiperparameter warisan E3).

Ide (kandidat C5 dari sesi BlackBox 2026-10-03): alih-alih melatih 1 model untuk y
(rata-rata 20 item kecemasan), latih **20 model**, satu per item target, lalu rata-ratakan
prediksinya. y = mean(20 item) secara eksak (terverifikasi: max|mean-item − y| = 0,0),
sehingga rata-rata 20 prediksi item = estimator y dengan noise item saling meniadakan.
Item-item TIDAK pernah dipakai sebagai fitur (bukan leakage) — mereka adalah target.

Yang diuji (semua pada fold IDENTIK dengan E3, jadi paired):
  A. itemavg_uniform   : 20 model item, bobot seragam, dirata-rata
  B. itemavg_weighted  : 20 model item, bobot bin per-item (skema E3), dirata-rata
  C. blendA/B          : 0,5 × (itemavg + model langsung E3)
  D. direct            : model langsung E3 (kontrol; harus mereproduksi pred_raw E3)

Hiperparameter: **warisan per ronde** dari checkpoint E3 (`outputs/exp3/checkpoints/exp3_<fp>/`),
jadi TANPA Optuna — biaya ±30 menit. Isolasi bersih: satu-satunya yang berubah = dekomposisi target.
Post-processing (clip → α terpool → snap 0,05) dihitung fold-safe per varian.

Kriteria pra-daftar: E4 dipakai bila MAE total (proc) < 0,2338 ATAU MAE ekor turun nyata,
tanpa menaikkan sisi lain > noise (0,003).

Pemakaian (EKSEKUSI = YUSUF):
  .venv/bin/python src/exp4_train.py --self-test
  .venv/bin/python src/exp4_train.py                 # ±30 menit (resume otomatis)
  .venv/bin/python src/exp4_train.py --fresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from exp2_features import FEATURES_EXP2  # noqa: E402
from exp2_train import GRID, _decode, _encode, _fit, _fit_alpha, _mae, _rmse, _simpan  # noqa: E402
from exp3_train import AMBANG_EKOR, bobot  # noqa: E402

EXP3_DIR = C.ROOT / "outputs" / "exp3"
EXP4_DIR = C.ROOT / "outputs" / "exp4"
CKPT_DIR = EXP4_DIR / "checkpoints"
OBJECTIVE = "reg:absoluteerror"
VARIAN = ("direct", "itemavg_uniform", "itemavg_weighted", "blendA", "blendB")


# ---------- util ----------

def _e3_terbaru() -> tuple[str, Path]:
    """Fingerprint + folder checkpoint E3 terbaru (E4 bergantung pada artefak E3)."""
    js = sorted(EXP3_DIR.glob("exp3_full_*.json"), key=lambda p: p.stat().st_mtime)
    assert js, f"artefak eksperimen-3 tidak ditemukan di {EXP3_DIR} — jalankan exp3 dulu"
    fp = json.loads(js[-1].read_text())["fingerprint"]
    ck = EXP3_DIR / "checkpoints" / f"exp3_{fp}"
    assert ck.is_dir(), f"checkpoint E3 {ck} tidak ada (dibutuhkan untuk hiperparameter warisan)"
    return fp, ck


def _item_cols(dc: pd.DataFrame) -> list[str]:
    items = [c for c in C.ANXIETY_ITEMS_ALL if c in dc.columns]
    assert len(items) == 20, f"item target harus 20, dapat {len(items)}"
    return items


def _fit_multi(params: dict, x: np.ndarray, ymat: np.ndarray,
               wmat: np.ndarray | None) -> list:
    return [_fit(params, x, ymat[:, j], OBJECTIVE, None if wmat is None else wmat[:, j])
            for j in range(ymat.shape[1])]


def _pred_multi(models: list, x: np.ndarray) -> np.ndarray:
    """Prediksi per item (n × k)."""
    return np.column_stack([m.predict(x) for m in models])


def _bobot_item(ymat: np.ndarray) -> np.ndarray:
    """Bobot bin per kolom item (skema E3 diterapkan pada distribusi tiap item)."""
    return np.column_stack([bobot(ymat[:, j]) for j in range(ymat.shape[1])])


# ---------- fingerprint ----------

def _fingerprint(e3_fp: str, repeats: int, folds: int, data_sha: str, items: list[str]) -> str:
    cfg = hashlib.sha256(json.dumps(
        {"e3": e3_fp, "items": items, "obj": OBJECTIVE, "base": C.BASE_XGB,
         "grid": GRID, "varian": VARIAN}, sort_keys=True).encode()).hexdigest()[:12]
    return hashlib.sha256(f"exp4|{repeats}|{folds}|{data_sha}|{cfg}".encode()).hexdigest()[:12]


# ---------- run ----------

def run_cv(repeats: int, folds: int, fresh: bool) -> dict:
    e3_fp, e3_ck = _e3_terbaru()
    feats = pd.read_parquet(FEATURES_EXP2)
    dc = pd.read_parquet(C.DATA_CLEAN)
    items = _item_cols(dc)
    y = feats[C.TARGET_CONT].to_numpy(float)
    ymat = dc[items].to_numpy(float)
    selisih = float(np.abs(ymat.mean(axis=1) - y).max())
    assert selisih < 1e-9, f"y != mean(item) (selisih {selisih}) — baris tidak sejajar"
    assert ymat.shape == (len(feats), 20) and not np.isnan(ymat).any()

    data_sha = hashlib.sha256(FEATURES_EXP2.read_bytes()).hexdigest()[:16]
    fp = _fingerprint(e3_fp, repeats, folds, data_sha, items)
    ckpt = CKPT_DIR / f"exp4_{fp}"
    if fresh and ckpt.exists():
        for f in ckpt.glob("ronde_*.json*"):
            f.unlink()
    ckpt.mkdir(parents=True, exist_ok=True)

    rounds: list[dict] = []
    t0 = time.time()
    for rep in range(repeats):
        outer = KFold(folds, shuffle=True, random_state=C.RANDOM_SEED + rep)
        for fold, (itr, ite) in enumerate(outer.split(feats)):
            f_json = ckpt / f"ronde_r{rep}_f{fold}.json"
            if f_json.exists():
                rounds.append(json.loads(f_json.read_text()))
                continue
            e3_rec = json.loads((e3_ck / f"ronde_r{rep}_f{fold}.json").read_text())
            params = e3_rec["params"]
            tr, te = feats.iloc[itr], feats.iloc[ite]
            xtr, xte = _encode(tr, te, "76")
            ytr, yte = y[itr], y[ite]
            ymtr, ymte = ymat[itr], ymat[ite]

            # D. model langsung (bobot E3) — reproduksi kontrol
            w_direct = bobot(ytr)
            m_direct = _fit(params, xtr, ytr, OBJECTIVE, w_direct)
            pred_direct = m_direct.predict(xte)

            # A/B. 20 model item
            mA = _fit_multi(params, xtr, ymtr, None)
            mB = _fit_multi(params, xtr, ymtr, _bobot_item(ymtr))
            pA_mat, pB_mat = _pred_multi(mA, xte), _pred_multi(mB, xte)
            pA, pB = pA_mat.mean(axis=1), pB_mat.mean(axis=1)

            # inner-OOF train-side (seed sama dengan E3) untuk alpha tiap varian
            inner = KFold(C.INNER_FOLDS, shuffle=True,
                          random_state=C.RANDOM_SEED + 1000 * rep + fold)
            ipA, ipB, iy = [], [], []
            for jtr, jva in inner.split(xtr):
                iA = _fit_multi(params, xtr[jtr], ymtr[jtr], None)
                iB = _fit_multi(params, xtr[jtr], ymtr[jtr], _bobot_item(ymtr[jtr]))
                ipA.extend(_pred_multi(iA, xtr[jva]).mean(axis=1).tolist())
                ipB.extend(_pred_multi(iB, xtr[jva]).mean(axis=1).tolist())
                iy.extend(ytr[jva].tolist())
            iA_v, iB_v, iD_v, iy_v = (np.array(ipA), np.array(ipB),
                                      np.array(e3_rec["inner_p"]), np.array(e3_rec["inner_y"]))
            assert np.allclose(iy_v, iy), "inner OOF y tidak cocok dengan checkpoint E3"

            rec = {"rep": rep, "fold": fold, "idx": [int(i) for i in ite],
                   "y_true": yte.tolist(), "mu": float(ytr.mean()),
                   "lo": float(ytr.min()), "hi": float(ytr.max()),
                   "pred": {"direct": pred_direct.tolist(), "itemavg_uniform": pA.tolist(),
                            "itemavg_weighted": pB.tolist(),
                            "blendA": (0.5 * (pA + pred_direct)).tolist(),
                            "blendB": (0.5 * (pB + pred_direct)).tolist()},
                   "inner": {"direct": iD_v.tolist(), "itemavg_uniform": iA_v.tolist(),
                             "itemavg_weighted": iB_v.tolist(),
                             "blendA": (0.5 * (iA_v + iD_v)).tolist(),
                             "blendB": (0.5 * (iB_v + iD_v)).tolist()},
                   "inner_y": iy,
                   # diagnostik per item (MAE test-side tiap item) — matriks sudah dihitung
                   "mae_item_uniform": [float(_mae(ymte[:, j], pA_mat[:, j])) for j in range(20)],
                   "mae_item_weighted": [float(_mae(ymte[:, j], pB_mat[:, j])) for j in range(20)]}
            _simpan(f_json, rec)
            rounds.append(rec)
            print(f"[exp4] rep{rep + 1}/{repeats} fold{fold + 1}/{folds} "
                  f"MAE direct={_mae(yte, pred_direct):.4f} A={_mae(yte, pA):.4f} "
                  f"B={_mae(yte, pB):.4f} ({time.time() - t0:.0f}s)", flush=True)

    alphas = {v: _fit_alpha([(np.array(r["inner"][v]), np.array(r["inner_y"]), r["mu"])
                             for r in rounds]) for v in VARIAN}

    rows = []
    for r in rounds:
        for i, idx in enumerate(r["idx"]):
            row = {"row_id": idx, "rep": r["rep"], "fold": r["fold"], "y": r["y_true"][i]}
            for v in VARIAN:
                raw = r["pred"][v][i]
                row[f"{v}_raw"] = raw
                row[f"{v}_proc"] = float(_decode(np.array([raw]), r["mu"], r["lo"], r["hi"],
                                                 alphas[v])[0])
            rows.append(row)
    df = pd.DataFrame(rows)
    ekor = df["y"] >= AMBANG_EKOR

    hasil = {"eksperimen": 4, "mode": "exp4_item_decomposition", "basis_e3": e3_fp,
             "fingerprint": fp, "alpha": alphas, "repeats": repeats, "folds": folds,
             "items": items, "objective": OBJECTIVE,
             "catatan": "hiperparameter warisan per-ronde dari checkpoint E3; tanpa Optuna",
             "mae_item_rata2": {
                 "uniform": float(np.mean([r["mae_item_uniform"] for r in rounds], axis=0).mean()),
                 "weighted": float(np.mean([r["mae_item_weighted"] for r in rounds], axis=0).mean())}}
    for v in VARIAN:
        for suf in ("raw", "proc"):
            kol = df[f"{v}_{suf}"].to_numpy(float)
            err = (df["y"] - kol).abs()
            ens = df.groupby("row_id").agg(y=("y", "first"), pred=(f"{v}_{suf}", "mean"))
            ens_err = (ens["y"] - ens["pred"]).abs()
            hasil[f"{v}_{suf}"] = {
                "mae_pooled": _mae(df["y"].to_numpy(float), kol),
                "rmse_pooled": _rmse(df["y"].to_numpy(float), kol),
                "mae_ekor": float(err[ekor].mean()),
                "mae_non_ekor": float(err[~ekor].mean()),
                "mae_ensemble": float(ens_err.mean()),
                "mae_ensemble_ekor": float(ens_err[ens["y"] >= AMBANG_EKOR].mean()),
                "mae_per_repeat": [float(v2) for v2 in df.assign(e=err).groupby("rep")["e"].mean()]}
    _simpan(EXP4_DIR / f"exp4_full_{fp}.json", hasil)
    df.to_csv(EXP4_DIR / f"exp4_full_{fp}_per_siswa.csv", index=False)
    print(f"\nSELESAI [exp4] -> {EXP4_DIR / f'exp4_full_{fp}.json'}")
    print(json.dumps({v: {k: round(hasil[f'{v}_proc'][k], 4) for k in
                          ('mae_pooled', 'mae_ekor', 'mae_ensemble', 'mae_ensemble_ekor')}
                      for v in VARIAN}, indent=2))
    return hasil


# ---------- self-test (sintetis) ----------

def _self_test() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 6))
    ymat = rng.integers(1, 5, size=(60, 3)).astype(float)
    y = ymat.mean(axis=1)
    assert abs(np.abs(ymat.mean(axis=1) - y).max()) < 1e-12
    w = _bobot_item(ymat)
    assert w.shape == ymat.shape and np.allclose(w.mean(axis=0), 1.0)
    params = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.2,
              "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
              "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0}
    mA = _fit_multi(params, x, ymat, None)
    pA = _pred_multi(mA, x)
    assert pA.shape == (60, 3) and np.isfinite(pA).all()
    avg = pA.mean(axis=1)
    assert avg.shape == (60,)
    mB = _fit_multi(params, x, ymat, w)
    pB = _pred_multi(mB, x)
    assert np.isfinite(pB).all()
    a1, a2 = _pred_multi(mA, x[:5]).mean(axis=1), _pred_multi(mB, x[:5]).mean(axis=1)
    assert np.allclose(0.5 * (a1 + a2), 0.5 * a1 + 0.5 * a2)
    assert not np.allclose(a1, a2), "bobot seragam vs per-item harus berbeda"
    print("self-test OK: item matrix, bobot per-item, fit 3-item, blend")


def main() -> None:
    ap = argparse.ArgumentParser(description="Eksperimen 4 — dekomposisi target item-level")
    ap.add_argument("--repeats", type=int, default=C.OUTER_REPEATS)
    ap.add_argument("--folds", type=int, default=C.OUTER_FOLDS)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    EXP4_DIR.mkdir(parents=True, exist_ok=True)
    run_cv(a.repeats, a.folds, a.fresh)


if __name__ == "__main__":
    main()
