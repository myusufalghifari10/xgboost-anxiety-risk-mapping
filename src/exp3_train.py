"""Eksperimen 3 — Weighted-Loss (DenseWeight-style) + Warm-Start Optuna.

Perubahan atas pemenang eksperimen-2 (arm c3: 76 fitur + reg:absoluteerror +
post-processing fold-safe), sesuai keputusan 2026-10-04:

1. SAMPLE WEIGHT: skema bin (10 bin lebar 0,25 pada skala 1-4) dengan
   w = min(3, sqrt(ref / n_bin))  -> dinormalisasi rata-rata = 1.
   ref = median jumlah per bin pada FOLD LATIH (fold-safe, tanpa melihat test).
   Akar kuadrat + cap 3 = kompromi DenseWeight (Steininger et al. 2021):
   penyeimbangan moderat, tanpa data sintetis, tanpa bin 3-siswa mendominasi.
2. WARM-START OPTUNA: hyperparameter terbaik eksperimen-1 (best_params_1se +
   best_params_raw dari outputs/tuning/best_params.json) di-ENQUEUE sebagai
   2 trial pertama tiap study.

Tanpa ladder/arm — langsung nested CV 5-fold x 10 ulangan (juri tunggal),
checkpoint per ronde (resume otomatis). Kemenangan dinilai dari:
  (a) MAE total tidak naik > noise (~0,003) versus eksperimen-2 (0,2357), DAN
  (b) MAE siswa y>=2,5 (ekor) turun nyata.

Pemakaian (EKSEKUSI = YUSUF):
  .venv/bin/python src/exp3_train.py --self-test
  .venv/bin/python src/exp3_train.py                   # full run (~5 jam)
  .venv/bin/python src/exp3_train.py --fresh           # ulang dari nol
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
import optuna
import xgboost as xgb
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from exp2_features import FEATURES_EXP2  # noqa: E402
from exp2_train import (  # noqa: E402
    GRID, _decode, _encode, _fit, _fit_alpha, _mae, _rmse, _simpan, _suggest,
)

EXP3_DIR = C.ROOT / "outputs" / "exp3"
CKPT_DIR = EXP3_DIR / "checkpoints"

BIN_EDGES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
WEIGHT_CAP = 3.0
OBJECTIVE = "reg:absoluteerror"
AMBANG_EKOR = C.ANXIETY_HIGH_THRESHOLD  # 2.5
STARTER_KEYS = ("best_params", "best_params_raw")  # eksperimen-1


# ---------- bobot (fold-safe) ----------

def bobot(y_train: np.ndarray) -> np.ndarray:
    """w = min(CAP, sqrt(ref/n_bin)) per bin; ref = median bin fold latih;
    dinormalisasi rata-rata 1. Hanya memakai y_train — fold-safe."""
    y = np.asarray(y_train, float)
    counts, _ = np.histogram(y, bins=BIN_EDGES)
    aktif = counts > 0
    assert aktif.any(), "semua bin kosong di fold latih"
    ref = float(np.median(counts[aktif]))
    w_bin = np.ones(len(counts), float)
    w_bin[aktif] = np.minimum(WEIGHT_CAP, np.sqrt(ref / counts[aktif]))
    idx = np.clip(np.digitize(y, BIN_EDGES) - 1, 0, len(counts) - 1)
    w = w_bin[idx]
    return w / w.mean()


# ---------- warm-start dari eksperimen-1 ----------

def starter_params() -> list[dict]:
    d = json.loads(C.BEST_PARAMS_JSON.read_text())
    out = []
    for k in STARTER_KEYS:
        p = d.get(k)
        if not isinstance(p, dict):
            continue
        bersih = {}
        for nama, spec in C.XGB_SEARCH_SPACE.items():
            if nama not in p:
                continue
            bersih[nama] = int(p[nama]) if spec[0] == "int" else float(p[nama])
        if len(bersih) == len(C.XGB_SEARCH_SPACE) and bersih not in out:
            out.append(bersih)
    assert out, "starter params tidak ditemukan di best_params.json"
    return out


# ---------- tuning dengan bobot + enqueue ----------

def _tune(x: np.ndarray, y: np.ndarray, w: np.ndarray, n_trials: int, seed: int,
          starters: list[dict]) -> dict:
    inner = KFold(C.INNER_FOLDS, shuffle=True, random_state=seed)
    splits = list(inner.split(x))

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        vals = []
        for k, (itr, iva) in enumerate(splits):
            m = _fit(params, x[itr], y[itr], OBJECTIVE, w[itr])
            vals.append(_mae(y[iva], m.predict(x[iva])))
            trial.report(float(np.mean(vals)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(vals))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    for sp in starters:  # warm-start: 2 trial pertama = jagoan eksperimen-1
        study.enqueue_trial(sp)
    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    selesai = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert selesai, "tidak ada trial COMPLETE"
    return dict(study.best_params)


# ---------- fingerprint + checkpoint ----------

def _fingerprint(n_trials: int, repeats: int, folds: int, data_sha: str, starters: list[dict]) -> str:
    cfg_sha = hashlib.sha256(json.dumps(
        {"bins": BIN_EDGES, "cap": WEIGHT_CAP, "obj": OBJECTIVE,
         "space": C.XGB_SEARCH_SPACE, "base": C.BASE_XGB, "grid": GRID,
         "starters": starters}, sort_keys=True).encode()).hexdigest()[:12]
    return hashlib.sha256(f"exp3|{n_trials}|{repeats}|{folds}|{data_sha}|{cfg_sha}".encode()).hexdigest()[:12]


# ---------- CV utama ----------

def run_cv(n_trials: int, repeats: int, folds: int, fresh: bool) -> dict:
    feats = pd.read_parquet(FEATURES_EXP2)
    y = feats[C.TARGET_CONT].to_numpy(float)
    assert np.allclose(y / GRID, np.round(y / GRID)), "y off-grid"
    starters = starter_params()

    data_sha = hashlib.sha256(FEATURES_EXP2.read_bytes()).hexdigest()[:16]
    fp = _fingerprint(n_trials, repeats, folds, data_sha, starters)
    ckpt = CKPT_DIR / f"exp3_{fp}"
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
            tr, te = feats.iloc[itr], feats.iloc[ite]
            xtr, xte = _encode(tr, te, "76")
            ytr, yte = y[itr], y[ite]
            wtr = bobot(ytr)
            params = _tune(xtr, ytr, wtr, n_trials, C.RANDOM_SEED + 1000 * rep + fold, starters)
            m = _fit(params, xtr, ytr, OBJECTIVE, wtr)
            pred_raw = m.predict(xte)

            # pasangan inner-OOF train-side untuk alpha terpool (post-processing)
            inner = KFold(C.INNER_FOLDS, shuffle=True, random_state=C.RANDOM_SEED + 1000 * rep + fold)
            ip, iy = [], []
            for jtr, jva in inner.split(xtr):
                mj = _fit(params, xtr[jtr], ytr[jtr], OBJECTIVE, wtr[jtr])
                ip.extend(mj.predict(xtr[jva]).tolist())
                iy.extend(ytr[jva].tolist())

            rec = {"rep": rep, "fold": fold, "idx": [int(i) for i in ite],
                   "y_true": yte.tolist(), "pred_raw": [float(v) for v in pred_raw],
                   "mu": float(ytr.mean()), "lo": float(ytr.min()), "hi": float(ytr.max()),
                   "params": params, "inner_p": ip, "inner_y": iy}
            _simpan(f_json, rec)
            rounds.append(rec)
            ekor = yte >= AMBANG_EKOR
            print(f"[exp3] rep{rep + 1}/{repeats} fold{fold + 1}/{folds} "
                  f"MAE={_mae(yte, pred_raw):.4f} MAE_ekor={_mae(yte[ekor], pred_raw[ekor]) if ekor.any() else float('nan'):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    alpha = _fit_alpha([(np.array(r["inner_p"]), np.array(r["inner_y"]), r["mu"]) for r in rounds])

    rows = []
    for r in rounds:
        proc = _decode(np.array(r["pred_raw"]), r["mu"], r["lo"], r["hi"], alpha)
        for i, idx in enumerate(r["idx"]):
            rows.append({"row_id": idx, "rep": r["rep"], "fold": r["fold"], "y": r["y_true"][i],
                         "pred_raw": r["pred_raw"][i], "pred_proc": float(proc[i])})
    df = pd.DataFrame(rows)
    ekor_mask = df["y"] >= AMBANG_EKOR

    hasil = {"eksperimen": 3, "mode": "exp3_full", "alpha": alpha, "n_trials": n_trials,
             "repeats": repeats, "folds": folds, "fingerprint": fp, "objective": OBJECTIVE,
             "weight_scheme": f"min({WEIGHT_CAP}, sqrt(ref/n_bin)) per bin {BIN_EDGES}, fold-safe, mean-1",
             "starters": starters}
    for nama in ("pred_raw", "pred_proc"):
        kol = df[nama].to_numpy(float)
        err = (df["y"] - kol).abs()
        per_siswa = df.assign(e=err).groupby("row_id")["e"].mean()
        ens = df.groupby("row_id").agg(y=("y", "first"), pred=(nama, "mean"))
        ens_err = (ens["y"] - ens["pred"]).abs()
        hasil[nama] = {
            "mae_pooled": _mae(df["y"].to_numpy(float), kol),
            "rmse_pooled": _rmse(df["y"].to_numpy(float), kol),
            "mae_ekor": float(err[ekor_mask].mean()),
            "mae_non_ekor": float(err[~ekor_mask].mean()),
            "mae_per_siswa_rata2": float(per_siswa.mean()),
            "mae_ensemble": float(ens_err.mean()),
            "mae_ensemble_ekor": float(ens_err[ens["y"] >= AMBANG_EKOR].mean()),
            "mae_per_repeat": [float(v) for v in df.assign(e=err).groupby("rep")["e"].mean()],
        }
    _simpan(EXP3_DIR / f"exp3_full_{fp}.json", hasil)
    df.to_csv(EXP3_DIR / f"exp3_full_{fp}_per_siswa.csv", index=False)
    print(f"\nSELESAI [exp3] -> {EXP3_DIR / f'exp3_full_{fp}.json'}")
    print(json.dumps({k: v for k, v in hasil.items() if k.startswith("pred")}, indent=2))
    return hasil


# ---------- self-test (sintetis, tanpa data proyek) ----------

def _self_test() -> None:
    # 3 bin padat (n=4) + 2 bin langka (n=1) -> rasio bobot harus jelas terbalik
    y = np.array([1.1, 1.6, 1.6, 1.6, 1.6, 1.9, 1.9, 1.9, 1.9, 2.1, 2.1, 2.1, 2.1, 3.4])
    w = bobot(y)
    assert w.shape == y.shape
    assert abs(w.mean() - 1.0) < 1e-9, "bobot harus ternormalisasi mean 1"
    # ref = median bin aktif = 4 -> bin n=1 dapat sqrt(4/1)=2x, bin n=4 dapat 1x
    w_langka, w_padat = float(w[y == 3.4].mean()), float(w[y == 1.6].mean())
    assert w_langka > w_padat * 1.9, f"bin langka harus ~2x bin padat: {w_langka} vs {w_padat}"
    assert abs(w_padat - float(w[y == 1.9].mean())) < 1e-9, "sesama bin n=4 harus sama"
    counts, _ = np.histogram(y, bins=BIN_EDGES)
    assert (counts > 0).sum() == 5
    st = starter_params()
    assert len(st) == 2 and all(len(p) == len(C.XGB_SEARCH_SPACE) for p in st)
    assert isinstance(st[0]["n_estimators"], int) and isinstance(st[0]["learning_rate"], float)
    # jalur tuning end-to-end pada data sintetis (warm-start ikut teruji)
    rng = np.random.default_rng(0)
    xs, ys = rng.normal(size=(40, 6)), rng.normal(size=40) + 2.0
    best = _tune(xs, ys, bobot(ys), n_trials=3, seed=1, starters=st)
    assert isinstance(best, dict) and "max_depth" in best
    print("self-test OK: bobot, starter params, tuning sintetis (warm-start)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Eksperimen 3 — weighted-loss + warm-start")
    ap.add_argument("--n-trials", type=int, default=C.N_TRIALS_PER_OUTER_FOLD)
    ap.add_argument("--repeats", type=int, default=C.OUTER_REPEATS)
    ap.add_argument("--folds", type=int, default=C.OUTER_FOLDS)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    EXP3_DIR.mkdir(parents=True, exist_ok=True)
    run_cv(a.n_trials, a.repeats, a.folds, a.fresh)


if __name__ == "__main__":
    main()
