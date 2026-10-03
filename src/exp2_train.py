"""Eksperimen 2 — "MAE-Aligned Item-Level Pipeline" (Keputusan BlackBox 2026-10-03).

Ladder arm kumulatif (pemenang dipilih lewat screening murah, lalu full run):
  c0 = baseline dalam-budget : 18 fitur, reg:squarederror, pilih RMSE  (= percobaan-1)
  c1 = c0 + objective MAE    : reg:absoluteerror, pilih MAE (re-tuning penuh)
  c2 = c1 + fitur item-level : 58 item mentah -> 76 fitur
  c3 = c2 + post-processing  : clip rentang train-fold -> shrinkage alpha terpool
                               -> snap grid 0,05 (terverifikasi 100% valid)
  c4 = c3 + bobot gaya        : sample_weight straightliner (w_style)

Aturan jujur (hasil red-team 2026-10-03):
  * SEMUA fitting post-processing (alpha, rentang clip, rata-rata train) hanya
    dari data TRAIN tiap fold luar (inner-OOF) — tanpa bocor ke fold test.
  * alpha DIPOOL untuk seluruh run (estimasi per-fold ±245 baris terlalu bising).
  * Pemenang dipilih dari screening; angka final diambil dari outer loop yang
    TIDAK dipakai memilih (mode full) — outer loop hanya juri sekali pakai.
  * Tanpa aturan 1-SE: tujuan eksperimen ini akurasi MAE murni.

Mode:
  screen : outer 5-fold x 3 ulangan, 150 trial/ronde  (±20 menit/arm)
  full   : outer 5-fold x 10 ulangan, 500 trial/ronde (setara percobaan-1)
  final  : tuning Optuna penuh di 306 siswa + simpan model & decoder deployment

Pemakaian (EKSEKUSI = YUSUF):
  .venv/bin/python src/exp2_features.py                 # sekali, preprocessing
  .venv/bin/python src/exp2_train.py --self-test        # uji fungsi tanpa data
  .venv/bin/python src/exp2_train.py --arm c0 --mode screen   # ... c1..c4
  .venv/bin/python src/exp2_train.py --arm <pemenang> --mode full
  .venv/bin/python src/exp2_train.py --arm <pemenang> --mode final

Checkpoint per ronde di outputs/exp2/checkpoints/ (resume otomatis; --fresh hapus).
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
import optuna
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from exp2_features import FEATURES_EXP2, ITEM_COLS  # noqa: E402

EXP2_DIR = C.ROOT / "outputs" / "exp2"
CKPT_DIR = EXP2_DIR / "checkpoints"
MODEL_DIR = EXP2_DIR / "model"
FINAL_DB = CKPT_DIR / "exp2_final.db"
GRID = 0.05  # granularitas y terverifikasi: 0/306 nilai off-grid

ARMS = {
    "c0": {"fitur": "18", "objective": "reg:squarederror", "select": "rmse", "post": False, "w": False},
    "c1": {"fitur": "18", "objective": "reg:absoluteerror", "select": "mae", "post": False, "w": False},
    "c2": {"fitur": "76", "objective": "reg:absoluteerror", "select": "mae", "post": False, "w": False},
    "c3": {"fitur": "76", "objective": "reg:absoluteerror", "select": "mae", "post": True, "w": False},
    "c4": {"fitur": "76", "objective": "reg:absoluteerror", "select": "mae", "post": True, "w": True},
}


# ---------- fungsi murni (diuji --self-test, tanpa data) ----------

def _decode(pred: np.ndarray, mu: float, lo: float, hi: float, alpha: float) -> np.ndarray:
    """clip ke rentang train-fold -> shrink ke mean train -> snap grid -> clip lagi."""
    p = np.clip(np.asarray(pred, float), lo, hi)
    p = mu + alpha * (p - mu)
    p = np.round(p / GRID) * GRID
    return np.clip(p, lo, hi)


def _fit_alpha(pairs: list[tuple[np.ndarray, np.ndarray, float]]) -> float:
    """alpha terpool: satu alpha untuk seluruh run, dari pasangan inner-OOF train-side.
    MAE terhadap alpha piecewise-linear konveks -> grid 0..1 langkah 0,01 cukup."""
    assert pairs, "pairs kosong"
    best_a, best_e = 0.0, float("inf")
    for a in np.linspace(0.0, 1.0, 101):
        err = 0.0
        n = 0
        for p, y, mu in pairs:
            err += float(np.abs(mu + a * (p - mu) - y).sum())
            n += len(y)
        if err / n < best_e:
            best_a, best_e = a, err / n
    return float(best_a)


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _self_test() -> None:
    p = np.array([0.5, 1.02, 2.024, 3.9])
    d = _decode(p, mu=2.0, lo=1.0, hi=3.5, alpha=1.0)
    assert np.all(d >= 1.0) and np.all(d <= 3.5), "clip gagal"
    assert np.allclose(d / GRID, np.round(d / GRID)), "snap grid gagal"
    d0 = _decode(p, mu=2.0, lo=1.0, hi=3.5, alpha=0.0)
    assert np.allclose(d0, 2.0), "alpha=0 harus mengembalikan mu"
    perfect = [(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 2.0)]
    assert _fit_alpha(perfect) > 0.9, "alpha pada data tanpa noise harus ~1"
    noise = [(np.array([2.0, 2.0, 2.0, 2.0]), np.array([1.0, 1.4, 2.6, 3.0]), 2.0)]
    assert _fit_alpha(noise) < 0.11, "alpha pada prediksi konstan harus ~0"
    assert set(ARMS) == {"c0", "c1", "c2", "c3", "c4"}
    assert _mae(np.array([1.0, 2.0]), np.array([1.5, 2.5])) == 0.5
    assert abs(_rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) - 3.5355) < 1e-3
    # uji jalur tuning end-to-end pada data SINTETIS (tanpa data proyek) — menangkap
    # bug kelas accessor API tak-terimport (mis. optuna.tp) yang lolos py_compile
    rng = np.random.default_rng(0)
    xs, ys = rng.normal(size=(40, 6)), rng.normal(size=40) + 2.0
    best = _tune(xs, ys, n_trials=3, seed=1, arm=ARMS["c1"], w=None)
    assert isinstance(best, dict) and "max_depth" in best, "_tune gagal pada data sintetis"
    m = _fit(best, xs, ys, "reg:absoluteerror", np.ones(40))
    assert _decode(m.predict(xs[:3]), float(ys.mean()), float(ys.min()), float(ys.max()), 0.5).shape == (3,)
    print("self-test OK: decode, alpha, arm table, metrik, tuning sintetis")


# ---------- encoding + tuning ----------

def _encode(train: pd.DataFrame, test: pd.DataFrame, fitur: str):
    num = C.NUMERIC_FEATURES + (ITEM_COLS if fitur == "76" else [])
    cat = C.CATEGORICAL_FEATURES
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    xtr = np.hstack([train[num].to_numpy(float), enc.fit_transform(train[cat])])
    xte = np.hstack([test[num].to_numpy(float), enc.transform(test[cat])])
    return xtr, xte


def _suggest(trial: optuna.Trial) -> dict:
    params = {}
    for nama, spec in C.XGB_SEARCH_SPACE.items():
        if spec[0] == "int":
            params[nama] = trial.suggest_int(nama, spec[1], spec[2])
        elif len(spec) == 4 and spec[3] == "log":
            params[nama] = trial.suggest_float(nama, spec[1], spec[2], log=True)
        else:
            params[nama] = trial.suggest_float(nama, spec[1], spec[2])
    return params


def _fit(params: dict, x: np.ndarray, y: np.ndarray, objective: str, w: np.ndarray | None):
    return xgb.XGBRegressor(**C.BASE_XGB, objective=objective, **params).fit(x, y, sample_weight=w)


def _tune(x: np.ndarray, y: np.ndarray, n_trials: int, seed: int, arm: dict, w: np.ndarray | None) -> dict:
    inner = KFold(C.INNER_FOLDS, shuffle=True, random_state=seed)
    splits = list(inner.split(x))
    metric = arm["select"]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        vals = []
        for k, (itr, iva) in enumerate(splits):
            m = _fit(params, x[itr], y[itr], arm["objective"], None if w is None else w[itr])
            p = m.predict(x[iva])
            vals.append(_rmse(y[iva], p) if metric == "rmse" else _mae(y[iva], p))
            trial.report(float(np.mean(vals)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(vals))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    selesai = [t for t in study.trials if t.state.is_finished() and not t.state == optuna.trial.TrialState.PRUNED]
    assert selesai, "tidak ada trial COMPLETE"
    return dict(study.best_params)


# ---------- checkpoint ----------

def _fingerprint(arm: str, mode: str, n_trials: int, repeats: int, folds: int, data_sha: str) -> str:
    cfg_sha = hashlib.sha256(json.dumps(
        {"fitur": ARMS[arm]["fitur"], "items": ITEM_COLS if ARMS[arm]["fitur"] == "76" else [],
         "space": C.XGB_SEARCH_SPACE, "base": C.BASE_XGB, "grid": GRID}, sort_keys=True).encode()
    ).hexdigest()[:12]
    return hashlib.sha256(f"{arm}|{mode}|{n_trials}|{repeats}|{folds}|{data_sha}|{cfg_sha}".encode()).hexdigest()[:12]


def _simpan(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


# ---------- CV utama ----------

def run_cv(arm_name: str, mode: str, n_trials: int, repeats: int, folds: int, fresh: bool) -> dict:
    arm = ARMS[arm_name]
    feats = pd.read_parquet(FEATURES_EXP2)
    y = feats[C.TARGET_CONT].to_numpy(float)
    assert np.allclose(y / GRID, np.round(y / GRID)), "y off-grid — periksa ulang granularitas"
    w = feats["w_style"].to_numpy(float) if arm["w"] else None

    data_sha = hashlib.sha256(FEATURES_EXP2.read_bytes()).hexdigest()[:16]
    fp = _fingerprint(arm_name, mode, n_trials, repeats, folds, data_sha)
    ckpt = CKPT_DIR / f"{mode}_{arm_name}_{fp}"
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
            xtr, xte = _encode(tr, te, arm["fitur"])
            ytr, yte = y[itr], y[ite]
            wtr = None if w is None else w[itr]
            params = _tune(xtr, ytr, n_trials, C.RANDOM_SEED + 1000 * rep + fold, arm, wtr)
            m = _fit(params, xtr, ytr, arm["objective"], wtr)
            pred_raw = m.predict(xte)

            rec = {"rep": rep, "fold": fold, "idx": [int(i) for i in ite],
                   "y_true": yte.tolist(), "pred_raw": [float(v) for v in pred_raw],
                   "mu": float(ytr.mean()), "lo": float(ytr.min()), "hi": float(ytr.max()),
                   "params": params}
            if arm["post"]:  # pasangan inner-OOF train-side untuk alpha terpool
                inner = KFold(C.INNER_FOLDS, shuffle=True, random_state=C.RANDOM_SEED + 1000 * rep + fold)
                ip, iy = [], []
                for jtr, jva in inner.split(xtr):
                    mj = _fit(params, xtr[jtr], ytr[jtr], arm["objective"], None if wtr is None else wtr[jtr])
                    ip.extend(mj.predict(xtr[jva]).tolist())
                    iy.extend(ytr[jva].tolist())
                rec["inner_p"], rec["inner_y"] = ip, iy
            _simpan(f_json, rec)
            rounds.append(rec)
            print(f"[{arm_name}/{mode}] rep{rep + 1}/{repeats} fold{fold + 1}/{folds} "
                  f"raw MAE={_mae(yte, pred_raw):.4f} RMSE={_rmse(yte, pred_raw):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- alpha terpool + dekode ----
    alpha = 0.0
    if arm["post"]:
        pairs = [(np.array(r["inner_p"]), np.array(r["inner_y"]), r["mu"]) for r in rounds]
        alpha = _fit_alpha(pairs)

    rows = []
    for r in rounds:
        proc = _decode(np.array(r["pred_raw"]), r["mu"], r["lo"], r["hi"], alpha) if arm["post"] else None
        for i, idx in enumerate(r["idx"]):
            rows.append({"row_id": idx, "rep": r["rep"], "fold": r["fold"], "y": r["y_true"][i],
                         "pred_raw": r["pred_raw"][i],
                         "pred_proc": float(proc[i]) if proc is not None else r["pred_raw"][i]})
    df = pd.DataFrame(rows)
    hasil = {"arm": arm_name, "mode": mode, "alpha": alpha, "n_trials": n_trials,
             "repeats": repeats, "folds": folds, "fingerprint": fp,
             "objective": arm["objective"], "n_fitur": 18 if arm["fitur"] == "18" else 76}
    for nama in ("pred_raw", "pred_proc"):
        kol = df[nama].to_numpy(float)
        per_siswa = df.assign(e=(df["y"] - kol).abs()).groupby("row_id")["e"].mean()
        hasil[nama] = {
            "mae_pooled": _mae(df["y"].to_numpy(float), kol),
            "rmse_pooled": _rmse(df["y"].to_numpy(float), kol),
            "mae_per_siswa_rata2": float(per_siswa.mean()),
        }
        per_rep = df.assign(e=(df["y"] - kol).abs()).groupby("rep")["e"].mean()
        hasil[nama]["mae_per_repeat"] = [float(v) for v in per_rep]
    out = EXP2_DIR / f"{mode}_{arm_name}_{fp}.json"
    _simpan(out, hasil)
    df.to_csv(EXP2_DIR / f"{mode}_{arm_name}_{fp}_per_siswa.csv", index=False)
    print(f"\nSELESAI [{arm_name}/{mode}] -> {out}")
    print(json.dumps({k: v for k, v in hasil.items() if k.startswith("pred")}, indent=2))
    return hasil


# ---------- final: tuning penuh + model deployment ----------

def run_final(arm_name: str, n_trials_final: int) -> None:
    arm = ARMS[arm_name]
    feats = pd.read_parquet(FEATURES_EXP2)
    y = feats[C.TARGET_CONT].to_numpy(float)
    w = feats["w_style"].to_numpy(float) if arm["w"] else None
    x_all, _ = _encode(feats, feats, arm["fitur"])  # encoder final diganti di bawah

    seed = C.RANDOM_SEED
    nama_study = "exp2_final_" + _fingerprint(arm_name, "final", n_trials_final, 0, 0,
                                              hashlib.sha256(FEATURES_EXP2.read_bytes()).hexdigest()[:16])
    study = optuna.create_study(study_name=nama_study, storage=f"sqlite:///{FINAL_DB}",
                                load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    sudah = sum(1 for t in study.trials if t.state.is_finished())
    sisa = max(0, n_trials_final - sudah)
    print(f"final tuning: {sudah} trial tersimpan, menjalankan {sisa} lagi")

    inner = KFold(C.INNER_FOLDS, shuffle=True, random_state=seed)
    splits = list(inner.split(x_all))
    metric = arm["select"]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        vals = []
        for k, (itr, iva) in enumerate(splits):
            m = _fit(params, x_all[itr], y[itr], arm["objective"], None if w is None else w[itr])
            p = m.predict(x_all[iva])
            vals.append(_rmse(y[iva], p) if metric == "rmse" else _mae(y[iva], p))
            trial.report(float(np.mean(vals)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(vals))

    if sisa:
        study.optimize(objective, n_trials=sisa, catch=(ValueError,))
    params = dict(study.best_params)

    # alpha deployment dari inner-OOF penuh (train-side, fold-safe)
    alpha = 0.0
    if arm["post"]:
        ip, iy = [], []
        for jtr, jva in splits:
            mj = _fit(params, x_all[jtr], y[jtr], arm["objective"], None if w is None else w[jtr])
            ip.extend(mj.predict(x_all[jva]).tolist())
            iy.extend(y[jva].tolist())
        alpha = _fit_alpha([(np.array(ip), np.array(iy), float(y.mean()))])

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = C.NUMERIC_FEATURES + (ITEM_COLS if arm["fitur"] == "76" else [])
    x_enc = np.hstack([feats[num].to_numpy(float), enc.fit_transform(feats[C.CATEGORICAL_FEATURES])])
    reg = _fit(params, x_enc, y, arm["objective"], w)
    yb = feats[C.TARGET_BIN].to_numpy(int)
    spw = float((yb == 0).sum() / max(1, (yb == 1).sum()))
    clf = xgb.XGBClassifier(**C.BASE_XGB, objective="binary:logistic", scale_pos_weight=spw, **params)
    clf.fit(x_enc, yb)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    reg.save_model(MODEL_DIR / "model_exp2.ubj")
    clf.save_model(MODEL_DIR / "model_exp2_clf.ubj")
    import joblib
    joblib.dump({"enc": enc, "num": num, "cat": C.CATEGORICAL_FEATURES},
                MODEL_DIR / "preprocessor_exp2.joblib")
    _simpan(MODEL_DIR / "best_params_exp2.json", {
        "arm": arm_name, "params": params, "alpha": alpha, "mu": float(y.mean()),
        "lo": float(y.min()), "hi": float(y.max()), "grid": GRID,
        "inner_cv_" + metric: float(study.best_value),
        "catatan": "dekode deployment: clip(lo,hi) -> mu + alpha*(p-mu) -> snap grid -> clip",
    })
    print(f"SELESAI final -> {MODEL_DIR} (alpha={alpha:.2f}, best inner {metric}={study.best_value:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Eksperimen 2 — ladder MAE-aligned item-level")
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--mode", choices=["screen", "full", "final"], default="screen")
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--n-trials-final", type=int, default=C.N_TRIALS_FINAL)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--folds", type=int, default=C.OUTER_FOLDS)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        _self_test()
        return
    assert a.arm, "--arm wajib (c0..c4)"
    if a.mode == "final":
        run_final(a.arm, a.n_trials_final)
        return
    n_trials = a.n_trials or (150 if a.mode == "screen" else C.N_TRIALS_PER_OUTER_FOLD)
    repeats = a.repeats or (3 if a.mode == "screen" else C.OUTER_REPEATS)
    run_cv(a.arm, a.mode, n_trials, repeats, a.folds, a.fresh)


if __name__ == "__main__":
    main()
