"""03_train_tune.py — Training XGBoost + tuning masif Optuna (nested CV).

Skema (PLAN bagian 5-6):
    Outer: OUTER_FOLDS x OUTER_REPEATS, shuffle, seed per ulangan.
           Test fold = 61 baris, TIDAK PERNAH dipakai tuning maupun pemilihan setelan.
    Inner: tuning Optuna HANYA pada data train; skor tiap trial = RMSE pada
           inner-validation fold (out-of-sample), bukan in-sample.
           N_TRIALS_PER_OUTER_FOLD trial per ronde outer; N_TRIALS_FINAL trial
           untuk tuning model akhir di seluruh 306 baris.
    1-SE rule: di antara trial dalam 1 standard error dari yang terbaik, pilih
               model paling sederhana.
    Biner: XGBClassifier memakai setelan sama + scale_pos_weight yang dihitung
           dari data latih tiap fold (kelas positif hanya 14,1%).
    Final: model dilatih ulang di seluruh 306 baris dengan parameter terpilih.

Artefak:
    outputs/optuna_trials.csv     (kontrak: trial_number, params, mean_test_rmse, std_test_rmse, duration)
    outputs/oof_predictions.csv   (prediksi out-of-fold per baris/ulangan -> dipakai 04_evaluate)
    outputs/model_final.ubj       (XGBRegressor akhir)
    outputs/model_final_clf.ubj   (XGBClassifier akhir, target biner)
    outputs/preprocessor.joblib   (encoder kategorikal ter-fit di seluruh data)
    outputs/best_params.json      (kunci kontrak: best_params, best_params_raw,
                                   feature_columns + jejak tambahan)

CATATAN: skrip ini MELETAKKAN training. Belum boleh dijalankan sampai disetujui.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.pruning import TrialPruned
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

# Artefak tambahan di luar kontrak — dibutuhkan 05/06 agar encoding & SHAP konsisten.
# Path diambil dari config.py (sumber kebenaran tunggal).
OOF_PATH = cfg.OOF_CSV
MODEL_CLF_PATH = cfg.MODEL_CLF_PATH
PREPROCESSOR_PATH = cfg.PREPROCESSOR_PATH
BEST_PARAMS_JSON = cfg.BEST_PARAMS_JSON
PER_FOLD_CSV = cfg.PER_FOLD_TUNING_CSV

BASE_XGB = {"random_state": cfg.RANDOM_SEED, "n_jobs": -1, "verbosity": 0}


# ---------------------------------------------------------------- data & encoding
def load_features(path: Path | None = None) -> pd.DataFrame:
    """Baca features.parquet dan validasi Against kontrak (fail-fast)."""
    path = path or cfg.FEATURES
    if not path.exists():
        raise FileNotFoundError(
            f"{path} belum ada. Jalankan 01_clean.py lalu 02_features.py lebih dulu."
        )
    df = pd.read_parquet(path)
    missing = [c for c in cfg.FEATURE_COLS + [cfg.TARGET_CONT, cfg.TARGET_BIN] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang di features.parquet: {missing}")
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """One-hot untuk kategorikal, numerik dibiarkan.

    OneHot dipilih (bukan ordinal) karena XGBoost membelah nilai kategorikal secara
    acak-ordinal sehingga urutan kategori palsu mengacaukansplit; handle_unknown
    penting karena kategori langka (mis. tinggal grup n=13) bisa absen di fold train.
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", cfg.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                cfg.CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def encoded_feature_names(pre: ColumnTransformer) -> list[str]:
    """Nama fitur setelah encoding.

    Dipanggil di main() lalu disimpan ke best_params.json["feature_columns"]
    (fix blocker rev-model #4: sebelumnya fungsi ini tidak pernah dipanggil,
    sehingga 05/06 tidak punya daftar kolom untuk memetakan SHAP).
    """
    return [str(c) for c in pre.get_feature_names_out()]


# ---------------------------------------------------------------- tuning
def suggest_params(trial: optuna.Trial) -> dict:
    """Ambil satu setelan dari ruang pencarian config.XGB_SEARCH_SPACE."""
    params = {}
    for name, spec in cfg.XGB_SEARCH_SPACE.items():
        kind, lo, hi = spec[0], spec[1], spec[2]
        log = len(spec) > 3 and spec[3] == "log"
        if kind == "int":
            params[name] = trial.suggest_int(name, lo, hi, log=log)
        else:
            params[name] = trial.suggest_float(name, lo, hi, log=log)
    return params


def fit_score(params: dict, X_tr, y_tr, X_va, y_va) -> float:
    """Latih XGBRegressor dengan satu setelan -> RMSE OUT-OF-SAMPLE di (X_va, y_va).

    Fix blocker rev-model #1: error WAJIB dihitung pada inner-validation fold.
    Versi lama menghitung RMSE pada data yang sama dengan data latih (in-sample),
    sehingga objective/pruning/skor trials mengukur error train, bukan validasi.
    """
    model = XGBRegressor(**params, **BASE_XGB)
    model.fit(X_tr, y_tr)
    return float(mean_squared_error(y_va, model.predict(X_va)) ** 0.5)


def tune_on_train(X, y, seed: int, n_trials: int) -> tuple[dict, dict, list[dict]]:
    """Tuning Optuna HANYA pada data train. Kembalikan (params_1se, params_best, trials).

    Pruning Median: objective melaporkan rata-rata running RMSE tiap inner fold;
    trial yang jauh lebih buruk dari median trial sebelumnya dibuang lebih awal.
    """
    inner = KFold(n_splits=cfg.INNER_FOLDS, shuffle=True, random_state=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),  # seed sampler dari config
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = []
        for fold, (tr, va) in enumerate(inner.split(X)):
            # Skor pada va (baris yang TIDAK ikut dilatih), bukan pada tr.
            scores.append(fit_score(params, X[tr], y[tr], X[va], y[va]))
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                trial.set_user_attr("fold_scores", [float(s) for s in scores])
                raise TrialPruned()
        trial.set_user_attr("fold_scores", [float(s) for s in scores])
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    trials = study.trials
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("Tidak ada trial Optuna yang selesai; periksa ruang pencarian.")

    best = min(complete, key=lambda t: t.value)
    # 1-SE: ambang = rmse terbaik + 1 standard error BENAR dari fold-fold trial terbaik.
    # Fix blocker rev-model #2: SE dari daftar skor per-fold yang tersimpan,
    # bukan dari running mean kumulatif (yang membuat ambang terlalu sempit).
    best_folds = trial_fold_scores(best)
    inner_se = float(np.std(best_folds, ddof=1) / np.sqrt(len(best_folds))) if len(best_folds) > 1 else 0.0
    threshold = best.value + inner_se
    within = [t for t in complete if t.value <= threshold]
    chosen = min(within, key=lambda t: simplicity_key(t.params))
    return dict(chosen.params), dict(best.params), [trial_record(t) for t in trials]


def trial_fold_scores(trial: optuna.Trial) -> list[float]:
    """Daftar RMSE per inner fold milik sebuah trial (disimpan di user_attrs)."""
    return list(trial.user_attrs.get("fold_scores", []))


def std_of_best_folds(trial: optuna.Trial) -> float:
    """Deviasi standar RMSE antar inner fold (ddof=1) dari daftar skor per-fold.

    Fix blocker rev-model #2: sumbernya daftar skor per-fold, bukan intermediate
    values yang berisi running mean kumulatif.
    """
    values = trial_fold_scores(trial)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def simplicity_key(params: dict) -> tuple:
    """Urutan 'paling sederhana dulu' untuk aturan 1-SE: pohon lebih sedikit & dangkal,
    regularisasi lebih kuat, anak lebih banyak (lebih sedikit split)."""
    return (
        params["n_estimators"],
        params["max_depth"],
        params["reg_alpha"],
        params["reg_lambda"],
        params["min_child_weight"],
        -params["learning_rate"],
    )


def trial_record(trial: optuna.Trial) -> dict:
    """Baris untuk optuna_trials.csv.

    Nama kolom 'mean_test_rmse' mengikuti kontrak, tetapi NILAINYA adalah RMSE
    inner-CV pada data train — test fold sengaja belum tersentuh saat tuning.
    """
    return {
        "trial_number": trial.number,
        "params": json.dumps(trial.params, sort_keys=True),
        "mean_test_rmse": trial.value if trial.value is not None else "",
        "std_test_rmse": std_of_best_folds(trial) if trial.state == optuna.trial.TrialState.COMPLETE else "",
        "duration": getattr(trial, "duration", None).total_seconds() if getattr(trial, "duration", None) else "",
    }


# ---------------------------------------------------------------- nested CV
def scale_pos_weight(y_bin: np.ndarray, konteks: str) -> float:
    """Bobot kelas untuk data biner timpang, dihitung dari data latih yang diberikan.

    Fix temuan round-2 R2-4: target biner hanya 43/306 siswa (14,1%) positif.
    Tanpa class weighting, XGBClassifier condong ke kelas mayoritas sehingga
    balanced_accuracy ~0,5 dan Brier didominasi kelas 0 — keduanya tidak
    interpretable di metrics.json. Nilai dihitung per fold dari data TRAIN,
    bukan dari keseluruhan data (info test tidak boleh bocor).
    """
    pos = int(np.sum(y_bin == 1))
    neg = int(np.sum(y_bin == 0))
    if pos == 0 or neg == 0:
        raise ValueError(
            f"Data latih biner untuk {konteks} hanya punya satu kelas "
            f"(pos={pos}, neg={neg}); scale_pos_weight tidak bisa dihitung."
        )
    return neg / pos


def nested_cv(df: pd.DataFrame, n_trials: int, repeats: int, folds: int) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Outer loop 5-fold x N ulangan. Test fold tidak pernah masuk tuning.

    Mengembalikan (oof_predictions, per_fold_info, trial_log).
    """
    X_raw = df[cfg.FEATURE_COLS]
    y_cont = df[cfg.TARGET_CONT].to_numpy(dtype=float)
    y_bin = df[cfg.TARGET_BIN].to_numpy(dtype=int)
    row_ids = df["row_id"].to_numpy()

    oof_rows: list[dict] = []
    per_fold: list[dict] = []
    trial_offset = 0
    trial_log: list[dict] = []

    for rep in range(repeats):
        outer = KFold(n_splits=folds, shuffle=True, random_state=cfg.RANDOM_SEED + rep)
        for fold, (tr, va) in enumerate(outer.split(X_raw)):
            t0 = time.time()
            seed = cfg.RANDOM_SEED + 1000 * rep + fold

            pre = build_preprocessor(X_raw.iloc[tr])
            X_tr = pre.fit_transform(X_raw.iloc[tr])
            X_va = pre.transform(X_raw.iloc[va])

            params_1se, _params_best, trials = tune_on_train(X_tr, y_cont[tr], seed, n_trials)
            for rec in trials:
                rec["trial_number"] += trial_offset
            trial_log.extend(trials)
            trial_offset += len(trials)

            reg = XGBRegressor(**params_1se, **BASE_XGB)
            reg.fit(X_tr, y_cont[tr])
            # Bobot kelas dihitung dari data latih fold ini (R2-4).
            clf = XGBClassifier(
                **params_1se,
                objective="binary:logistic",
                scale_pos_weight=scale_pos_weight(y_bin[tr], f"rep {rep + 1} fold {fold + 1}"),
                **BASE_XGB,
            )
            clf.fit(X_tr, y_bin[tr])

            pred_cont = reg.predict(X_va)
            pred_bin = clf.predict_proba(X_va)[:, 1]
            fold_rmse = float(mean_squared_error(y_cont[va], pred_cont) ** 0.5)

            for i, idx in enumerate(va):
                oof_rows.append({
                    "row_id": row_ids[idx],
                    "repeat": rep,
                    "fold": fold,
                    "y_kontinu": y_cont[idx],
                    "pred_kontinu": float(pred_cont[i]),
                    "y_biner": int(y_bin[idx]),
                    "pred_proba_biner": float(pred_bin[i]),
                })
            per_fold.append({
                "repeat": rep, "fold": fold,
                "n_train": int(len(tr)), "n_test": int(len(va)),
                "test_rmse": fold_rmse,
                "inner_best_rmse": min(
                    t["mean_test_rmse"] for t in trials if t["mean_test_rmse"] != ""
                ),
                "duration_sec": round(time.time() - t0, 1),
            })
            print(
                f"[nested_cv] rep {rep + 1}/{repeats} fold {fold + 1}/{folds} "
                f"test_rmse={fold_rmse:.4f} ({per_fold[-1]['duration_sec']}s)",
                flush=True,
            )

    return pd.DataFrame(oof_rows), per_fold, trial_log


# ---------------------------------------------------------------- final model
def train_final(df: pd.DataFrame, params: dict) -> list[str]:
    """Latih model akhir di seluruh data dan simpan artefak.

    Preprocessor di-fit sekali di seluruh data lalu disimpan; nama kolom hasil
    encoding dikembalikan untuk ditulis ke best_params.json.
    """
    pre = build_preprocessor(df)
    X = pre.fit_transform(df[cfg.FEATURE_COLS])
    joblib.dump(pre, PREPROCESSOR_PATH)

    reg = XGBRegressor(**params, **BASE_XGB)
    reg.fit(X, df[cfg.TARGET_CONT].to_numpy(dtype=float))
    reg.save_booster(cfg.MODEL_PATH)

    y_bin_full = df[cfg.TARGET_BIN].to_numpy(dtype=int)
    clf = XGBClassifier(
        **params,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight(y_bin_full, "model final"),
        **BASE_XGB,
    )
    clf.fit(X, y_bin_full)
    clf.save_booster(MODEL_CLF_PATH)
    return encoded_feature_names(pre)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Training XGBoost + tuning Optuna (nested CV).")
    ap.add_argument("--n-trials", type=int, default=cfg.N_TRIALS_PER_OUTER_FOLD,
                    help="Trial Optuna per outer fold (default dari config).")
    ap.add_argument("--n-trials-final", type=int, default=cfg.N_TRIALS_FINAL,
                    help="Trial Optuna untuk tuning model final di seluruh data.")
    ap.add_argument("--repeats", type=int, default=cfg.OUTER_REPEATS, help="Ulangan outer CV.")
    ap.add_argument("--folds", type=int, default=cfg.OUTER_FOLDS, help="Lipatan outer CV.")
    args = ap.parse_args(argv)

    assert len(cfg.FEATURE_COLS) == 18, f"Fitur harus 18, dapat {len(cfg.FEATURE_COLS)}"

    df = load_features()
    y = df[cfg.TARGET_CONT].to_numpy(dtype=float)
    assert y.min() >= cfg.ANXIETY_VALID_RANGE[0] and y.max() <= cfg.ANXIETY_VALID_RANGE[1], \
        f"y_kontinu di luar rentang {cfg.ANXIETY_VALID_RANGE}"
    assert set(df[cfg.TARGET_BIN].unique()) <= {0, 1}, "y_biner harus 0/1"

    # Angka resmi lebih dulu (nested CV), lalu tuning di data penuh untuk model final.
    oof, per_fold, trials = nested_cv(df, args.n_trials, args.repeats, args.folds)
    oof.to_csv(OOF_PATH, index=False)
    pd.DataFrame(trials, columns=["trial_number", "params", "mean_test_rmse", "std_test_rmse", "duration"]).to_csv(cfg.TRIALS_CSV, index=False)
    pd.DataFrame(per_fold).to_csv(PER_FOLD_CSV, index=False)

    pre = build_preprocessor(df)
    X_full = pre.fit_transform(df[cfg.FEATURE_COLS])
    params_1se, params_best, _ = tune_on_train(X_full, y, cfg.RANDOM_SEED, args.n_trials_final)
    feature_columns = train_final(df, params_1se)
    BEST_PARAMS_JSON.write_text(json.dumps({
        # Kunci WAJIB kontrak (dipakai 05_explain.py & 06_report.py):
        "best_params": params_1se,
        "best_params_raw": params_best,
        "feature_columns": feature_columns,
        # Kunci lama dipertahankan sebagai jejak tambahan (tidak dipakai konsumen).
        "chosen_1se": params_1se,
        "optuna_best_raw": params_best,
        "clifier_uses": "parameter sama dengan regressor (dokumentasi di docstring)",
        "n_trials_per_outer_fold": args.n_trials,
        "n_trials_final": args.n_trials_final,
        "outer": {"folds": args.folds, "repeats": args.repeats},
    }, indent=2), encoding="utf-8")
    print(f"Selesai. Artefak: {cfg.MODEL_PATH}, {OOF_PATH}, {cfg.TRIALS_CSV}, {BEST_PARAMS_JSON}")


if __name__ == "__main__":
    main()