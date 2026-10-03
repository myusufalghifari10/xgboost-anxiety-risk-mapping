"""03_train_tune.py — Training XGBoost + tuning masif Optuna (nested CV).

Skema (PLAN bagian 5-6):
    Outer: OUTER_FOLDS x OUTER_REPEATS, shuffle, seed per ulangan.
           Test fold = 62/61 baris (306 dibagi 5 -> [62,61,61,61,61], fold pertama dapat 62),
           TIDAK PERNAH dipakai tuning maupun pemilihan setelan.
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
    outputs/checkpoints/          (status RESUME: progres.json, ronde_*.json,
                                   optuna_final.db — bukan deliverable)

CHECKPOINT/RESUME: tiap ronde outer-CV disimpan begitu selesai; tuning final
memakai SQLite. Proses terputus? Jalankan ulang perintah yang sama — yang sudah
selesai dilewati, hanya sisa yang dijalankan. --fresh menghapus semua checkpoint.

CATATAN: skrip ini menjalankan training — urutan eksekusi pipeline:
01 -> 02 -> 03 (skrip ini) -> 04 -> 05 -> 06.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import TrialPruned  # TrialPruned selalu diekspor optuna.exceptions (bukan optuna.pruners)
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
CKPT_DIR = cfg.CHECKPOINT_DIR  # checkpoint/resume (status antara, bukan deliverable)

BASE_XGB = cfg.BASE_XGB  # fix C4: satu sumber di config (dulu duplikat di 03 dan 05)


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


def _make_objective(X, y, inner: KFold):
    """Objective Optuna: RMSE inner-CV dengan pruning median.

    Dipakai tune_on_train (per ronde outer) DAN tuning_final (model akhir)
    supaya kedua jalur tuning tidak mungkin menyimpang satu sama lain.
    """
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

    return objective


def _pilih_1se(study: optuna.Study) -> tuple[dict, dict]:
    """Pilih (params_1se, params_best) dari study: aturan 1-SE + simplicity_key.

    1-SE: ambang = rmse terbaik + 1 standard error BENAR dari fold-fold trial terbaik.
    Fix blocker rev-model #2: SE dari daftar skor per-fold yang tersimpan,
    bukan dari running mean kumulatif (yang membuat ambang terlalu sempit).
    """
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("Tidak ada trial Optuna yang selesai; periksa ruang pencarian.")

    best = min(complete, key=lambda t: t.value)
    best_folds = trial_fold_scores(best)
    # Fix B1: jangan diam-diam jatuh ke argmin saat daftar skor per-fold hilang/incomplete.
    if len(best_folds) != cfg.INNER_FOLDS:
        raise RuntimeError(
            f"fold_scores trial terbaik tidak lengkap: {len(best_folds)} != {cfg.INNER_FOLDS}; "
            "standard error aturan 1-SE tidak bisa dihitung dengan benar."
        )
    inner_se = float(np.std(best_folds, ddof=1) / np.sqrt(len(best_folds)))
    threshold = best.value + inner_se
    within = [t for t in complete if t.value <= threshold]
    chosen = min(within, key=lambda t: simplicity_key(t.params))
    return dict(chosen.params), dict(best.params)


def tune_on_train(X, y, seed: int, n_trials: int) -> tuple[dict, dict, list[dict]]:
    """Tuning Optuna HANYA pada data train. Kembalikan (params_1se, params_best, trials).

    Pruning Median: objective melaporkan rata-rata running RMSE tiap inner fold;
    trial yang jauh lebih buruk dari median trial sebelumnya dibuang lebih awal.
    """
    inner = KFold(n_splits=cfg.INNER_FOLDS, shuffle=True, random_state=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),  # seed sampler dari config
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),  # fix F4: sebelumnya Nopruner (docstring bohong)
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(_make_objective(X, y, inner), n_trials=n_trials, gc_after_trial=True)
    params_1se, params_best = _pilih_1se(study)
    return params_1se, params_best, [trial_record(t) for t in study.trials]


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
    """Urutan 'paling sederhana dulu' untuk aturan 1-SE (fix A1).

    Definisi 'sederhana' = estimasi total daun ``n_estimators * 2**max_depth`` lebih kecil
    dulu; bila total daun sama, regularisasi lebih kuat (reg_alpha/reg_lambda/
    min_child_weight lebih BESAR) dan learning_rate lebih kecil dipilih lebih dulu.
    Kunci lama (n_estimators dulu) terbukti bisa memilih model 1,2 juta daun di atas
    model 4.800 daun pada tie 1-SE. min() memilih nilai terkecil, jadi kunci
    'lebih sederhana' harus mengecil saat model menyederhana.
    """
    return (
        params["n_estimators"] * 2 ** params["max_depth"],
        -params["reg_alpha"],
        -params["reg_lambda"],
        -params["min_child_weight"],
        params["learning_rate"],
    )


def trial_record(trial: optuna.Trial) -> dict:
    """Baris untuk optuna_trials.csv.

    Nama kolom 'mean_test_rmse' mengikuti kontrak, tetapi NILAINYA adalah RMSE
    inner-CV pada data train — test fold sengaja belum tersentuh saat tuning.
    Fix B2/D1: kolom numerik selalu float NaN (bukan string ""). Eksperimen
    langsung di optuna 5.0 membuktikan trial PRUNED tetap punya
    ``value = intermediate TERAKHIR`` (rata-rata parsial <5 fold, BUKAN None),
    jadi mean_test_rmse/std_test_rmse hanya diisi bila state COMPLETE; selain
    itu NaN. Kolom 'state' hanya dipakai penyaring in-memory (CSV tetap 5 kolom
    kontrak — kolom state tidak ikut ditulis).
    """
    durasi = getattr(trial, "duration", None)
    lengkap = trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    return {
        "trial_number": trial.number,
        "params": json.dumps(trial.params, sort_keys=True),
        "state": trial.state.name,
        "mean_test_rmse": float(trial.value) if lengkap else np.nan,
        "std_test_rmse": (
            std_of_best_folds(trial) if lengkap else np.nan
        ),
        "duration": durasi.total_seconds() if durasi is not None else np.nan,
    }


def _best_complete_rmse(trials: list[dict]) -> float:
    """RMSE inner-CV terbaik dari trial COMPLETE saja (fix D1 + guard D3d).

    Trial PRUNED di optuna 5.0 membawa value = rata-rata parsial fold yang belum
    lengkap (terverifikasi eksperimen), sehingga penyaring NaN saja tidak cukup —
    barisnya harus state COMPLETE. Bila tidak ada satu pun trial COMPLETE,
    gagal dengan pesan jelas (guard D3d), bukan ValueError min() yang telanjang.
    """
    nilai = [
        t["mean_test_rmse"] for t in trials
        if t["state"] == "COMPLETE" and not pd.isna(t["mean_test_rmse"])
    ]
    if not nilai:
        raise RuntimeError(
            "Tidak ada trial Optuna COMPLETE pada ronde ini — inner_best_rmse tidak bisa dihitung."
        )
    return float(min(nilai))


# ---------------------------------------------------------------- checkpoint / resume
def _fingerprint(n_trials: int, n_trials_final: int, repeats: int, folds: int) -> dict:
    """Identitas run: resume hanya sah untuk konfigurasi + data yang sama persis.

    Fix review F3: fingerprint juga mengunci NAMA fitur, ruang pencarian, INNER_FOLDS,
    dan BASE_XGB (bukan hanya jumlah fitur) — mengubah salah satunya membuat checkpoint
    lama tidak sah. Fix review F4: pemakaian dipisah di main() — fp_cv (tanpa
    n_trials_final) untuk checkpoint ronde, fp_final untuk study tuning final,
    sehingga mengubah --n-trials-final tidak menghapus ronde CV yang sudah selesai.
    """
    data_hash = hashlib.sha256(cfg.FEATURES.read_bytes()).hexdigest()[:16]
    cfg_hash = hashlib.sha256(json.dumps({
        "num": cfg.NUMERIC_FEATURES, "cat": cfg.CATEGORICAL_FEATURES,
        "space": cfg.XGB_SEARCH_SPACE, "inner": cfg.INNER_FOLDS, "base": cfg.BASE_XGB,
    }, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "n_trials": int(n_trials), "n_trials_final": int(n_trials_final),
        "repeats": int(repeats), "folds": int(folds),
        "seed": cfg.RANDOM_SEED, "n_fitur": len(cfg.FEATURE_COLS), "data_sha256_16": data_hash,
        "cfg_sha256": cfg_hash,
    }


def _checkpoint_path(rep: int, fold: int) -> Path:
    return CKPT_DIR / f"ronde_r{rep}_f{fold}.json"


def _simpan_checkpoint(rep: int, fold: int, oof_round: list[dict], trials: list[dict],
                       meta: dict, fp: dict) -> None:
    """Simpan hasil SATU ronde outer-CV secara atomik (tmp + os.replace).

    Ronde tanpa checkpoint = belum selesai -> akan diulang penuh saat resume.
    """
    path = _checkpoint_path(rep, fold)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {"fingerprint": fp, "oof": oof_round, "trials": trials, "meta": meta},
            # Safety net: numpy scalar (np.int64/np.float64) -> Python scalar.
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


# ---------------------------------------------------------------- nested CV
def scale_pos_weight(y_bin: np.ndarray, konteks: str) -> float:
    """Bobot kelas untuk data biner timpang, dihitung dari data latih yang diberikan.

    Fix temuan round-2 R2-4: target biner hanya 43/306 siswa (14,1%) positif.
    Tanpa class weighting, XGBClassifier condong ke kelas mayoritas sehingga
    balanced_accuracy ~0,5 (tidak informatif).

    Fix A2 (klaim lama terbalik): weighting membantu balanced_accuracy, tetapi
    justru MENGGESER probabilitas ke prior 50/50 sehingga probabilitas TIDAK lagi
    terkalibrasi dan Brier dari model berbobot BUKAN ukuran kualitas probabilitas
    untuk prevalensi asli 14,1% — Brier harus dikutip dengan caveat kalibrasi.
    AUC tetap sahih karena hanya bergantung urutan peringkat. Nilai dihitung per
    fold dari data TRAIN, bukan dari keseluruhan data (info test tidak boleh bocor).
    """
    pos = int(np.sum(y_bin == 1))
    neg = int(np.sum(y_bin == 0))
    if pos == 0 or neg == 0:
        raise ValueError(
            f"Data latih biner untuk {konteks} hanya punya satu kelas "
            f"(pos={pos}, neg={neg}); scale_pos_weight tidak bisa dihitung."
        )
    return neg / pos


def nested_cv(df: pd.DataFrame, n_trials: int, repeats: int, folds: int, fp: dict) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Outer loop 5-fold x N ulangan. Test fold tidak pernah masuk tuning.

    RESUME: tiap ronde yang selesai langsung disimpan ke outputs/checkpoints/.
    Bila proses terputus, jalankan ulang perintah yang sama — ronde yang sudah
    ber-checkpoint dilewati. Ronde saling independen (seed deterministik per
    ronde), sehingga resume menghasilkan angka IDENTIK dengan run penuh.

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

    # Resume hanya sah untuk fingerprint yang sama (konfigurasi + data identik).
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    marker = CKPT_DIR / "progres.json"
    resume = False
    if marker.exists():
        try:
            fp_lama = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fp_lama = None
        if fp_lama == fp:
            resume = True
            print("[nested_cv] checkpoint ditemukan — melanjutkan ronde yang belum selesai.", flush=True)
        else:
            for p in sorted(CKPT_DIR.glob("ronde_*.json*")):  # + *.json.tmp yatim (fix review F8)
                p.unlink()
            print("[nested_cv] konfigurasi/data berbeda dari checkpoint lama — memulai dari nol.", flush=True)
    marker.write_text(json.dumps(fp, sort_keys=True), encoding="utf-8")

    for rep in range(repeats):
        outer = KFold(n_splits=folds, shuffle=True, random_state=cfg.RANDOM_SEED + rep)
        for fold, (tr, va) in enumerate(outer.split(X_raw)):
            cpath = _checkpoint_path(rep, fold)
            if resume and cpath.exists():
                isi = json.loads(cpath.read_text(encoding="utf-8"))
                assert isi["fingerprint"] == fp, f"checkpoint {cpath.name} tidak cocok fingerprint"
                oof_rows.extend(isi["oof"])
                per_fold.append(isi["meta"])
                trial_log.extend(isi["trials"])
                trial_offset += len(isi["trials"])
                print(
                    f"[nested_cv] rep {rep + 1}/{repeats} fold {fold + 1}/{folds} "
                    "dilewati (checkpoint)",
                    flush=True,
                )
                continue

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

            oof_round: list[dict] = []
            for i, idx in enumerate(va):
                oof_round.append({
                    "row_id": int(row_ids[idx]),
                    "repeat": rep,
                    "fold": fold,
                    "y_kontinu": float(y_cont[idx]),
                    "pred_kontinu": float(pred_cont[i]),
                    "y_biner": int(y_bin[idx]),
                    "pred_proba_biner": float(pred_bin[i]),
                })
            oof_rows.extend(oof_round)
            per_fold.append({
                "repeat": rep, "fold": fold,
                "n_train": int(len(tr)), "n_test": int(len(va)),
                "test_rmse": fold_rmse,
                # Fix D1: hanya trial COMPLETE (baris PRUNED = NaN, jangan tercampur).
                "inner_best_rmse": _best_complete_rmse(trials),
                "duration_sec": round(time.time() - t0, 1),
            })
            print(
                f"[nested_cv] rep {rep + 1}/{repeats} fold {fold + 1}/{folds} "
                f"test_rmse={fold_rmse:.4f} ({per_fold[-1]['duration_sec']}s)",
                flush=True,
            )
            _simpan_checkpoint(rep, fold, oof_round, trials, per_fold[-1], fp)

    return pd.DataFrame(oof_rows), per_fold, trial_log


def tuning_final(X, y, n_trials_final: int, fp: dict) -> tuple[dict, dict]:
    """Tuning model akhir di seluruh data dengan RESUME (SQLite).

    Trial tersimpan di cfg.OPTUNA_FINAL_DB; bila proses terputus, jalankan ulang
    perintah yang sama — trial yang sudah selesai dipertahankan dan hanya sisa
    trial yang dijalankan. Study name memuat digest fingerprint sehingga
    konfigurasi/data yang berbeda tidak pernah mencampur trial lama.
    Catatan jujur: stream sampling TPE setelah resume bisa berbeda tipis dari run
    penuh tak terputus (RNG sampler tidak di-bridge); run penuh tetap golden
    reference untuk reproduksi paper.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    inner = KFold(n_splits=cfg.INNER_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
    digest = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:12]
    study = optuna.create_study(
        study_name=f"final_{digest}",
        storage=f"sqlite:///{cfg.OPTUNA_FINAL_DB}",
        direction="minimize",
        sampler=TPESampler(seed=cfg.RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Fix review F1: trial RUNNING yatim (kill -9) tidak pernah dibersihkan otomatis
    # (heartbeat mati) — hanya hitung yang is_finished(), jangan ikut menghitung hantu.
    sudah = sum(1 for t in study.trials if t.state.is_finished())
    sisa = int(n_trials_final) - sudah
    if sisa > 0:
        print(f"[tuning_final] {sudah}/{n_trials_final} trial tersimpan — menjalankan {sisa} sisa.", flush=True)
        study.optimize(_make_objective(X, y, inner), n_trials=sisa, gc_after_trial=True)
    else:
        print(f"[tuning_final] {sudah} trial sudah tersimpan — tidak ada yang perlu dijalankan.", flush=True)
    return _pilih_1se(study)


# ---------------------------------------------------------------- final model
def train_final(df: pd.DataFrame, params: dict):
    """Latih model akhir di seluruh data. Mengembalikan (nama_kolom, pre, reg, clf).

    Fix review F2: penyimpanan TIDAK dilakukan di sini — dipindah ke main() setelah
    best_params.json ditulis, supaya tiga artefak model tidak pernah tersimpan tanpa
    best_params.json yang cocok (dulu menjadi lubang di guard generasi 06).
    """
    pre = build_preprocessor(df)
    X = pre.fit_transform(df[cfg.FEATURE_COLS])

    reg = XGBRegressor(**params, **BASE_XGB)
    reg.fit(X, df[cfg.TARGET_CONT].to_numpy(dtype=float))

    y_bin_full = df[cfg.TARGET_BIN].to_numpy(dtype=int)
    clf = XGBClassifier(
        **params,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight(y_bin_full, "model final"),
        **BASE_XGB,
    )
    clf.fit(X, y_bin_full)
    return encoded_feature_names(pre), pre, reg, clf


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Training XGBoost + tuning Optuna (nested CV).")
    ap.add_argument("--n-trials", type=int, default=cfg.N_TRIALS_PER_OUTER_FOLD,
                    help="Trial Optuna per outer fold (default dari config).")
    ap.add_argument("--n-trials-final", type=int, default=cfg.N_TRIALS_FINAL,
                    help="Trial Optuna untuk tuning model final di seluruh data.")
    ap.add_argument("--repeats", type=int, default=cfg.OUTER_REPEATS, help="Ulangan outer CV.")
    ap.add_argument("--folds", type=int, default=cfg.OUTER_FOLDS, help="Lipatan outer CV.")
    ap.add_argument("--fresh", action="store_true",
                    help="Abaikan dan hapus checkpoint lama; mulai dari nol.")
    args = ap.parse_args(argv)

    assert len(cfg.FEATURE_COLS) == 18, f"Fitur harus 18, dapat {len(cfg.FEATURE_COLS)}"

    df = load_features()
    y = df[cfg.TARGET_CONT].to_numpy(dtype=float)
    assert y.min() >= cfg.ANXIETY_VALID_RANGE[0] and y.max() <= cfg.ANXIETY_VALID_RANGE[1], \
        f"y_kontinu di luar rentang {cfg.ANXIETY_VALID_RANGE}"
    assert set(df[cfg.TARGET_BIN].unique()) <= {0, 1}, "y_biner harus 0/1"

    # Checkpoint/resume: fingerprint menentukan apakah checkpoint lama masih sah.
    # Fix review F4: pisahkan kunci — mengubah --n-trials-final tidak boleh menghapus
    # ronde CV yang sudah selesai, dan mengubah --n-trials tidak boleh membuang
    # trial tuning final.
    fp = _fingerprint(args.n_trials, args.n_trials_final, args.repeats, args.folds)
    fp_cv = {k: v for k, v in fp.items() if k != "n_trials_final"}
    fp_final = {k: fp[k] for k in ("n_trials_final", "seed", "cfg_sha256", "data_sha256_16")}
    if args.fresh:
        for p in sorted(CKPT_DIR.glob("ronde_*.json*")):
            p.unlink()
        (CKPT_DIR / "progres.json").unlink(missing_ok=True)
        cfg.OPTUNA_FINAL_DB.unlink(missing_ok=True)
        print("[main] --fresh: checkpoint lama dibuang.", flush=True)

    # Angka resmi lebih dulu (nested CV), lalu tuning di data penuh untuk model final.
    # Fix D2a: oof/trials/per_fold TIDAK ditulis di sini — semua artefak data ditulis
    # di akhir setelah best_params.json sukses (satu titik komit), supaya run terputus
    # (mis. di tengah tuning final berjam-jam) tidak meninggalkan artefak generasi baru;
    # 06_report menolak campur generasi (guard D2b).
    oof, per_fold, trials = nested_cv(df, args.n_trials, args.repeats, args.folds, fp_cv)

    pre = build_preprocessor(df)
    X_full = pre.fit_transform(df[cfg.FEATURE_COLS])
    params_1se, params_best = tuning_final(X_full, y, args.n_trials_final, fp_final)
    feature_columns, pre_final, reg_final, clf_final = train_final(df, params_1se)
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

    # Fix review F2: simpan model/preprocessor SETELAH best_params.json — ketiganya
    # kini ikut dijaga guard generasi 06 (dulu tersembunyi dan bisa campur generasi).
    joblib.dump(pre_final, PREPROCESSOR_PATH)
    reg_final.save_model(cfg.MODEL_PATH)  # fix F1: save_booster() tidak ada di xgboost 3.x
    clf_final.save_model(MODEL_CLF_PATH)

    # Fix D2a: satu titik komit — tulis artefak data SETELAH best_params.json sukses.
    oof.to_csv(OOF_PATH, index=False)
    pd.DataFrame(trials, columns=["trial_number", "params", "mean_test_rmse", "std_test_rmse", "duration"]).to_csv(cfg.TRIALS_CSV, index=False)
    pd.DataFrame(per_fold).to_csv(PER_FOLD_CSV, index=False)
    print(f"Selesai. Artefak: {cfg.MODEL_PATH}, {OOF_PATH}, {cfg.TRIALS_CSV}, {BEST_PARAMS_JSON}")


if __name__ == "__main__":
    main()