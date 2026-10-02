"""04_evaluate.py — Metrik resmi HANYA dari test fold (out-of-fold).

Sumber angka = outputs/oof_predictions.csv yang ditulis 03_train_tune.py
(prediksi tiap baris saat baris itu menjadi test fold; test tidak pernah masuk tuning).
Di sini TIDAK ada training maupun tuning — hanya penghitungan metrik + bootstrap CI.

Artefak: outputs/metrics.json (skema kontrak). CATATAN: tabel_performa.csv TIDAK ditulis
di sini (pemilik tunggal 06_report.py, sesuai revisi kontrak).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

OOF_PATH = cfg.OOF_CSV  # fix R3-6: satu sumber kebenaran path dari config
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = cfg.RANDOM_SEED


def load_oof(path: Path | None = None) -> pd.DataFrame:
    path = path or OOF_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} belum ada. Jalankan 03_train_tune.py lebih dulu.")
    df = pd.read_csv(path)
    required = {"row_id", "repeat", "fold", "y_kontinu", "pred_kontinu", "y_biner", "pred_proba_biner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom OOF hilang: {missing}")
    return df


def continuous_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def binary_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    yhat = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "brier": float(brier_score_loss(y, proba)),
    }


def bootstrap_ci95(
    y: np.ndarray, pred: np.ndarray, row_ids: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> dict:
    """CI95 percentile bootstrap dengan resampling per SISWA (cluster bootstrap).

    Fix blocker rev-model #6: versi lama me-resample baris OOF (3060 = 306 siswa x 10
    ulangan) seolah independen, padahal 10 baris milik siswa yang sama berkorelasi
    kuat -> CI terlalu sempit. Di sini yang di-resample adalah row_id unik; semua
    prediksi out-of-fold milik siswa yang terpilih ikut dibawa.
    """
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(row_ids)
    by_id = {rid: np.flatnonzero(row_ids == rid) for rid in unique_ids}
    n = len(unique_ids)
    stats = {"rmse": [], "mae": [], "r2": []}
    for _ in range(n_boot):
        picked = rng.choice(unique_ids, size=n, replace=True)
        idx = np.concatenate([by_id[rid] for rid in picked])
        m = continuous_metrics(y[idx], pred[idx])
        for k in stats:
            stats[k].append(m[k])
    out = {}
    for k, vals in stats.items():
        arr = np.asarray(vals)
        out[k] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    return out


def per_repeat_table(oof: pd.DataFrame) -> list[dict]:
    """Metrik per ulangan = bukti stabilitas (PLAN bagian 7).

    AUC per ulangan hanya dihitung bila kedua kelas ada di test fold tersebut.
    """
    rows = []
    for rep, g in oof.groupby("repeat"):
        row = {"repeat": int(rep), "n_test": int(len(g))}
        row.update(continuous_metrics(g["y_kontinu"].to_numpy(), g["pred_kontinu"].to_numpy()))
        yb, pb = g["y_biner"].to_numpy(int), g["pred_proba_biner"].to_numpy(float)
        row["auc"] = float(roc_auc_score(yb, pb)) if len(np.unique(yb)) == 2 else None
        rows.append(row)
    return rows


def main() -> None:
    oof = load_oof()
    y = oof["y_kontinu"].to_numpy(dtype=float)
    pred = oof["pred_kontinu"].to_numpy(dtype=float)
    yb = oof["y_biner"].to_numpy(dtype=int)
    pb = oof["pred_proba_biner"].to_numpy(dtype=float)
    row_ids = oof["row_id"].to_numpy()

    # assert level-1 (fix blocker rev-model #7; kontrak aturan 5)
    n_siswa = int(len(np.unique(row_ids)))
    n_repeat = int(oof["repeat"].nunique())
    assert len(oof) == n_siswa * n_repeat, (
        f"Jumlah baris OOF {len(oof)} harus = siswa ({n_siswa}) x ulangan ({n_repeat})."
    )
    # Fix R3-8: defensif — pasangan (row_id, repeat) harus unik (1 prediksi per siswa per ulangan).
    assert not oof.duplicated(["row_id", "repeat"]).any(), \
        "ada pasangan (row_id, repeat) terduplikasi di OOF — fold/ulangan tidak konsisten"
    assert y.min() >= cfg.ANXIETY_VALID_RANGE[0] and y.max() <= cfg.ANXIETY_VALID_RANGE[1], \
        f"y_kontinu di luar rentang {cfg.ANXIETY_VALID_RANGE}"
    assert 0.0 <= pb.min() and pb.max() <= 1.0, "probabilitas biner harus di rentang 0-1"

    cont = continuous_metrics(y, pred)
    cont["ci95"] = bootstrap_ci95(y, pred, row_ids)
    # n_evals = satuan independen (siswa), bukan jumlah baris OOF (blocker #6)
    cont["n_evals"] = n_siswa
    cont["n_prediksi"] = int(len(oof))
    binm = binary_metrics(yb, pb)

    metrics = {
        "continuous": cont,
        "binary": binm,
        "per_repeat": per_repeat_table(oof),
    }
    cfg.METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({"continuous": {k: v for k, v in cont.items() if k != "ci95"},
                      "binary": binm}, indent=2))
    print(f"Selesai. metrics.json -> {cfg.METRICS_JSON}")


if __name__ == "__main__":
    main()