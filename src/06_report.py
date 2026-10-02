"""Tahap 8 — Laporan ringkas, tabel, dan peta risiko.

Menggabungkan artefak Model #1 menjadi keluaran untuk Bu Rita, paper Q3, dan
policy brief. Hanya BACA artefak yang sudah ada (dari 04_evaluate.py dan
05_explain.py); skrip ini tidak melatih model.

Keluaran (nama persis sesuai CONTRACT.md):
  outputs/tables/report_ringkas.md    laporan ringkas Bahasa Indonesia
  outputs/tables/tabel_performa.csv   ringkasan metrik dari metrics.json
  outputs/tables/tabel_faktor.csv     peringkat faktor risiko (SHAP + stabilitas)
  outputs/tables/risk_map_jurusan.csv peta risiko-zona per jurusan x jenis kelamin
  outputs/tables/risk_map_jeniskelamin.csv (sama, dikelompokkan per jenis kelamin)

Definisi zona risiko (dokumentasi, PLAN 8 + keputusan #8):
  Dipakai pada prediksi model (skor kecemasan 1-4). Ambang diambil dari config.py
  (ZONE_HIJAU_MAX, ZONE_KUNING_MAX) agar tidak menyimpang dari definisi di config:
    hijau  : skor < ZONE_HIJAU_MAX
    kuning : ZONE_HIJAU_MAX <= skor < ZONE_KUNING_MAX
    merah  : skor >= ZONE_KUNING_MAX (= ANXIETY_HIGH_THRESHOLD)
  Zona bersifat agregat per kelompok; TIDAK dipakai sebagai label individual.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import config as C

ZONE_HIJAU_MAX = C.ZONE_HIJAU_MAX
ZONE_KUNING_MAX = C.ZONE_KUNING_MAX


# ---------------------------------------------------------------------------
# Pembaca artefak
# ---------------------------------------------------------------------------
def require(path: Path, dibuat_oleh: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} belum ada. Jalankan {dibuat_oleh} lebih dulu."
        )
    return path


def load_metrics() -> Dict:
    return json.loads(require(C.METRICS_JSON, "04_evaluate.py").read_text(encoding="utf-8"))


def load_shap_ranking() -> pd.DataFrame:
    """Peringkat faktor dari outputs/shap_global_ranking.csv (ditulis 05_explain.py).

    Fix blocker rev-explain B1: versi lama membaca stability_ranking.csv yang tidak
    punya kolom mean_abs_shap sehingga 06 crash tepat sebelum menulis laporan.
    """
    return pd.read_csv(require(C.SHAP_GLOBAL_RANKING, "05_explain.py"))


def load_drop_hb() -> Dict:
    return json.loads(
        require(C.DROP_HB_JSON, "05_explain.py").read_text(encoding="utf-8")
    )


def load_stability() -> pd.DataFrame:
    return pd.read_csv(require(C.STABILITY_RANKING, "05_explain.py"))


def load_model_meta() -> Dict:
    """Baca best_params.json (kunci kontrak) untuk jejak tuning pada laporan."""
    path = require(C.BEST_PARAMS_JSON, "03_train_tune.py")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not meta.get("best_params"):
        raise KeyError("best_params.json tidak punya kunci 'best_params'.")
    return meta


def load_features() -> pd.DataFrame:
    return pd.read_parquet(require(C.FEATURES, "02_features.py"))


# ---------------------------------------------------------------------------
# Zona risiko
# ---------------------------------------------------------------------------
def zona_skor(skor: float) -> str:
    """Petakan skor prediksi (1-4) ke zona merah/kuning/hijau (lihat docstring modul).

    Fix catatan rev-explain N8: skor NaN tidak boleh jatuh ke zona "hijau" (zona
    terbaik) karena kedua perbandingan terhadap NaN bernilai False.
    """
    if skor is None or not np.isfinite(skor):
        raise ValueError("Skor prediksi NaN/tidak valid; sumbernya oof_predictions.csv tidak lengkap.")
    if skor >= ZONE_KUNING_MAX:
        return "merah"
    if skor >= ZONE_HIJAU_MAX:
        return "kuning"
    return "hijau"


def _oof_skor_per_siswa() -> pd.Series:
    """Skor kecemasan out-of-fold per siswa = rata-rata prediksi OOF miliknya.

    Fix catatan rev-explain N6: versi lama men-score 306 baris dengan model_final.ubj
    yang justru dilatih pada 306 baris itu (in-sample) sehingga proporsi zona merah
    pada policy brief Systematicly terinflasi. Sekarang memakai prediksi out-of-fold
    dari oof_predictions.csv (rata-rata antar ulangan).
    """
    oof = pd.read_csv(require(C.OOF_CSV, "03_train_tune.py"))
    for col in ("row_id", "pred_kontinu"):
        if col not in oof.columns:
            raise KeyError(f"oof_predictions.csv tidak punya kolom '{col}'.")
    per_siswa = oof.groupby("row_id")["pred_kontinu"].mean()
    if per_siswa.isna().any():
        raise ValueError("Ada siswa tanpa prediksi OOF; periksa keluaran 03_train_tune.py.")
    return per_siswa.rename("skor_prediksi")


# ---------------------------------------------------------------------------
# Tabel
# ---------------------------------------------------------------------------
def _ci_bounds(ci) -> tuple:
    """Terima dua bentuk ci95 dari 04_evaluate: [bawah, atas] atau {bawah, atas}."""
    if ci is None:
        return None, None
    if isinstance(ci, dict):
        return ci.get("bawah", ci.get("lower")), ci.get("atas", ci.get("upper"))
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return ci[0], ci[1]
    return None, None


def tabel_performa(metrics: Dict, n_trial: int, n_outer: int) -> pd.DataFrame:
    """Ringkasan metrik resmi (dari test) + jejak tuning, satu baris per metrik."""
    cont = metrics.get("continuous", {})
    binary = metrics.get("binary", {})
    rows: List[Dict] = []
    for key in C.REPORT_METRICS:
        if key in cont:
            row = {"target": "kontinu", "metrik": key, "nilai": cont[key]}
            ci = (cont.get("ci95") or {}).get(key)
            bawah, atas = _ci_bounds(ci)
            if bawah is not None:
                row["ci95_bawah"], row["ci95_atas"] = bawah, atas
            rows.append(row)
    for key in C.REPORT_METRICS_BIN:
        if key in binary:
            rows.append({"target": "biner", "metrik": key, "nilai": binary[key]})
    # Fix catatan rev-explain N15: tulis "trial per ronde x jumlah ronde", bukan
    # total baris trials CSV (yang dulu ditulis seolah-olah 250.000 setelan unik).
    rows.append({"target": "tuning", "metrik": "n_trials", "nilai": int(n_trial)})
    rows.append({"target": "tuning", "metrik": "n_ronde_outer", "nilai": int(n_outer)})
    return pd.DataFrame(rows)


def tabel_faktor(ranking: pd.DataFrame, kestabilan: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan peringkat SHAP global dengan kestabilannya antar fold.

    Fix blocker rev-explain B2: versi lama menghitung 'arah' dari shap_local.csv
    yang hanya berisi top-5 per siswa, sehingga arah tiap fitur bias seleksi dan
    fitur yang tak pernah masuk top-5 diberi arah karangan. Sekarang arah diambil
    dari SHAP global semua siswa (shap_global_ranking.csv) yang juga menjadi sumber
    shap_global.png. Fix N11: urutan mengikuti mean_abs_shap, bukan proporsi_top5.
    Kolom kestabilan tidak di-hardcode agar tetap sinkron bila 05 mengubah
    jumlah fitur teratas yang dihitung stabil.
    """
    out = ranking.merge(kestabilan, on="fitur", how="left")
    return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def risk_map(df: pd.DataFrame, skor: pd.Series, by: List[str]) -> pd.DataFrame:
    """Peta zona per kelompok (mis. ['d_jurusan','d_jenis_kelamin']).

    Ambang tingkat risiko kelompok diambil dari config (GROUP_RISK_HIGH/MODERATE).
    Kolom n_hilang ditambahkan supaya penyebut proporsi selalu sama dengan n_siswa.
    """
    tmp = df[["row_id"] + by].copy()
    tmp = tmp.merge(skor.rename_axis("row_id").reset_index(), on="row_id", how="left")
    tmp["zona"] = tmp["skor_prediksi"].apply(zona_skor)
    grouped = (
        tmp.groupby(by)
        .agg(
            n_siswa=("row_id", "count"),
            skor_rata2=("skor_prediksi", "mean"),
            n_merah=("zona", lambda z: int((z == "merah").sum())),
            n_kuning=("zona", lambda z: int((z == "kuning").sum())),
            n_hijau=("zona", lambda z: int((z == "hijau").sum())),
            n_hilang=("skor_prediksi", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    grouped["proporsi_merah"] = grouped["n_merah"] / grouped["n_siswa"]
    grouped["tingkat_risiko_kelompok"] = np.where(
        grouped["proporsi_merah"] >= C.GROUP_RISK_HIGH,
        "TINGGI",
        np.where(grouped["proporsi_merah"] >= C.GROUP_RISK_MODERATE, "SEDANG", "RENDAH"),
    )
    return grouped.sort_values("proporsi_merah", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Laporan ringkas
# ---------------------------------------------------------------------------
def tulis_laporan(
    performa: pd.DataFrame, faktor: pd.DataFrame, map_j: pd.DataFrame, drop: Dict, n_trial: int, n_outer: int
) -> Path:
    m = {}
    for _, r in performa.iterrows():
        m[(r["target"], r["metrik"])] = r["nilai"]

    top5 = faktor.head(5)[["fitur", "arah", "mean_abs_shap"]]
    baris_top = "\n".join(
        f"| {i+1} | {r.fitur} | {r.arah.replace('_', ' ')} | {r.mean_abs_shap:.4f} |"
        for i, r in enumerate(top5.itertuples())
    )
    map_tinggi = map_j[map_j["tingkat_risiko_kelompok"] == "TINGGI"]
    baris_map = (
        "\n".join(
            f"- {r.d_jurusan} / {r.d_jenis_kelamin}: {r.proporsi_merah:.0%} siswa zona merah"
            for r in map_tinggi.head(5).itertuples()
        )
        or "- Tidak ada kelompok dengan proporsi zona merah tinggi."
    )

    isi = f"""# Laporan Ringkas — Model 1: Peta Risiko Kecemasan (XGBoost)

## Ringkasan performa (dari data test yang tidak pernah dipakai training/tuning)
- Skor kecemasan (kontinu): RMSE = **{m.get(('kontinu','rmse'), float('nan')):.3f}**, MAE = {m.get(('kontinu','mae'), float('nan')):.3f}, R2 = {m.get(('kontinu','r2'), float('nan')):.3f}
- Kategori cemas tinggi (biner): AUC = **{m.get(('biner','auc'), float('nan')):.3f}**
- Total hyperparameter yang dicoba: **{n_trial:,} trial x {n_outer} ronde outer-CV** (Optuna; test tidak pernah dilihat proses ini)

## 5 faktor risiko teratas (SHAP)
| Peringkat | Faktor | Arah | Rata-rata \\|SHAP\\| |
|---|---|---|---|
{baris_top}

## Uji klaim perilaku sehat (drop-HB)
- Menghapus fitur `f_perilaku_sehat` mengubah RMSE test sebesar **{drop['selisih_rmse_mean']:+.4f}** (rata-rata {drop['n_folds']} fold).
- {'Arah positif berarti fitur perilaku sehat membantu prediksi pada data ini.' if drop['selisih_rmse_mean'] > 0 else 'Arah mendekati nol / negatif berarti tidak ada kontribusi bermakna pada data ini — jawaban jujur untuk klaim protektif.'}

## Kelompok dengan risiko tinggi (zona merah proporsi tinggi)
{baris_map}

## Catatan penggunaan
- Zona merah/kuning/hijau bersifat **agregat per kelompok** untuk policy brief dan triase guru BK; bukan label diagnosis individual.
- Ambang zona: hijau < {ZONE_HIJAU_MAX}, kuning {ZONE_HIJAU_MAX}-{ZONE_KUNING_MAX}, merah >= {ZONE_KUNING_MAX} (skala 1-4, mengikuti ambang kategori tinggi di config).
- Skor risiko per siswa memakai prediksi out-of-fold (bukan prediksi in-sample), sehingga proporsi zona merah tidak terinflasi oleh data latih.
- Menggunakan bahasa asosiatif: data cross-sectional, korelasi bukan sebab-akibat.
"""
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = C.REPORT_RINGKAS
    path.write_text(isi, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Orkestrasi
# ---------------------------------------------------------------------------
def run_all() -> None:
    # assert level-1 sesuai CONTRACT aturan 5 (temuan round-2 R2-7: 06 tanpa assert)
    assert C.ZONE_HIJAU_MAX < C.ZONE_KUNING_MAX, "ambang zona harus terurut"
    assert zona_skor(C.ZONE_KUNING_MAX) == "merah", "batas bawah zona merah"
    assert zona_skor(C.ZONE_HIJAU_MAX) == "kuning", "batas bawah zona kuning"
    assert zona_skor(C.ZONE_HIJAU_MAX - 1e-9) == "hijau", "batas bawah zona hijau"

    metrics = load_metrics()
    meta = load_model_meta()
    outer = meta.get("outer") or {}
    n_trial = int(meta.get("n_trials_per_outer_fold", 0))
    n_outer = int(outer.get("folds", 0)) * int(outer.get("repeats", 0))
    ranking = load_shap_ranking()
    kestabilan = load_stability()
    drop = load_drop_hb()
    df = load_features()
    skor = _oof_skor_per_siswa()

    assert {"fitur", "mean_abs_shap", "arah"} <= set(ranking.columns), \
        f"shap_global_ranking.csv kolom tak lengkap: {list(ranking.columns)}"
    assert "fitur" in kestabilan.columns, \
        f"stability_ranking.csv kolom tak lengkap: {list(kestabilan.columns)}"
    assert len(skor) == len(df), "prediksi OOF harus menutup semua siswa"

    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tabel_performa(metrics, n_trial, n_outer).to_csv(C.TABEL_PERFORMA, index=False)

    faktor = tabel_faktor(ranking, kestabilan)
    faktor.to_csv(C.TABEL_FAKTOR, index=False)

    map_j = risk_map(df, skor, ["d_jurusan", "d_jenis_kelamin"])
    map_j.to_csv(C.TABLE_DIR / "risk_map_jurusan.csv", index=False)

    map_k = risk_map(df, skor, ["d_jenis_kelamin"])
    map_k.to_csv(C.TABLE_DIR / "risk_map_jeniskelamin.csv", index=False)

    tulis_laporan(
        tabel_performa(metrics, n_trial, n_outer), faktor, map_j, drop, n_trial, n_outer
    )


if __name__ == "__main__":  # pragma: no cover - dijalankan setelah pipeline lengkap
    run_all()
    print(f"[06_report] selesai; laporan di {C.TABLE_DIR / 'report_ringkas.md'}")