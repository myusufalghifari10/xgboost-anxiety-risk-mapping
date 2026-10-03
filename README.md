# XGBoost Anxiety Risk Mapping

Pemodelan risiko kecemasan remaja dengan **XGBoost + SHAP** — bagian analisis AI dari penelitian *"Pemodelan Konseptual Perilaku Sehat Remaja dalam Pencegahan Stres dan Kecemasan"* (Dr. Rita).

## Tentang

- **Data:** survei 306 siswa SMK, Kec. Kresek, Kab. Tangerang (cross-sectional, self-report) — *tidak disertakan dalam repo ini (data sensitif responden di bawah umur)*.
- **Input:** 18 fitur — 6 faktor tekanan, perilaku sehat, dukungan lingkungan, strategi coping, demografi.
- **Target:** skor kecemasan/stres remaja.
- **Model:** XGBoost dengan hyperparameter tuning masif (Optuna) + penjelasan SHAP (faktor risiko global, penjelasan per siswa, efek interaksi).
- **Tujuan:** pemetaan pola risiko untuk kebijakan promotif-preventif kesehatan mental remaja — **alat analisis eksploratif, bukan diagnosis klinis**.

## Struktur

```
src/         kode pipeline (cleaning → fitur → training → evaluasi → SHAP → laporan)
data/        data mentah (lokal saja, tidak di-push)
outputs/     artefak run terkini — rapi per tahap:
  data/        hasil 01-02 (data_clean, features, qa_report)
  model/       model final + encoder (03)
  tuning/      jejak Optuna + best_params (03)
  evaluasi/    metrics + OOF (04)
  explain/     SHAP, stability, drop-HB (05)
  tables/      tabel & laporan (06)
  figures/     grafik SHAP/PDP
  checkpoints/ status resume 03 (bukan deliverable)
  exp2/        Eksperimen 2 (fitur 76, ladder C0-C4, model & decoder exp2)
eksperimen/  arsip per percobaan (lokal saja) — lihat eksperimen/README.md
docs/        codebook & dokumentasi
```

## Eksperimen 2 — MAE-Aligned Item-Level Pipeline

Perbaikan atas baseline (MAE 0,243): **fitur item-level** (58 item mentah → 76 fitur), **objective MAE** (`reg:absoluteerror`, re-tuning penuh), dan **post-processing fold-safe** (clip → shrinkage α terpool → snap grid 0,05 → rata-rata 10 model). Dianalisis lewat ladder arm kumulatif C0–C4 (`src/exp2_train.py`), pemenang dipilih di screening murah, angka final dari outer CV yang tidak dipakai memilih.

```
.venv/bin/python src/exp2_features.py                       # preprocessing (sekali)
.venv/bin/python src/exp2_train.py --self-test              # uji fungsi tanpa data
.venv/bin/python src/exp2_train.py --arm c0 --mode screen   # ulangi c1..c4
.venv/bin/python src/exp2_train.py --arm <pemenang> --mode full
.venv/bin/python src/exp2_train.py --arm <pemenang> --mode final
```

Target MAE realistis: 0,18–0,22 (lantai teoretis ±0,12 dari reliabilitas α=0,89).

## Status

Dalam pengembangan — pipeline Model #1 (XGBoost risk mapping).
