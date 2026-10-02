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
src/       kode pipeline (cleaning → fitur → training → evaluasi → SHAP → laporan)
data/      data mentah (lokal saja, tidak di-push)
outputs/   dataset bersih, model, figur, tabel
docs/      codebook & dokumentasi
```

## Status

Dalam pengembangan — pipeline Model #1 (XGBoost risk mapping).
