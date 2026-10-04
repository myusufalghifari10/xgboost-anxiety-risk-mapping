# Eksperimen — Aturan Pencatatan

Folder ini adalah **buku besar eksperimen** Model 1 (dan seterusnya). Setiap percobaan
yang menghasilkan angka untuk dibandingkan WAJIB diarsipkan di sini sebelum memulai
percobaan berikutnya.

## Daftar eksperimen

| # | Folder | Inti | Status | MAE ens (ekor) |
|---|---|---|---|---|
| 01 | `eksperimen-01-xgboost-baseline-2026-10-03` | XGBoost baseline, 18 fitur agregat, RMSE objective, 30.000 trial | DIPAKAI (baseline) | 0,2366 (0,4293) |
| 02 | `eksperimen-02-mae-aligned-item-level-2026-10-04` | 76 fitur item-level + objective MAE + post-processing fold-safe | digantikan E3 | 0,2278 (0,4170) |
| 03 | `eksperimen-03-weighted-loss-warmstart-2026-10-04` | Bobot bin √+cap3 (fold-safe) + warm-start Optuna | **DIPAKAI** | **0,2257 (0,3707)** |

## Penamaan

```
eksperimen-<NN>-<kode-pendek>-<tanggal>
contoh: eksperimen-01-xgboost-baseline-2026-10-03
        eksperimen-02-xgboost-tuned-depth-2026-10-05
```

## Isi tiap folder eksperimen

| File/folder | Isi |
|---|---|
| `RINGKASAN.md` | **Wajib.** Tujuan, konfigurasi, hasil, verdict, kelanjutan |
| `outputs/` | Salinan beku artefak run (struktur sama dengan `outputs/` kerja) |

`outputs/checkpoints/` TIDAK diarsipkan (status resume, bukan hasil).

## Isi minimal `RINGKASAN.md`

1. **Tujuan** — hipotesis/perbaikan yang diuji (1-2 kalimat)
2. **Konfigurasi** — yang dibedakan dari eksperimen sebelumnya
3. **Hasil** — RMSE (CI), MAE, R², AUC + tabel faktor utama
4. **Perbandingan** — vs eksperimen sebelumnya: lebih baik? sebabnya?
5. **Verdict** — dipakai / ditolak / diulang + alasan
6. **Reproduce** — perintah persis untuk meregenerasi

## Aturan

- Nomor urut TIDAK pernah dipakai ulang, meski eksperimen gagal/berantakan.
- Angka di `RINGKASAN.md` harus diambil dari artefak, bukan dari ingatan.
- Folder ini **lokal saja** (gitignored) — berisi turunan data responden di bawah umur.
- Jangan menimpa `outputs/` eksperimen lama dengan hasil baru — buat folder nomor baru.
