# Codebook — Model #1 XGBoost Risk Mapping

Dihasilkan otomatis oleh `src/02_features.py`. Sumber data: 306 siswa SMK, Kec. Kresek, Kab. Tangerang (survei cross-sectional).

## Target

| Variabel | Definisi | Rentang | Distribusi |
|---|---|---|---|
| `y_kontinu` | Rata-rata 20 item kecemasan/stres (skala 1=Tidak pernah … 4=Sangat sering) | 1.00–4.00 | 1.00–3.50 (mean 1.97, SD 0.47) |
| `y_biner` | 1 bila `y_kontinu` ≥ 2.5 (proxy kategori *cemas tinggi*) | 0/1 | 1 = 43 siswa (14.1%) |

> **Catatan ketidakseimbangan kelas:** hanya 43/306 siswa (14.1%) berada di kategori positif. Model biner wajib memakai `scale_pos_weight` atau `class_weight`, dan dievaluasi dengan AUC + Brier score (bukan accuracy) — lihat `config.py`.

## 18 Fitur Input

| # | Fitur | Sumber item | Skala | Rentang data |
|---|---|---|---|---|
| 1 | `f_akademik` | Faktor tekanan akademik — AKADEMIK1, AKADEMIK2, AKADEMIK3 | 1–5 | 1.00–4.67 (mean 2.46, SD 0.89) |
| 2 | `f_keluarga` | Faktor tekanan keluarga — KELUARGA1, KELUARGA2, KELUARGA3 | 1–5 | 1.00–5.00 (mean 2.31, SD 1.08) |
| 3 | `f_sosial` | Faktor tekanan sosial (SOSIAL3 sudah di-reverse) — SOSIAL1, SOSIAL2, SOSIAL3 | 1–5 | 1.00–5.00 (mean 2.74, SD 0.72) |
| 4 | `f_ekonomi` | Faktor tekanan ekonomi — EKONOMI1, EKONOMI2, EKONOMI3 | 1–5 | 1.00–5.00 (mean 2.03, SD 1.00) |
| 5 | `f_digital` | Faktor tekanan digital — DIGITAL1, DIGITAL2, DIGITAL3 | 1–5 | 1.00–4.67 (mean 1.77, SD 0.79) |
| 6 | `f_masadepan` | Faktor tekanan masa depan — MASADEPAN1, MASADEPAN2, MASADEPAN3 | 1–5 | 1.00–4.67 (mean 2.37, SD 0.84) |
| 7 | `f_perilaku_sehat` | Perilaku sehat (konstruk inti penelitian) — PERILAKU1–PERILAKU10 (10 item) | 1–5 | 1.30–4.70 (mean 2.96, SD 0.57) |
| 8 | `f_duk_keluarga` | Dukungan lingkungan — keluarga — LINGKUNGAN1–4 | 1–7 | 1.00–7.00 (mean 4.81, SD 1.26) |
| 9 | `f_duk_teman` | Dukungan lingkungan — teman — LINGKUNGAN5–8 | 1–7 | 1.00–7.00 (mean 4.33, SD 1.22) |
| 10 | `f_duk_orang_dekat` | Dukungan lingkungan — orang terdekat — LINGKUNGAN9–12 | 1–7 | 1.00–7.00 (mean 4.96, SD 1.38) |
| 11 | `f_coping_adaptif` | Coping adaptif (fokus masalah, religius, cari bantuan) — COPING1–12 | 1–5 | 1.58–4.67 (mean 3.34, SD 0.57) |
| 12 | `f_coping_ekspresi` | Coping ekspresi emosi — COPING13–15 | 1–5 | 1.00–5.00 (mean 2.59, SD 1.00) |
| 13 | `f_coping_negatif` | Coping negatif (menyendiri, self-blame, helpless) — COPING16–18 | 1–5 | 1.00–5.00 (mean 2.72, SD 0.95) |
| 14 | `d_umur` | Umur siswa — Umur | 14–17 | 4 kategori |
| 15 | `d_jenis_kelamin` | Jenis kelamin — JK | L / P | 2 kategori |
| 16 | `d_jurusan` | Jurusan — Jurusan | 5 kategori | 5 kategori |
| 17 | `d_tinggal_grup` | Tinggal dengan siapa (dipadatkan 11 → 5 grup) — TINGGAL → TINGGAL_RECODE | 1–5 | 5 kategori |
| 18 | `d_status_ortu` | Status orang tua (dipadatkan 5 → 4 kode) — ORANGTUA → ORANGTUA_RECODE | 1–4 | 4 kategori |

## Aturan Cleaning (Tahap 1)

| Aturan | Perlakuan |
|---|---|
| Item kecemasan 13+14 dan 19+20 | Kalimat terbelah jadi 2 kolom; digabung (rata-rata nilai sah), kolom pasangan (14, 20) dibuang — keduanya berisi 0 di semua baris |
| `KECEMASANSTRESS12` = 5 | Di luar skala 1–4 → imputasi median kolom |
| Item terbalik (SOSIAL3) | Reverse-score: nilai_baru = 6 − nilai |
| `VAR00001` | Kolom kosong → dibuang |

- **Audit arah item (hasil verifikasi).** `REVERSE_ITEMS=[SOSIAL3]` sudah lengkap: semua item bernegasi lain (KECEMASANSTRESS2/9/16/17/22, MASADEPAN3, COPING16–18) sengaja TIDAK di-reverse karena tetap searah konstruk/subskalanya (COPING16–18 = subskala `coping_negatif`, skor tinggi = makin negatif).
- **Sensitivitas item 12 (nilai 5).** Nilai invalid diisi median kolom (=2); alternatif isi 4 teruji TIDAK mengubah label `y_biner` siswa mana pun.

## Asumsi yang Perlu Dikonfirmasi Dosen

- **Label status orang tua.** Kategori 1 berlabel *"kedua orang tua meninggal dunia"* tapi 226 dari 240 siswa kategori tersebut tinggal bersama ayah dan ibu → label kemungkinan salah ketik. Asumsi kerja: kategori 1 = **keluarga utuh**. Kategori 2 (9 siswa, pola mencampur) dan kategori 5 (4 siswa, tanpa label) digabung ke **lainnya**. Sisa kategori 3 & 4 dipakai sesuai label aslinya. 3 versi coding di `config.ORANGTUA_RECODE_VARIANTS` disiapkan untuk sensitivity analysis, namun BELUM dijalankan (fase robustness terpisah).
- **Ambang `y_biner`.** 2,5 dipilih sebagai titik tengah antara pilihan "kadang-kadang" (2) dan "sering" (3) — siswa dengan rata-rata skor di atasnya masuk kategori cemas tinggi; ambang alternatif 2,0 dan 2,3 menghasilkan 42,2% dan 25,8% positif — dipakai sebagai bahan diskusi bila dosen ingin prevalensi lebih tinggi.
- **Temuan penting — `f_sosial` berbeda dari analisis sebelumnya.** Rata-rata blok sosial TANPA reverse-scoring adalah **2,15** (angka yang dipakai di analisis deskriptif & regresi lama), sedangkan pipeline kita memakai mean **2,74** setelah `SOSIAL3` di-reverse. Item itu berbunyi *"Saya merasa TIDAK tertekan ketika mengalami masalah…"* — arahnya berlawanan, sehingga skor sebelumnya sempat menyamakan "sering tidak tertekan" dengan "tinggi tekanan sosial". Karena faktor sosial menjadi prediktor terkuat di regresi lama (β=0,241), temuan itu perlu dihitung ulang. Pipeline ini sengaja mengikuti aturan `config.REVERSE_ITEMS` (reverse = benar).
- **Baris duplikat.** Dua baris identik di seluruh kolom tidak dihapus (N tetap 306) agar selaras dengan analisis deskriptif sebelumnya; ditandai di `outputs/qa_report.md`. Implikasi evaluasi: pasangan kembar bisa terbagi ke train dan test (bias optimis <0,5%).

## Pemakaian Etis

Output model bersifat **agregat dan eksploratif** untuk policy promotif-preventif; bukan alat diagnosis klinis dan bukan label per individu.
