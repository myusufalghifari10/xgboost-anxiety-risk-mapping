# Laporan Ringkas — Model 1: Peta Risiko Kecemasan (XGBoost)

## Ringkasan performa (prediksi out-of-fold, repeated nested CV 5 fold x 10 ulangan)
- Skor kecemasan (kontinu): RMSE = **0.317** (CI95 0.288–0.344), MAE = 0.243, R2 = 0.538
- Catatan desain (fix review F2): setiap siswa menjadi data test 10x (sekali per ulangan) dan data latih pada ulangan lain — angka adalah estimasi resampling internal pada 306 siswa yang sama, BUKAN test set eksternal terpisah.
- Kategori cemas tinggi (biner): AUC = **0.784**
- Konteks: N = 306 siswa (3060 prediksi out-of-fold); kategori cemas tinggi 43 siswa (14.1%).
- Hyperparameter Optuna (nested-CV; test fold TIDAK PERNAH dilihat proses ini): **500 trial per ronde x 50 ronde = 25.000 trial**.
- Tuning model final terpisah: **5.000 trial** pada seluruh 306 siswa (tidak ada angka test yang dikutip dari model final).

## 5 faktor paling berpengaruh (SHAP)
| Peringkat | Faktor | Arah | Rata-rata \|SHAP\| (poin skor) | Stabil (masuk top-5 antar fold) |
|---|---|---|---|---|
| 1 | f_coping_negatif | kontribusi rata2 menurunkan | 0.1081 | 100% |
| 2 | f_digital | kontribusi rata2 menurunkan | 0.1003 | 100% |
| 3 | d_jenis_kelamin | kontribusi rata2 menurunkan | 0.0697 | 96% |
| 4 | f_keluarga | kontribusi rata2 menaikkan | 0.0520 | 92% |
| 5 | f_akademik | kontribusi rata2 menaikkan | 0.0447 | 44% ⚠ kurang stabil |

_Arah = kontribusi rata-rata SHAP pada sampel ini (menaikkan/menurunkan prediksi skor kecemasan), bukan uji sebab-akibat, dan nilainya mendekati nol secara matematis — untuk arah hubungan gunakan grafik PDP (outputs/figures/pdp_*.png), bukan kolom ini (bisa bertentangan). Peringkat SHAP dihitung pada model final (in-sample); angka performa di atas berasal dari prediksi out-of-fold._
- Fitur yang TIDAK dipakai model sama sekali (|SHAP| = 0): f_sosial, f_duk_keluarga, f_perilaku_sehat, f_coping_adaptif, d_tinggal_grup, d_status_ortu — nol berarti fitur tidak pernah dipakai model untuk membelah (49 simpul split), BUKAN terbukti tidak berpengaruh.

## Uji klaim perilaku sehat (drop-HB)
- Menghapus fitur `f_perilaku_sehat` mengubah RMSE test sebesar **+0.0017** (rata-rata 50 fold; SD antar fold = 0.0096; 56% fold menunjukkan HB membantu).
- Selisih rata-rata lebih kecil daripada sebaran antar fold (< 1 SD): tidak ada kesepakatan arah antar fold — tidak ada kontribusi pasti pada data ini (deskriptif, bukan uji signifikansi).
- Interpretasi jujur: model final tidak pernah membelah pada `f_perilaku_sehat` (SHAP identik nol, PDP datar), jadi uji ini konsisten dengan "fitur tidak dipakai"; bukti utamanya ada di SHAP/PDP, sedangkan selisih ≈ 0 juga bisa muncul dari lotre subsampling kolom (`colsample_bytree`) — jangan dijadikan satu-satunya dasar.

## Interaksi antar faktor — TIDAK TERUKUR
- Sembilan pasang fokus (perilaku sehat x dukungan/tekanan) bernilai 0 karena fitur tidak pernah dipakai model untuk membelah, BUKAN karena interaksi terbukti tidak ada. Pertanyaan interaksi pada PLAN 8.4 tidak bisa dijawab oleh model ini; jawabannya butuh model yang memang memakai fitur tersebut.

## Kelompok dengan risiko tinggi (zona merah proporsi tinggi)
- Tidak ada sel yang mencapai ambang TINGGI (proporsi zona merah >= 50%).

Tiga sel dengan proporsi zona merah tertinggi (ambang TINGGI = proporsi zona merah >= 50%):
- Teknik Kimia Industri / P: 22% siswa zona merah (n=32)
- Teknik Komputer Jaringan / P: 9% siswa zona merah (n=150)
- Teknik Sepeda Motor / L: 2% siswa zona merah (n=58)

- Konteks peta: 22 dari 306 siswa (7.2%) masuk zona merah, sementara prevalensi aktual kategori cemas tinggi 43 siswa (14.1%). Skor per siswa = rata-rata 10 prediksi OOF (ensemble, bukan prediksi satu model) sehingga ekor atas agak ter-mampat — peta untuk MEMBANDINGKAN kelompok, bukan menghitung jumlah kasus.

## Catatan penggunaan
- Zona merah/kuning/hijau bersifat **agregat per kelompok** untuk policy brief dan triase guru BK; bukan label diagnosis individual.
- Ambang zona: hijau < 1.5; kuning >= 1.5 dan < 2.5; merah >= 2.5 (skala 1-4, mengikuti ambang kategori tinggi di config).
- Skor risiko per siswa memakai prediksi out-of-fold (bukan prediksi in-sample), sehingga proporsi zona merah tidak terinflasi oleh data latih.
- Brier score memakai konvensi scikit-learn, yaitu rata-rata (p - y)^2 (setengah dari definisi Brier klasik 2 kelas) — jangan dibandingkan langsung dengan literatur yang memakai definisi lain tanpa konversi.
- Probabilitas biner berasal dari model dengan `scale_pos_weight` (target timpang): probabilitas BELUM terkalibrasi terhadap prevalensi asli sehingga tidak boleh dibaca sebagai risiko absolut per siswa.
- `balanced_accuracy` dihitung pada ambang tetap 0.5 (bukan ambang optimal), sedangkan AUC tetap sahih karena hanya bergantung urutan peringkat.
- Menggunakan bahasa asosiatif: data cross-sectional, korelasi bukan sebab-akibat.
