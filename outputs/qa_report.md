# Laporan QA — Tahap 1 Pembersihan Data

## Shape Awal
- **baris**: 306
- **kolom**: 87
- **sumber**: INPUT DATA  (1).sav

## Duplikat
- **baris_duplikat**: 2
- **kebijakan**: Tidak dihapus. Kemiripan jawaban seluruh kolom pada 2 baris bisa jadi duplikasi input atau bisa jadi dua siswa yang kebetulan menjawab identik; dropping akan mengubah N=306 menjadi 305 dan membuat deskriptif sebelumnya tidak sinkron. Ditandai untuk diperiksa dosen.

## Perbaikan Rentang
- **rentang_valid**: 1-4
- **kolom_diperbaiki**: {'KECEMASANSTRESS12': 1}
- **nilai_missing_sebelum_imputasi**: {'KECEMASANSTRESS12': 0}
- **total_kasus**: 1

## Item Terbelah
- **pasangan**: {'KECEMASANSTRESS13': {'pasangan': 'KECEMASANSTRESS13+KECEMASANSTRESS14', 'kedua_nilai_valid': 0, 'satu_nilai_valid_fallback': 306, 'tanpa_nilai_valid_diimputasi': 0, 'nilai_nol_di_kolom': {'KECEMASANSTRESS13': 0, 'KECEMASANSTRESS14': 306}}, 'KECEMASANSTRESS19': {'pasangan': 'KECEMASANSTRESS19+KECEMASANSTRESS20', 'kedua_nilai_valid': 0, 'satu_nilai_valid_fallback': 306, 'tanpa_nilai_valid_diimputasi': 0, 'nilai_nol_di_kolom': {'KECEMASANSTRESS19': 0, 'KECEMASANSTRESS20': 306}}}
- **catatan**: Kolom pasangan (14 dan 20) berisi 0 pada seluruh 306 baris — artefak input, bukan jawaban siswa. Setelah digabung, kedua kolom dibuang.

## Kolom Dibuang
- **mortal_kolom_anxiety**: ['KECEMASANSTRESS14', 'KECEMASANSTRESS20']
- **kosong**: ['VAR00001']
- **total**: 3

## Reverse Item
- **item**: ['SOSIAL3']
- **rumus**: (min+max)-nilai = 6-nilai
- **alasan**: Item SOSIAL3 berbunyi 'merasa TIDAK tertekan' — arahnya berlawanan dengan konstruk.

## Recode Demografi
- **tinggal_grup**: {'ayah_ibu': 234, 'satu_orang_tua': 34, 'ortu_bekerja_luar': 6, 'diasuh_keluarga_lembaga': 19, 'lainnya': 13}
- **status_ortu**: {'utuh': 240, 'terpisah': 33, 'salah_satu_meninggal': 20, 'lainnya': 13}
- **distribusi_jenis_kelamin**: {'P': 183, 'L': 123}
- **distribusi_tinggal_mentah**: {1: 234, 2: 5, 3: 24, 4: 5, 5: 6, 6: 9, 7: 6, 9: 4, 10: 1, 11: 12}
- **crosstab_tinggal_x_orangtua**: {'tinggal_1': {'orangtua_1': 226, 'orangtua_2': 3, 'orangtua_3': 4, 'orangtua_4': 1}, 'tinggal_2': {'orangtua_2': 1, 'orangtua_3': 1, 'orangtua_4': 3}, 'tinggal_3': {'orangtua_2': 2, 'orangtua_3': 15, 'orangtua_4': 7}, 'tinggal_4': {'orangtua_1': 2, 'orangtua_3': 1, 'orangtua_4': 2}, 'tinggal_5': {'orangtua_1': 2, 'orangtua_2': 2, 'orangtua_3': 2}, 'tinggal_6': {'orangtua_1': 2, 'orangtua_2': 1, 'orangtua_3': 4, 'orangtua_4': 2}, 'tinggal_7': {'orangtua_1': 2, 'orangtua_3': 1, 'orangtua_4': 2, 'orangtua_5': 1}, 'tinggal_9': {'orangtua_1': 4}, 'tinggal_10': {'orangtua_3': 1}, 'tinggal_11': {'orangtua_1': 2, 'orangtua_3': 4, 'orangtua_4': 3, 'orangtua_5': 3}}
- **catatan_status_ortu**: Label kategori 1 asumsi salah ketik ('kedua orang tua meninggal' -> 'utuh'); kategori 2 dan 5 digabung ke 'lainnya'. Menunggu konfirmasi Bu Rita; 3 versi coding tersedia di config.ORANGTUA_RECODE_VARIANTS.

## Validasi Akhir
- **kecemasan_1_4**: {'rentang': [1, 4], 'pelanggaran': 0, 'kolom_bermasalah': {}}
- **likert_1_5**: {'rentang': [1, 5], 'pelanggaran': 0, 'kolom_bermasalah': {}}
- **dukungan_1_7**: {'rentang': [1, 7], 'pelanggaran': 0, 'kolom_bermasalah': {}}

## Missing Setelah Cleaning
- (tidak ada)
