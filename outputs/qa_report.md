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
- **total_kasus**: 1

## Item Terbelah
- **pasangan**: {'KECEMASANSTRESS13': {'pasangan': 'KECEMASANSTRESS13+KECEMASANSTRESS14', 'kedua_nilai_valid': 0, 'satu_nilai_valid_fallback': 306, 'tanpa_nilai_valid_diimputasi': 0}, 'KECEMASANSTRESS19': {'pasangan': 'KECEMASANSTRESS19+KECEMASANSTRESS20', 'kedua_nilai_valid': 0, 'satu_nilai_valid_fallback': 306, 'tanpa_nilai_valid_diimputasi': 0}}
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
- **catatan_status_ortu**: Label kategori 1 asumsi salah ketik ('kedua orang tua meninggal' -> 'utuh'); kategori 2 dan 5 digabung ke 'lainnya'. Menunggu konfirmasi Bu Rita; 3 versi coding tersedia di config.ORANGTUA_RECODE_VARIANTS.

## Validasi Akhir
- **kecemasan_1_4**: {'rentang': [1, 4], 'pelanggaran': 0, 'kolom_bermasalah': {}}
- **likert_1_5**: {'rentang': [1, 5], 'pelanggaran': 0, 'kolom_bermasalah': {}}
- **dukungan_1_7**: {'rentang': [1, 7], 'pelanggaran': 0, 'kolom_bermasalah': {}}

## Missing Setelah Cleaning
