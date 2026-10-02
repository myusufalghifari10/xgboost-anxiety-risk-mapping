"""Tahap 2 — Feature engineering: data bersih -> 18 fitur + 2 target (features.parquet).

Definisi setiap fitur dan target dibangun dari ``config`` supaya tidak ada angka
yang ditulis dua kali di tempat berbeda (lihat PLAN.md bagian 4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def _fail(msg: str) -> None:
    raise ValueError(f"[02_features] {msg}")


def load_clean(path: Path = C.DATA_CLEAN) -> pd.DataFrame:
    """Baca hasil tahap 1; pastikan file ada dan kolom kunci tersedia."""
    if not path.exists():
        _fail(f"file data bersih tidak ditemukan: {path} — jalankan 01_clean.py lebih dulu")
    df = pd.read_parquet(path)
    for col in ("Kelas", "Jurusan", "Umur", "JK", "TINGGAL", "ORANGTUA"):
        if col not in df.columns:
            _fail(f"kolom demografi mentah hilang di data bersih: {col}")
    missing_items = [
        i for items in C.FACTOR_ITEMS.values() for i in items
    ] + C.PERILAKU_ITEMS + C.LINGKUNGAN_ITEMS + sorted(C.COPING_ITEMS)
    hilang = [c for c in missing_items if c not in df.columns]
    if hilang:
        _fail(f"item berikut hilang di data bersih: {hilang}")
    return df


def anxiety_item_columns() -> list[str]:
    """20 item kecemasan yang dipakai menghitung target (kolom mati sudah dibuang di tahap 1)."""
    cols = [c for c in C.ANXIETY_ITEMS_ALL if c not in C.ANXIETY_DEAD_COLS]
    if len(cols) != 20:
        _fail(f"harus ada 20 item kecemasan valid, ditemukan {len(cols)}")
    return cols


def _mean_block(df: pd.DataFrame, cols: list[str], nama: str) -> pd.Series:
    """Rata-rata satu blok item; gagal cepat bila ada kolom tidak lengkap."""
    hilang = [c for c in cols if c not in df.columns]
    if hilang:
        _fail(f"item blok {nama} tidak lengkap: {hilang}")
    return df[cols].mean(axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bangun tabel fitur sesuai skema kontrak (nama kolom wajib, urutan bebas)."""
    out = pd.DataFrame({"row_id": range(len(df))})

    # 6 faktor tekanan (skala 1-5)
    for nama, items in C.FACTOR_ITEMS.items():
        out[f"f_{nama}"] = _mean_block(df, items, nama)

    # Perilaku sehat (skala 1-5) — konstruk inti penelitian
    out["f_perilaku_sehat"] = _mean_block(df, C.PERILAKU_ITEMS, "perilaku_sehat")

    # Dukungan lingkungan MSPSS (skala 1-7), 3 subskala
    for nama, items in C.LINGKUNGAN_SUBSCALES.items():
        out[f"f_{nama}"] = _mean_block(df, items, nama)

    # Strategi coping (skala 1-5), 3 subskala
    for nama, items in C.COPING_SUBSCALES.items():
        out[f"f_{nama}"] = _mean_block(df, items, nama)

    # Demografi
    out["d_umur"] = df["d_umur"] if "d_umur" in df.columns else df["Umur"].astype(int)
    out["d_jenis_kelamin"] = df["d_jenis_kelamin"].astype(str)
    out["d_jurusan"] = df["d_jurusan"].astype(str)
    out["d_tinggal_grup"] = df["d_tinggal_grup"].astype(int)
    out["d_status_ortu"] = df["d_status_ortu"].astype(int)

    # Target: skor kecemasan kontinu + turunan biner
    y = _mean_block(df, anxiety_item_columns(), "kecemasan")
    out[C.TARGET_CONT] = y.astype(float)
    out[C.TARGET_BIN] = (y >= C.ANXIETY_HIGH_THRESHOLD).astype(int)

    missing_cols = [c for c in C.FEATURE_COLS if c not in out.columns]
    if missing_cols:
        _fail(f"fitur di config belum dibangun: {missing_cols}")
    kolom_tidak_diminta = [c for c in out.columns if c not in C.FEATURE_COLS + [C.TARGET_CONT, C.TARGET_BIN, "row_id"]]
    if kolom_tidak_diminta:
        _fail(f"kolom keluar dari skema kontrak: {kolom_tidak_diminta}")
    return out


def build_codebook(features: pd.DataFrame, path: Path = C.CODEBOOK) -> None:
    """Tulis kamus variabel langsung dari config + statistik data (dokumentasi hidup)."""
    def desc(kolom: str) -> str:
        s = features[kolom].astype(float)
        return f"{s.min():.2f}–{s.max():.2f} (mean {s.mean():.2f}, SD {s.std():.2f})"

    n_pos = int(features[C.TARGET_BIN].sum())
    baris = [
        "# Codebook — Model #1 XGBoost Risk Mapping",
        "",
        "Dihasilkan otomatis oleh `src/02_features.py`. Sumber data: 306 siswa SMK, "
        "Kec. Kresek, Kab. Tangerang (survei cross-sectional).",
        "",
        "## Target",
        "",
        "| Variabel | Definisi | Rentang | Distribusi |",
        "|---|---|---|---|",
        f"| `{C.TARGET_CONT}` | Rata-rata 20 item kecemasan/stres (skala 1=Tidak pernah … 4=Sangat sering) "
        f"| 1.00–4.00 | {desc(C.TARGET_CONT)} |",
        f"| `{C.TARGET_BIN}` | 1 bila `{C.TARGET_CONT}` ≥ {C.ANXIETY_HIGH_THRESHOLD} (proxy kategori *cemas tinggi*) "
        f"| 0/1 | 1 = {n_pos} siswa ({100*n_pos/len(features):.1f}%) |",
        "",
        f"> **Catatan ketidakseimbangan kelas:** hanya {n_pos}/{len(features)} siswa ({100*n_pos/len(features):.1f}%) "
        "berada di kategori positif. Model biner wajib memakai `scale_pos_weight` atau `class_weight`, "
        "dan dievaluasi dengan AUC + Brier score (bukan accuracy) — lihat `config.py`.",
        "",
        "## 18 Fitur Input",
        "",
        "| # | Fitur | Sumber item | Skala | Rentang data |",
        "|---|---|---|---|---|",
    ]
    urutan = [
        ("f_akademik", "Faktor tekanan akademik", ", ".join(C.FACTOR_ITEMS["akademik"]), "1–5"),
        ("f_keluarga", "Faktor tekanan keluarga", ", ".join(C.FACTOR_ITEMS["keluarga"]), "1–5"),
        ("f_sosial", "Faktor tekanan sosial (SOSIAL3 sudah di-reverse)", ", ".join(C.FACTOR_ITEMS["sosial"]), "1–5"),
        ("f_ekonomi", "Faktor tekanan ekonomi", ", ".join(C.FACTOR_ITEMS["ekonomi"]), "1–5"),
        ("f_digital", "Faktor tekanan digital", ", ".join(C.FACTOR_ITEMS["digital"]), "1–5"),
        ("f_masadepan", "Faktor tekanan masa depan", ", ".join(C.FACTOR_ITEMS["masadepan"]), "1–5"),
        ("f_perilaku_sehat", "Perilaku sehat (konstruk inti penelitian)",
         f"{C.PERILAKU_ITEMS[0]}–{C.PERILAKU_ITEMS[-1]} (10 item)", "1–5"),
        ("f_duk_keluarga", "Dukungan lingkungan — keluarga", "LINGKUNGAN1–4", "1–7"),
        ("f_duk_teman", "Dukungan lingkungan — teman", "LINGKUNGAN5–8", "1–7"),
        ("f_duk_orang_dekat", "Dukungan lingkungan — orang terdekat", "LINGKUNGAN9–12", "1–7"),
        ("f_coping_adaptif", "Coping adaptif (fokus masalah, religius, cari bantuan)", "COPING1–12", "1–5"),
        ("f_coping_ekspresi", "Coping ekspresi emosi", "COPING13–15", "1–5"),
        ("f_coping_negatif", "Coping negatif (menyendiri, self-blame, helpless)", "COPING16–18", "1–5"),
        ("d_umur", "Umur siswa", "Umur", "14–17"),
        ("d_jenis_kelamin", "Jenis kelamin", "JK", "L / P"),
        ("d_jurusan", "Jurusan", "Jurusan", "5 kategori"),
        ("d_tinggal_grup", "Tinggal dengan siapa (dipadatkan 11 → 5 grup)", "TINGGAL → TINGGAL_RECODE", "1–5"),
        ("d_status_ortu", "Status orang tua (dipadatkan 5 → 4 kode)", "ORANGTUA → ORANGTUA_RECODE", "1–4"),
    ]
    for i, (kolom, nama, sumber, skala) in enumerate(urutan, 1):
        rentang = desc(kolom) if kolom.startswith("f_") else f"{features[kolom].nunique()} kategori"
        baris.append(f"| {i} | `{kolom}` | {nama} — {sumber} | {skala} | {rentang} |")

    baris += [
        "",
        "## Aturan Cleaning (Tahap 1)",
        "",
        "| Aturan | Perlakuan |",
        "|---|---|",
        "| Item kecemasan 13+14 dan 19+20 | Kalimat terbelah jadi 2 kolom; digabung (rata-rata nilai sah), "
        "kolom pasangan (14, 20) dibuang — keduanya berisi 0 di semua baris |",
        "| `KECEMASANSTRESS12` = 5 | Di luar skala 1–4 → imputasi median kolom |",
        f"| Item terbalik ({', '.join(C.REVERSE_ITEMS)}) | Reverse-score: nilai_baru = {sum(C.LIKERT5_RANGE)} − nilai |",
        "| `VAR00001` | Kolom kosong → dibuang |",
        "",
        "## Asumsi yang Perlu Dikonfirmasi Dosen",
        "",
        "- **Label status orang tua.** Kategori 1 berlabel *\"kedua orang tua meninggal dunia\"* tapi "
        "226 dari 240 siswa kategori tersebut tinggal bersama ayah dan ibu → label kemungkinan salah "
        "ketik. Asumsi kerja: kategori 1 = **keluarga utuh**. Kategori 2 (9 siswa, pola mencampur) dan "
        "kategori 5 (4 siswa, tanpa label) digabung ke **lainnya**. Sisa kategori 3 & 4 dipakai sesuai "
        "label aslinya. Tersedia 3 versi coding di `config.ORANGTUA_RECODE_VARIANTS` untuk sensitivity analysis.",
        "- **Ambang `y_biner`.** 2,5 dipilih agar \"sering ke atas\" masuk kategori cemas tinggi; "
        "ambang alternatif 2,0 dan 2,3 menghasilkan 42,2% dan 25,8% positif — dipakai sebagai bahan "
        "diskusi bila dosen ingin prevalensi lebih tinggi.",
        "- **Temuan penting — `f_sosial` berbeda dari analisis sebelumnya.** Rata-rata blok sosial "
        "TANPA reverse-scoring adalah **2,15** (angka yang dipakai di analisis deskriptif & regresi lama), "
        "sedangkan pipeline kita memakai mean **2,74** setelah `SOSIAL3` di-reverse. Item itu berbunyi "
        "*\"Saya merasa TIDAK tertekan ketika mengalami masalah…\"* — arahnya berlawanan, sehingga skor "
        "sebelumnya sempat menyamakan \"sering tidak tertekan\" dengan \"tinggi tekanan sosial\". Karena "
        "faktor sosial menjadi prediktor terkuat di regresi lama (β=0,241), temuan itu perlu dihitung ulang. "
        "Pipeline ini sengaja mengikuti aturan `config.REVERSE_ITEMS` (reverse = benar).",
        "- **Baris duplikat.** Dua baris identik di seluruh kolom tidak dihapus (N tetap 306) agar "
        "selaras dengan analisis deskriptif sebelumnya; ditandai di `outputs/qa_report.md`.",
        "",
        "## Pemakaian Etis",
        "",
        "Output model bersifat **agregat dan eksploratif** untuk policy promotif-preventif; "
        "bukan alat diagnosis klinis dan bukan label per individu.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(baris) + "\n", encoding="utf-8")


def main() -> None:
    df = load_clean()
    feats = build_features(df)
    feats.to_parquet(C.FEATURES, index=False)
    build_codebook(feats)
    print(f"[02_features] ditulis: {C.FEATURES}  shape={feats.shape}")
    print(f"[02_features] target: mean={feats[C.TARGET_CONT].mean():.3f}, "
          f"positif={int(feats[C.TARGET_BIN].sum())}/{len(feats)}")
    # Bukti level-1: skema & rentang (ATURAN 5 kontrak)
    assert len(C.FEATURE_COLS) == 18, f"config harus punya 18 fitur, ada {len(C.FEATURE_COLS)}"
    assert feats[C.TARGET_CONT].between(1.0, 4.0).all(), "y_kontinu di luar rentang 1-4"
    assert feats[C.TARGET_BIN].isin([0, 1]).all(), "y_biner harus 0/1"
    assert not feats.isna().any().any(), "fitur masih ada missing value"


if __name__ == "__main__":
    main()