"""Tahap 1 — Pembersihan data mentah (.sav) -> outputs/data_clean.parquet.

Aturan cleaning diambil dari ``config`` (sumber kebenaran tunggal), lihat PLAN.md bagian 3.
Urutan langkah penting: nilai invalid dibetulkan dulu, baru item terbelah digabung,
baru kolom mati dibuang — supaya penggabungan tidak membaca nilai yang sudah rusak.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


# ---------- utilitas ----------


def _fail(msg: str) -> None:
    """Gagal cepat dengan pesan spesifik (ATURAN 4 kontrak: tanpa except kosong)."""
    raise ValueError(f"[01_clean] {msg}")


def _read_sav(path: Path) -> pd.DataFrame:
    """Baca .sav dan pastikan semua kolom yang dibutuhkan benar-benar ada."""
    if not path.exists():
        _fail(f"file data mentah tidak ditemukan: {path}")
    df, _meta = pyreadstat.read_sav(str(path))

    needed = (
        set(C.ANXIETY_ITEMS_ALL)
        | set(C.ANXIETY_DEAD_COLS)
        | {i for items in C.FACTOR_ITEMS.values() for i in items}
        | set(C.PERILAKU_ITEMS)
        | set(C.LINGKUNGAN_ITEMS)
        | set(C.COPING_ITEMS)
        | {"Kelas", "Jurusan", "Umur", "JK", "TINGGAL", "ORANGTUA"}
    )
    missing = sorted(needed - set(df.columns))
    if missing:
        _fail(f"kolom berikut tidak ada di .sav: {missing}")
    if df.empty:
        _fail("data mentah kosong")
    return df


def _fix_out_of_range(df: pd.DataFrame, qa: dict) -> pd.DataFrame:
    """Nilai di luar rentang skala -> NaN, lalu imputasi median kolom.

    Hanya kolom yang terdaftar di ``ANXIETY_OUT_OF_RANGE`` yang diperiksa (skala 1-4).
    """
    df = df.copy()
    fixed: dict[str, int] = {}
    lo, hi = C.ANXIETY_VALID_RANGE
    for col in C.ANXIETY_OUT_OF_RANGE:
        if col not in df.columns:
            _fail(f"kolom untuk perbaikan rentang tidak ada: {col}")
        bad = ~df[col].between(lo, hi) & df[col].notna()
        n_bad = int(bad.sum())
        if n_bad:
            df.loc[bad, col] = np.nan
            median = df[col].median()
            if pd.isna(median):
                _fail(f"tidak ada nilai valid untuk imputasi median pada kolom {col}")
            df[col] = df[col].fillna(median)
            fixed[col] = n_bad
    qa["perbaikan_rentang"] = {
        "rentang_valid": f"{lo}-{hi}",
        "kolom_diperbaiki": fixed,
        "total_kasus": sum(fixed.values()),
    }
    return df


def _merge_split_anxiety_items(df: pd.DataFrame, qa: dict) -> pd.DataFrame:
    """Gabungkan item kecemasan yang kalimatnya terbelah jadi 2 kolom (13+14, 19+20).

    Pada data ini semua baris jatuh ke satu nilai valid karena kolom pasangannya artefak
    isian (nilai 0 di luar skala), tapi aturannya dibuat umum: per baris dipakai rata-rata
    nilai yang berada di dalam rentang sah; bila tidak ada satu pun, hasilnya NaN
    (diimputasi median).
    """
    df = df.copy()
    lo, hi = C.ANXIETY_VALID_RANGE
    merged: dict[str, dict[str, int]] = {}
    for keep, partner in C.ANXIETY_SPLIT_PAIRS:
        for col in (keep, partner):
            if col not in df.columns:
                _fail(f"kolom item terbelah tidak ada: {col}")
        valid = df[[keep, partner]].apply(lambda r: [v for v in r if lo <= v <= hi], axis=1)
        both = int((valid.map(len) == 2).sum())
        one = int((valid.map(len) == 1).sum())
        none = int((valid.map(len) == 0).sum())
        df[keep] = valid.map(lambda vs: float(np.mean(vs)) if vs else np.nan)
        if df[keep].isna().any():  # jaga-jaga: tidak boleh ada baris tanpa nilai
            med = df[keep].median()
            if pd.isna(med):
                _fail(f"tidak ada nilai valid untuk imputasi median pada kolom {keep}")
            df[keep] = df[keep].fillna(med)
        merged[keep] = {
            "pasangan": f"{keep}+{partner}",
            "kedua_nilai_valid": both,
            "satu_nilai_valid_fallback": one,
            "tanpa_nilai_valid_diimputasi": none,
        }
    qa["item_terbelah"] = {
        "pasangan": merged,
        "catatan": (
            "Kolom pasangan (14 dan 20) berisi 0 pada seluruh 306 baris — artefak input, "
            "bukan jawaban siswa. Setelah digabung, kedua kolom dibuang."
        ),
    }
    return df


def _drop_dead_columns(df: pd.DataFrame, qa: dict) -> pd.DataFrame:
    """Buang kolom mati (konstan) dan kolom kosong VAR00001."""
    df = df.copy()
    dropped: list[str] = []
    for col in list(C.ANXIETY_DEAD_COLS) + ["VAR00001"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped.append(col)
    qa["kolom_dibuang"] = {
        "mortal_kolom_anxiety": list(C.ANXIETY_DEAD_COLS),
        "kosong": ["VAR00001"],
        "total": len(dropped),
    }
    return df


def _reverse_items(df: pd.DataFrame, qa: dict) -> pd.DataFrame:
    """Reverse-score item berlawanan arah: nilai_baru = (min+max) - nilai_lama."""
    df = df.copy()
    done: list[str] = []
    lo, hi = C.LIKERT5_RANGE
    for col in C.REVERSE_ITEMS:
        if col not in df.columns:
            _fail(f"kolom item terbalik tidak ada: {col}")
        before = df[col].copy()
        df[col] = (lo + hi) - df[col]
        if not (df[col] + before == lo + hi).all():  # sanity: transformasi harus involusi di titik uji
            _fail(f"reverse-score gagal pada kolom {col}")
        done.append(col)
    qa["reverse_item"] = {
        "item": done,
        "rumus": f"(min+max)-nilai = {lo + hi}-nilai",
        "alasan": "Item SOSIAL3 berbunyi 'merasa TIDAK tertekan' — arahnya berlawanan dengan konstruk.",
    }
    return df


def _recode_demographics(df: pd.DataFrame, qa: dict) -> pd.DataFrame:
    """Buat kolom demografi d_* sesuai kontrak; kolom mentah dipertahankan untuk sensitivity."""
    df = df.copy()

    unknown_tinggal = sorted(set(df["TINGGAL"].dropna().unique()) - set(C.TINGGAL_RECODE))
    if unknown_tinggal:
        _fail(f"kode TINGGAL di luar peta recode: {unknown_tinggal}")
    unknown_ortu = sorted(set(df["ORANGTUA"].dropna().unique()) - set(C.ORANGTUA_RECODE))
    if unknown_ortu:
        _fail(f"kode ORANGTUA di luar peta recode: {unknown_ortu}")

    df["d_jenis_kelamin"] = df["JK"].astype(str)
    df["d_umur"] = df["Umur"].astype(int)
    df["d_jurusan"] = df["Jurusan"].astype(str)
    df["d_tinggal_grup"] = df["TINGGAL"].astype(int).map(C.TINGGAL_RECODE).astype(int)
    df["d_status_ortu"] = df["ORANGTUA"].astype(int).map(C.ORANGTUA_RECODE).astype(int)

    qa["recode_demografi"] = {
        "tinggal_grup": {C.TINGGAL_LABELS[k]: int(v) for k, v in
                         df["d_tinggal_grup"].value_counts().sort_index().items()},
        "status_ortu": {C.ORANGTUA_LABELS[k]: int(v) for k, v in
                        df["d_status_ortu"].value_counts().sort_index().items()},
        "distribusi_jenis_kelamin": {k: int(v) for k, v in
                                     df["d_jenis_kelamin"].value_counts().items()},
        "catatan_status_ortu": (
            "Label kategori 1 asumsi salah ketik ('kedua orang tua meninggal' -> 'utuh'); "
            "kategori 2 dan 5 digabung ke 'lainnya'. Menunggu konfirmasi Bu Rita; "
            "3 versi coding tersedia di config.ORANGTUA_RECODE_VARIANTS."
        ),
    }
    return df


def _validate(df: pd.DataFrame, qa: dict) -> pd.DataFrame:
    """Cek rentang semua item + kolom hasil; catat pelanggaran tanpa membiarkan data rusak lewat."""
    df = df.copy()
    checks: dict[str, dict] = {}

    anxiety_cols = [c for c in C.ANXIETY_ITEMS_ALL if c not in C.ANXIETY_DEAD_COLS]
    likert5_cols = [i for items in C.FACTOR_ITEMS.values() for i in items] + C.PERILAKU_ITEMS + sorted(C.COPING_ITEMS)
    likert7_cols = C.LINGKUNGAN_ITEMS
    for nama, cols, rentang in (
        ("kecemasan_1_4", anxiety_cols, C.ANXIETY_VALID_RANGE),
        ("likert_1_5", likert5_cols, C.LIKERT5_RANGE),
        ("dukungan_1_7", likert7_cols, C.LIKERT7_RANGE),
    ):
        lo, hi = rentang
        violating = {c: int((~df[c].between(lo, hi) & df[c].notna()).sum()) for c in cols}
        total = sum(violating.values())
        checks[nama] = {"rentang": [lo, hi], "pelanggaran": total,
                        "kolom_bermasalah": {k: v for k, v in violating.items() if v}}
        if total:
            _fail(f"validasi rentang gagal pada {nama}: {checks[nama]['kolom_bermasalah']}")

    for col, valid in (("d_tinggal_grup", set(C.TINGGAL_LABELS)), ("d_status_ortu", set(C.ORANGTUA_LABELS))):
        bad = sorted(set(df[col].unique()) - valid)
        if bad:
            _fail(f"kode hasil recode tidak valid pada {col}: {bad}")

    qa["validasi_akhir"] = checks
    return df


def _write_qa_report(qa: dict, path: Path) -> None:
    """Tulis laporan QA sederhana (markdown) berisi jumlah kasus per aturan cleaning."""
    lines = ["# Laporan QA — Tahap 1 Pembersihan Data", ""]
    for section, isi in qa.items():
        lines.append(f"## {section.replace('_', ' ').title()}")
        if isinstance(isi, dict):
            for k, v in isi.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(f"- {isi}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------- pipeline ----------


def run() -> tuple[pd.DataFrame, dict]:
    """Jalankan seluruh aturan cleaning; kembalikan (dataframe bersih, kumpulan laporan QA)."""
    qa: dict = {}
    df = _read_sav(C.DATA_RAW)
    qa["shape_awal"] = {"baris": int(df.shape[0]), "kolom": int(df.shape[1]),
                        "sumber": str(C.DATA_RAW.name)}

    dup_mask = df.duplicated(keep=False)
    qa["duplikat"] = {
        "baris_duplikat": int(dup_mask.sum()),
        "kebijakan": (
            "Tidak dihapus. Kemiripan jawaban seluruh kolom pada 2 baris bisa jadi duplikasi "
            "input atau bisa jadi dua siswa yang kebetulan menjawab identik; dropping akan "
            "mengubah N=306 menjadi 305 dan membuat deskriptif sebelumnya tidak sinkron. "
            "Ditandai untuk diperiksa dosen."
        ),
    }

    df = _fix_out_of_range(df, qa)
    df = _merge_split_anxiety_items(df, qa)
    df = _drop_dead_columns(df, qa)
    df = _reverse_items(df, qa)
    df = _recode_demographics(df, qa)
    df = _validate(df, qa)

    sisa = df.isna().sum()
    qa["missing_setelah_cleaning"] = {c: int(n) for c, n in sisa.items() if n}
    return df, qa


def main() -> None:
    df, qa = run()
    C.DATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.DATA_CLEAN, index=False)
    _write_qa_report(qa, C.DATA_CLEAN.parent / "qa_report.md")
    print(f"[01_clean] ditulis: {C.DATA_CLEAN}  shape={df.shape}")
    assert len(df) == 306, f"jumlah baris berubah: {len(df)} (harus 306)"
    assert not df.isna().any().any(), "masih ada missing value setelah cleaning"


if __name__ == "__main__":
    main()