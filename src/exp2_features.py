"""Eksperimen 2 — pembangun fitur item-level (76 fitur) + bobot gaya menjawab.

Keputusan BlackBox 2026-10-03: lever terbesar penurunan MAE adalah memasukkan
58 item mentah blok non-target sebagai fitur (selama ini hanya 13 rata-rata skala),
plus style-signature sebagai sample_weight (BUKAN fitur — red-team menunjukkan
fitur style rawan keying campuran; bobot lebih aman dan tetap mengubah loss).

Keluaran: outputs/exp2/features_exp2.parquet
  kolom = row_id + 18 fitur lama (dari features.parquet) + 58 item mentah
          + y_kontinu + y_biner + w_style

Anti-leakage: 20 item target TIDAK PERNAH disentuh di file ini (hanya blok
non-target). SOSIAL3 sudah di-reverse di 01_clean, jadi arah item sudah konsisten.

Skrip ini = preprocessing murni (boleh dijalankan siapa saja; tidak ada training).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

EXP2_DIR = C.ROOT / "outputs" / "exp2"
FEATURES_EXP2 = EXP2_DIR / "features_exp2.parquet"

# 58 item non-target, urutan tetap (satu sumber kebenaran = config.py)
FACTOR_COLS = [c for cols in C.FACTOR_ITEMS.values() for c in cols]
ITEM_COLS = FACTOR_COLS + sorted(C.COPING_ITEMS) + C.PERILAKU_ITEMS + C.LINGKUNGAN_ITEMS
BIG_BLOCKS = {
    "coping": sorted(C.COPING_ITEMS),
    "perilaku": C.PERILAKU_ITEMS,
    "lingkungan": C.LINGKUNGAN_ITEMS,
}


def _longstring(values: list[float]) -> int:
    """Run nilai identik berurutan terpanjang dalam satu blok."""
    best = cur = 1
    for a, b in zip(values, values[1:]):
        cur = cur + 1 if b == a else 1
        best = max(best, cur)
    return best


def _style_weights(clean: pd.DataFrame) -> pd.DataFrame:
    """Bobot gaya menjawab. Aturan sengaja sederhana (ponytail):
    straightliner = seluruh isi blok panjang (COPING/PERILAKU/LINGKUNGAN) identik
    -> w_style = 0.5, selain itu 1.0. Blok faktor 3-item sengaja tidak dipakai
    (terlalu pendek; jawaban seragam sering sah). Gerbang screening memutuskan
    apakah bobot ini membantu — kalau tidak, arm C4 dibuang."""
    rows = []
    for _, r in clean.iterrows():
        flags = []
        longs = {}
        for nama, cols in BIG_BLOCKS.items():
            vals = [float(r[c]) for c in cols]
            longs[nama] = _longstring(vals)
            flags.append(len(set(vals)) == 1)
        rows.append({
            "w_style": 0.5 if any(flags) else 1.0,
            "longstring_max": max(longs.values()),
            "flag_straightliner": int(any(flags)),
        })
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    feats = pd.read_parquet(C.FEATURES)
    clean = pd.read_parquet(C.DATA_CLEAN).reset_index(drop=True)
    assert len(feats) == len(clean) == 306, f"jumlah baris tak terduga: {len(feats)} / {len(clean)}"
    assert len(ITEM_COLS) == 58, f"jumlah item non-target tak terduga: {len(ITEM_COLS)}"

    missing = [c for c in ITEM_COLS if c not in clean.columns]
    assert not missing, f"kolom item hilang di data_clean: {missing}"
    target_like = [c for c in ITEM_COLS if c.startswith(("KECEMASAN", "KECEMSAN"))]
    assert not target_like, f"item target bocor ke fitur: {target_like}"  # anti-leakage

    items = clean[ITEM_COLS].astype(float)
    assert not items.isna().any().any(), "ada missing pada item non-target"
    assert items.to_numpy().min() >= 1 and items.to_numpy().max() <= 7, "nilai item di luar 1-7"

    style = _style_weights(clean)
    out = pd.concat(
        [feats[["row_id"] + C.FEATURE_COLS].reset_index(drop=True),
         items.reset_index(drop=True),
         feats[[C.TARGET_CONT, C.TARGET_BIN]].reset_index(drop=True),
         style],
        axis=1,
    )
    assert len(out.columns) == 1 + 18 + 58 + 2 + 3, f"skema tak terduga: {len(out.columns)} kolom"
    return out


def main() -> None:
    EXP2_DIR.mkdir(parents=True, exist_ok=True)
    out = build()
    out.to_parquet(FEATURES_EXP2, index=False)
    n_flag = int(out["flag_straightliner"].sum())
    print(f"OK features_exp2.parquet: {out.shape[0]} baris x {out.shape[1]} kolom")
    print(f"  fitur = 18 lama + 58 item mentah = 76; target terpisah; w_style aktif untuk {n_flag} siswa")
    print(f"  -> {FEATURES_EXP2}")


if __name__ == "__main__":
    main()
