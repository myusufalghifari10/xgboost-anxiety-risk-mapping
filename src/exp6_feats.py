"""Eksperimen 6 — pembangun fitur BARU dari kolom yang SUDAH ADA (tanpa data baru).

Tiga keluarga fitur, semuanya turunan dari 76 kolom yang sudah ada. Tidak ada
satu pun yang memakai informasi di luar X (aturan proyek: bangun dari data yang ada).

  B1 gaya respons  : ciri gaya menjawab per blok item (SD, rentang, run identik,
                     rasio jawaban ekstrem, rasio kosong). Murni PER-BARIS —
                     nilai seorang siswa tidak bergantung pada siswa lain.
  B2 komponen      : PCA(10) + PLS(5) dari 58 item non-target. WAJIB di-fit di
                     dalam fold (train saja) — dipakai lewat _fitur_fold().
  B3 interaksi     : 6 interaksi/rasio domain a-priori dari kolom f_* (tekanan,
                     dukungan, keseimbangan coping, digital vs dukungan keluarga).

Anti-bocor:
  * 20 item KECEMASANSTRESS (target) TIDAK PERNAH dipakai sebagai sumber fitur —
    dispersinya pun tidak (itu informasi yang tidak tersedia saat prediksi).
  * B1/B3 hanya fungsi per-baris -> aman tanpa fitting.
  * B2 di-fit HANYA di train fold; transformasi baris test juga tidak boleh
    bergantung pada baris test lain (dijamin sklearn; diuji di --self-test).

Pemakaian: diimpor exp6_train.py. Jalankan uji mandiri: `python src/exp6_feats.py --self-test`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from exp2_features import FACTOR_COLS, ITEM_COLS  # noqa: E402

# ---------- definisi blok & rentang (satu sumber: config.py) ----------

BLOK = {
    "faktor": FACTOR_COLS,
    "coping": sorted(C.COPING_ITEMS),
    "perilaku": C.PERILAKU_ITEMS,
    "lingkungan": C.LINGKUNGAN_ITEMS,
}
RENTANG = {
    "faktor": C.LIKERT5_RANGE,
    "coping": C.LIKERT5_RANGE,
    "perilaku": C.LIKERT5_RANGE,
    "lingkungan": C.LIKERT7_RANGE,
}

KOL_B1 = ["r_sd_faktor", "r_range_faktor", "r_sd_coping", "r_long_coping",
          "r_sd_perilaku", "r_sd_lingkungan", "r_range_lingkungan",
          "r_ekstrem_lingkungan", "r_na_frac"]
KOL_B2 = [f"pca{i}" for i in range(1, 11)] + [f"pls{i}" for i in range(1, 6)]
KOL_B3 = ["x_tekanan_total", "x_duk_total", "x_beban_vs_dukungan",
          "x_coping_balance", "x_coping_rasio", "x_digital_minus_duk"]
N_PCA, N_PLS = 10, 5

assert len(ITEM_COLS) == 58, f"harus 58 item non-target, dapat {len(ITEM_COLS)}"
assert len(KOL_B2) == N_PCA + N_PLS


# ---------- B1: gaya respons (per-baris) ----------

def _longstring(v: np.ndarray) -> float:
    """Run nilai identik berurutan terpanjang (NaN memutus run). 0 bila < 2 nilai."""
    best = cur = 1
    prev = None
    for x in v:
        if not np.isfinite(x):
            cur, prev = 1, None
            continue
        cur = cur + 1 if (prev is not None and x == prev) else 1
        best = max(best, cur)
        prev = x
    return float(best if np.isfinite(v).sum() >= 2 else 0.0)


def gaya_respons(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur gaya menjawab per siswa — hanya fungsi baris itu sendiri."""
    out = pd.DataFrame(index=df.index)
    for nama, cols in BLOK.items():
        M = df[cols].to_numpy(float)
        sd = np.nanstd(M, axis=1, ddof=1)
        rng_ = np.nanmax(M, axis=1) - np.nanmin(M, axis=1)
        out[f"r_sd_{nama}"] = np.where(np.isfinite(sd), sd, 0.0)
        out[f"r_range_{nama}"] = np.where(np.isfinite(rng_), rng_, 0.0)
        if nama == "coping":
            out["r_long_coping"] = np.array([_longstring(r) for r in M])
        if nama == "lingkungan":
            lo, hi = RENTANG[nama]
            ekst = np.isclose(M, lo) | np.isclose(M, hi)
            out["r_ekstrem_lingkungan"] = np.nanmean(ekst.astype(float), axis=1)
    out["r_na_frac"] = df[ITEM_COLS].isna().to_numpy(float).mean(axis=1)
    return out[KOL_B1].astype(float)


# ---------- B3: interaksi/rasio domain (per-baris) ----------

def interaksi(df: pd.DataFrame) -> pd.DataFrame:
    """6 interaksi a-priori (bukan hasil pencarian) dari kolom f_* yang sudah ada."""
    butuh = ["f_akademik", "f_keluarga", "f_sosial", "f_ekonomi", "f_digital",
             "f_masadepan", "f_duk_keluarga", "f_duk_teman", "f_duk_orang_dekat",
             "f_coping_adaptif", "f_coping_negatif"]
    hilang = [c for c in butuh if c not in df.columns]
    assert not hilang, f"kolom sumber interaksi hilang: {hilang}"
    out = pd.DataFrame(index=df.index)
    tekan = df[["f_akademik", "f_keluarga", "f_sosial", "f_ekonomi",
                "f_digital", "f_masadepan"]].to_numpy(float).sum(axis=1)
    duk = df[["f_duk_keluarga", "f_duk_teman", "f_duk_orang_dekat"]].to_numpy(float).sum(axis=1)
    posit, negat = df["f_coping_adaptif"].to_numpy(float), df["f_coping_negatif"].to_numpy(float)
    out["x_tekanan_total"] = tekan
    out["x_duk_total"] = duk
    out["x_beban_vs_dukungan"] = tekan - duk
    out["x_coping_balance"] = posit - negat
    out["x_coping_rasio"] = posit / (posit + negat + 1e-6)
    out["x_digital_minus_duk"] = df["f_digital"].to_numpy(float) - df["f_duk_keluarga"].to_numpy(float)
    return out[KOL_B3].astype(float)


# ---------- B2: komponen fold-safe ----------

def komponen(tr_items: np.ndarray, te_items: np.ndarray, y_tr: np.ndarray,
             n_pca: int = N_PCA, n_pls: int = N_PLS) -> tuple[np.ndarray, np.ndarray]:
    """PCA + PLS di-fit HANYA pada train fold. Mengembalikan (z_tr, z_te).

    Dua jaminan yang diuji di --self-test:
      (1) transform bergantung pada train (bukan hanya baris itu sendiri);
      (2) transform satu baris test tidak bergantung pada baris test lain.
    """
    tr = np.asarray(tr_items, float)
    te = np.asarray(te_items, float)
    y = np.asarray(y_tr, float).ravel()
    assert tr.ndim == te.ndim == 2 and tr.shape[1] == te.shape[1], "dimensi item tidak cocok"
    assert len(tr) == len(y), "y_tr tidak sepanjang train"
    assert len(tr) > n_pls + 1, "train terlalu kecil untuk PLS"
    assert np.isfinite(tr).all() and np.isfinite(te).all(), (
        "komponen: item masih mengandung NaN — imputasi dulu (median train fold, lihat _isi_nan)")
    sc = StandardScaler().fit(tr)
    a, b = sc.transform(tr), sc.transform(te)
    pca = PCA(n_components=n_pca, random_state=C.RANDOM_SEED).fit(a)
    pa, pb = pca.transform(a), pca.transform(b)
    pls = PLSRegression(n_components=n_pls, scale=False).fit(a, y)
    qa, qb = pls.transform(a), pls.transform(b)
    ztr = np.hstack([pa, qa])
    zte = np.hstack([pb, qb])
    assert ztr.shape == (len(tr), n_pca + n_pls), f"z_tr bentuk salah: {ztr.shape}"
    assert zte.shape == (len(te), n_pca + n_pls), f"z_te bentuk salah: {zte.shape}"
    assert np.isfinite(ztr).all() and np.isfinite(zte).all(), "komponen mengandung NaN/inf"
    return ztr.astype(float), zte.astype(float)


# ---------- self-test ----------

def _self_test() -> None:
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(index=range(n))
    for c in ITEM_COLS:
        hi = 7 if c in C.LINGKUNGAN_ITEMS else 5
        df[c] = rng.integers(1, hi + 1, n)
    df.loc[3:6, "COPING1"] = np.nan  # uji jalur kosong

    g = gaya_respons(df)
    assert list(g.columns) == KOL_B1 and len(g) == n
    assert np.isfinite(g.to_numpy()).all(), "B1 mengandung NaN"
    # (a) determinisme + independensi per-baris: dihitung ulang dari himpunan baris lain
    anak = gaya_respons(df.loc[[5, 1, 30]])
    assert np.allclose(anak.loc[1].to_numpy(), g.loc[1].to_numpy()), "B1 tidak per-baris"
    assert np.allclose(anak.loc[30].to_numpy(), g.loc[30].to_numpy()), "B1 tidak per-baris"
    # (b) nilai yang diharapkan pada blok yang sengaja konstan
    df2 = df.copy()
    for c in C.LINGKUNGAN_ITEMS:
        df2[c] = 4
    g2 = gaya_respons(df2)
    assert np.allclose(g2["r_sd_lingkungan"], 0.0) and np.allclose(g2["r_range_lingkungan"], 0.0)
    assert np.allclose(g2["r_long_coping"].iloc[0], g["r_long_coping"].iloc[0])
    # ranjang run identik: 12 jawaban sama -> longstring 12
    v = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3.0])
    assert _longstring(v) == 12.0 and _longstring(np.array([1.0, np.nan, 1.0])) == 1.0

    # B3
    for c in ["f_akademik", "f_keluarga", "f_sosial", "f_ekonomi", "f_digital",
              "f_masadepan", "f_duk_keluarga", "f_duk_teman", "f_duk_orang_dekat",
              "f_coping_adaptif", "f_coping_negatif"]:
        df[c] = rng.normal(3, 1, n)
    x = interaksi(df)
    assert list(x.columns) == KOL_B3 and np.isfinite(x.to_numpy()).all()
    assert np.allclose(x["x_tekanan_total"], df[["f_akademik", "f_keluarga", "f_sosial",
                                                 "f_ekonomi", "f_digital", "f_masadepan"]].sum(axis=1))
    assert np.allclose(x["x_beban_vs_dukungan"], x["x_tekanan_total"] - x["x_duk_total"])
    rasio = interaksi(df.assign(f_coping_adaptif=5.0, f_coping_negatif=5.0))
    assert np.allclose(rasio["x_coping_rasio"], 0.5, atol=1e-6), "rasio coping salah"

    # B2: fold-safety
    items = df[ITEM_COLS].to_numpy(float)
    y = rng.normal(2, 0.5, n)
    tr, te = np.arange(0, 30), np.arange(30, 40)
    # kontrak: pemanggil WAJIB mengimputasi item dulu (median train fold) — seperti exp6_train._rakit
    tolak = False
    try:
        komponen(items[tr], items[te], y[tr])
    except AssertionError as e:
        tolak = "NaN" in str(e)
        assert tolak, f"komponen menolak NaN dengan pesan yang salah: {e}"
    # tidak boleh lolos hampa: kalau komponen TIDAK melempar, assert di bawah yang jatuh
    assert tolak, "komponen harus menolak item yang masih NaN (kontrak pemanggil)"
    med = np.nanmedian(items[tr], axis=0)
    items = np.where(np.isnan(items), med, items)
    ztr, zte = komponen(items[tr], items[te], y[tr])
    assert ztr.shape == (30, 15) and zte.shape == (10, 15)
    # (1) transform WAJIB berubah bila train berbeda (bukti bergantung pada train)
    z2_tr, z2_te = komponen(items[5:35], items[te], y[5:35])
    assert not np.allclose(z2_te, zte), "komponen tidak bergantung pada train -> mencurigakan"
    # (2) transform satu baris test TIDAK bergantung pada baris test lain
    urut = np.array([4, 0, 9, 1, 8, 2, 7, 3, 6, 5])
    _, z3_te = komponen(items[tr], items[te][urut], y[tr])
    assert np.allclose(z3_te[urut.argsort()], zte, atol=1e-9), "komponen bocor antar baris test"

    print("self-test OK: B1 per-baris (deterministik, NaN-safe), B3 interaksi, "
          "B2 komponen fold-safe (tergantung train, invarian urutan baris test)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Eksperimen 6 — pembangun fitur baru")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    print("modul fitur E6 — impor dari exp6_train.py; gunakan --self-test untuk uji mandiri")


if __name__ == "__main__":
    main()
