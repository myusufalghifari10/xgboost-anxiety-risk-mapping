"""Tahap 6 — Penjelasan model (SHAP) dan uji Scientific kunci.

Dijalankan SETELAH 03_train_tune.py menghasilkan model_final.ubj + best_params.json
(PLAN bagian 8). Skrip ini tidak melakukan hyperparameter tuning. Training yang
diperbolehkan di sini hanya untuk dua hal: perekaman SHAP per fold (ukur stabilitas
peringkat) dan uji drop-HB (PLAN 8.5).

Keluaran (nama persis sesuai CONTRACT.md / konstanta path di config.py):
  outputs/shap_global_ranking.csv       peringkat 18 fitur + arah (8.1, sumber 06)
  outputs/stability_ranking.csv         stabilitas peringkat fitur antar fold (8.2)
  outputs/shap_local.csv                top faktor per siswa untuk triase (8.3)
  outputs/interaction_summary.json      angka interaksi fokus (8.4)
  outputs/drop_hb_test.json             uji klaim protektif perilaku sehat (8.5)
  outputs/figures/shap_global.png       bar chart peringkat + arah (8.1)
  outputs/figures/shap_interaction_*.png  interaksi perilaku sehat x dukungan/tekanan (8.4)
  outputs/figures/pdp_*.png             partial dependence fitur kunci (8.6)

Kedelapan artefak di atas ditulis seluruhnya oleh run_all() dalam sekali jalan;
fungsi shap_global_binary (eksperimen model biner) sengaja TIDAK dipanggil run_all
(di luar cakupan PLAN bagian 8 fase ini).

Sudut yang disengaja:
  - TreeSHAP dipilih karena model XGBoost berbasis pohon; interaksi diambil dari
    shap_interaction_values (tanpa dependensi tambahan) dan selalu disimetresisasi
    karena matriks interaksi SHAP simetris.
  - Stabilitas peringkat (8.2) diukur sebagai proporsi fold CV tempat fitur masuk
    5 teratas berdasarkan |SHAP| rata-rata. Metode deterministik dan murah di N=306.
  - Uji drop-HB memakai skema outer CV yang identik dengan 03 (jumlah fold, jumlah
    ulangan, dan seed sama) sehingga selisih RMSE tidak bias oleh perbedaan skenario.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import config as C

STABILITY_TOP_K = 5          # berapa fitur teratas yang dihitung "stabil"
LOCAL_TOP_N = 5              # berapa faktor utama per siswa (bahan triase)
N_FEATURES_EXPECTED = 18
PD_FEATURES: Sequence[str] = (
    "f_perilaku_sehat",
    "f_sosial",
    "f_digital",
    "f_coping_negatif",
    "f_duk_keluarga",
)
HB_FEATURE = "f_perilaku_sehat"
SUPPORT_FEATURES: Sequence[str] = ("f_duk_keluarga", "f_duk_teman", "f_duk_orang_dekat")
STRESS_FEATURES: Sequence[str] = tuple(f"f_{k}" for k in C.FACTOR_ITEMS)
FOCUS_PAIRS: Sequence[Tuple[str, str]] = tuple(
    [(HB_FEATURE, f) for f in SUPPORT_FEATURES + STRESS_FEATURES]
)

# Jalur model klasifikasi (XGBClassifier): path-nya sudah ditetapkan di config.py
# (C.MODEL_CLF_PATH) dan ditulis 03_train_tune.py. Penjelasan SHAP biner sengaja
# tidak dijalankan di run_all() — di luar cakupan PLAN bagian 8 pada fase ini.
CLF_MODEL_PATH: Path | None = None

BASE_XGB = C.BASE_XGB  # fix C4: satu sumber di config (dulu duplikat identik dengan 03)
# Path artefak diambil dari config.py (sumber kebenaran tunggal), bukan literal.
OUT_GLOBAL_RANKING = C.SHAP_GLOBAL_RANKING
OUT_INTERACTION = C.INTERACTION_SUMMARY
OUT_STABILITY = C.STABILITY_RANKING
OUT_SHAP_LOCAL = C.SHAP_LOCAL
OUT_DROP_HB = C.DROP_HB_JSON


# ---------------------------------------------------------------------------
# Artefak & matriks fitur
# ---------------------------------------------------------------------------
def require_artifact(path: Path, dibuat_oleh: str) -> Path:
    """Fail-fast dengan pesan spesifik bila artefak model belum ada."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} belum ada. Jalankan {dibuat_oleh} lebih dulu; "
            "penjelasan model hanya bisa dijalankan setelah model terlatih."
        )
    return path


def load_features() -> pd.DataFrame:
    """Baca features.parquet dan pastikan 18 fitur + 2 target lengkap."""
    df = pd.read_parquet(require_artifact(C.FEATURES, "02_features.py"))
    missing = [
        c for c in list(C.FEATURE_COLS) + [C.TARGET_CONT, C.TARGET_BIN] if c not in df.columns
    ]
    if missing:
        raise KeyError(f"features.parquet tidak lengkap, kolom hilang: {missing}")
    return df


def load_preprocessor():
    """Muat encoder kategorikal yang SAMA persis dengan saat training (kontrak revisi).

    Objek ini adalah sumber skema kolom ter-encode; dipakai lewat pre.transform()
    di encode_features (fix temuan round-2 R2-1: versi lama memuatnya lalu
    membuangnya, dan masih membangun one-hot sendiri via pd.get_dummies).
    """
    import joblib

    return joblib.load(require_artifact(C.PREPROCESSOR_PATH, "03_train_tune.py"))


def encode_features(df: pd.DataFrame, pre, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Encode fitur dengan preprocessor ter-training (preprocessor.joblib).

    Fix temuan round-2 R2-1: versi lama TIDAK memakai parameter `pre` sama
    sekali — masih membangun one-hot sendiri dengan pd.get_dummies (duplikasi
    yang bisa berdrift dari skema model). Sekarang transform() dari objek
    preprocessor yang sama dipakai 03_train_tune.py yang jadi acuan, dan
    kesesuaian kolom diverifikasi fail-fast.
    """
    if feature_columns is None:
        raise KeyError(
            "best_params.json tidak punya kunci 'feature_columns'; "
            "jalankan ulang 03_train_tune.py (versi terkini) terlebih dahulu."
        )
    hilang = [c for c in C.FEATURE_COLS if c not in df.columns]
    if hilang:
        raise KeyError(f"features.parquet tidak punya kolom fitur: {hilang}")
    # Nama kolom dari preprocessor WAJIB sama persis dengan yang direkam 03 di
    # best_params.json["feature_columns"]; jika tidak, skema penjelasan menyimpang.
    expected = [str(c) for c in pre.get_feature_names_out()]
    if expected != list(feature_columns):
        raise ValueError(
            "Skema kolom preprocessor tidak sama dengan feature_columns di "
            f"best_params.json. Preprocessor: {expected}; best_params.json: {list(feature_columns)}."
        )
    return pd.DataFrame(
        pre.transform(df[list(C.FEATURE_COLS)]),
        columns=list(feature_columns),
        index=df.index,
    )


def read_model_meta() -> Dict:
    """Baca best_params.json (kunci kontrak: best_params, best_params_raw, feature_columns).

    Fix blocker rev-explain B3: versi lama kembali {} bila berkas hilang sehingga
    stability_ranking & drop_hb_test diam-diam memakai model DEFAULT tanpa tuning.
    Sekarang fail-fast: penjelasan hanya sah dengan setelan pemenang yang sama.
    """
    meta = json.loads(
        require_artifact(C.BEST_PARAMS_JSON, "03_train_tune.py").read_text(encoding="utf-8")
    )
    for key in ("best_params", "feature_columns"):
        if not meta.get(key):
            raise KeyError(
                f"best_params.json tidak punya kunci '{key}'; "
                "jalankan ulang 03_train_tune.py (versi terkini) terlebih dahulu."
            )
    return meta


def assert_outer_config(meta: Dict) -> None:
    """Pastikan skema CV 05 sama dengan yang dipakai 03 (catatan rev-explain N4).

    Fix temuan round-2 R2-2: versi lama punya `if outer and (...)` sehingga bila
    kunci 'outer' hilang/kosong pengecekan DIAM-DIAM dilewati — justru kasus yang
    paling perlu gagal. Sekarang fail-fast kalau jejak skema tidak ada.
    """
    outer = meta.get("outer") or {}
    if not outer:
        raise KeyError(
            "best_params.json tidak punya jejak 'outer' (folds/repeats); "
            "jalankan ulang 03_train_tune.py (versi terkini) terlebih dahulu."
        )
    if outer.get("folds") != C.OUTER_FOLDS or outer.get("repeats") != C.OUTER_REPEATS:
        raise ValueError(
            "Skema outer CV 05_explain ("
            f"folds={C.OUTER_FOLDS}, repeats={C.OUTER_REPEATS}) berbeda dari yang dipakai "
            f"03_train_tune.py (folds={outer.get('folds')}, repeats={outer.get('repeats')}). "
            "Jalankan 03 dengan argumen default atau sesuaikan config.py."
        )


def tuned_model_factory() -> Callable[[], object]:
    """Fabrik XGBRegressor dengan setelan pemenang tuning + seed/n_jobs deterministik.

    Fix catatan rev-explain N2: sebelumnya hanya 9 param ruang pencarian yang
    di-merge, sehingga random_state jatuh ke default xgboost (0) dan model bisa
    tidak deterministik saat subsample < 1.
    """
    import xgboost as xgb

    params = dict(read_model_meta()["best_params"])
    return lambda: xgb.XGBRegressor(**params, **BASE_XGB)


def assert_schema_matches_model(model, X_encoded: pd.DataFrame) -> None:
    """Pastikan skema kolom ter-encode identik dengan yang dilatih model (kontrak revisi).

    Fix C5: model dilatih pada numpy (tanpa nama kolom) sehingga booster.feature_names
    = None dan cek nama lama selalu dilewati tanpa menahan apa pun. Cek jumlah kolom
    terhadap booster.num_features() benar-benar menahan mismatch skema.
    """
    booster = model.get_booster()
    names = booster.feature_names
    if names is not None and list(X_encoded.columns) != list(names):
        raise ValueError(
            "Skema penjelasan != skema model. "
            f"Model: {list(names)}; hasil encode: {list(X_encoded.columns)}."
        )
    if X_encoded.shape[1] != booster.num_features():
        raise ValueError(
            f"Jumlah kolom hasil encode ({X_encoded.shape[1]}) != jumlah fitur model "
            f"({booster.num_features()})."
        )


def _source_of(col: str, cols: Sequence[str]) -> str:
    """Petakan satu kolom ter-encode ke fitur sumbernya di C.FEATURE_COLS."""
    if col in C.FEATURE_COLS:
        return col
    matches = [f for f in C.FEATURE_COLS if col.startswith(f"{f}_")]
    if len(matches) != 1:
        raise KeyError(f"Kolom ter-encode '{col}' tidak bisa dipetakan unik ke fitur sumber.")
    return matches[0]


def source_features(feature_columns: Sequence[str]) -> List[str]:
    """Daftar 18 fitur sumber (urutan config) dari daftar kolom ter-encode.

    Fix blocker rev-explain B5: one-hot harus dikembalikan ke fitur asalnya agar
    artefak SHAP benar-benar 18 baris sesuai PLAN 8.1, bukan ~30 kolom terpotong.
    """
    cols = list(feature_columns)
    for col in cols:
        _source_of(col, cols)
    missing = [f for f in C.FEATURE_COLS if f not in {_source_of(c, cols) for c in cols}]
    if missing:
        raise KeyError(f"Fitur sumber tidak muncul di feature_columns: {missing}")
    return list(C.FEATURE_COLS)


def aggregate_shap_to_source(
    shap_values: np.ndarray, feature_columns: Sequence[str]
) -> np.ndarray:
    """Jumlahkan SHAP kolom one-hot ke fitur sumbernya (18 kolom).

    Kontribusi one-hot bersifat aditif, sehingga penjumlahan adalah agregasi yang
    benar secara explainability: total kontribusi satu fitur = jumlah kontribusi
    seluruh kategorinya.
    """
    cols = list(feature_columns)
    out = np.zeros((shap_values.shape[0], len(C.FEATURE_COLS)), dtype=float)
    for j, col in enumerate(cols):
        out[:, C.FEATURE_COLS.index(_source_of(col, cols))] += shap_values[:, j]
    return out


def load_regressor() -> object:
    """Muat XGBRegressor terlatih dari outputs/model_final.ubj."""
    import xgboost as xgb

    require_artifact(C.MODEL_PATH, "03_train_tune.py")
    model = xgb.XGBRegressor()
    model.load_model(C.MODEL_PATH)
    return model


# ---------------------------------------------------------------------------
# 8.1 / 8.3 — SHAP
# ---------------------------------------------------------------------------
def compute_shap(model, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Hitung nilai SHAP dengan TreeSHAP; kembalikan matriks (n_siswa, n_fitur)."""
    import shap

    values = shap.TreeExplainer(model).shap_values(X)
    if hasattr(values, "values"):        # shap.Explanation (shap >= 0.45)
        values = values.values
    # Fix catatan rev-explain N3: shap < 0.45 mengembalikan list per kelas; np.asarray(list)
    # akan memotong sumbu yang salah sehingga angka diam-diam salah.
    if isinstance(values, list):
        values = values[1]              # kelas positif (cemas tinggi)
    values = np.asarray(values)
    if values.ndim == 3:                 # (n, fitur, kelas) -> ambil kelas positif
        values = values[:, :, 1]
    if values.shape[0] != len(X):
        raise ValueError(
            f"Dimensi SHAP tidak sesuai jumlah baris: {values.shape} vs {len(X)} baris."
        )
    return values, list(X.columns)


def global_ranking(shap_values: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    """Rata-rata |SHAP| per fitur sebagai besar pengaruh, plus arah rata-ratanya."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    out = pd.DataFrame(
        {
            "fitur": list(columns),
            "mean_abs_shap": mean_abs,
            "mean_shap_signed": mean_signed,
            "arah": np.where(mean_signed >= 0, "kontribusi_rata2_menaikkan", "kontribusi_rata2_menurunkan"),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    out.insert(0, "peringkat", np.arange(1, len(out) + 1))
    return out


def plot_shap_global(ranking: pd.DataFrame, top_n: int = N_FEATURES_EXPECTED) -> Path:
    """Bar chart 18 fitur: panjang = pengaruh, warna = arah."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = ranking.head(top_n)
    colors = [
        "#c0392b" if a == "kontribusi_rata2_menaikkan" else "#1e8449" for a in top["arah"]
    ]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(top) + 1.5))
    ax.barh(top["fitur"][::-1], top["mean_abs_shap"][::-1], color=colors[::-1])
    ax.set_xlabel("Rata-rata |SHAP| terhadap skor kecemasan")
    ax.set_title("Peringkat faktor risiko (merah = menaikkan, hijau = menurunkan)")
    fig.tight_layout()
    path = C.FIG_DIR / "shap_global.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def shap_global_binary(model_clf, X: pd.DataFrame) -> pd.DataFrame:
    """Peringkat faktor untuk XGBClassifier (kelas cemas tinggi).

    Nilai SHAP pada klasifikasi berada pada skala log-odds: nilai positif menaikkan
    peluang masuk kategori cemas tinggi. Fungsi ini menerima objek model, bukan
    path file, karena jalur artefak klasifikasi belum disepakati di kontrak.
    """
    values, columns = compute_shap(model_clf, X)
    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    out = pd.DataFrame(
        {
            "fitur": columns,
            "mean_abs_shap": mean_abs,
            "mean_shap_signed": mean_signed,
            "arah": np.where(mean_signed >= 0, "kontribusi_rata2_menaikkan", "kontribusi_rata2_menurunkan"),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    out.insert(0, "peringkat", np.arange(1, len(out) + 1))
    return out


def shap_local_table(
    shap_values: np.ndarray, columns: Sequence[str], df: pd.DataFrame
) -> pd.DataFrame:
    """Top-N kontributor per siswa — bahan triase guru BK (bukan diagnosis)."""
    raw = df[list(C.FEATURE_COLS)]
    top_idx = np.argsort(-np.abs(shap_values), axis=1)[:, :LOCAL_TOP_N]
    rows: List[Dict] = []
    for i in range(len(df)):
        for rank, j in enumerate(top_idx[i], start=1):
            feature = columns[j]
            rows.append(
                {
                    "row_id": int(df["row_id"].iloc[i]),
                    "peringkat_faktor": rank,
                    "fitur": feature,
                    "nilai_fitur": (
                        raw.iloc[i][feature] if feature in C.FEATURE_COLS else np.nan
                    ),
                    "shap": float(shap_values[i, j]),
                    "arah": "menaikkan" if shap_values[i, j] >= 0 else "menurunkan",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8.2 — Stabilitas peringkat antar fold
# ---------------------------------------------------------------------------
def outer_cv_splits(n_rows: int, seed: int = C.RANDOM_SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Outer CV: OUTER_FOLDS x OUTER_REPEATS, identik dengan skema di 03."""
    from sklearn.model_selection import KFold

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(C.OUTER_REPEATS):
        kf = KFold(n_splits=C.OUTER_FOLDS, shuffle=True, random_state=seed + repeat)
        for train_idx, test_idx in kf.split(np.arange(n_rows)):
            splits.append((train_idx, test_idx))
    return splits


def stability_ranking(
    model_factory: Callable[[], object], df: pd.DataFrame, X_encoded: pd.DataFrame
) -> pd.DataFrame:
    """Proporsi fold CV tempat fitur masuk TOP-K by |SHAP| (model dilatih per fold).

    Peringkat dihitung pada 18 fitur sumber (one-hot diagabung), konsisten dengan
    shap_global.png; tanpa ini tabel jadi ~30 baris dan tidak bisa dibandingkan.
    """
    y = df[C.TARGET_CONT].to_numpy()
    columns = list(X_encoded.columns)
    top_k_count = {c: 0 for c in C.FEATURE_COLS}
    n_folds = 0
    for train_idx, test_idx in outer_cv_splits(len(df)):
        model = model_factory()
        model.fit(X_encoded.iloc[train_idx], y[train_idx])
        values, _ = compute_shap(model, X_encoded.iloc[test_idx])
        mean_abs = np.abs(aggregate_shap_to_source(values, columns)).mean(axis=0)
        for j in np.argsort(-mean_abs)[:STABILITY_TOP_K]:
            top_k_count[C.FEATURE_COLS[j]] += 1
        n_folds += 1
    out = pd.DataFrame(
        {
            "fitur": list(C.FEATURE_COLS),
            f"top{STABILITY_TOP_K}_count": [top_k_count[c] for c in C.FEATURE_COLS],
            f"proporsi_top{STABILITY_TOP_K}": [top_k_count[c] / n_folds for c in C.FEATURE_COLS],
            "n_folds": n_folds,
        }
    ).sort_values(f"proporsi_top{STABILITY_TOP_K}", ascending=False, ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# 8.4 — Interaksi
# ---------------------------------------------------------------------------
def interaction_matrix(model, X: pd.DataFrame) -> np.ndarray:
    """Matriks interaksi SHAP (n, fitur, fitur), disimetresisasi dan diag nol."""
    import shap

    inter = shap.TreeExplainer(model).shap_interaction_values(X)
    if isinstance(inter, list):          # XGBClassifier -> ambil kelas positif
        inter = inter[1]
    inter = np.asarray(inter)
    if inter.ndim != 3:
        raise ValueError(f"Dimensi interaksi SHAP tidak diharapkan: {inter.shape}")
    inter = 0.5 * (inter + inter.transpose(0, 2, 1))   # simetris
    idx = np.arange(inter.shape[1])
    inter[:, idx, idx] = 0.0                            # buang main effect
    return inter


def aggregate_interactions_to_source(inter: np.ndarray, feature_columns: Sequence[str]) -> np.ndarray:
    """Agregasi tensor interaksi ter-encode ke 18 fitur sumber (jumlah blok one-hot)."""
    cols = list(feature_columns)
    src_idx = [C.FEATURE_COLS.index(_source_of(c, cols)) for c in cols]
    out = np.zeros((inter.shape[0], len(C.FEATURE_COLS), len(C.FEATURE_COLS)), dtype=float)
    for a, ia in enumerate(src_idx):
        for b, ib in enumerate(src_idx):
            out[:, ia, ib] += inter[:, a, b]
    return out


def focus_interactions(inter: np.ndarray, columns: Sequence[str]) -> Dict[str, object]:
    """Rata-rata |interaksi| untuk pasangan fokus (perilaku sehat x Dukungan/tekanan).

    Konvensi SHAP interaction (catatan N3/F5 review):
    - Nilai luar diagonal = SETENGAH efek interaksi pasangan (definisi SHAP membagi
      simetris; efek penuh = 2x nilai bila dikutip di paper).
    - Diagonal fitur kategorikal bisa != 0 setelah agregasi one-hot ke fitur sumber
      (pasangan dummy satu-fitur jatuh ke diagonal) — bukan main effect murni.
    """
    resolved = []
    for a, b in FOCUS_PAIRS:
        if a not in columns or b not in columns:
            continue
        ia, ib = list(columns).index(a), list(columns).index(b)
        resolved.append((a, b, float(np.abs(inter[:, ia, ib]).mean())))
    return {
        "pairs": [{"fitur_a": a, "fitur_b": b, "mean_abs_interaksi": v} for a, b, v in resolved],
        "semua_fokus_tersedia": len(resolved) == len(FOCUS_PAIRS),
        "cara_baca": (
            "Nilai = rata-rata |SHAP interaction| luar diagonal = SETENGAH efek interaksi "
            "pasangan (konvensi SHAP membagi simetris; efek penuh = 2x). Diagonal fitur "
            "kategorikal bisa != 0 setelah agregasi one-hot (bukan main effect murni)."
        ),
    }


def plot_interaction_heatmap(inter: np.ndarray, columns: Sequence[str], top_n: int = 10) -> Path:
    """Heatmap rata-rata |interaksi| untuk 10 fitur teratas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strength = np.abs(inter).mean(axis=0).sum(axis=1)
    top_idx = np.argsort(-strength)[:top_n]
    sub = np.abs(inter).mean(axis=0)[np.ix_(top_idx, top_idx)]
    labels = [columns[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(sub, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Rata-rata |interaksi SHAP| (tanpa main effect)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = C.FIG_DIR / "shap_interaction_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_interaction_scatter(
    inter: np.ndarray, tensor_cols: Sequence[str], values: pd.DataFrame, feat_a: str, feat_b: str
) -> Path:
    """Scatter nilai interaksi (a,b) terhadap nilai fitur b — bentuk non-linear interaksi.

    Fix temuan round-2 R2-3: versi lama mencari indeks fitur di `X.columns`
    (ruang ter-encode, ~30 kolom) lalu memakainya untuk mengindeks tensor yang
    sudah diagabung ke 18 fitur sumber — benar hanya karena kebetulan urutan
    kolom. Sekarang indeks diturunkan dari `tensor_cols` (nama sumbu tensor),
    sedangkan nilai sumbu x diambil dari features.parquet (nilai yang dibaca manusia).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = list(tensor_cols)
    for f in (feat_a, feat_b):
        if f not in cols:
            raise KeyError(f"Fitur '{f}' tidak ada di sumbu tensor interaksi: {cols}")
    if feat_b not in values.columns:
        raise KeyError(f"Fitur '{feat_b}' tidak ada di features.parquet untuk nilai sumbu x.")
    ia, ib = cols.index(feat_a), cols.index(feat_b)
    y_vals = inter[:, ia, ib]
    x_vals = values[feat_b].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_vals, y_vals, s=12, alpha=0.55, color="#1f618d")
    ax.axhline(0.0, color="#7f8c8d", linewidth=1)
    ax.set_xlabel(feat_b)
    ax.set_ylabel(f"SHAP interaction: {feat_a} x {feat_b}")
    ax.set_title(
        f"Apakah pengaruh {feat_b} berubah saat {feat_a} tinggi?"
    )
    fig.tight_layout()
    path = C.FIG_DIR / f"shap_interaction_{feat_a}__{feat_b}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 8.6 — Partial dependence
# ---------------------------------------------------------------------------
def plot_partial_dependence(
    model, X: pd.DataFrame, features: Sequence[str] = PD_FEATURES
) -> List[Path]:
    """Kurva partial dependence (rata-rata) untuk fitur kunci."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import partial_dependence

    paths: List[Path] = []
    for feat in features:
        if feat not in X.columns:
            # Fix C5: deliverable tidak boleh hilang senyap — fail-fast seperti scatter.
            raise KeyError(
                f"Fitur PDP '{feat}' tidak ada di matriks ter-encode; "
                "periksa PD_FEATURES terhadap FEATURE_COLS/encoding."
            )
        # Fix F2(a): sklearn 1.9 mengembalikan Bunch/dict; unpack posisi menghasilkan
        # string key (silent bug: grafik berisi 1 titik sampah). Akses by-key.
        pdp = partial_dependence(model, X, [feat], kind="average", grid_resolution=20)
        average = np.asarray(pdp["average"]).ravel()
        grid = np.asarray(pdp["grid_values"][0]).ravel()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        # Fix F2(b): posisi ke-3 scatter = ukuran marker, bukan format string -> plot.
        ax.plot(grid, average, ".-", color="#c0392b", markersize=4)
        ax.set_xlabel(feat)
        ax.set_ylabel("Prediksi rata-rata (skor kecemasan)")
        ax.set_title(f"Partial dependence: {feat}")
        fig.tight_layout()
        path = C.FIG_DIR / f"pdp_{feat}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 8.5 — Uji drop-HB
# ---------------------------------------------------------------------------
def drop_hb_test(
    model_factory_full: Callable[[], object],
    model_factory_without_hb: Callable[[], object],
    df: pd.DataFrame,
    X_encoded: pd.DataFrame,
) -> Dict:
    """Bandingkan RMSE test model lengkap vs tanpa f_perilaku_sehat pada CV identik.

    selisih_rmse_mean > 0 berarti menghapus perilaku sehat MEMBURUKKAN prediksi,
    yaitu bukti kontribusi positif fitur tersebut pada data ini. Nilai ~0 berarti
    tidak ada kontribusi bermakna — jawaban jujur untuk klaim protektif riset.
    """
    from sklearn.metrics import root_mean_squared_error

    if HB_FEATURE not in X_encoded.columns:
        raise KeyError(f"Fitur {HB_FEATURE} tidak ditemukan di matriks ter-encode.")
    keep = [c for c in X_encoded.columns if c != HB_FEATURE]
    y = df[C.TARGET_CONT].to_numpy()
    rmse_full: List[float] = []
    rmse_tanpa: List[float] = []
    for train_idx, test_idx in outer_cv_splits(len(df)):
        m_full = model_factory_full()
        m_full.fit(X_encoded.iloc[train_idx], y[train_idx])
        rmse_full.append(
            root_mean_squared_error(y[test_idx], m_full.predict(X_encoded.iloc[test_idx]))
        )
        m_hb = model_factory_without_hb()
        m_hb.fit(X_encoded.iloc[train_idx][keep], y[train_idx])
        rmse_tanpa.append(
            root_mean_squared_error(
                y[test_idx], m_hb.predict(X_encoded.iloc[test_idx][keep])
            )
        )
    diff = np.array(rmse_tanpa) - np.array(rmse_full)
    return {
        "fitur": HB_FEATURE,
        "rmse_lengkap_mean": float(np.mean(rmse_full)),
        "rmse_tanpa_hb_mean": float(np.mean(rmse_tanpa)),
        "selisih_rmse_mean": float(diff.mean()),
        "selisih_rmse_std": float(diff.std(ddof=1)),  # Fix D3c: SD sampel (ddof=1), dikutip laporan
        "proporsi_fold_hb_membantu": float(np.mean(diff > 0)),
        "n_folds": int(len(diff)),
        "cara_baca": (
            "selisih_rmse_mean positif berarti menghapus f_perilaku_sehat memperburuk "
            "prediksi, jadi fitur tersebut membantu. Nilai mendekati nol berarti tidak "
            "ada kontribusi bermakna pada data ini."
        ),
    }


# ---------------------------------------------------------------------------
# Orkestrasi
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)


def run_all() -> Dict[str, object]:
    """Jalankan seluruh analisis penjelasan; kembalikan ringkasan status."""
    ensure_dirs()
    meta = read_model_meta()
    assert_outer_config(meta)
    feature_columns = list(meta["feature_columns"])

    df = load_features()
    pre = load_preprocessor()
    X_encoded = encode_features(df, pre, feature_columns)
    model = load_regressor()
    assert_schema_matches_model(model, X_encoded)

    # assert level-1 sesuai kontrak
    assert len(C.FEATURE_COLS) == N_FEATURES_EXPECTED, "jumlah fitur harus 18"
    assert df[C.TARGET_CONT].between(*C.ANXIETY_VALID_RANGE).all(), "target di luar 1-4"

    # SHAP dihitung di ruang ter-encode, lalu diagabung ke 18 fitur sumber (B5)
    shap_enc, cols_enc = compute_shap(model, X_encoded)
    shap_src = aggregate_shap_to_source(shap_enc, cols_enc)
    source_cols = source_features(cols_enc)

    ranking = global_ranking(shap_src, source_cols)
    # B1: simpan ranking global — satu-satunya sumber arah & prioritas bagi 06_report.py
    ranking[["fitur", "mean_abs_shap", "mean_shap_signed", "arah"]].to_csv(
        OUT_GLOBAL_RANKING, index=False
    )
    plot_shap_global(ranking)
    shap_local_table(shap_src, source_cols, df).to_csv(OUT_SHAP_LOCAL, index=False)
    stability_ranking(tuned_model_factory(), df, X_encoded).to_csv(OUT_STABILITY, index=False)

    inter_enc = interaction_matrix(model, X_encoded)
    inter_src = aggregate_interactions_to_source(inter_enc, cols_enc)
    plot_interaction_heatmap(inter_src, source_cols)
    plot_interaction_scatter(inter_src, source_cols, df, HB_FEATURE, "f_duk_keluarga")
    plot_interaction_scatter(inter_src, source_cols, df, HB_FEATURE, "f_sosial")

    plot_partial_dependence(model, X_encoded)

    # drop-HB memakai setelan pemenang yang sama dengan SHAP global (B3)
    drop = drop_hb_test(tuned_model_factory(), tuned_model_factory(), df, X_encoded)
    OUT_DROP_HB.write_text(json.dumps(drop, indent=2), encoding="utf-8")

    # N13: persist angka interaksi fokus (jawaban "perilaku sehat x dukungan")
    fokus = focus_interactions(inter_src, source_cols)
    OUT_INTERACTION.write_text(json.dumps(fokus, indent=2), encoding="utf-8")

    return {
        "peringkat_shap": ranking.head(5).to_dict("records"),
        "interaksi_fokus": fokus,
        "drop_hb": drop,
    }


if __name__ == "__main__":  # pragma: no cover - dijalankan setelah training disetujui
    print(json.dumps(run_all(), indent=2, default=str))