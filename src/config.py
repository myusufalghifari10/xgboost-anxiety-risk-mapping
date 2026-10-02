"""Konfigurasi pusat — SATU-SATUNYA sumber kebenaran untuk semua skrip pipeline.

Kontrak antar-skrip (lihat CONTRACT.md). Semua path relatif terhadap root repo.
"""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "INPUT DATA  (1).sav"
DATA_CLEAN = ROOT / "outputs" / "data_clean.parquet"
FEATURES = ROOT / "outputs" / "features.parquet"
MODEL_PATH = ROOT / "outputs" / "model_final.ubj"
TRIALS_CSV = ROOT / "outputs" / "optuna_trials.csv"
METRICS_JSON = ROOT / "outputs" / "metrics.json"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
CODEBOOK = ROOT / "docs" / "codebook.md"

# ---------- Reproducibility ----------
RANDOM_SEED = 42

# ---------- Item inventories (nama kolom persis seperti di .sav) ----------
# Item kecemasan (skala 1-4). Kolom 14 & 20 adalah artefak kalimat terbelah (semua 0).
ANXIETY_ITEMS_ALL = [f"KECEMASANSTRESS{i}" for i in range(1, 23)]
ANXIETY_ITEMS_ALL[8] = "KECEMSANSTRESS9"  # typo di data asli kolom 9
ANXIETY_SPLIT_PAIRS = [("KECEMASANSTRESS13", "KECEMASANSTRESS14"),
                       ("KECEMASANSTRESS19", "KECEMASANSTRESS20")]
ANXIETY_DEAD_COLS = ["KECEMASANSTRESS14", "KECEMASANSTRESS20"]
ANXIETY_VALID_RANGE = (1, 4)
ANXIETY_OUT_OF_RANGE_FIX = {"KECEMASANSTRESS12": 5}  # 1 kasus -> missing -> median

# Item faktor tekanan (skala 1-5)
FACTOR_ITEMS = {
    "akademik": ["AKADEMIK1", "AKADEMIK2", "AKADEMIK3"],
    "keluarga": ["KELUARGA1", "KELUARGA2", "KELUARGA3"],
    "sosial": ["SOSIAL1", "SOSIAL2", "SOSIAL3"],
    "ekonomi": ["EKONOMI1", "EKONOMI2", "EKONOMI3"],
    "digital": ["DIGITAL1", "DIGITAL2", "DIGITAL3"],
    "masadepan": ["MASADEPAN1", "MASADEPAN2", "MASADEPAN3"],
}
LIKERT5_RANGE = (1, 5)

# Coping (skala 1-5)
COPING_ITEMS = {f"COPING{i}" for i in range(1, 19)}
COPING_SUBSCALES = {
    "coping_adaptif": [f"COPING{i}" for i in range(1, 13)],   # 1-12
    "coping_ekspresi": [f"COPING{i}" for i in range(13, 16)],  # 13-15
    "coping_negatif": [f"COPING{i}" for i in range(16, 19)],   # 16-18
}

# Perilaku sehat (skala 1-5)
PERILAKU_ITEMS = [f"PERILAKU{i}" for i in range(1, 11)]

# Dukungan lingkungan MSPSS (skala 1-7)
LINGKUNGAN_ITEMS = [f"LINGKUNGAN{i}" for i in range(1, 13)]
LINGKUNGAN_SUBSCALES = {
    "duk_keluarga": [f"LINGKUNGAN{i}" for i in range(1, 5)],   # 1-4
    "duk_teman": [f"LINGKUNGAN{i}" for i in range(5, 9)],      # 5-8
    "duk_orang_dekat": [f"LINGKUNGAN{i}" for i in range(9, 13)],  # 9-12
}
LIKERT7_RANGE = (1, 7)

# Item berlawanan arah -> reverse-score: nilai_baru = (min+max) - nilai_lama
REVERSE_ITEMS = ["SOSIAL3"]  # "saya merasa TIDAK tertekan ketika ..."

# ---------- Aturan cleaning ----------
ANXIETY_OUT_OF_RANGE = {"KECEMASANSTRESS12": [5]}  # nilai di luar 1-4 -> missing -> median kolom

# Recode ORANGTUA -> 4 kelompok (asumsi label, menunggu konfirmasi; lihat PLAN §1)
ORANGTUA_RECODE = {
    1: 1,  # utuh (asumsi: "kedua orang tua masih hidup/bersama")
    2: 4,  # lainnya / tidak diketahui (gabungan 2+5)
    3: 2,  # orang tua terpisah
    4: 3,  # salah satu meninggal
    5: 4,  # lainnya / tidak diketahui
}
ORANGTUA_LABELS = {1: "utuh", 2: "terpisah", 3: "salah_satu_meninggal", 4: "lainnya"}
# 3 versi coding untuk sensitivity analysis (PLAN §9)
ORANGTUA_RECODE_VARIANTS = {
    "v1_default": ORANGTUA_RECODE,
    "v2_cat5_meninggal": {1: 1, 2: 4, 3: 2, 4: 3, 5: 3},
    "v3_drop_uncertain": {1: 1, 2: None, 3: 2, 4: 3, 5: None},
}

# Padatkan TINGGAL (11 kategori) -> 5 grup
TINGGAL_RECODE = {
    1: 1,   # ayah+ibu
    2: 2, 3: 2, 4: 2,          # satu orang tua / orang tua tiri
    5: 3,                      # orang tua bekerja luar kota/negeri
    6: 4, 7: 4, 8: 4, 9: 4,    # diasuh keluarga/lembaga
    10: 5, 11: 5,              # lainnya / tinggal sendiri
}
TINGGAL_LABELS = {
    1: "ayah_ibu", 2: "satu_orang_tua", 3: "ortu_bekerja_luar",
    4: "diasuh_keluarga_lembaga", 5: "lainnya",
}

# ---------- Target ----------
# y_kontinu = rata-rata 20 item kecemasan valid (1-4)
# y_biner   = 1 jika y_kontinu >= ambang; ambang ditetapkan di 02_features (codebook)
ANXIETY_HIGH_THRESHOLD = 2.5  # >= "sering" -> cemas tinggi (definisi di codebook)

# ---------- 18 fitur input (urutan kolom features.parquet) ----------
NUMERIC_FEATURES = [
    "f_akademik", "f_keluarga", "f_sosial", "f_ekonomi", "f_digital", "f_masadepan",
    "f_perilaku_sehat",
    "f_duk_keluarga", "f_duk_teman", "f_duk_orang_dekat",
    "f_coping_adaptif", "f_coping_ekspresi", "f_coping_negatif",
    "d_umur",
]
CATEGORICAL_FEATURES = ["d_jenis_kelamin", "d_jurusan", "d_tinggal_grup", "d_status_ortu"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_CONT = "y_kontinu"
TARGET_BIN = "y_biner"

# ---------- Skema evaluasi (PLAN §5) ----------
OUTER_FOLDS = 5
OUTER_REPEATS = 10
INNER_FOLDS = 5

# ---------- Tuning (PLAN §6) ----------
N_TRIALS = 5000
XGB_SEARCH_SPACE = {
    "n_estimators": ("int", 100, 2000),
    "max_depth": ("int", 2, 10),
    "learning_rate": ("float", 0.01, 0.3, "log"),
    "min_child_weight": ("float", 1, 20),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0),
    "gamma": ("float", 0, 5),
    "reg_alpha": ("float", 1e-4, 10, "log"),
    "reg_lambda": ("float", 0.1, 100, "log"),
}
ONE_SE_RULE = True  # pilih model lebih sederhana bila skor dalam 1 SE dari yang terbaik

# ---------- Metrik ----------
PRIMARY_METRIC = "rmse"   # fungsi latih/tuning (kontinu)
REPORT_METRICS = ["rmse", "mae", "r2"]           # kontinu
REPORT_METRICS_BIN = ["auc", "balanced_accuracy", "brier"]  # biner
