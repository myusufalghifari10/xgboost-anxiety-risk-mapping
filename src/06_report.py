"""Tahap 8 — Laporan ringkas, tabel, dan peta risiko.

Menggabungkan artefak Model #1 menjadi keluaran untuk Bu Rita, paper Q3, dan
policy brief. Hanya BACA artefak yang sudah ada (dari 04_evaluate.py dan
05_explain.py); skrip ini tidak melatih model.

Keluaran (nama persis sesuai CONTRACT.md):
  outputs/tables/report_ringkas.md    laporan ringkas Bahasa Indonesia
  outputs/tables/tabel_performa.csv   ringkasan metrik dari metrics.json
  outputs/tables/tabel_faktor.csv     peringkat faktor risiko (SHAP + stabilitas)
  outputs/tables/risk_map_jurusan.csv peta risiko-zona per jurusan x jenis kelamin
  outputs/tables/risk_map_jeniskelamin.csv (sama, dikelompokkan per jenis kelamin)

Definisi zona risiko (dokumentasi, PLAN 8 + keputusan #8):
  Dipakai pada prediksi model (skor kecemasan 1-4). Ambang diambil dari config.py
  (ZONE_HIJAU_MAX, ZONE_KUNING_MAX) agar tidak menyimpang dari definisi di config:
    hijau  : skor < ZONE_HIJAU_MAX
    kuning : ZONE_HIJAU_MAX <= skor < ZONE_KUNING_MAX
    merah  : skor >= ZONE_KUNING_MAX (= ANXIETY_HIGH_THRESHOLD)
  Zona bersifat agregat per kelompok; TIDAK dipakai sebagai label individual.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import config as C

ZONE_HIJAU_MAX = C.ZONE_HIJAU_MAX
ZONE_KUNING_MAX = C.ZONE_KUNING_MAX


# ---------------------------------------------------------------------------
# Pembaca artefak
# ---------------------------------------------------------------------------
def require(path: Path, dibuat_oleh: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} belum ada. Jalankan {dibuat_oleh} lebih dulu."
        )
    return path


def load_metrics() -> Dict:
    return json.loads(require(C.METRICS_JSON, "04_evaluate.py").read_text(encoding="utf-8"))


def _assert_metrics_complete(metrics: Dict) -> None:
    """Fail-fast kontrak metrics.json (fix B3/D3a) — laporan tidak boleh berisi 'nan' senyap."""
    cont = metrics.get("continuous") or {}
    binary = metrics.get("binary") or {}
    kurang = [k for k in C.REPORT_METRICS if k not in cont]
    kurang += [k for k in C.REPORT_METRICS_BIN if k not in binary]
    # Fix D3a: n_evals/n_prediksi dipakai tulis_laporan tapi sebelumnya tak divalidasi.
    kurang += [k for k in ("n_evals", "n_prediksi") if k not in cont]
    if kurang:
        raise KeyError(f"metrics.json kurang kunci metrik: {kurang}")
    ci = cont.get("ci95")
    if not isinstance(ci, dict) or not all(k in ci for k in ("rmse", "mae", "r2")):
        raise KeyError(f"metrics.json ci95 tidak lengkap: {ci!r}")
    # Fix D3a: nilai metrik harus finite — "nan" tidak boleh sampai ter-render.
    for key in C.REPORT_METRICS:
        if not np.isfinite(float(cont[key])):
            raise ValueError(f"metrics.json continuous.{key} = {cont[key]!r} bukan angka finite")
    for key in C.REPORT_METRICS_BIN:
        if not np.isfinite(float(binary[key])):
            raise ValueError(f"metrics.json binary.{key} = {binary[key]!r} bukan angka finite")


def _assert_artifacts_same_generation() -> None:
    """Guard D2b: tolak campur-generasi artefak (run terputus / tahap dilewati).

    Urutan normal: 03 (oof/trials/per_fold/best_params ditulis bersamaan pada satu
    titik komit) -> 04 (metrics.json) -> 05 (shap/stability/drop_hb) -> 06 (skrip ini).
    """
    # (i) artefak 03 harus ditulis bersamaan: rentang mtime maksimal 600 detik.
    path_03 = {
        "oof_predictions.csv": require(C.OOF_CSV, "03_train_tune.py"),
        "optuna_trials.csv": require(C.TRIALS_CSV, "03_train_tune.py"),
        "per_fold_tuning.csv": require(C.PER_FOLD_TUNING_CSV, "03_train_tune.py"),
        "best_params.json": require(C.BEST_PARAMS_JSON, "03_train_tune.py"),
        # Fix review F1: tiga artefak model ikut dijaga (dulu tersembunyi dari guard
        # sehingga model baru bisa berpasangan dengan metrik/SHAP generasi lama).
        "model_final.ubj": require(C.MODEL_PATH, "03_train_tune.py"),
        "model_final_clf.ubj": require(C.MODEL_CLF_PATH, "03_train_tune.py"),
        "preprocessor.joblib": require(C.PREPROCESSOR_PATH, "03_train_tune.py"),
    }
    mtime = {nama: p.stat().st_mtime for nama, p in path_03.items()}
    rentang = max(mtime.values()) - min(mtime.values())
    if rentang > 600:
        lama = min(mtime, key=mtime.get)
        baru = max(mtime, key=mtime.get)
        raise RuntimeError(
            "artefak campur generasi — jalankan ulang 03-05 secara lengkap "
            f"(rentang mtime artefak 03 = {rentang:.0f} dtk > 600 dtk: "
            f"{lama} vs {baru})."
        )
    # (ibis) Fix review F1: features.parquet tidak boleh lebih baru dari best_params.json —
    # mencegah 02 dijalankan ulang (mis. ganti ambang) lalu 06 mencampur label baru
    # dengan model/metrik lama. Arah ini aman: pada run penuh 02 selalu lebih dulu.
    if require(C.FEATURES, "02_features.py").stat().st_mtime > mtime["best_params.json"] + 2.0:
        raise RuntimeError(
            "artefak campur generasi — features.parquet lebih baru dari best_params.json "
            "(02_features.py dijalankan ulang setelah training? jalankan ulang 03-05)."
        )
    # (ii) output 04/05 tidak boleh lebih lama dari best_params.json (toleransi 2 dtk).
    bawah = mtime["best_params.json"] - 2.0
    pasca = {
        "metrics.json": (C.METRICS_JSON, "04_evaluate.py"),
        "shap_global_ranking.csv": (C.SHAP_GLOBAL_RANKING, "05_explain.py"),
        "stability_ranking.csv": (C.STABILITY_RANKING, "05_explain.py"),
        "drop_hb_test.json": (C.DROP_HB_JSON, "05_explain.py"),
    }
    for nama, (path, dibuat) in pasca.items():
        if require(path, dibuat).stat().st_mtime < bawah:
            raise RuntimeError(
                "artefak campur generasi — jalankan ulang 03-05 secara lengkap "
                f"({nama} lebih lama dari best_params.json; tahap 04/05 dilewati "
                "atau 03 dijalankan ulang tanpa 04/05)."
            )


def load_shap_ranking() -> pd.DataFrame:
    """Peringkat faktor dari outputs/shap_global_ranking.csv (ditulis 05_explain.py).

    Fix blocker rev-explain B1: versi lama membaca stability_ranking.csv yang tidak
    punya kolom mean_abs_shap sehingga 06 crash tepat sebelum menulis laporan.
    """
    return pd.read_csv(require(C.SHAP_GLOBAL_RANKING, "05_explain.py"))


def load_drop_hb() -> Dict:
    drop = json.loads(
        require(C.DROP_HB_JSON, "05_explain.py").read_text(encoding="utf-8")
    )
    # Fix D3a: kunci yang dipakai tulis_laporan divalidasi SEBELUM tulis pertama.
    kurang = [
        k for k in ("selisih_rmse_mean", "selisih_rmse_std", "proporsi_fold_hb_membantu", "n_folds")
        if k not in drop
    ]
    if kurang:
        raise KeyError(
            f"drop_hb_test.json kurang kunci: {kurang} — jalankan ulang 05_explain.py."
        )
    return drop


def load_stability() -> pd.DataFrame:
    return pd.read_csv(require(C.STABILITY_RANKING, "05_explain.py"))


def load_model_meta() -> Dict:
    """Baca best_params.json (kunci kontrak) untuk jejak tuning pada laporan."""
    path = require(C.BEST_PARAMS_JSON, "03_train_tune.py")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not meta.get("best_params"):
        raise KeyError("best_params.json tidak punya kunci 'best_params'.")
    return meta


def load_features() -> pd.DataFrame:
    return pd.read_parquet(require(C.FEATURES, "02_features.py"))


# ---------------------------------------------------------------------------
# Zona risiko
# ---------------------------------------------------------------------------
def zona_skor(skor: float) -> str:
    """Petakan skor prediksi (1-4) ke zona merah/kuning/hijau (lihat docstring modul).

    Fix catatan rev-explain N8: skor NaN tidak boleh jatuh ke zona "hijau" (zona
    terbaik) karena kedua perbandingan terhadap NaN bernilai False.
    """
    if skor is None or not np.isfinite(skor):
        raise ValueError("Skor prediksi NaN/tidak valid; sumbernya oof_predictions.csv tidak lengkap.")
    if skor >= ZONE_KUNING_MAX:
        return "merah"
    if skor >= ZONE_HIJAU_MAX:
        return "kuning"
    return "hijau"


def _oof_skor_per_siswa() -> pd.Series:
    """Skor kecemasan out-of-fold per siswa = rata-rata prediksi OOF miliknya.

    Fix catatan rev-explain N6: versi lama men-score 306 baris dengan model_final.ubj
    yang justru dilatih pada 306 baris itu (in-sample) sehingga proporsi zona merah
    pada policy brief Systematicly terinflasi. Sekarang memakai prediksi out-of-fold
    dari oof_predictions.csv (rata-rata antar ulangan).
    """
    oof = pd.read_csv(require(C.OOF_CSV, "03_train_tune.py"))
    for col in ("row_id", "pred_kontinu"):
        if col not in oof.columns:
            raise KeyError(f"oof_predictions.csv tidak punya kolom '{col}'.")
    # Fix B5: NaN sebagian (sebagian ulangan) dulu disembunyikan .mean(skipna=True);
    # sekarang tolak apa pun yang tidak lengkap SEBELUM agregasi per siswa.
    if oof["pred_kontinu"].isna().any():
        raise ValueError(
            "oof_predictions.csv punya pred_kontinu NaN (termasuk NaN sebagian) — "
            "periksa keluaran 03_train_tune.py."
        )
    per_siswa = oof.groupby("row_id")["pred_kontinu"].mean()
    return per_siswa.rename("skor_prediksi")


# ---------------------------------------------------------------------------
# Tabel
# ---------------------------------------------------------------------------
def _ci_bounds(ci) -> tuple:
    """Terima dua bentuk ci95 dari 04_evaluate: [bawah, atas] atau {bawah, atas}.

    Fix B3: bentuk tak dikenal -> raise (dulu return (None, None) dan CI hilang senyap).
    """
    if ci is None:
        return None, None
    if isinstance(ci, dict):
        return ci.get("bawah", ci.get("lower")), ci.get("atas", ci.get("upper"))
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return ci[0], ci[1]
    raise ValueError(f"bentuk ci95 tidak dikenal: {ci!r}")


def tabel_performa(metrics: Dict, n_trial: int, n_outer: int) -> pd.DataFrame:
    """Ringkasan metrik resmi (dari test) + jejak tuning, satu baris per metrik."""
    cont = metrics.get("continuous", {})
    binary = metrics.get("binary", {})
    rows: List[Dict] = []
    for key in C.REPORT_METRICS:
        if key in cont:
            row = {"target": "kontinu", "metrik": key, "nilai": cont[key]}
            ci = (cont.get("ci95") or {}).get(key)
            bawah, atas = _ci_bounds(ci)
            if bawah is not None:
                row["ci95_bawah"], row["ci95_atas"] = bawah, atas
            rows.append(row)
    for key in C.REPORT_METRICS_BIN:
        if key in binary:
            rows.append({"target": "biner", "metrik": key, "nilai": binary[key]})
    # Fix catatan rev-explain N15: tulis "trial per ronde x jumlah ronde", bukan
    # total baris trials CSV (yang dulu ditulis seolah-olah 250.000 setelan unik).
    # Fix B9/D4c: baris hitungan ditulis sebagai str(int(...)) (bukan "500.0")
    # agar tidak salah baca sebagai nilai metrik.
    rows.append({"target": "tuning", "metrik": "n_trials_per_ronde", "nilai": str(int(n_trial))})
    rows.append({"target": "tuning", "metrik": "n_ronde_outer", "nilai": str(int(n_outer))})
    return pd.DataFrame(rows)


def tabel_faktor(ranking: pd.DataFrame, kestabilan: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan peringkat SHAP global dengan kestabilannya antar fold.

    Fix blocker rev-explain B2: versi lama menghitung 'arah' dari shap_local.csv
    yang hanya berisi top-5 per siswa, sehingga arah tiap fitur bias seleksi dan
    fitur yang tak pernah masuk top-5 diberi arah karangan. Sekarang arah diambil
    dari SHAP global semua siswa (shap_global_ranking.csv) yang juga menjadi sumber
    shap_global.png. Fix N11: urutan mengikuti mean_abs_shap, bukan proporsi_top5.
    Kolom kestabilan tidak di-hardcode agar tetap sinkron bila 05 mengubah
    jumlah fitur teratas yang dihitung stabil.
    """
    out = ranking.merge(kestabilan, on="fitur", how="left")
    return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def risk_map(df: pd.DataFrame, skor: pd.Series, by: List[str]) -> pd.DataFrame:
    """Peta zona per kelompok (mis. ['d_jurusan','d_jenis_kelamin']).

    Ambang tingkat risiko kelompok diambil dari config (GROUP_RISK_HIGH/MODERATE).
    Kolom n_hilang adalah invarian: harus SELALU 0 karena zona_skor fail-fast pada
    skor NaN, sehingga siswa tanpa prediksi tidak pernah sampai ke groupby.

    Fix A6: kelompok dengan n < GROUP_RISK_MIN_N diberi label "n kecil" karena sel
    n=1-4 bisa berlabel TINGGI dengan proporsi 100% dan mendominasi policy brief.
    """
    tmp = df[["row_id"] + by].copy()
    tmp = tmp.merge(skor.rename_axis("row_id").reset_index(), on="row_id", how="left")
    tmp["zona"] = tmp["skor_prediksi"].apply(zona_skor)
    grouped = (
        tmp.groupby(by)
        .agg(
            n_siswa=("row_id", "count"),
            skor_rata2=("skor_prediksi", "mean"),
            n_merah=("zona", lambda z: int((z == "merah").sum())),
            n_kuning=("zona", lambda z: int((z == "kuning").sum())),
            n_hijau=("zona", lambda z: int((z == "hijau").sum())),
            n_hilang=("skor_prediksi", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    # Fix B4: groupby dropna=True bisa membuang baris senyap (kolom kelompok NaN).
    assert grouped["n_siswa"].sum() == len(df), \
        "ada baris terbuang saat groupby risk_map (kolom kelompok mengandung NaN?)"
    grouped["proporsi_merah"] = grouped["n_merah"] / grouped["n_siswa"]
    assert grouped["proporsi_merah"].notna().all(), "proporsi_merah mengandung NaN"
    grouped["tingkat_risiko_kelompok"] = np.where(
        grouped["proporsi_merah"] >= C.GROUP_RISK_HIGH,
        "TINGGI",
        np.where(grouped["proporsi_merah"] >= C.GROUP_RISK_MODERATE, "SEDANG", "RENDAH"),
    )
    # Fix A6: tandai sel kecil — jangan buang barisnya (data apa adanya).
    kecil_mask = grouped["n_siswa"] < C.GROUP_RISK_MIN_N
    grouped.loc[kecil_mask, "tingkat_risiko_kelompok"] = (
        grouped.loc[kecil_mask, "tingkat_risiko_kelompok"] + " (n kecil — interpretasi hati-hati)"
    )
    return grouped.sort_values("proporsi_merah", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Laporan ringkas
# ---------------------------------------------------------------------------
def tulis_laporan(
    performa: pd.DataFrame, faktor: pd.DataFrame, map_j: pd.DataFrame, drop: Dict,
    n_trial: int, n_outer: int, n_trial_final: int, metrics: Dict, df: pd.DataFrame,
) -> Path:
    m = {}
    for _, r in performa.iterrows():
        m[(r["target"], r["metrik"])] = r["nilai"]

    # Fix B6: konteks statistik dari metrics.json/features (bukan hardcode selain definisi).
    cont = metrics["continuous"]
    n_evals = int(cont["n_evals"])
    n_prediksi = int(cont["n_prediksi"])
    n_pos = int(df[C.TARGET_BIN].sum())
    prevalence = 100 * n_pos / len(df)
    bawah_rmse, atas_rmse = _ci_bounds(cont["ci95"]["rmse"])

    # Fix B9/D4b: notasi desimal seragam gaya teknis (titik) -> konsisten dengan RMSE/CI.
    hijau_s = f"{ZONE_HIJAU_MAX:.1f}"
    kuning_s = f"{ZONE_KUNING_MAX:.1f}"

    # Fix A8: total trial eksplisit dan pemisahan klaim test untuk nested vs model final.
    nested_total = f"{n_trial * n_outer:,}".replace(",", ".")
    final_total = f"{n_trial_final:,}".replace(",", ".")
    # Fix D4b: pemisah ribuan gaya Indonesia (titik) untuk semua angka teknis.
    n_trial_indo = f"{n_trial:,}".replace(",", ".")

    # Fix B7: tampilkan stabilitas (proporsi_top5) + caveat in-sample pada top-5 SHAP.
    # Fix review F5: beri penanda stabilitas rendah dan fitur yang tidak dipakai model;
    # arah dari mean_shap_signed bisa bertentangan dengan bentuk PDP -> arahkan ke grafik.
    top5 = faktor.head(5)[["fitur", "arah", "mean_abs_shap", "proporsi_top5"]]

    def _baris_top(i: int, r) -> str:
        if float(r.mean_abs_shap) == 0.0:
            arah_s, stab_s = "tidak dipakai model", "—"
        else:
            arah_s = r.arah.replace("_", " ")
            stab_s = f"{r.proporsi_top5:.0%}" + (" ⚠ kurang stabil" if r.proporsi_top5 < 0.6 else "")
        return f"| {i+1} | {r.fitur} | {arah_s} | {r.mean_abs_shap:.4f} | {stab_s} |"

    baris_top = "\n".join(_baris_top(i, r) for i, r in enumerate(top5.itertuples()))
    fitur_nol = faktor.loc[faktor["mean_abs_shap"] == 0, "fitur"].tolist()
    baris_nol = (
        f"- Fitur yang TIDAK dipakai model sama sekali (|SHAP| = 0): {', '.join(fitur_nol)}"
        if fitur_nol else ""
    )
    # Fix F5: label 'arah' = kontribusi rata-rata pada sampel ini, BUKAN arah efek/kausal.
    catatan_arah = (
        "_Arah = kontribusi rata-rata SHAP pada sampel ini (menaikkan/menurunkan prediksi "
        "skor kecemasan), bukan uji sebab-akibat, dan nilainya mendekati nol secara "
        "matematis — untuk arah hubungan gunakan grafik PDP (outputs/figures/pdp_*.png), "
        "bukan kolom ini (bisa bertentangan). Peringkat SHAP dihitung pada model final "
        "(in-sample); angka performa di atas berasal dari prediksi out-of-fold._"
    )
    if baris_nol:
        catatan_arah = catatan_arah + "\n" + baris_nol + " — nol berarti fitur tidak pernah " \
            "dipakai model untuk membelah (49 simpul split), BUKAN terbukti tidak berpengaruh."
    # Fix A7: klaim drop-HB dibandingkan dengan 1 SD antar fold, bukan hanya tanda rata-rata.
    selisih = float(drop["selisih_rmse_mean"])
    sd_selisih = float(drop["selisih_rmse_std"])
    proporsi_hb = float(drop["proporsi_fold_hb_membantu"])
    if selisih > sd_selisih:
        kalimat_hb = (
            "Arah positif dan lebih besar dari 1 SD antar fold: fitur perilaku sehat "
            "membantu prediksi pada data ini."
        )
    elif selisih < -sd_selisih:
        kalimat_hb = (
            "Selisih negatif dan melebihi 1 SD antar fold: menghapus fitur justru "
            "memperbaiki prediksi (tidak membantu pada data ini)."
        )
    else:
        kalimat_hb = (
            "Selisih rata-rata lebih kecil daripada sebaran antar fold (< 1 SD): "
            "tidak ada kesepakatan arah antar fold — tidak ada kontribusi pasti "
            "pada data ini (deskriptif, bukan uji signifikansi)."
        )
    # Fix A6: label TINGGI kini berakhiran " (n kecil ...)" pada sel kecil -> pakai startswith.
    map_tinggi = map_j[map_j["tingkat_risiko_kelompok"].str.startswith("TINGGI")]
    # Fix review F4: selalu tampilkan top-3 sel + ambangnya (bukan hanya yang TINGGI),
    # supaya "tidak ada yang tinggi" tetap bisa diinterpretasi pembaca.
    ambang_tinggi = f"{C.GROUP_RISK_HIGH:.0%}"
    baris_tinggi = (
        "\n".join(
            f"- {r.d_jurusan} / {r.d_jenis_kelamin}: {r.proporsi_merah:.0%} siswa zona merah "
            f"(n={r.n_siswa}) — {r.tingkat_risiko_kelompok}"
            for r in map_tinggi.head(5).itertuples()
        )
        or f"- Tidak ada sel yang mencapai ambang TINGGI (proporsi zona merah >= {ambang_tinggi})."
    )
    baris_top3 = "\n".join(
        f"- {r.d_jurusan} / {r.d_jenis_kelamin}: {r.proporsi_merah:.0%} siswa zona merah (n={r.n_siswa})"
        for r in map_j.head(3).itertuples()
    )
    baris_map = (
        f"{baris_tinggi}\n\nTiga sel dengan proporsi zona merah tertinggi "
        f"(ambang TINGGI = proporsi zona merah >= {ambang_tinggi}):\n{baris_top3}"
    )
    # Fix review F3: disclosure ekor-atas — skor = rata-rata 10 prediksi OOF (ensemble).
    n_merah_total = int(map_j["n_merah"].sum())
    baris_zona_konteks = (
        f"- Konteks peta: {n_merah_total} dari {len(df)} siswa ({100 * n_merah_total / len(df):.1f}%) "
        f"masuk zona merah, sementara prevalensi aktual kategori cemas tinggi {n_pos} siswa "
        f"({prevalence:.1f}%). Skor per siswa = rata-rata 10 prediksi OOF (ensemble, bukan "
        "prediksi satu model) sehingga ekor atas agak ter-mampat — peta untuk MEMBANDINGKAN "
        "kelompok, bukan menghitung jumlah kasus."
    )

    isi = f"""# Laporan Ringkas — Model 1: Peta Risiko Kecemasan (XGBoost)

## Ringkasan performa (prediksi out-of-fold, repeated nested CV 5 fold x 10 ulangan)
- Skor kecemasan (kontinu): RMSE = **{m[('kontinu','rmse')]:.3f}** (CI95 {bawah_rmse:.3f}–{atas_rmse:.3f}), MAE = {m[('kontinu','mae')]:.3f}, R2 = {m[('kontinu','r2')]:.3f}
- Catatan desain (fix review F2): setiap siswa menjadi data test 10x (sekali per ulangan) dan data latih pada ulangan lain — angka adalah estimasi resampling internal pada {n_evals} siswa yang sama, BUKAN test set eksternal terpisah.
- Kategori cemas tinggi (biner): AUC = **{m[('biner','auc')]:.3f}**
- Konteks: N = {n_evals} siswa ({n_prediksi} prediksi out-of-fold); kategori cemas tinggi {n_pos} siswa ({prevalence:.1f}%).
- Hyperparameter Optuna (nested-CV; test fold TIDAK PERNAH dilihat proses ini): **{n_trial_indo} trial per ronde x {n_outer} ronde = {nested_total} trial**.
- Tuning model final terpisah: **{final_total} trial** pada seluruh 306 siswa (tidak ada angka test yang dikutip dari model final).

## 5 faktor paling berpengaruh (SHAP)
| Peringkat | Faktor | Arah | Rata-rata \\|SHAP\\| (poin skor) | Stabil (masuk top-5 antar fold) |
|---|---|---|---|---|
{baris_top}

{catatan_arah}

## Uji klaim perilaku sehat (drop-HB)
- Menghapus fitur `f_perilaku_sehat` mengubah RMSE test sebesar **{drop['selisih_rmse_mean']:+.4f}** (rata-rata {drop['n_folds']} fold; SD antar fold = {sd_selisih:.4f}; {proporsi_hb:.0%} fold menunjukkan HB membantu).
- {kalimat_hb}
- Interpretasi jujur: model final tidak pernah membelah pada `f_perilaku_sehat` (SHAP identik nol, PDP datar), jadi uji ini konsisten dengan "fitur tidak dipakai"; bukti utamanya ada di SHAP/PDP, sedangkan selisih ≈ 0 juga bisa muncul dari lotre subsampling kolom (`colsample_bytree`) — jangan dijadikan satu-satunya dasar.

## Interaksi antar faktor — TIDAK TERUKUR
- Sembilan pasang fokus (perilaku sehat x dukungan/tekanan) bernilai 0 karena fitur tidak pernah dipakai model untuk membelah, BUKAN karena interaksi terbukti tidak ada. Pertanyaan interaksi pada PLAN 8.4 tidak bisa dijawab oleh model ini; jawabannya butuh model yang memang memakai fitur tersebut.

## Kelompok dengan risiko tinggi (zona merah proporsi tinggi)
{baris_map}

{baris_zona_konteks}

## Catatan penggunaan
- Zona merah/kuning/hijau bersifat **agregat per kelompok** untuk policy brief dan triase guru BK; bukan label diagnosis individual.
- Ambang zona: hijau < {hijau_s}; kuning >= {hijau_s} dan < {kuning_s}; merah >= {kuning_s} (skala 1-4, mengikuti ambang kategori tinggi di config).
- Skor risiko per siswa memakai prediksi out-of-fold (bukan prediksi in-sample), sehingga proporsi zona merah tidak terinflasi oleh data latih.
- Brier score memakai konvensi scikit-learn, yaitu rata-rata (p - y)^2 (setengah dari definisi Brier klasik 2 kelas) — jangan dibandingkan langsung dengan literatur yang memakai definisi lain tanpa konversi.
- Probabilitas biner berasal dari model dengan `scale_pos_weight` (target timpang): probabilitas BELUM terkalibrasi terhadap prevalensi asli sehingga tidak boleh dibaca sebagai risiko absolut per siswa.
- `balanced_accuracy` dihitung pada ambang tetap 0.5 (bukan ambang optimal), sedangkan AUC tetap sahih karena hanya bergantung urutan peringkat.
- Menggunakan bahasa asosiatif: data cross-sectional, korelasi bukan sebab-akibat.
"""
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = C.REPORT_RINGKAS
    path.write_text(isi, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Orkestrasi
# ---------------------------------------------------------------------------
def run_all() -> None:
    # assert level-1 sesuai CONTRACT aturan 5 (temuan round-2 R2-7: 06 tanpa assert)
    assert C.ZONE_HIJAU_MAX < C.ZONE_KUNING_MAX, "ambang zona harus terurut"
    assert zona_skor(C.ZONE_KUNING_MAX) == "merah", "batas bawah zona merah"
    assert zona_skor(C.ZONE_HIJAU_MAX) == "kuning", "batas bawah zona kuning"
    assert zona_skor(C.ZONE_HIJAU_MAX - 1e-9) == "hijau", "batas bawah zona hijau"
    # Fix D2b: sebelum membaca/menulis apa pun, pastikan artefak satu generasi.
    _assert_artifacts_same_generation()

    metrics = load_metrics()
    _assert_metrics_complete(metrics)
    meta = load_model_meta()
    outer = meta.get("outer") or {}
    # Fix R3-4: fail-fast bila meta tuning hilang/tidak valid (sebelumnya fail-open
    # -> laporan menulis "0 trial x 0 ronde" tanpa gagal).
    if "n_trials_per_outer_fold" not in meta:
        raise KeyError("best_params.json tidak punya 'n_trials_per_outer_fold' — jalankan 03_train_tune.py versi lengkap.")
    if not outer or "folds" not in outer or "repeats" not in outer:
        raise KeyError("best_params.json tidak punya 'outer.folds/repeats' — jalankan 03_train_tune.py versi lengkap.")
    n_trial = int(meta["n_trials_per_outer_fold"])
    # Fix B3: jangan fail-open — kunci hilang harus gagal, bukan diam-diam 0.
    if "n_trials_final" not in meta:
        raise KeyError(
            "best_params.json tidak punya 'n_trials_final' — jalankan 03_train_tune.py versi lengkap."
        )
    n_trial_final = int(meta["n_trials_final"])
    n_outer = int(outer["folds"]) * int(outer["repeats"])
    if n_trial <= 0 or n_outer <= 0:
        raise ValueError(f"meta tuning tidak valid: n_trial={n_trial}, n_outer={n_outer}")
    ranking = load_shap_ranking()
    kestabilan = load_stability()
    drop = load_drop_hb()
    df = load_features()
    skor = _oof_skor_per_siswa()

    assert {"fitur", "mean_abs_shap", "arah"} <= set(ranking.columns), \
        f"shap_global_ranking.csv kolom tak lengkap: {list(ranking.columns)}"
    assert "proporsi_top5" in kestabilan.columns, \
        f"stability_ranking.csv tidak punya proporsi_top5: {list(kestabilan.columns)}"
    # Fix B4: fitur yang tidak ada di kedua artefak akan jadi NaN senyap saat merge.
    assert set(ranking["fitur"]) == set(kestabilan["fitur"]), (
        "fitur shap_global_ranking.csv != stability_ranking.csv: "
        f"{sorted(set(ranking['fitur']) ^ set(kestabilan['fitur']))}"
    )
    assert len(skor) == len(df), "prediksi OOF harus menutup semua siswa"

    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tabel_performa(metrics, n_trial, n_outer).to_csv(C.TABEL_PERFORMA, index=False)

    faktor = tabel_faktor(ranking, kestabilan)
    # Fix B10: cegah AttributeError dari r.arah.replace bila kolom arah diedit manual.
    assert faktor["arah"].notna().all(), "kolom arah mengandung NaN (CSV hasil suntingan?)"
    faktor.to_csv(C.TABEL_FAKTOR, index=False)

    map_j = risk_map(df, skor, ["d_jurusan", "d_jenis_kelamin"])
    map_j.to_csv(C.RISK_MAP_JURUSAN, index=False)

    map_k = risk_map(df, skor, ["d_jenis_kelamin"])
    map_k.to_csv(C.RISK_MAP_JENISKELAMIN, index=False)

    tulis_laporan(
        tabel_performa(metrics, n_trial, n_outer), faktor, map_j, drop,
        n_trial, n_outer, n_trial_final, metrics, df,
    )


if __name__ == "__main__":  # pragma: no cover - dijalankan setelah pipeline lengkap
    run_all()
    print(f"[06_report] selesai; laporan di {C.TABLE_DIR / 'report_ringkas.md'}")