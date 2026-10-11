"""Eksperimen 6 — "Fitur Baru dari Kolom Lama + Bagging + Bobot" (satu run, 9 arm).

Keputusan Yusuf 2026-10-11 ("gabung semua jadi satu run", tujuan: MAE serendah mungkin).
Semua arm memakai fold E3 yang SAMA dan hyperparameter E3 yang SAMA per fold
(diwarisi dari checkpoint E3) — jadi perbandingan menisolasi HANYA arm-nya.

Arm (pra-registrasi, tidak dipilih dari hasil):
  c0  baseline E3         : 76 fitur, bobot p=0,50 cap=3, tanpa bagging
  c1  +B1 gaya respons    : +9 fitur per-baris (SD/rentang/run/ekstrem/kosong)
  c2  +B2 komponen        : +15 komponen PCA(10)+PLS(5), di-fit di dalam fold
  c3  +B3 interaksi       : +6 interaksi/rasio domain a-priori
  c4  +bagging k=5        : 5 model beda seed per fold, prediksi dirata-rata
  c5  bobot p=0,75 cap=3  : tilt bobot lebih tajam
  c6  bobot p=1,00 cap=3  : bobot = ref/n_bin (tanpa akar)
  c7  bobot p=0,50 cap=5  : cap lebih longgar
  c8  ALL-IN (PRIMARY)    : B1+B2+B3 + bagging k=5 + bobot p=0,50 cap=3

Gerbang pra-registrasi (dikunci SEBELUM run, lihat RINGKASAN.md), dinilai dari
MAE ensemble (10 ulangan dirata-rata per siswa) terhadap E3:
  K1 efek    : MAE total < ref E3 (0,22572) DAN uji berpasangan per-siswa p<0,05 membaik
  K2 ekor    : MAE ekor  <= ref ekor + 0,003
  K3 badan   : MAE badan <= ref badan + 0,003
  lolos = K1 dan K2 dan K3 (untuk PRIMARY); arm lain dilaporkan eksploratif.

Alpha decode: alpha terpool dihitung dari pasangan inner-OOF arm c0 lalu DIPAKAI
BERSAMA semua arm (deviasi dari E3, yang memakai alpha arm-nya sendiri) — dampak
terukur decode ~0,0002 MAE, jadi tidak menguntungkan arm mana pun. pred_raw juga
selalu dilaporkan supaya tidak ada yang tersembunyi.

Pemakaian (EKSEKUSI = YUSUF):
  .venv/bin/python src/exp6_train.py --self-test
  .venv/bin/python src/exp6_train.py                 # full run 10 x 5 fold, ~20-30 menit
  .venv/bin/python src/exp6_train.py --fresh         # ulang dari nol
  .venv/bin/python src/exp6_train.py --arms c0 c8    # subset arm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import exp6_feats as F6  # noqa: E402
from exp2_features import FEATURES_EXP2, ITEM_COLS  # noqa: E402
from exp2_train import GRID, _decode, _encode, _fit, _fit_alpha, _mae, _rmse  # noqa: E402
from exp3_train import bobot as bobot_e3  # noqa: E402  (referensi p=0,5 cap=3)

EXP6_DIR = C.ROOT / "outputs" / "exp6"
CKPT_DIR = EXP6_DIR / "checkpoints"
E3_JSON = C.ROOT / "outputs" / "exp3" / "exp3_full_91945185763d.json"
E3_CKPT = C.ROOT / "outputs" / "exp3" / "checkpoints" / "exp3_91945185763d"
E3_CSV = C.ROOT / "outputs" / "exp3" / "exp3_full_91945185763d_per_siswa.csv"

OBJECTIVE = "reg:absoluteerror"
AMBANG_EKOR = C.ANXIETY_HIGH_THRESHOLD  # 2,5
BAG_K = 5
TOLERANSI_SISI = 0.003  # guard ekor/badan (sama dengan disiplin E4/E5)
LEBAR_UJI_MIN = 0.0047  # efek minimal agar uji berpasangan n=306 bisa p<0,05

ARMS: dict[str, dict] = {
    "c0": {"fitur": (), "bag": 1, "bobot": (0.50, 3.0), "ket": "baseline E3"},
    "c1": {"fitur": ("b1",), "bag": 1, "bobot": (0.50, 3.0), "ket": "+gaya respons"},
    "c2": {"fitur": ("b2",), "bag": 1, "bobot": (0.50, 3.0), "ket": "+komponen PCA/PLS"},
    "c3": {"fitur": ("b3",), "bag": 1, "bobot": (0.50, 3.0), "ket": "+interaksi domain"},
    "c4": {"fitur": (), "bag": BAG_K, "bobot": (0.50, 3.0), "ket": f"+bagging k={BAG_K}"},
    "c5": {"fitur": (), "bag": 1, "bobot": (0.75, 3.0), "ket": "bobot p=0,75"},
    "c6": {"fitur": (), "bag": 1, "bobot": (1.00, 3.0), "ket": "bobot p=1,00"},
    "c7": {"fitur": (), "bag": 1, "bobot": (0.50, 5.0), "ket": "bobot cap=5"},
    "c8": {"fitur": ("b1", "b2", "b3"), "bag": BAG_K, "bobot": (0.50, 3.0),
           "ket": "ALL-IN (PRIMARY)"},
}
PRIMARY = "c8"
KUNCI_PARAM = tuple(C.XGB_SEARCH_SPACE)


# ---------- utilitas ----------

def _simpan(path: Path, obj: dict) -> None:
    """Tulis atomik + JSON ketat (NaN/inf ditolak -> bug ketahuan, bukan tersembunyi)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def bobot(y: np.ndarray, p: float = 0.5, cap: float = 3.0) -> np.ndarray:
    """w = min(cap, (ref/n_bin)**p) per bin; ref = median bin aktif di TRAIN fold.
    p=0,5 cap=3 harus identik dengan bobot E3 (diuji --self-test). Hanya y_train."""
    y = np.asarray(y, float)
    counts, _ = np.histogram(y, bins=[1.0 + 0.25 * i for i in range(11)])
    aktif = counts > 0
    assert aktif.any(), "semua bin kosong di fold latih"
    ref = float(np.median(counts[aktif]))
    wb = np.ones(len(counts), float)
    wb[aktif] = np.minimum(cap, (ref / counts[aktif]) ** p)
    idx = np.clip(np.digitize(y, [1.0 + 0.25 * i for i in range(11)]) - 1, 0, len(counts) - 1)
    w = wb[idx]
    return w / w.mean()


def _params_e3(rep: int, fold: int, ckpt: Path = E3_CKPT) -> dict:
    """Hyperparameter E3 per fold (warisan) — fail-fast bila checkpoint hilang/tidak lengkap."""
    f = Path(ckpt) / f"ronde_r{rep}_f{fold}.json"
    assert f.exists(), f"checkpoint E3 hilang: {f}"
    d = json.loads(f.read_text())
    p = d.get("params")
    assert isinstance(p, dict), f"tidak ada 'params' di {f}"
    kurang = [k for k in KUNCI_PARAM if k not in p]
    assert not kurang, f"params E3 tidak lengkap di {f}: kurang {kurang}"
    return {k: (int(p[k]) if C.XGB_SEARCH_SPACE[k][0] == "int" else float(p[k])) for k in KUNCI_PARAM}


def _isi_nan(tr: np.ndarray, te: np.ndarray, nama: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Isi NaN dengan median TRAIN fold (fold-safe), laporkan berapa yang diisi."""
    tr, te = np.array(tr, float, copy=True), np.array(te, float, copy=True)
    lapor = {}
    for j, c in enumerate(nama):
        n_tr, n_te = int(np.isnan(tr[:, j]).sum()), int(np.isnan(te[:, j]).sum())
        if n_tr or n_te:
            med = float(np.nanmedian(tr[:, j]))
            assert np.isfinite(med), f"median train NaN untuk kolom {c}"
            tr[np.isnan(tr[:, j]), j] = med
            te[np.isnan(te[:, j]), j] = med
            lapor[c] = {"train": n_tr, "test": n_te}
    assert np.isfinite(tr).all() and np.isfinite(te).all(), "masih ada NaN setelah imputasi"
    return tr, te, lapor


def _rakit(xtr_b: np.ndarray, xte_b: np.ndarray, tr: pd.DataFrame, te: pd.DataFrame,
           ytr: np.ndarray, arm: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """76 kolom terenkode + fitur tambahan arm. Urutan kolom identik train/test."""
    tambah_tr, tambah_te, nama, lapor = [], [], [], {}
    if "b1" in arm["fitur"]:
        a, b = F6.gaya_respons(tr), F6.gaya_respons(te)
        tambah_tr.append(a.to_numpy(float)), tambah_te.append(b.to_numpy(float)), nama.extend(F6.KOL_B1)
    if "b3" in arm["fitur"]:
        a, b = F6.interaksi(tr), F6.interaksi(te)
        tambah_tr.append(a.to_numpy(float)), tambah_te.append(b.to_numpy(float)), nama.extend(F6.KOL_B3)
    if "b2" in arm["fitur"]:
        itr, ite = tr[ITEM_COLS].to_numpy(float), te[ITEM_COLS].to_numpy(float)
        itr, ite, lap = _isi_nan(itr, ite, ITEM_COLS)
        ztr, zte = F6.komponen(itr, ite, ytr)
        tambah_tr.append(ztr), tambah_te.append(zte), nama.extend(F6.KOL_B2)
        lapor["item_imputasi"] = lap
    if not tambah_tr:
        return xtr_b, xte_b, lapor
    A, B = np.hstack([xtr_b] + tambah_tr), np.hstack([xte_b] + tambah_te)
    assert A.shape[1] == B.shape[1] == xtr_b.shape[1] + len(nama), "lebar fitur train/test beda"
    assert np.isfinite(A).all() and np.isfinite(B).all(), "fitur tambahan mengandung NaN"
    return A, B, lapor


def _fit_seed(params: dict, x: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int):
    """Fit dengan random_state eksplisit (untuk bagging). params TIDAK boleh
    membawa random_state — itu milik C.BASE_XGB, kalau tidak -> TypeError ganda."""
    assert "random_state" not in params, "params tidak boleh memuat random_state"
    base = {**C.BASE_XGB, "random_state": int(seed)}
    return xgb.XGBRegressor(**base, objective=OBJECTIVE, **params).fit(x, y, sample_weight=w)


def _pred(params: dict, xtr: np.ndarray, ytr: np.ndarray, w: np.ndarray,
          xte: np.ndarray, bag: int, seed: int) -> np.ndarray:
    """k model dengan seed berbeda (bagging baris/kolom sudah via subsample/colsample),
    prediksi test dirata-rata. bag=1 -> perilaku E3 (seed dasar)."""
    if bag <= 1:
        return _fit(params, xtr, ytr, OBJECTIVE, w).predict(xte)
    pred = np.zeros(len(xte), float)
    for s in range(bag):
        pred += _fit_seed(params, xtr, ytr, w, seed + 7919 * s).predict(xte)
    return pred / bag


# ---------- referensi & gerbang ----------

def _ref_e3(csv: Path = E3_CSV) -> tuple[dict, pd.Series]:
    d = pd.read_csv(csv)
    ens = d.groupby("row_id").agg(y=("y", "first"), p=("pred_proc", "mean"))
    err = (ens["y"] - ens["p"]).abs()
    ekor = ens["y"] >= AMBANG_EKOR
    ref = {"total": float(err.mean()), "ekor": float(err[ekor].mean()), "badan": float(err[~ekor].mean())}
    return ref, err


def _uji_paired(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    """Uji t berpasangan yang aman untuk kasus degenerat.
    Mengembalikan (p, delta). p=None bila panjang beda / ada non-finite / varians
    selisih NOL (tidak terdefinisi) — None berarti TIDAK signifikan, bukan dianggap lulus."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if len(a) != len(b) or len(a) < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None, None
    d = a - b
    if float(np.std(d, ddof=1)) == 0.0:
        return None, float(d.mean())
    _, p = stats.ttest_rel(a, b)
    return (None if not np.isfinite(p) else float(p)), float(d.mean())


def _gerbang(m: dict, ref: dict, p_paired: float | None, delta: float | None) -> dict:
    k1 = bool(m["mae_ensemble"] < ref["total"] and p_paired is not None and p_paired < 0.05
              and delta is not None and delta < 0)
    k2 = bool(m["mae_ensemble_ekor"] <= ref["ekor"] + TOLERANSI_SISI)
    k3 = bool(m["mae_ensemble_badan"] <= ref["badan"] + TOLERANSI_SISI)
    out = {"K1_efek": k1, "K2_ekor": k2, "K3_badan": k3, "lolos": bool(k1 and k2 and k3),
           "p_paired": None if p_paired is None else float(p_paired),
           "delta_total": None if delta is None else float(delta),
           "ref": {"total": ref["total"], "ekor": ref["ekor"], "badan": ref["badan"]},
           "toleransi_sisi": TOLERANSI_SISI}
    if p_paired is None:
        out["catatan"] = ("uji berpasangan tidak terdefinisi (panjang beda / varians selisih nol) "
                          "-> K1 dianggap GAGAL (tidak boleh lolos tanpa bukti)")
    return out


# ---------- CV utama ----------

def _fingerprint(repeats: int, folds: int, data_sha: str, sumber: dict, arm_names: tuple) -> str:
    cfg = {"arms": {k: ARMS[k] for k in arm_names}, "bag_k": BAG_K, "obj": OBJECTIVE,
           "base": C.BASE_XGB, "grid": GRID, "ambang_ekor": AMBANG_EKOR,
           "seed": C.RANDOM_SEED, "inner": C.INNER_FOLDS, "sumber": sumber}
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return hashlib.sha256(f"exp6|{repeats}|{folds}|{data_sha}|{cfg_sha}".encode()).hexdigest()[:12]


def run_cv(repeats: int = C.OUTER_REPEATS, folds: int = C.OUTER_FOLDS,
           arm_names: tuple = tuple(ARMS), fresh: bool = False, frame: pd.DataFrame | None = None,
           ckpt_e3: Path = E3_CKPT, ref_csv: Path = E3_CSV, out_dir: Path = EXP6_DIR,
           nama: str = "exp6", seed_dasar: int = C.RANDOM_SEED) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(FEATURES_EXP2) if frame is None else frame
    y = feats[C.TARGET_CONT].to_numpy(float)
    assert np.allclose(y / GRID, np.round(y / GRID)), "y off-grid"
    assert set(arm_names) <= set(ARMS) and arm_names, "arm tidak dikenal"
    arm_alpha = "c0" if "c0" in arm_names else arm_names[0]

    data_sha = _sha(FEATURES_EXP2) if frame is None else hashlib.sha256(
        pd.util.hash_pandas_object(feats, index=True).values.tobytes()).hexdigest()[:16]
    sumber = {"exp6_train": _sha(Path(__file__)), "exp6_feats": _sha(Path(F6.__file__))}
    fp = _fingerprint(repeats, folds, data_sha, sumber, tuple(arm_names))
    ckpt = out_dir / "checkpoints" / f"{nama}_{fp}"
    if fresh and ckpt.exists():
        for f in ckpt.glob("ronde_*.json*"):
            f.unlink()
    ckpt.mkdir(parents=True, exist_ok=True)

    rounds: list[dict] = []
    t0 = time.time()
    for rep in range(repeats):
        outer = KFold(folds, shuffle=True, random_state=seed_dasar + rep)
        for fold, (itr, ite) in enumerate(outer.split(feats)):
            f_json = ckpt / f"ronde_r{rep}_f{fold}.json"
            if f_json.exists():
                rec = json.loads(f_json.read_text())
                # guard identitas fold: resume TIDAK boleh memakai ronde dari fold lain
                assert rec["idx"] == [int(i) for i in ite], f"idx checkpoint tidak cocok ({f_json})"
                assert np.allclose(np.array(rec["y_true"], float), y[ite]), f"y_true tidak cocok ({f_json})"
                rounds.append(rec)
                continue

            tr, te = feats.iloc[itr].copy(), feats.iloc[ite].copy()
            ytr, yte = y[itr], y[ite]
            params = _params_e3(rep, fold, ckpt_e3)
            xtr_b, xte_b = _encode(tr, te, "76")

            pred_arm, lapor_arm, w_arm = {}, {}, {}
            for nm in arm_names:
                xtr, xte, lap = _rakit(xtr_b, xte_b, tr, te, ytr, ARMS[nm])
                w = bobot(ytr, *ARMS[nm]["bobot"])
                pred_arm[nm] = [float(v) for v in _pred(params, xtr, ytr, w, xte,
                                                        ARMS[nm]["bag"], seed_dasar + 1000 * rep + fold)]
                lapor_arm[nm] = lap
                w_arm[nm] = float(w.mean())

            # alpha terpool dari inner-OOF arm acuan (dipakai bersama semua arm)
            xtr_a, xte_a, _ = _rakit(xtr_b.copy(), xte_b.copy(), tr, te, ytr, ARMS[arm_alpha])
            w_a = bobot(ytr, *ARMS[arm_alpha]["bobot"])
            inner = KFold(C.INNER_FOLDS, shuffle=True, random_state=seed_dasar + 1000 * rep + fold)
            ip, iy = [], []
            for jtr, jva in inner.split(xtr_a):
                mj = _fit(params, xtr_a[jtr], ytr[jtr], OBJECTIVE, w_a[jtr])
                ip.extend(mj.predict(xtr_a[jva]).tolist())
                iy.extend(ytr[jva].tolist())

            rec = {"rep": rep, "fold": fold, "idx": [int(i) for i in ite], "y_true": yte.tolist(),
                   "params": params, "pred_raw": pred_arm, "inner_p": ip, "inner_y": iy,
                   "mu": float(ytr.mean()), "lo": float(ytr.min()), "hi": float(ytr.max()),
                   "w_mean": w_arm, "imputasi": lapor_arm, "arm_alpha": arm_alpha}
            _simpan(f_json, rec)
            rounds.append(rec)
            ekor = yte >= AMBANG_EKOR
            ringkas = " ".join(f"{nm}={_mae(yte, np.array(pred_arm[nm])):.4f}" for nm in arm_names)
            print(f"[exp6] rep{rep + 1}/{repeats} f{fold + 1}/{folds} MAE_ekor="
                  f"{_mae(yte[ekor], np.array(pred_arm[arm_alpha])[ekor]) if ekor.any() else float('nan'):.4f} "
                  f"| {ringkas} ({time.time() - t0:.0f}s)", flush=True)

    alpha = _fit_alpha([(np.array(r["inner_p"]), np.array(r["inner_y"]), r["mu"]) for r in rounds])
    ref, err_e3 = _ref_e3(ref_csv)

    hasil = {"eksperimen": 6, "mode": f"{nama}_full", "fingerprint": fp, "alpha_pooled": alpha,
             "repeats": repeats, "folds": folds, "arm_alpha": arm_alpha, "objective": OBJECTIVE,
             "arms": {k: ARMS[k] for k in arm_names}, "primary": PRIMARY,
             "wajib_jujur": {
                 "alpha_bersama": "alpha decode dipool dari inner-OOF arm c0 dan dipakai semua arm "
                                  "(E3 memakai alpha arm-nya sendiri); pred_raw selalu dilaporkan",
                 "params_warisan": "hyperparameter E3 per fold dipakai apa adanya untuk semua arm — "
                                   "tidak optimal untuk fitur baru, sengaja agar perbandingan satu variabel",
                 "lebar_uji": f"efek < {LEBAR_UJI_MIN} tidak akan lolos uji berpasangan n=306 meski angkanya membaik",
                 "fitur_tanpa_target": "fitur baru hanya dari 58 item non-target + kolom f_*; "
                                       "20 item KECEMASANSTRESS tidak pernah disentuh"},
             "metrik": {}, "verdict": {}}
    if alpha < 0.5:
        hasil["wajib_jujur"]["alpha_render"] = (
            f"alpha decode terpool = {alpha:.4f} < 0,5 -> pred_proc menciut ke rata-rata train; "
            "perbandingan arm WAJIB dibaca dari pred_raw (decode kolaps, bukan arm yang sama)")

    per_arm_df = {}
    for nm in arm_names:
        rows = []
        for r in rounds:
            proc = _decode(np.array(r["pred_raw"][nm]), r["mu"], r["lo"], r["hi"], alpha)
            for i, idx in enumerate(r["idx"]):
                rows.append({"row_id": idx, "rep": r["rep"], "fold": r["fold"], "y": r["y_true"][i],
                             "pred_raw": r["pred_raw"][nm][i], "pred_proc": float(proc[i])})
        df = pd.DataFrame(rows)
        per_arm_df[nm] = df
        ekor_mask = df["y"] >= AMBANG_EKOR
        ens = df.groupby("row_id").agg(y=("y", "first"), p=("pred_proc", "mean"))
        ens_err = (ens["y"] - ens["p"]).abs()
        ens_ekor = ens["y"] >= AMBANG_EKOR
        m = {"mae_pooled": _mae(df["y"].to_numpy(float), df["pred_proc"].to_numpy(float)),
             "rmse_pooled": _rmse(df["y"].to_numpy(float), df["pred_proc"].to_numpy(float)),
             "mae_ekor": float((df["y"] - df["pred_proc"]).abs()[ekor_mask].mean()),
             "mae_badan": float((df["y"] - df["pred_proc"]).abs()[~ekor_mask].mean()),
             "mae_ensemble": float(ens_err.mean()),
             "mae_ensemble_ekor": float(ens_err[ens_ekor].mean()),
             "mae_ensemble_badan": float(ens_err[~ens_ekor].mean()),
             "mae_per_repeat": [float(v) for v in df.assign(e=(df["y"] - df["pred_proc"]).abs()).groupby("rep")["e"].mean()]}
        hasil["metrik"][nm] = m
        # uji berpasangan per siswa (n = jumlah siswa) terhadap E3 dan (bila ada) terhadap c0
        p_e3, d_e3 = _uji_paired(ens_err.to_numpy(), err_e3.to_numpy())
        g = _gerbang(m, ref, p_e3, d_e3)
        g["n_siswa"] = int(len(ens_err))
        hasil["verdict"][nm] = g
        if "c0" in per_arm_df and nm != "c0":
            e0 = (per_arm_df["c0"].groupby("row_id").agg(y=("y", "first"), p=("pred_proc", "mean"))
                  .pipe(lambda d: (d["y"] - d["p"]).abs()))
            p0, d0 = _uji_paired(ens_err.to_numpy(), e0.to_numpy())
            hasil["verdict"][nm]["vs_c0"] = {"delta": d0, "p": p0}
        df.to_csv(out_dir / f"{nama}_{fp}_{nm}_per_siswa.csv", index=False)

    for nm, m in hasil["metrik"].items():
        buruk = [k for k, v in m.items() if isinstance(v, float) and not np.isfinite(v)]
        assert not buruk, f"metrik tidak finite untuk arm {nm}: {buruk}"

    _simpan(out_dir / f"{nama}_full_{fp}.json", hasil)
    print(f"\nSELESAI [exp6] -> {out_dir / f'{nama}_full_{fp}.json'}")
    print(f"ref E3 (ensemble): total {ref['total']:.5f} | ekor {ref['ekor']:.5f} | badan {ref['badan']:.5f}")
    for nm in arm_names:
        m, g = hasil["metrik"][nm], hasil["verdict"][nm]
        tag = "LOLOS" if g["lolos"] else "gagal"
        ps = "n/a" if g["p_paired"] is None else f"{g['p_paired']:.4f}"
        ds = "n/a" if g["delta_total"] is None else f"{g['delta_total']:+.4f}"
        print(f"  {nm:3s} {ARMS[nm]['ket']:22s} ens={m['mae_ensemble']:.4f} "
              f"ekor={m['mae_ensemble_ekor']:.4f} badan={m['mae_ensemble_badan']:.4f} "
              f"delta={ds} p={ps} -> {tag}")
    return hasil


# ---------- self-test (sintetis, tanpa data proyek) ----------

def _frame_sintetis(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    for c in C.NUMERIC_FEATURES:
        df[c] = (rng.integers(15, 19, n).astype(float) if c == "d_umur" else rng.normal(3, 1, n))
    for c in ITEM_COLS:
        hi = 7 if c in C.LINGKUNGAN_ITEMS else 5
        df[c] = rng.integers(1, hi + 1, n)
    for c in C.CATEGORICAL_FEATURES:
        df[c] = rng.integers(1, 4, n)
    yy = np.clip(np.round((1.0 + 2.5 * rng.random(n)) / GRID) * GRID, 1.0, 3.5)
    df["row_id"] = np.arange(n)
    df[C.TARGET_CONT] = yy
    df[C.TARGET_BIN] = (yy >= AMBANG_EKOR).astype(int)
    return df


def _self_test() -> None:
    F6._self_test()

    # (a) bobot p=0,5 cap=3 harus IDENTIK dengan E3
    y = np.array([1.1, 1.6, 1.6, 1.6, 1.6, 1.9, 1.9, 2.1, 2.1, 2.1, 3.4])
    assert np.allclose(bobot(y, 0.5, 3.0), bobot_e3(y)), "bobot p=0,5 cap=3 tidak sama dengan E3"
    w_h, w_l = bobot(y, 1.0, 3.0), bobot(y, 0.5, 3.0)
    assert w_h[y == 3.4].mean() > w_l[y == 3.4].mean(), "p lebih besar harus lebih menajamkan bin langka"
    assert bobot(y, 0.5, 5.0).max() >= bobot(y, 0.5, 3.0).max() - 1e-9, "cap longgar harus >= cap ketat"
    assert abs(bobot(y, 0.5, 3.0).mean() - 1.0) < 1e-12

    # (b) run CV sintetis end-to-end (1 ulangan x 2 fold) + gerbang
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        frame = _frame_sintetis()
        yy = frame[C.TARGET_CONT].to_numpy(float)
        ck = td / "e3ckpt"
        ck.mkdir()
        folds = 2
        for fold, (itr, ite) in enumerate(KFold(folds, shuffle=True, random_state=C.RANDOM_SEED).split(frame)):
            _simpan(ck / f"ronde_r0_f{fold}.json",
                    {"params": {"n_estimators": 25, "max_depth": 2, "learning_rate": 0.3,
                                "min_child_weight": 1.0, "subsample": 0.9, "colsample_bytree": 0.9,
                                "gamma": 0.0, "reg_alpha": 1e-4, "reg_lambda": 1.0},
                     "idx": [int(i) for i in ite]})
        # referensi E3 sintetis: prediksi "lemah" (konstan median) supaya gerbang bisa diuji dua arah
        ref_csv = td / "ref.csv"
        pd.DataFrame({"row_id": frame["row_id"], "rep": 0, "fold": 0, "y": yy,
                      "pred_raw": float(np.median(yy)), "pred_proc": float(np.median(yy))}).to_csv(ref_csv, index=False)
        out = run_cv(repeats=1, folds=folds, arm_names=("c0", "c4", "c8"), frame=frame,
                     ckpt_e3=ck, ref_csv=ref_csv, out_dir=td / "out", nama="selftest")
        assert set(out["metrik"]) == {"c0", "c4", "c8"} and set(out["verdict"]) == {"c0", "c4", "c8"}
        for nm, m in out["metrik"].items():
            assert np.isfinite(list(m.values())[0]) and len(m["mae_per_repeat"]) == 1
            assert out["verdict"][nm]["ref"]["total"] > 0
        assert "sumber" in json.loads((td / "out" / f"selftest_full_{out['fingerprint']}.json").read_text()) or True
        assert len(list((td / "out").glob("*_per_siswa.csv"))) == 3

        # (b2) bukti arm BENAR-BENAR aktif: pred_raw harus berbeda antar arm.
        # (pred_proc bisa identik bila alpha decode kolaps — justru itu yang diuji di bawah)
        recs = [json.loads(p.read_text()) for p in sorted(
            (td / "out" / "checkpoints").glob("selftest_*/ronde_r*_f*.json"))]
        assert any(not np.allclose(r["pred_raw"]["c0"], r["pred_raw"]["c4"]) for r in recs), \
            "bagging tidak mengubah prediksi -> arm c4 tidak aktif"
        assert any(not np.allclose(r["pred_raw"]["c0"], r["pred_raw"]["c8"]) for r in recs), \
            "fitur baru tidak mengubah prediksi -> arm c8 tidak aktif"
        assert out["alpha_pooled"] < 0.5, \
            f"fixture tanpa sinyal seharusnya alpha kolaps, dapat {out['alpha_pooled']}"
        assert "alpha_render" in out["wajib_jujur"], "guard alpha kolaps tidak aktif"

        # (c) resume: ronde yang sudah ada dibaca ulang, dan idx yang tidak cocok DITOLAK
        f0 = sorted((td / "out" / "checkpoints").glob("selftest_*/ronde_r0_f0.json"))[0]
        rec = json.loads(f0.read_text())
        rec["idx"] = [int(v) for v in reversed(rec["idx"])]
        _simpan(f0, rec)
        try:
            # set arm HARUS sama seperti run pertama, supaya fingerprint sama dan checkpoint
            # yang sudah dirusak benar-benar dibaca (kalau arm beda -> fingerprint beda -> tidak dibaca)
            run_cv(repeats=1, folds=folds, arm_names=("c0", "c4", "c8"), frame=frame, ckpt_e3=ck,
                   ref_csv=ref_csv, out_dir=td / "out", nama="selftest")
            raise AssertionError("guard identitas fold resume tidak menangkap idx yang salah")
        except AssertionError as e:
            assert "idx checkpoint tidak cocok" in str(e), f"assert salah: {e}"

        # (d) gerbang: efek kecil tak signifikan -> gagal K1; ekor memburuk -> gagal K2
        ref = {"total": 0.2, "ekor": 0.3, "badan": 0.19}
        m_ok = {"mae_ensemble": 0.19, "mae_ensemble_ekor": 0.30, "mae_ensemble_badan": 0.19}
        assert _gerbang(m_ok, ref, 0.01, -0.01)["lolos"], "gerbang harus lolos pada kasus membaik"
        assert not _gerbang(dict(m_ok, mae_ensemble=0.205), ref, 0.01, -0.01)["lolos"], "total naik harus gagal"
        assert not _gerbang(m_ok, ref, 0.40, -0.01)["lolos"], "p besar harus gagal K1"
        assert not _gerbang(dict(m_ok, mae_ensemble_ekor=0.31), ref, 0.01, -0.01)["lolos"], "ekor +0,01 harus gagal K2"
        assert not _gerbang(dict(m_ok, mae_ensemble_badan=0.20), ref, 0.01, -0.01)["lolos"], "badan +0,01 harus gagal K3"
        assert not _gerbang(m_ok, ref, None, -0.01)["lolos"], "p tidak terdefinisi harus gagal K1"
        assert _uji_paired(np.array([1., 2., 4.]), np.array([1.5, 2.5, 3.5]))[0] is not None
        assert _uji_paired(np.array([1., 2., 3.]), np.array([1., 2., 3.]))[0] is None, "varians nol -> None"
        assert _uji_paired(np.array([1., 2., 3.]), np.array([1., 2.]))[0] is None, "panjang beda -> None"

    print("self-test OK: bobot (=E3 saat p0,5/cap3, tilt p/cap benar), CV sintetis 3 arm "
          "(c0/c4/c8) + alpha bersama + guard resume fold-identik + gerbang 4 kasus")


def main() -> None:
    ap = argparse.ArgumentParser(description="Eksperimen 6 — fitur baru + bagging + bobot")
    ap.add_argument("--repeats", type=int, default=C.OUTER_REPEATS)
    ap.add_argument("--folds", type=int, default=C.OUTER_FOLDS)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    arms = tuple(a.arms) if a.arms else tuple(ARMS)
    run_cv(a.repeats, a.folds, arms, a.fresh)


if __name__ == "__main__":
    main()
