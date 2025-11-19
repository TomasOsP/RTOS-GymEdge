# -*- coding: utf-8 -*-
import argparse, glob, os, re
import numpy as np, pandas as pd, yaml
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, welch
from scipy.integrate import trapezoid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

# ============= Señal =============
def _norm_wn(cut, fs):
    if fs is None or fs <= 0: fs = 50.0
    nyq = fs / 2.0
    wn = cut / nyq if nyq > 0 else 0.5
    return float(max(min(wn, 0.99), 1e-3))

def butter_highpass(cut, fs, order=4):
    b, a = butter(order, _norm_wn(cut, fs), btype="highpass"); return b, a

def butter_lowpass(cut, fs, order=4):
    b, a = butter(order, _norm_wn(cut, fs),  btype="lowpass");  return b, a

def psd_band_power(x, fs, flo, fhi):
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    m = (f>=flo) & (f<=fhi)
    return float(trapezoid(Pxx[m], f[m]) if m.any() else 0.0)

# ============= Utils tiempo/unidades =============
G = 9.80665
RAD = np.pi/180.0

def _detect_time_scale(colname, v):
    name = colname.lower()
    med = np.median(v)
    if "us" in name:      return 1e-6   # microseg → s
    if "ms" in name:      return 1e-3   # miliseg → s
    if "ns" in name:      return 1e-9
    # heurística por magnitud
    if med > 1e12:        return 1e-9   # ns
    if med > 1e9:         return 1e-6   # µs (epoch en µs)
    if med > 1e6:         return 1e-6   # µs
    if med > 1e3:         return 1e-3   # ms
    return 1.0            # ya en segundos

# ============= Carga CSV (formato nuevo) =============
def read_imu_csv(path):
    """
    Espera columnas: timestamp_*, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
    Devuelve: t(s), ax/ay/az (m/s^2), gx/gy/gz (rad/s)
    """
    df = pd.read_csv(path)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # localizar timestamp
    tcol = None
    for c in cols:
        if c.lower().startswith("timestamp"):
            tcol = c; break
    if tcol is None:
        raise ValueError(f"{path}: no encuentro columna de tiempo que empiece por 'timestamp'.")

    # convertir tiempo a segundos relativos (t0=0)
    scale = _detect_time_scale(tcol, pd.to_numeric(df[tcol], errors="coerce").values)
    t = pd.to_numeric(df[tcol], errors="coerce").values * scale
    t = t - np.nanmin(t)

    # mapear columnas físicas con tolerancia de nombre
    def pick(key):
        # busca coincidencia exacta; si no, por contiene
        for c in cols:
            if c.lower() == key: return c
        for c in cols:
            if key in c.lower(): return c
        return None

    axc, ayc, azc = pick("ax_g"), pick("ay_g"), pick("az_g")
    gxc, gyc, gzc = pick("gx_dps"), pick("gy_dps"), pick("gz_dps")
    need = [axc, ayc, azc, gxc, gyc, gzc]
    if any(c is None for c in need):
        raise ValueError(f"{path}: faltan columnas esperadas (ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps).")

    ax = pd.to_numeric(df[axc], errors="coerce").values * G
    ay = pd.to_numeric(df[ayc], errors="coerce").values * G
    az = pd.to_numeric(df[azc], errors="coerce").values * G
    gx = pd.to_numeric(df[gxc], errors="coerce").values * RAD
    gy = pd.to_numeric(df[gyc], errors="coerce").values * RAD
    gz = pd.to_numeric(df[gzc], errors="coerce").values * RAD

    out = pd.DataFrame({"t": t, "ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz})
    out = out.dropna().sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)
    return out

# ============= Recorte =============
def crop_df_by_intervals(df, intervals):
    if not intervals: return df
    parts=[]
    for t0,t1 in intervals:
        parts.append(df[(df["t"]>=t0)&(df["t"]<=t1)])
    return (pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)) if parts else df.iloc[0:0]

def trim_df(df, start_seconds=0.0, end_seconds=0.0):
    if df.empty: return df
    tmin,tmax = df["t"].min(), df["t"].max()
    return df[(df["t"]>=tmin+start_seconds)&(df["t"]<=tmax-end_seconds)].reset_index(drop=True)

def base_activity(name):
    # Datos_Trote2.csv -> trote
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r'^datos[_-]*','', name, flags=re.IGNORECASE)
    name = re.sub(r'\d+$','', name.lower())
    return name

def apply_time_config(activity_key, df, cfg):
    if cfg is None: return df
    sec = cfg.get(activity_key, None)
    if sec is None:
        sec = cfg.get(base_activity(activity_key), None)
        if sec is None: return df
    if "intervals" in sec: df = crop_df_by_intervals(df, sec["intervals"])
    if "trim" in sec:
        st = float(sec["trim"].get("start_seconds", 0.0))
        en = float(sec["trim"].get("end_seconds", 0.0))
        df = trim_df(df, st, en)
    return df

# ============= Prepro =============
def preprocess_df(df, fs, hp_cut=0.25, lp_cut=20.0):
    b,a = butter_highpass(hp_cut, fs)
    for c in ["ax","ay","az"]: df[c] = filtfilt(b,a, df[c].values)
    b2,a2 = butter_lowpass(lp_cut, fs)
    for c in ["gx","gy","gz"]: df[c] = filtfilt(b2,a2, df[c].values)
    df["a_mag"]=np.sqrt(df["ax"]**2+df["ay"]**2+df["az"]**2)
    df["g_mag"]=np.sqrt(df["gx"]**2+df["gy"]**2+df["gz"]**2)
    return df

# ============= Recorte automático por estabilidad =============
def auto_trim_by_stability(df, fs, win_sec=1.0, hold_sec=3.0, thr_g_rms=0.4, thr_a_std=0.6):
    if df.empty or fs is None or fs <= 0:
        return df
    win = max(1, int(round(win_sec * fs)))
    hold = max(win, int(round(hold_sec * fs)))
    g2 = pd.Series(df["g_mag"].values**2, copy=False)
    g_rms = np.sqrt(g2.rolling(win, min_periods=win).mean())
    a_std = pd.Series(df["a_mag"].values).rolling(win, min_periods=win).std()
    good = (g_rms.fillna(0.0).values >= thr_g_rms) & (a_std.fillna(0.0).values >= thr_a_std)
    i0 = 0
    for i in range(0, len(good) - hold + 1):
        if good[i:i+hold].all(): i0 = i; break
    i1 = len(good)
    for i in range(len(good) - hold, -1, -1):
        if i - hold >= 0 and good[i-hold:i].all(): i1 = i; break
    if i1 - i0 < hold:  # no hay zona estable suficientemente larga
        return df
    return df.iloc[i0:i1].reset_index(drop=True)

# ============= Ventaneo + features =============
@dataclass
class WindowingCfg:
    win_sec: float = 2.0
    hop_frac: float = 0.5

def estimate_fs(t):
    t = np.asarray(t)
    if t.size < 2: return 50.0
    dt = np.diff(t); dt = dt[np.isfinite(dt)]
    if dt.size == 0: return 50.0
    m = np.median(dt)
    return (1.0/m) if m > 0 else 50.0

def segment_indices(n, win, hop):
    out=[]; i=0
    while i+win<=n:
        out.append((i,i+win)); i+=hop
    return out

def extract_features(seg, fs):
    feats=[]; cols=["ax","ay","az","gx","gy","gz","a_mag","g_mag"]
    for c in cols:
        x=seg[c].values
        feats += [x.mean(), x.std(), x.min(), x.max(),
                  np.median(x), np.percentile(x,25), np.percentile(x,75),
                  np.ptp(x),
                  np.sum(np.abs(np.diff(x)))/(len(x)-1+1e-9),
                  (np.count_nonzero((x[1:]*x[:-1])<0)/(len(x)-1+1e-9))]
    feats += [psd_band_power(seg["a_mag"].values, fs, 1.5,3.5),
              psd_band_power(seg["a_mag"].values, fs, 3.5,6.0)]
    return np.array(feats, dtype=float)

def windows_from_df(df, label, cfg: WindowingCfg):
    if df.empty: return np.empty((0,)), np.array([]), None, None, None
    fs = estimate_fs(df["t"].values)
    win = int(cfg.win_sec * fs); hop = int(win * cfg.hop_frac)
    X_list, y_list = [], []
    for i0,i1 in segment_indices(len(df), win, hop):
        seg = df.iloc[i0:i1]
        X_list.append(extract_features(seg, fs))
        y_list.append(label)
    return (np.vstack(X_list) if X_list else np.empty((0,))), np.array(y_list), fs, win, hop

# ============= Descubrir archivos (formato nuevo) =============
def find_activity_files(data_dir):
    pats = glob.glob(os.path.join(data_dir, "Datos_*.csv"))  # <-- deja solo uno
    # pats += glob.glob(os.path.join(data_dir, "datos_*.csv"))  # quítalo

    files = []
    seen = set()
    for p in pats:
        key = os.path.normcase(os.path.abspath(p))  # dedupe robusto en Windows
        if key in seen: 
            continue
        seen.add(key)
        act = base_activity(os.path.basename(p))  # p.ej. Datos_Trote2.csv → 'trote'
        files.append((act, p))
    return files


# ============= Main =============
def main(args):
    # Config de tiempo
    cfg=None
    if args.time_cfg and os.path.exists(args.time_cfg):
        with open(args.time_cfg,"r",encoding="utf-8") as f: cfg=yaml.safe_load(f)
        print(f"Config de tiempo cargada: {args.time_cfg}")
    else:
        print("Sin config de tiempo (usar archivos completos).")

    files = find_activity_files(args.data_dir)
    if not files:
        raise SystemExit(f"No encontré CSV en formato 'Datos_*.csv' dentro de {args.data_dir}")

    # Clases detectadas y orden
    activities = sorted({act for act,_ in files})
    preferred = ["trote","escal","bici","elip","lazo","plancha"]
    activities = sorted(set(activities), key=lambda x: (preferred.index(x) if x in preferred else 999, x))
    name2id = {name:i for i,name in enumerate(activities)}
    print("Clases:", name2id)

    X_all, y_all = [], []
    any_fs=any_win=any_hop=None

    for act, path in files:
        # Carga (ya trae acel+gyro)
        df = read_imu_csv(path)

        # Recorte por config (clave por actividad)
        df = apply_time_config(act, df, cfg)

        if df.empty:
            print(f"[AVISO] {act}: sin datos tras recorte → omitido."); continue

        # Preprocesado
        fs = estimate_fs(df["t"].values)
        # print(f"   [{act}] fs≈{fs:.1f} Hz")  # depuración opcional
        df = preprocess_df(df, fs, hp_cut=args.hp_cut, lp_cut=args.lp_cut)

        # Auto-trim opcional
        if args.auto_trim:
            pre_len = len(df)
            df = auto_trim_by_stability(
                df, fs,
                win_sec=args.auto_win_sec,
                hold_sec=args.auto_hold_sec,
                thr_g_rms=args.auto_thr_g_rms,
                thr_a_std=args.auto_thr_a_std
            )
            post_len = len(df)
            print(f"   auto_trim: {act} => {pre_len}→{post_len} muestras")

        # Ventanas + label
        label = name2id[act]
        cfg_w = WindowingCfg(win_sec=args.win_sec, hop_frac=args.hop_frac)
        X, y, fs_used, win, hop = windows_from_df(df, label, cfg_w)
        if X.size==0:
            print(f"[AVISO] {act}: insuficiente para {args.win_sec}s → omitido."); continue

        X_all.append(X); y_all.append(y)
        any_fs, any_win, any_hop = fs_used, win, hop
        print(f"{act:>8}: ventanas={len(y)}  clase={label}")

    if not X_all: raise SystemExit("No se generaron ventanas. Revisa recortes/ventanas.")

    X = np.vstack(X_all); y = np.concatenate(y_all)
    print("\nDistribución por clase (id:conteo):", dict(zip(*np.unique(y, return_counts=True))))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight="balanced",
            random_state=42
        ))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    print("\nMatriz de confusión:")
    print(confusion_matrix(yte, pred))
    print("\nReporte de clasificación:")
    print(classification_report(yte, pred, digits=3, target_names=[activities[i] for i in sorted(name2id.values())]))
    scores = cross_val_score(pipe, X, y, cv=5)
    print(f"CV acc: {scores.mean():.3f} ± {scores.std():.3f}")

    dump({
        "pipe": pipe,
        "fs": any_fs, "win": any_win, "hop": any_hop,
        "cfg": {"win_sec": args.win_sec, "hop_frac": args.hop_frac},
        "feature_note": "stats(ax,ay,az,gx,gy,gz,a_mag,g_mag) + PSD [1.5-3.5,3.5-6.0] Hz",
        "label_names": activities
    }, args.out_model)
    print(f"\nModelo guardado en: {args.out_model}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entrenar multiclase desde CSV únicos (Datos_*.csv).")
    p.add_argument("--data_dir", type=str, default="DATA_IMU")
    p.add_argument("--time_cfg", type=str, default=None)
    p.add_argument("--win_sec", type=float, default=2.0)
    p.add_argument("--hop_frac", type=float, default=0.5)
    p.add_argument("--hp_cut", type=float, default=0.25)
    p.add_argument("--lp_cut", type=float, default=20.0)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--out_model", type=str, default="modelo_multiclase.joblib")

    # ---- Flags para auto-trim ----
    p.add_argument("--auto_trim", action="store_true",
                   help="Recorte automático por estabilidad (g_rms y a_std).")
    p.add_argument("--auto_win_sec", type=float, default=1.0)
    p.add_argument("--auto_hold_sec", type=float, default=3.0)
    p.add_argument("--auto_thr_g_rms", type=float, default=0.4)
    p.add_argument("--auto_thr_a_std", type=float, default=0.6)
    args = p.parse_args()
    main(args)
