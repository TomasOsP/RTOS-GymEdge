# -*- coding: utf-8 -*-
import argparse, pandas as pd, numpy as np
from joblib import load
from scipy.signal import butter, filtfilt, welch
from scipy.integrate import trapezoid
from math import pi
import pandas as pd
import matplotlib.pyplot as plt


# ========= utilidades de señal (robustas) =========
def _norm_wn(cut, fs):
    """Normaliza fc a Wn y la acota a (1e-3, 0.99) para evitar ValueError en SciPy."""
    if fs is None or fs <= 0: fs = 50.0
    nyq = fs / 2.0
    wn = (cut / nyq) if nyq > 0 else 0.5
    return float(max(min(wn, 0.99), 1e-3))

def butter_highpass(cut, fs, order=4):
    wn = _norm_wn(cut, fs)
    b, a = butter(order, wn, btype="highpass"); return b, a

def butter_lowpass(cut, fs, order=4):
    wn = _norm_wn(cut, fs)
    b, a = butter(order, wn, btype="lowpass");  return b, a

def psd_band_power(x, fs, flo, fhi):
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    m = (f >= flo) & (f <= fhi)
    return float(trapezoid(Pxx[m], f[m]) if m.any() else 0.0)

def estimate_fs(t):
    t = np.asarray(t)
    if t.size < 2: return 50.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0: return 50.0
    m = np.median(dt)  # mediana → robusto a huecos/outliers
    return (1.0/m) if m > 0 else 50.0

def preprocess(df, fs, hp_cut=0.25, lp_cut=20.0):
    # Si fs es bajo, limita lp_cut automáticamente
    lp_eff = min(lp_cut, 0.45 * fs)  # deja margen antes de Nyquist
    b,a = butter_highpass(hp_cut, fs)
    for c in ["ax","ay","az"]: df[c] = filtfilt(b,a, df[c].values)
    b2,a2 = butter_lowpass(lp_eff, fs)
    for c in ["gx","gy","gz"]: df[c] = filtfilt(b2,a2, df[c].values)
    df["a_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["g_mag"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    return df

def extract_features(seg, fs):
    feats=[]; cols=["ax","ay","az","gx","gy","gz","a_mag","g_mag"]
    for c in cols:
        x = seg[c].values
        feats += [x.mean(), x.std(), x.min(), x.max(),
                  np.median(x), np.percentile(x,25), np.percentile(x,75),
                  np.ptp(x),
                  np.sum(np.abs(np.diff(x))) / (len(x)-1 + 1e-9),
                  (np.count_nonzero((x[1:]*x[:-1])<0) / (len(x)-1 + 1e-9))]
    feats += [psd_band_power(seg["a_mag"].values, fs, 1.5, 3.5),
              psd_band_power(seg["a_mag"].values, fs, 3.5, 6.0)]
    return np.array(feats, dtype=float)

# ========= lecturas =========
def _to_seconds(ts, unit="auto"):
    ts = pd.to_numeric(ts, errors="coerce").dropna()
    if ts.empty: return None
    unit = unit.lower()
    if unit == "auto":
        name = str(getattr(ts, "name", "")).lower()
        if "us" in name: unit = "us"
        elif "ms" in name: unit = "ms"
        else:
            mx = ts.max()
            unit = "us" if mx > 1e7 else ("ms" if mx > 1e3 else "s")
    if unit == "us": return ts.values / 1e6
    if unit == "ms": return ts.values / 1e3
    return ts.values.astype(float)  # "s"

def read_single_file(path, ts_unit="auto"):
    """
    Espera columnas como: timestamp_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
    Convierte g→m/s^2, dps→rad/s, tiempo→s.
    """
    df = pd.read_csv(path)
    # tiempo
    ts_col = None
    for c in df.columns:
        cl = c.lower()
        if "timestamp" in cl or cl == "t":
            ts_col = c; break
    if ts_col is None:
        raise ValueError("No encuentro columna de tiempo (timestamp_* o t).")

    t_sec = _to_seconds(df[ts_col], ts_unit)
    if t_sec is None: raise ValueError("Columna de tiempo vacía o no numérica.")

    # ejes
    def pick(prefix):
        cols = [c for c in df.columns if c.lower().startswith(prefix)]
        if not cols: raise ValueError(f"No encuentro columna que empiece por '{prefix}'")
        return pd.to_numeric(df[cols[0]], errors="coerce")

    ax = pick("ax"); ay = pick("ay"); az = pick("az")
    gx = pick("gx"); gy = pick("gy"); gz = pick("gz")

    out = pd.DataFrame({
        "t": t_sec,
        "ax": ax * 9.80665,             # g → m/s^2
        "ay": ay * 9.80665,
        "az": az * 9.80665,
        "gx": gx * (pi/180.0),          # dps → rad/s
        "gy": gy * (pi/180.0),
        "gz": gz * (pi/180.0),
    }).dropna().reset_index(drop=True)
    return out

def read_csv_flexible(path, cols_expected):
    df = pd.read_csv(path)
    if len(df.columns) < 4:
        raise ValueError(f"{path} no tiene 4 columnas (t + 3 ejes).")
    df = df.rename(columns={
        df.columns[0]: "t",
        df.columns[1]: cols_expected[0],
        df.columns[2]: cols_expected[1],
        df.columns[3]: cols_expected[2],
    })
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)

# ========= main =========
def main(args):
    md = load(args.model)
    pipe = md["pipe"]
    label_names = md["label_names"]
    win_sec = md["cfg"]["win_sec"]
    hop_frac = md["cfg"]["hop_frac"]

    # lectura 1-CSV o 2-CSV
    if args.file:
        df = read_single_file(args.file, ts_unit=args.ts_unit)
    else:
        a = read_csv_flexible(args.acel,  ["ax","ay","az"])
        g = read_csv_flexible(args.giros, ["gx","gy","gz"])
        a["t_r"] = a["t"].round(3); g["t_r"] = g["t"].round(3)
        df = pd.merge_asof(a.sort_values("t_r"), g.sort_values("t_r"),
                           on="t_r", direction="nearest", tolerance=0.002).dropna()
        if "t_x" in df.columns: df = df.rename(columns={"t_x":"t"})
        elif "t" not in df.columns: df["t"] = df["t_r"]
        df = df[["t","ax","ay","az","gx","gy","gz"]].reset_index(drop=True)

    if len(df) < 10:
        raise SystemExit("Muy pocos datos tras la lectura/alineación.")

    df = df.sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)
    fs = estimate_fs(df["t"].values)

    # prepro con cortes configurables (y protección Wn)
    hp = args.hp_cut if args.hp_cut is not None else 0.25
    lp = args.lp_cut if args.lp_cut is not None else 20.0
    df = preprocess(df, fs, hp_cut=hp, lp_cut=lp)

    win = int(win_sec * fs); hop = int(win * hop_frac)
    if win <= 1 or hop < 1:
        raise SystemExit(f"Ventana inválida: win={win}, hop={hop} (fs≈{fs:.2f} Hz).")

    preds, probs, t_starts = [], [], []
    for i0 in range(0, len(df) - win + 1, hop):
        seg = df.iloc[i0:i0+win]
        X = extract_features(seg, fs).reshape(1, -1)
        proba = pipe.predict_proba(X)[0]
        k = int(np.argmax(proba))
        preds.append(k); probs.append(float(proba[k]))
        t0 = float(seg["t"].iloc[0]); t_starts.append(t0)
        print(f"{t0:7.2f}s -> {label_names[k]:16s} ({proba[k]:.2f})")

    if args.out:
        pd.DataFrame({
            "time_s": t_starts,
            "pred": [label_names[i] for i in preds],
            "prob": probs
        }).to_csv(args.out, index=False)
        print(f"\nResultados guardados en {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test del modelo multiclase (1 CSV combinado o 2 CSV).")
    p.add_argument("--model", type=str, default="modelo_multiclase.joblib")
    # opción 1: un solo archivo combinado
    p.add_argument("--file", type=str, help="CSV único con timestamp_* y *_g, *_dps")
    p.add_argument("--ts_unit", type=str, default="auto", choices=["auto","us","ms","s"],
                   help="Unidades del timestamp del CSV único (auto por defecto).")
    # opción 2: dos archivos separados (modo legado)
    p.add_argument("--acel", type=str, help="CSV de acelerómetro (t, X, Y, Z)")
    p.add_argument("--giros", type=str, help="CSV de giroscopio (t, X, Y, Z)")
    # cortes opcionales para este test
    p.add_argument("--hp_cut", type=float, default=None, help="Corte high-pass (Hz). Por defecto 0.25.")
    p.add_argument("--lp_cut", type=float, default=None, help="Corte low-pass (Hz). Por defecto 20.0.")
    p.add_argument("--out", type=str, default="resultados_test.csv")
    args = p.parse_args()

    if not args.file and not (args.acel and args.giros):
        raise SystemExit("Debes pasar --file (CSV único) o --acel y --giros (dos CSV).")
    if args.file and (args.acel or args.giros):
        raise SystemExit("Usa solo --file o ( --acel y --giros ), no ambos.")

    main(args)

# Leer resultados del test
df = pd.read_csv("res_lazo.csv")

# Agrupar por clase y calcular probabilidad media
mean_probs = df.groupby("pred")["prob"].mean().sort_values(ascending=False)

# Gráfico de barras
plt.figure(figsize=(8,5))
plt.bar(mean_probs.index, mean_probs.values)
plt.title("Probabilidad media por clase detectada")
plt.xlabel("Clase predicha")
plt.ylabel("Probabilidad media")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# python test_multiclase.py --file DATA_IMU/Datos_lazo.csv --ts_unit auto --out res_lazo.csv


