# -*- coding: utf-8 -*-
import argparse, pandas as pd, numpy as np
from joblib import load
from scipy.signal import butter, filtfilt, welch
from scipy.integrate import trapezoid

# ======== Utilidades de señal ========
def butter_highpass(cut, fs, order=4):
    b, a = butter(order, cut / (fs / 2), btype="highpass"); return b, a

def butter_lowpass(cut, fs, order=4):
    b, a = butter(order, cut / (fs / 2), btype="lowpass"); return b, a

def psd_band_power(x, fs, flo, fhi):
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    m = (f >= flo) & (f <= fhi)
    return float(trapezoid(Pxx[m], f[m]) if m.any() else 0.0)

def estimate_fs(t):
    t = np.asarray(t)
    if t.size < 2:
        return 50.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return 50.0
    m = dt.mean()
    return (1.0 / m) if m > 0 else 50.0

def preprocess(df, fs):
    b, a = butter_highpass(0.25, fs)
    for c in ["ax", "ay", "az"]:
        df[c] = filtfilt(b, a, df[c].values)
    b2, a2 = butter_lowpass(20.0, fs)
    for c in ["gx", "gy", "gz"]:
        df[c] = filtfilt(b2, a2, df[c].values)
    df["a_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["g_mag"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    return df

def extract_features(seg, fs):
    feats = []; cols = ["ax","ay","az","gx","gy","gz","a_mag","g_mag"]
    for c in cols:
        x = seg[c].values
        feats += [
            x.mean(), x.std(), x.min(), x.max(),
            np.median(x), np.percentile(x,25), np.percentile(x,75),
            np.ptp(x),
            np.sum(np.abs(np.diff(x))) / (len(x)-1 + 1e-9),
            (np.count_nonzero((x[1:]*x[:-1]) < 0) / (len(x)-1 + 1e-9)),
        ]
    feats += [
        psd_band_power(seg["a_mag"].values, fs, 1.5, 3.5),
        psd_band_power(seg["a_mag"].values, fs, 3.5, 6.0),
    ]
    return np.array(feats, dtype=float)

# ======== Lectura flexible ========
def read_csv_flexible(path, cols_expected):
    """
    Renombra por posición: col0->t, col1/2/3->ejes.
    Fuerza numérico y elimina filas no numéricas.
    """
    df = pd.read_csv(path)
    if len(df.columns) < 4:
        raise ValueError(f"{path} no tiene 4 columnas (t + 3 ejes). Tiene {len(df.columns)}.")
    df = df.rename(columns={
        df.columns[0]: "t",
        df.columns[1]: cols_expected[0],
        df.columns[2]: cols_expected[1],
        df.columns[3]: cols_expected[2],
    })
    df = df[["t"] + cols_expected]
    # Forzar a numérico
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df

# ======== Main ========
def main(args):
    # Cargar modelo y config
    model_data = load(args.model)
    pipe = model_data["pipe"]
    label_names = model_data["label_names"]
    win_sec = model_data["cfg"]["win_sec"]
    hop_frac = model_data["cfg"]["hop_frac"]

    # Leer CSVs
    a = read_csv_flexible(args.acel,  ["ax","ay","az"])
    g = read_csv_flexible(args.giros, ["gx","gy","gz"])

    # Alinear por tiempo redondeado
    a["t_r"] = a["t"].round(3); g["t_r"] = g["t"].round(3)
    df = pd.merge_asof(
        a.sort_values("t_r"), g.sort_values("t_r"),
        on="t_r", direction="nearest", tolerance=0.002
    ).dropna()

    # 🔧 asegurar columna 't' tras el merge
    if "t_x" in df.columns:
        df = df.rename(columns={"t_x": "t"})
    elif "t" not in df.columns:
        df["t"] = df["t_r"]

    # Quedarnos con lo necesario
    keep_cols = ["t","ax","ay","az","gx","gy","gz"]
    df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

    if len(df) < 10:
        raise SystemExit("Muy pocos datos tras la alineación. Revisa tolerancia/redondeo o los CSV.")

    # Preprocesar
    fs = estimate_fs(df["t"].values)
    df = preprocess(df, fs)

    # Ventaneo según config del modelo
    win = int(win_sec * fs)
    hop = int(win * hop_frac)
    if win <= 1 or hop < 1:
        raise SystemExit(f"Ventana inválida: win={win}, hop={hop}. Revisa win_sec/hop_frac en el modelo.")

    preds, probs, t_starts = [], [], []

    # Recorrer ventanas
    for i0 in range(0, len(df) - win + 1, hop):
        seg = df.iloc[i0:i0+win]
        X = extract_features(seg, fs).reshape(1, -1)
        proba = pipe.predict_proba(X)[0]
        k = int(np.argmax(proba))
        preds.append(k)
        probs.append(float(proba[k]))
        t_starts.append(float(seg["t"].iloc[0]))
        print(f"{t_starts[-1]:7.2f}s -> {label_names[k]:8s} ({probs[-1]:.2f})")

    # Exportar resultados
    if args.out:
        pd.DataFrame({
            "time_s": t_starts,
            "pred": [label_names[i] for i in preds],
            "prob": probs
        }).to_csv(args.out, index=False)
        print(f"\nResultados guardados en {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test modelo multiclase con un par de CSV.")
    p.add_argument("--model", type=str, default="modelo_multiclase.joblib")
    p.add_argument("--acel", type=str, required=True)
    p.add_argument("--giros", type=str, required=True)
    p.add_argument("--out", type=str, default="resultados_test.csv")
    args = p.parse_args()
    main(args)

# python test_multiclase.py --acel data/escal_acel_pedro.csv --giros data/escal_giros_pedro.csv --out resultados_escal.csv
