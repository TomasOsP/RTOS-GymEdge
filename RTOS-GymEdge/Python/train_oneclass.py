import glob, os, argparse
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, rfftfreq
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def butter_highpass(x, fs, cutoff=0.3, order=2):
    b,a = butter(order, cutoff/(fs/2), btype='high')
    return filtfilt(b,a,x)

def feat_basic(x, fs):
    f = {}
    f['mean']=x.mean(); f['std']=x.std(); f['rms']=np.sqrt((x**2).mean())
    f['absmean']=np.abs(x).mean(); f['min']=x.min(); f['max']=x.max()
    f['iqr']=np.subtract(*np.percentile(x,[75,25]))
    X = rfft(x); freqs = rfftfreq(len(x), 1/fs); m = freqs<=6.0
    if m.any() and np.any(np.abs(X[m])>0):
        idx = np.argmax(np.abs(X[m]))
        f['f_peak']=float(freqs[m][idx]); f['a_peak']=float(np.abs(X[m][idx]))
    else:
        f['f_peak']=0.0; f['a_peak']=0.0
    return f

def window_features(df, fs, win_s=1.2, step_s=0.6):
    Nw, Ns = int(win_s*fs), int(step_s*fs)
    X = []
    for i in range(0, len(df)-Nw+1, Ns):
        seg = df.iloc[i:i+Nw].copy()
        # quitar gravedad
        for c in ['ax','ay','az']:
            seg[c] = butter_highpass(seg[c].values, fs, cutoff=0.3)
        # magnitudes
        seg['a_mag'] = np.sqrt((seg[['ax','ay','az']]**2).sum(axis=1))
        seg['g_mag'] = np.sqrt((seg[['gx','gy','gz']]**2).sum(axis=1))
        feats = {}
        for c in ['ax','ay','az','gx','gy','gz','a_mag','g_mag']:
            feats |= {f'{c}_{k}':v for k,v in feat_basic(seg[c].values, fs).items()}
        # correlaciones
        for u,v in [('ax','ay'),('ax','az'),('ay','az'),('gx','gy'),('gx','gz'),('gy','gz')]:
            feats[f'corr_{u}_{v}'] = float(np.corrcoef(seg[u], seg[v])[0,1])
        X.append(feats)
    return pd.DataFrame(X)

def load_and_concat(csv_paths):
    dfs=[]
    for p in csv_paths:
        df=pd.read_csv(p)
        need = {'timestamp','ax','ay','az','gx','gy','gz'}
        assert need.issubset(df.columns), f"Faltan columnas en {p}"
        dfs.append(df[list(need)].sort_values('timestamp').reset_index(drop=True))
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Patrón de archivos CSV positivos (p.ej. data/pos_*.csv)")
    ap.add_argument("--fs", type=int, default=50, help="Frecuencia de muestreo (Hz)")
    ap.add_argument("--nu", type=float, default=0.1, help="Fracción de outliers esperada")
    ap.add_argument("--model_out", default="modelo.pkl")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    assert files, "No se encontraron CSV. Revisa el patrón --glob"
    df = load_and_concat(files)
    X = window_features(df, fs=args.fs, win_s=1.2, step_s=0.6)
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0.0)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", OneClassSVM(kernel="rbf", gamma="scale", nu=args.nu))
    ])
    pipe.fit(X.values)

    joblib.dump({"pipeline": pipe, "fs": args.fs,
                 "win_s": 1.2, "step_s": 0.6, "feature_cols": list(X.columns)}, args.model_out)
    print(f"Modelo guardado en {args.model_out}. Ventanas entrenadas: {len(X)}")
