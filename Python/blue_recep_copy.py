# gym_live_dashboard_web.py
# Versión limpia y funcional: FastAPI + BLE + WebSocket + inferencia (si existe el modelo)

import time
import threading
import asyncio
from pathlib import Path
from collections import deque
from threading import Lock
from datetime import datetime
import json
import sys

import numpy as np
import pandas as pd
from joblib import load

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from bleak import BleakScanner, BleakClient

# DSP
from scipy.signal import butter, filtfilt, welch, lfilter
from scipy.integrate import trapezoid

# ---------------- Config ----------------
HOST = "0.0.0.0"
PORT = 8000

BLE_DEVICE_NAME = "ESP32-MPU-Mac"
TX_CHAR_UUID = "abcdef12-3456-7890-abcd-ef1234567890"
MODEL_PATH = Path(__file__).with_name("modelo_multiclase.joblib")

WINDOW_SEC = 10
SAMPLE_HZ_GUESS = 50.0
SAMPLE_HZ = 50.0
MAX_POINTS = int(60 * SAMPLE_HZ)

# ---------------- Globals / buffers ----------------
buf_lock = Lock()
t_buf   = deque(maxlen=MAX_POINTS)
ax_buf  = deque(maxlen=MAX_POINTS)
ay_buf  = deque(maxlen=MAX_POINTS)
az_buf  = deque(maxlen=MAX_POINTS)
gx_buf  = deque(maxlen=MAX_POINTS)
gy_buf  = deque(maxlen=MAX_POINTS)
gz_buf  = deque(maxlen=MAX_POINTS)

rx_text_buffer = ""
rx_lock = Lock()

is_logging = False
program_exit = False
t0_millis = None

csv_file = None
csv_writer = None

model_pipe = None
label_names = None
model_win_sec = 2.0
model_hop_frac = 0.5

G = 9.80665
RAD = np.pi / 180.0
HEADERS = ["timestamp_ms","ax_g","ay_g","az_g","gx_dps","gy_dps","gz_dps"]

# WS clients
ws_connections = set()
ws_lock = asyncio.Lock()

# Metrics
packet_count = 0
last_fps_time = time.time()
current_fps = 0
last_device_ts_ms = None
last_recv_time = None

# Simple rep counter
reps = 0
last_a_mag = 0.0
rep_threshold = 1.35
last_rep_time = 0.0

# ---------------- Utility / DSP functions ----------------
def _norm_wn(cut, fs):
    if fs is None or fs <= 0:
        fs = SAMPLE_HZ_GUESS
    nyq = fs / 2.0
    wn = cut / nyq if nyq > 0 else 0.5
    return float(max(min(wn, 0.99), 1e-6))

def butter_highpass(cut, fs, order=4):
    wn = _norm_wn(cut, fs)
    return butter(order, wn, btype="highpass")

def butter_lowpass(cut, fs, order=4):
    wn = _norm_wn(cut, fs)
    return butter(order, wn, btype="lowpass")

def filtfilt_safe(b, a, x):
    x = np.asarray(x)
    if x.size < 3:
        return x.copy()
    padlen = 3 * (max(len(a), len(b)) - 1)
    if x.size <= padlen:
        return lfilter(b, a, x)
    return filtfilt(b, a, x)

def estimate_fs(t):
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return SAMPLE_HZ
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt>0)]
    if dt.size == 0:
        return SAMPLE_HZ
    m = np.median(dt)
    return 1.0 / m if m > 0 else SAMPLE_HZ

def preprocess_df(df, fs, hp_cut=0.25, lp_cut=20.0):
    try:
        b,a  = butter_highpass(hp_cut, fs)
        b2,a2= butter_lowpass(lp_cut, fs)
        for c in ["ax","ay","az"]:
            df[c] = filtfilt_safe(b, a, df[c].values)
        for c in ["gx","gy","gz"]:
            df[c] = filtfilt_safe(b2, a2, df[c].values)
    except Exception:
        pass
    df["a_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["g_mag"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    return df

def psd_band_power(x, fs, flo, fhi):
    if len(x) < 4:
        return 0.0
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    m = (f >= flo) & (f <= fhi)
    return float(trapezoid(Pxx[m], f[m]) if m.any() else 0.0)

def extract_features(seg, fs):
    feats = []
    cols = ["ax","ay","az","gx","gy","gz","a_mag","g_mag"]
    for c in cols:
        x = seg[c].values
        if len(x) == 0:
            feats += [0.0]*10
            continue
        feats += [
            float(x.mean()), float(x.std()), float(x.min()), float(x.max()),
            float(np.median(x)), float(np.percentile(x,25)), float(np.percentile(x,75)),
            float(np.ptp(x)),
            float(np.sum(np.abs(np.diff(x))) / (len(x)-1 + 1e-9)),
            float(np.count_nonzero((x[1:]*x[:-1])<0) / (len(x)-1 + 1e-9))
        ]
    feats += [
        psd_band_power(seg["a_mag"].values, fs, 1.5, 3.5),
        psd_band_power(seg["a_mag"].values, fs, 3.5, 6.0),
    ]
    return np.array(feats, dtype=float)

# ---------------- Prediction Top-K ----------------
def predict_topk_now(k_top=3):
    global model_pipe, label_names, model_win_sec

    if model_pipe is None or label_names is None:
        return None, None, "Modelo no cargado."

    with buf_lock:
        t_vals  = np.array(t_buf, dtype=float)
        ax_vals = np.array(ax_buf, dtype=float)
        ay_vals = np.array(ay_buf, dtype=float)
        az_vals = np.array(az_buf, dtype=float)
        gx_vals = np.array(gx_buf, dtype=float)
        gy_vals = np.array(gy_buf, dtype=float)
        gz_vals = np.array(gz_buf, dtype=float)

    if t_vals.size < 3:
        return None, None, "Muy pocos datos en buffer."

    fs = estimate_fs(t_vals)
    win = int(max(1, round(model_win_sec * fs)))

    if t_vals.size < win:
        return None, None, f"Faltan muestras para ventana de {model_win_sec:.1f}s."

    i1 = t_vals.size
    i0 = i1 - win

    seg = pd.DataFrame({
        "t":  t_vals[i0:i1],
        "ax": ax_vals[i0:i1],
        "ay": ay_vals[i0:i1],
        "az": az_vals[i0:i1],
        "gx": gx_vals[i0:i1],
        "gy": gy_vals[i0:i1],
        "gz": gz_vals[i0:i1],
    })

    seg = preprocess_df(seg, fs)
    X = extract_features(seg, fs).reshape(1, -1)

    try:
        proba = model_pipe.predict_proba(X)[0]
    except Exception as e:
        return None, None, f"predict_proba error: {e}"

    idxs = np.argsort(proba)[::-1][:k_top]
    top = [(label_names[int(i)], float(proba[int(i)])) for i in idxs]

    return float(seg["t"].iloc[0]), top, None

# ---------------- RX parsing (BLE notifications) ----------------
def process_records(text_chunk: str):
    global rx_text_buffer, t0_millis, csv_writer, is_logging
    global packet_count, last_fps_time, current_fps, last_device_ts_ms, last_recv_time
    global reps, last_a_mag, last_rep_time

    with rx_lock:
        rx_text_buffer += text_chunk
        parts = rx_text_buffer.split(';')
        rx_text_buffer = parts[-1]

    for rec in parts[:-1]:
        rec = rec.strip().replace('\r','').replace('\n','')
        if not rec:
            continue

        cols = rec.split(',')
        if len(cols) != 7:
            continue

        if is_logging and csv_writer:
            try:
                csv_writer.writerow(cols)
                csv_file.flush()
            except Exception:
                pass

        try:
            ts_ms = int(cols[0])
            ax_g, ay_g, az_g = map(float, cols[1:4])
            gx_d, gy_d, gz_d = map(float, cols[4:7])

            ax = ax_g * G
            ay = ay_g * G
            az = az_g * G
            gx = gx_d * RAD
            gy = gy_d * RAD
            gz = gz_d * RAD
        except Exception:
            continue

        if t0_millis is None:
            t0_millis = ts_ms
        t_rel = (ts_ms - t0_millis) / 1000.0

        with buf_lock:
            t_buf.append(t_rel)
            ax_buf.append(ax); ay_buf.append(ay); az_buf.append(az)
            gx_buf.append(gx); gy_buf.append(gy); gz_buf.append(gz)

        now = time.time()
        packet_count += 1
        last_recv_time = now
        last_device_ts_ms = ts_ms

        if now - last_fps_time >= 1.0:
            current_fps = packet_count
            packet_count = 0
            last_fps_time = now

            # 1) Low-pass para suavizar
            a_mag = (ax*ax + ay*ay + az*az)**0.5
            a_mag_f = 0.8 * last_a_mag + 0.2 * a_mag  # filtro IIR simple

            # 2) detección de reps basada en picos
            if a_mag_f > rep_threshold and last_a_mag <= rep_threshold:
                if now - last_rep_time > 0.35:   # evita doble conteo
                    reps += 1
                    last_rep_time = now

            last_a_mag = a_mag_f

        if a_mag > rep_threshold and last_a_mag <= rep_threshold:
            if now - last_rep_time > 0.35:
                reps += 1
                last_rep_time = now
        last_a_mag = a_mag

# ---------------- BLE handling ----------------
async def ble_main():
    print("BLE: buscando periféricos (5s)...")
    try:
        devices = await BleakScanner.discover(timeout=5.0)
    except Exception as e:
        print("BLE discovery error:", e)
        return

    device = next((d for d in devices if d.name == BLE_DEVICE_NAME), None)
    if device is None:
        print(f"BLE: no se encontró {BLE_DEVICE_NAME}")
        return

    print(f"BLE: conectando a {device.name} ({device.address})")

    try:
        async with BleakClient(device) as client:
            if not client.is_connected:
                print("BLE: no conectado")
                return

            print("BLE: conectado, suscribiendo notify...")

            def handle_notify(_, data: bytearray):
                try:
                    chunk = data.decode("utf-8", errors="ignore")
                    process_records(chunk)
                except Exception:
                    pass

            await client.start_notify(TX_CHAR_UUID, handle_notify)
            print("BLE: suscrito. Enviando datos...")

            while not program_exit and client.is_connected:
                await asyncio.sleep(0.2)

            try:
                await client.stop_notify(TX_CHAR_UUID)
            except Exception:
                pass

    except Exception as e:
        print("BLE connection error:", e)

    print("BLE: terminado")

# ---------------- FastAPI app & templates ----------------
BASE = Path(__file__).resolve().parent
app = FastAPI()

app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})
# ---------------- WebSocket ----------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    async with ws_lock:
        ws_connections.add(ws)

    try:
        await ws.send_text(json.dumps({"type": "info", "text": "Conectado al servidor"}))
        while True:
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    finally:
        async with ws_lock:
            ws_connections.discard(ws)

# ---------------- broadcaster ----------------
async def _broadcast(text: str):
    async with ws_lock:
        conns = list(ws_connections)

    dead = []
    for ws in conns:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)

    if dead:
        async with ws_lock:
            for d in dead:
                ws_connections.discard(d)

async def broadcaster_loop():
    global current_fps, last_device_ts_ms, last_recv_time, reps

    while not program_exit:

        with buf_lock:
            if len(t_buf) == 0:
                sample = None
            else:
                sample = {
                    "t": float(t_buf[-1]),
                    "ax": float(ax_buf[-1]),
                    "ay": float(ay_buf[-1]),
                    "az": float(az_buf[-1]),
                    "gx": float(gx_buf[-1]),
                    "gy": float(gy_buf[-1]),
                    "gz": float(gz_buf[-1]),
                }
                sample["a_mag"] = (sample["ax"]**2 + sample["ay"]**2 + sample["az"]**2)**0.5
                sample["g_mag"] = (sample["gx"]**2 + sample["gy"]**2 + sample["gz"]**2)**0.5

        if sample is not None:
            msg = {"type": "sample", **sample}
            await _broadcast(json.dumps(msg))

        t0, top, err = predict_topk_now()
        if err is None and top:
            best_label, best_prob = top[0]
            msgp = {"type":"pred", "best_label": best_label, "best_prob": best_prob, "topk": top}
        else:
            msgp = {"type":"pred", "best_label": None, "best_prob": 0.0, "topk": []}
        await _broadcast(json.dumps(msgp))

        now = time.time()
        latency_ms = int(now*1000.0 - last_device_ts_ms) if last_device_ts_ms else 0

        status = {
            "type": "status",
            "fps": int(current_fps),
            "loss": 0.0,
            "latency_ms": latency_ms,
            "reps": int(reps)
        }
        await _broadcast(json.dumps(status))

        await asyncio.sleep(0.1)

# ---------------- startup / threads ----------------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcaster_loop())

    def run_ble():
        asyncio.run(ble_main())

    t = threading.Thread(target=run_ble, daemon=True)
    t.start()

    print("Servidor: broadcaster y BLE iniciados.")

# ---------------- CSV helpers ----------------
def init_user_folder_interactive():
    try:
        user = input("Ingrese su nombre de usuario: ").strip()
    except EOFError:
        user = "usuario"

    if not user:
        user = "usuario"

    base_dir = Path("data") / user
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return user, base_dir, base_dir / f"{ts}.csv"

def open_csv(csv_path):
    global csv_file, csv_writer, is_logging
    try:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        import csv as _csv
        csv_writer = _csv.writer(csv_file)
        csv_writer.writerow(HEADERS)
        csv_file.flush()
        print("CSV opened:", csv_path)
        is_logging = True
    except Exception as e:
        print("CSV open failed:", e)
        csv_file = None
        csv_writer = None
        is_logging = False

def close_csv():
    global csv_file
    try:
        if csv_file:
            csv_file.flush()
            csv_file.close()
    except Exception:
        pass

# ---------------- main launcher ----------------
def main():
    global program_exit, model_pipe, label_names, model_win_sec, model_hop_frac

    if MODEL_PATH.exists():
        try:
            md = load(MODEL_PATH)
            model_pipe = md.get("pipe", None)
            label_names = md.get("label_names", None)
            cfg = md.get("cfg", {})
            model_win_sec = float(cfg.get("win_sec", model_win_sec))
            model_hop_frac = float(cfg.get("hop_frac", model_hop_frac))
            print(f"Modelo cargado. Ventana={model_win_sec:.1f}s hop={model_hop_frac:.2f}")
        except Exception as e:
            print("Model load error:", e)
            model_pipe = None
    else:
        print("No se encontró modelo_multiclase.joblib — ejecutando solo gráficas.")

    print("=== SISTEMA DE RECEPCIÓN (web) ===")
    user_name, user_dir, csv_path = init_user_folder_interactive()
    open_csv(csv_path)

    print(f"Iniciando servidor en http://{HOST}:{PORT} ...")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

    program_exit = True
    close_csv()
    print("Servidor detenido.")

if __name__ == "__main__":
    main()
