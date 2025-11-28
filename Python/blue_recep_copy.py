# gym_live_dashboard_web.py
# Servidor web + BLE + inferencia en tiempo real
# Servido en http://0.0.0.0:8000 (puerto elegido: 8000)

import time
import threading
import asyncio
from pathlib import Path
from collections import deque
from threading import Lock
from datetime import datetime
import json
import sys
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

import numpy as np
import pandas as pd
from joblib import load

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from bleak import BleakScanner, BleakClient

# DSP
from scipy.signal import butter, filtfilt, welch, lfilter
from scipy.integrate import trapezoid

# ================= Configuration =================
HOST = "0.0.0.0"
PORT = 8000

BLE_DEVICE_NAME = "ESP32-MPU-Mac"
TX_CHAR_UUID = "abcdef12-3456-7890-abcd-ef1234567890"
MODEL_PATH = Path(__file__).with_name("modelo_multiclase.joblib")

WINDOW_SEC = 10
SAMPLE_HZ_GUESS = 50.0
SAMPLE_HZ = 50.0
MAX_POINTS = int(60 * SAMPLE_HZ)

# Buffers and locks
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

# States
is_logging = False
program_exit = False
t0_millis = None

# CSV
csv_file = None
csv_writer = None

# Model
model_pipe = None
label_names = None
model_win_sec = 2.0
model_hop_frac = 0.5

# Constants
G = 9.80665
RAD = np.pi / 180.0
HEADERS = ["timestamp_ms","ax_g","ay_g","az_g","gx_dps","gy_dps","gz_dps"]

# FastAPI app
app = FastAPI()
# connected websockets
ws_connections = set()
ws_lock = asyncio.Lock()

# ================= DSP / features (same as your code) ================
def _norm_wn(cut, fs):
    if fs is None or fs <= 0:
        fs = SAMPLE_HZ_GUESS
    nyq = fs / 2.0
    wn = cut / nyq if nyq > 0 else 0.5
    return float(max(min(wn, 0.99), 1e-6))

def butter_highpass(cut, fs, order=4):
    wn = _norm_wn(cut, fs)
    b, a = butter(order, wn, btype="highpass")
    return b, a

def butter_lowpass(cut, fs, order=4):
    wn = _norm_wn(cut, fs)
    b, a = butter(order, wn, btype="lowpass")
    return b, a

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

# ================= Prediction Top-K =================
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
        return None, None, f"No hay muestras suficientes para {model_win_sec:.1f}s (win={win})."
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
    t0 = float(seg["t"].iloc[0])
    return t0, top, None

# ================= RX Parsing =================
def process_records(text_chunk: str):
    global rx_text_buffer, t0_millis, csv_writer, is_logging
    with rx_lock:
        rx_text_buffer += text_chunk
        parts = rx_text_buffer.split(';')
        rx_text_buffer = parts[-1]
    for rec in parts[:-1]:
        rec = rec.strip().replace('\r','').replace('\n','')
        if not rec: continue
        cols = rec.split(',')
        if len(cols) != 7:
            continue
        try:
            if is_logging and csv_writer:
                csv_writer.writerow(cols)
                csv_file.flush()
        except Exception:
            pass
        try:
            ts_ms = int(cols[0])
            ax_g = float(cols[1]); ay_g = float(cols[2]); az_g = float(cols[3])
            gx_d = float(cols[4]); gy_d = float(cols[5]); gz_d = float(cols[6])
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

# ================= BLE =================
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
                except Exception:
                    return
                process_records(chunk)
            await client.start_notify(TX_CHAR_UUID, handle_notify)
            print("BLE: suscrito. Enviando datos al dashboard...")
            while not program_exit:
                await asyncio.sleep(0.2)
            try:
                await client.stop_notify(TX_CHAR_UUID)
            except Exception:
                pass
    except Exception as e:
        print("BLE connection error:", e)
    print("BLE: terminado")

# ============== FastAPI: pages & ws ===================
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>GymEdge Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial; margin: 12px; background:#f7f7f8; }
    .row { display:flex; gap:12px; }
    .card { background:white; padding:12px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.08); flex:1; }
    canvas { width:100%; height:240px; }
    h2 { margin:6px 0 10px 0; font-size:18px; }
    .pred { font-size:18px; font-weight:600; }
    .topk { margin-top:8px; font-size:14px; }
  </style>
</head>
<body>
  <h1>GymEdge Live Dashboard</h1>
  <div class="row">
    <div class="card" style="flex:2">
      <h2>Acelerómetro</h2>
      <canvas id="accChart"></canvas>
    </div>
    <div class="card" style="flex:2">
      <h2>Giroscopio</h2>
      <canvas id="gyroChart"></canvas>
    </div>
    <div class="card" style="flex:1">
      <h2>Predicción</h2>
      <div id="pred" class="pred">Esperando datos...</div>
      <div id="topk" class="topk"></div>
      <hr/>
      <div id="info"></div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const ws = new WebSocket("ws://" + location.host + "/ws");
    const maxPoints = 200;

    function makeChart(ctx, labels, colors) {
      return new Chart(ctx, {
        type: 'line',
        data: { datasets: labels.map((lab, i) => ({
            label: lab,
            data: [],
            borderColor: colors[i],
            fill: false,
            tension: 0.1,
            pointRadius: 0
        })) },
        options: {
          animation: false,
          parsing: false,
          normalized: true,
          scales: {
            x: { type: 'linear', title: {display:true, text:'t (s)'}},
            y: { title: {display:true, text: 'valor'}}
          }
        }
      });
    }

    const accChart = makeChart(document.getElementById('accChart').getContext('2d'),
      ['ax','ay','az','|a|'], ['#1f77b4','#ff7f0e','#2ca02c','#000000']);

    const gyroChart = makeChart(document.getElementById('gyroChart').getContext('2d'),
      ['gx','gy','gz','|g|'], ['#1f77b4','#ff7f0e','#2ca02c','#000000']);

    function pushPoint(chart, t, values){
      values.forEach((v,i)=>{
        const ds = chart.data.datasets[i];
        ds.data.push({x: t, y: v});
        if (ds.data.length > maxPoints) ds.data.shift();
      });
      chart.update('none');
    }

    ws.onopen = () => {
      console.log("WS conectado");
      document.getElementById('info').innerText = "WS conectado a servidor.";
    };
    ws.onclose = () => {
      console.log("WS desconectado");
      document.getElementById('info').innerText = "WS desconectado.";
    };
    ws.onerror = (e) => console.warn("WS err", e);

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'sample') {
          const t = msg.t;
          pushPoint(accChart, t, [msg.ax, msg.ay, msg.az, msg.a_mag]);
          pushPoint(gyroChart, t, [msg.gx, msg.gy, msg.gz, msg.g_mag]);
        } else if (msg.type === 'pred') {
          document.getElementById('pred').innerText = msg.best_label ? (msg.best_label.toUpperCase() + " (" + (msg.best_prob*100).toFixed(1) + "%)") : "N/D";
          let html = "";
          if (msg.topk){
            msg.topk.forEach((p,i)=>{
              html += `${i+1}) ${p[0]} — ${(p[1]*100).toFixed(1)}%<br/>`;
            });
          }
          document.getElementById('topk').innerHTML = html;
        } else if (msg.type === 'info'){
          document.getElementById('info').innerText = msg.text;
        }
      } catch(e){
        console.warn("Bad WS msg", e);
      }
    };
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    async with ws_lock:
        ws_connections.add(ws)
    try:
        # send a small welcome/info
        await ws.send_text(json.dumps({"type":"info","text":"Conectado al servidor"}))
        while True:
            # Keep connection alive waiting for client messages (we don't expect any)
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                # ignore or could handle commands from client
                await ws.send_text(json.dumps({"type":"info","text":"ok"}))
            except asyncio.TimeoutError:
                # ping periodically (no-op) to keep connection
                await ws.send_text(json.dumps({"type":"info","text":"ping"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with ws_lock:
            if ws in ws_connections:
                ws_connections.remove(ws)

# Background broadcaster : reads buffers and sends JSON to all websockets
async def broadcaster_loop():
    """Task started on app startup that periodically reads the latest sample
    and prediction and broadcasts to all connected websockets."""
    while not program_exit:
        # snapshot latest sample
        with buf_lock:
            if len(t_buf) == 0:
                sample = None
            else:
                sample = {
                    "t": float(t_buf[-1]),
                    "ax": float(ax_buf[-1]) if len(ax_buf)>0 else 0.0,
                    "ay": float(ay_buf[-1]) if len(ay_buf)>0 else 0.0,
                    "az": float(az_buf[-1]) if len(az_buf)>0 else 0.0,
                    "gx": float(gx_buf[-1]) if len(gx_buf)>0 else 0.0,
                    "gy": float(gy_buf[-1]) if len(gy_buf)>0 else 0.0,
                    "gz": float(gz_buf[-1]) if len(gz_buf)>0 else 0.0,
                }
                sample["a_mag"] = (sample["ax"]**2 + sample["ay"]**2 + sample["az"]**2)**0.5
                sample["g_mag"] = (sample["gx"]**2 + sample["gy"]**2 + sample["gz"]**2)**0.5

        # send sample
        if sample is not None:
            msg = json.dumps({"type":"sample", **sample})
            await _broadcast(msg)

        # prediction (top1 + top3)
        t0, top, err = predict_topk_now(k_top=3)
        if err is None and top:
            best_label, best_prob = top[0]
            msgp = json.dumps({"type":"pred", "best_label":best_label, "best_prob":best_prob, "topk": top})
        else:
            msgp = json.dumps({"type":"pred", "best_label": None, "best_prob":0.0, "topk": []})
        await _broadcast(msgp)

        await asyncio.sleep(0.05)  # ~20 Hz update to clients

async def _broadcast(text: str):
    # send to all websocket connections (copy to avoid mutation while iterating)
    async with ws_lock:
        conns = list(ws_connections)
    for ws in conns:
        try:
            await ws.send_text(text)
        except Exception:
            # drop broken connection
            try:
                async with ws_lock:
                    if ws in ws_connections:
                        ws_connections.remove(ws)
            except Exception:
                pass

# FastAPI startup/shutdown events
@app.on_event("startup")
async def startup_event():
    # start broadcaster task
    asyncio.create_task(broadcaster_loop())
    print("Web server: broadcaster started.")

# ================ Threads to run BLE and optionally CSV ================
def init_user_folder_interactive():
    try:
        user = input("Ingrese su nombre de usuario: ").strip()
    except EOFError:
        user = ""
    if not user:
        user = "usuario"
    base_dir = Path("data") / user
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file_path = base_dir / f"{ts}.csv"
    return user, base_dir, csv_file_path

def start_ble_thread():
    # run asyncio event loop for bleak in separate thread
    def run():
        asyncio.run(ble_main())
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t

def open_csv(csv_path):
    global csv_file, csv_writer
    try:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        import csv as _csv
        csv_writer = _csv.writer(csv_file)
        csv_writer.writerow(HEADERS)
        csv_file.flush()
        print("CSV opened:", csv_path)
    except Exception as e:
        print("CSV open failed:", e)
        csv_file = None
        csv_writer = None

def close_csv():
    global csv_file
    try:
        if csv_file:
            csv_file.flush()
            csv_file.close()
    except Exception:
        pass

# --------------- main launcher ---------------
def main():
    global is_logging, program_exit, model_pipe, label_names, model_win_sec, model_hop_frac

    # load model if available
    if MODEL_PATH.exists():
        try:
            md = load(MODEL_PATH)
            model_pipe = md.get("pipe", None)
            label_names = md.get("label_names", None)
            cfg = md.get("cfg", {})
            if "win_sec" in cfg:
                model_win_sec = float(cfg["win_sec"])
            if "hop_frac" in cfg:
                model_hop_frac = float(cfg["hop_frac"])
            print(f"Modelo cargado. Ventana={model_win_sec:.1f}s hop={model_hop_frac:.2f}")
        except Exception as e:
            print("Model load error:", e)
            model_pipe = None
    else:
        print("No se encontró modelo_multiclase.joblib — ejecutando solo gráficas.")

    # init csv folder
    print("=== SISTEMA DE RECEPCIÓN (web) ===")
    user_name, user_dir, csv_path = init_user_folder_interactive()
    open_csv(csv_path)
    is_logging = True

    # start BLE thread
    start_ble_thread()

    # start uvicorn (this blocks)
    print(f"Iniciando servidor en http://{HOST}:{PORT} ...")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

    # on exit
    program_exit = True
    close_csv()
    print("Servidor detenido.")

if __name__ == "__main__":
    main()
