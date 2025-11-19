# -*- coding: utf-8 -*-
# === CODIGO RECEPCION + INFERENCIA CONTINUA (tecla 'd' = Top-3 cada 1s) ===

import sys
import csv
import time
import threading
import asyncio
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

# === BLE ===
from bleak import BleakScanner, BleakClient

# === Graficación ===
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# === ML ===
from joblib import load

# === DSP ===
from scipy.signal import butter, filtfilt, welch, lfilter
from scipy.integrate import trapezoid

# =================== CONFIGURACIÓN ===================
BLE_DEVICE_NAME = "ESP32-MPU-Mac"   # <-- tu nombre de periférico BLE
UART_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
TX_CHAR_UUID      = "abcdef12-3456-7890-abcd-ef1234567890"   # NOTIFY (la del sketch)

CSV_NAME = f"mpu_log_{int(time.time())}.csv"

# Ventana de datos visibles en la gráfica (segundos)
WINDOW_SEC = 30
SAMPLE_HZ = 5.0
MAX_POINTS = int(WINDOW_SEC * SAMPLE_HZ * 2)
# =====================================================

HEADERS = ["timestamp_ms","ax_g","ay_g","az_g","gx_dps","gy_dps","gz_dps"]

# ----- Estado de control -----
is_logging = False
plotting_active = False
program_exit = False
start_plot_event = threading.Event()

# ----- Modelo (globales) -----
MODEL_PATH = Path(__file__).with_name("modelo_multiclase.joblib")
model_pipe = None
label_names = None
model_win_sec = 2.0
model_hop_frac = 0.5

# ----- Diagnóstico continuo (Top-3 cada 1s) -----
diag_active = False
diag_thread = None

# Buffers (con lock)
from threading import Lock
buf_lock = Lock()
t_buf   = deque(maxlen=MAX_POINTS)
ax_buf  = deque(maxlen=MAX_POINTS)
ay_buf  = deque(maxlen=MAX_POINTS)
az_buf  = deque(maxlen=MAX_POINTS)
gx_buf  = deque(maxlen=MAX_POINTS)
gy_buf  = deque(maxlen=MAX_POINTS)
gz_buf  = deque(maxlen=MAX_POINTS)

t0_millis = None

csv_file = None
csv_writer = None
csv_path = None

# Buffer texto para rearmar paquetes ';'
rx_text_buffer = ""
rx_lock = Lock()

# ------------------ Utilidades de señal / features ------------------
def _norm_wn(cut, fs):
    if fs is None or fs <= 0:
        fs = 50.0
    nyq = fs / 2.0
    wn = cut / nyq if nyq > 0 else 0.5
    return float(max(min(wn, 0.99), 1e-3))

def butter_highpass(cut, fs, order=4):
    wn = _norm_wn(cut, fs)
    b, a = butter(order, wn, btype="highpass"); return b, a

def butter_lowpass(cut, fs, order=4):
    wn = _norm_wn(cut, fs)
    b, a = butter(order, wn, btype="lowpass");  return b, a

def filtfilt_safe(b, a, x):
    x = np.asarray(x)
    if x.size < 3:
        return x.copy()
    padlen = 3 * (max(len(a), len(b)) - 1)
    if x.size <= padlen:
        pl = max(0, min(padlen, x.size - 2))
        if pl == 0:
            return lfilter(b, a, x)
        return filtfilt(b, a, x, padlen=pl)
    return filtfilt(b, a, x)

def estimate_fs(t):
    t = np.asarray(t)
    if t.size < 2: return 50.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0: return 50.0
    m = np.median(dt)  # robusto
    return (1.0/m) if m > 0 else 50.0

def preprocess_df(df, fs, hp_cut=0.25, lp_cut=20.0):
    b,a  = butter_highpass(hp_cut, fs)
    b2,a2= butter_lowpass(lp_cut, fs)
    for c in ["ax","ay","az"]:
        df[c] = filtfilt_safe(b, a, df[c].values)
    for c in ["gx","gy","gz"]:
        df[c] = filtfilt_safe(b2, a2, df[c].values)
    df["a_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["g_mag"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    return df

def psd_band_power(x, fs, flo, fhi):
    f, Pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    m = (f>=flo) & (f<=fhi)
    return float(trapezoid(Pxx[m], f[m]) if m.any() else 0.0)

def extract_features(seg, fs):
    feats=[]; cols=["ax","ay","az","gx","gy","gz","a_mag","g_mag"]
    for c in cols:
        x = seg[c].values
        feats += [
            x.mean(), x.std(), x.min(), x.max(),
            np.median(x), np.percentile(x,25), np.percentile(x,75),
            np.ptp(x),
            np.sum(np.abs(np.diff(x))) / (len(x)-1 + 1e-9),
            (np.count_nonzero((x[1:]*x[:-1])<0) / (len(x)-1 + 1e-9))
        ]
    feats += [
        psd_band_power(seg["a_mag"].values, fs, 1.5, 3.5),
        psd_band_power(seg["a_mag"].values, fs, 3.5, 6.0),
    ]
    return np.array(feats, dtype=float)

# ------------------ Predicción (Top-3) ------------------
def predict_topk_now(k_top=3):
    """Toma la última ventana del buffer y devuelve Top-k (label, prob)."""
    if model_pipe is None or label_names is None:
        return None, None, "⚠️  Modelo no cargado."

    with buf_lock:
        t_vals  = np.array(t_buf, dtype=float)
        ax_vals = np.array(ax_buf, dtype=float)
        ay_vals = np.array(ay_buf, dtype=float)
        az_vals = np.array(az_buf, dtype=float)
        gx_vals = np.array(gx_buf, dtype=float)
        gy_vals = np.array(gy_buf, dtype=float)
        gz_vals = np.array(gz_buf, dtype=float)

    if t_vals.size < 5:
        return None, None, "⚠️  Muy pocos datos en buffer para predecir."

    fs = estimate_fs(t_vals)
    win = int(max(1, round(model_win_sec * fs)))
    if t_vals.size < win:
        return None, None, f"⚠️  Aún no hay muestras suficientes para {model_win_sec:.1f}s (win={win}, fs≈{fs:.1f}Hz)."

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
    proba = model_pipe.predict_proba(X)[0]  # shape (C,)

    # Top-k índices
    idxs = np.argsort(proba)[::-1][:k_top]
    top = [(label_names[i], float(proba[i])) for i in idxs]
    t0 = float(seg["t"].iloc[0])
    return t0, top, None

def diag_loop():
    """Imprime cada 1s el Top-3 mientras diag_active esté True."""
    global diag_active
    print("🟡 Diagnóstico continuo: Top-3 cada 1s (sal con 'x').")
    while diag_active and not program_exit:
        t0, top, err = predict_topk_now(k_top=3)
        if err:
            print(err)
        else:
            # Ejemplo de salida: [123.45s] 1) escaleras 0.92 | 2) lazo 0.05 | 3) remo 0.03
            pretty = " | ".join([f"{i+1}) {lbl} {prob:.2f}" for i,(lbl,prob) in enumerate(top)])
            print(f"[{t0:7.2f}s] {pretty}")
        time.sleep(1.0)
    print("🔵 Diagnóstico continuo detenido.")

# ------------------ Entrada por teclado ------------------
def user_input_thread():
    global is_logging, plotting_active, program_exit, csv_file
    global diag_active, diag_thread

    print("Controles: 's' = empezar/reanudar, 'q' = pausar, 'd' = Top-3 cada 1s, 'x' = salir.\n")
    while not program_exit:
        try:
            cmd = input().strip().lower()
        except EOFError:
            cmd = "x"

        if cmd == "s":
            if not plotting_active:
                plotting_active = True
                is_logging = True
                start_plot_event.set()
                try:
                    if csv_file:
                        csv_file.flush()
                except Exception:
                    pass
                print("▶️  REANUDADO: CSV + gráficas.")
            else:
                print("Ya estaba reproduciendo.")
        elif cmd == "q":
            if plotting_active or is_logging:
                plotting_active = False
                is_logging = False
                try:
                    if csv_file:
                        csv_file.flush()
                except Exception:
                    pass
                print("⏸️  PAUSA: sin cerrar ventanas.")
            else:
                print("Ya estaba en pausa.")
        elif cmd == "d":
            if diag_active:
                # Apagar
                diag_active = False
                print("⏹️  Diagnóstico continuo OFF.")
            else:
                # Encender
                diag_active = True
                if (diag_thread is None) or (not diag_thread.is_alive()):
                    diag_thread = threading.Thread(target=diag_loop, daemon=True)
                    diag_thread.start()
                print("▶️  Diagnóstico continuo ON (Top-3 cada 1s).")

        elif cmd == "x":
            program_exit = True
            diag_active = False
            print("👋 Saliendo…")
        else:
            print("Comando no reconocido. Usa 's', 'q', 'd' o 'x'.")

# ------------------ Graficador ------------------
def plotter_loop():
    start_plot_event.wait()
    if program_exit:
        return

    plt.ion()

    fig_acc, ax_acc = plt.subplots(num="Aceleraciones (g)")
    ax_acc.set_title("Aceleraciones — ax, ay, az (g)")
    ax_acc.set_xlabel("Tiempo (s)")
    ax_acc.set_ylabel("g")
    line_ax = Line2D([], [], label="ax_g", color='tab:blue', linewidth=1.8)
    line_ay = Line2D([], [], label="ay_g", color='tab:red',  linewidth=1.8)
    line_az = Line2D([], [], label="az_g", color='tab:green',linewidth=1.8)
    ax_acc.add_line(line_ax); ax_acc.add_line(line_ay); ax_acc.add_line(line_az)
    ax_acc.legend(loc="upper right", frameon=True, framealpha=0.85)
    ax_acc.grid(True, alpha=0.3)

    fig_gyro, ax_gyro = plt.subplots(num="Giroscopio (deg/s)")
    ax_gyro.set_title("Giroscopio — gx, gy, gz (deg/s)")
    ax_gyro.set_xlabel("Tiempo (s)")
    ax_gyro.set_ylabel("deg/s")
    line_gx = Line2D([], [], label="gx_dps", color='tab:blue',  linewidth=1.8)
    line_gy = Line2D([], [], label="gy_dps", color='tab:red',   linewidth=1.8)
    line_gz = Line2D([], [], label="gz_dps", color='tab:green', linewidth=1.8)
    ax_gyro.add_line(line_gx); ax_gyro.add_line(line_gy); ax_gyro.add_line(line_gz)
    ax_gyro.legend(loc="upper right", frameon=True, framealpha=0.85)
    ax_gyro.grid(True, alpha=0.3)

    while not program_exit:
        if plotting_active:
            with buf_lock:
                t_vals  = list(t_buf)
                ax_vals = list(ax_buf)
                ay_vals = list(ay_buf)
                az_vals = list(az_buf)
                gx_vals = list(gx_buf)
                gy_vals = list(gy_buf)
                gz_vals = list(gz_buf)

            if len(t_vals) >= 2:
                t_max = t_vals[-1]
                t_min = max(0.0, t_max - WINDOW_SEC)

                idx0 = 0
                for i in range(len(t_vals)-1, -1, -1):
                    if t_vals[i] < t_min:
                        idx0 = i + 1
                        break

                t_win  = t_vals[idx0:]
                ax_win = ax_vals[idx0:]
                ay_win = ay_vals[idx0:]
                az_win = az_vals[idx0:]
                gx_win = gx_vals[idx0:]
                gy_win = gy_vals[idx0:]
                gz_win = gz_vals[idx0:]

                # Acc
                line_ax.set_data(t_win, ax_win)
                line_ay.set_data(t_win, ay_win)
                line_az.set_data(t_win, az_win)
                ax_acc.set_xlim(t_min, t_max)
                if ax_win:
                    ymin = min(min(ax_win), min(ay_win), min(az_win))
                    ymax = max(max(ax_win), max(ay_win), max(az_win))
                    if ymin == ymax:
                        ymin -= 0.1; ymax += 0.1
                    margin = 0.06 * max(1e-6, abs(ymax - ymin))
                    ax_acc.set_ylim(ymin - margin, ymax + margin)

                # Gyro
                line_gx.set_data(t_win, gx_win)
                line_gy.set_data(t_win, gy_win)
                line_gz.set_data(t_win, gz_win)
                ax_gyro.set_xlim(t_min, t_max)
                if gx_win:
                    ymin = min(min(gx_win), min(gy_win), min(gz_win))
                    ymax = max(max(gx_win), max(gy_win), max(gz_win))
                    if ymin == ymax:
                        ymin -= 1.0; ymax += 1.0
                    margin = 0.06 * max(1e-6, abs(ymax - ymin))
                    ax_gyro.set_ylim(ymin - margin, ymax + margin)

                fig_acc.canvas.draw();  fig_acc.canvas.flush_events()
                fig_gyro.canvas.draw(); fig_gyro.canvas.flush_events()

        time.sleep(0.05)

    try:
        plt.ioff()
    except Exception:
        pass

# ------------------ RX BLE ------------------
# al inicio del archivo ya tienes: import numpy as np

G   = 9.80665
RAD = np.pi / 180.0

def process_records(text_chunk: str):
    """Acumula texto de notificaciones, separa por ';', parsea y actualiza buffers/CSV."""
    global rx_text_buffer, t0_millis

    with rx_lock:
        rx_text_buffer += text_chunk
        parts = rx_text_buffer.split(';')
        rx_text_buffer = parts[-1]  # posible parte incompleta

    for rec in parts[:-1]:
        rec = rec.strip()
        if not rec:
            continue
        rec = rec.replace('\r', '').replace('\n', '')
        cols = rec.split(',')
        if len(cols) != 7:
            continue

        # CSV si está en logging
        if is_logging and csv_writer:
            csv_writer.writerow(cols)

        try:
            ts_ms = int(cols[0])
            # ⚠️ CONVERSIÓN A LAS MISMAS UNIDADES DEL ENTRENAMIENTO
            ax_g = float(cols[1]); ay_g = float(cols[2]); az_g = float(cols[3])
            gx_d = float(cols[4]); gy_d = float(cols[5]); gz_d = float(cols[6])

            ax = ax_g * G
            ay = ay_g * G
            az = az_g * G
            gx = gx_d * RAD
            gy = gy_d * RAD
            gz = gz_d * RAD
        except ValueError:
            continue

        if t0_millis is None:
            t0_millis = ts_ms
        t_rel = (ts_ms - t0_millis) / 1000.0

        with buf_lock:
            t_buf.append(t_rel)
            ax_buf.append(ax); ay_buf.append(ay); az_buf.append(az)
            gx_buf.append(gx); gy_buf.append(gy); gz_buf.append(gz)

# ------------------ BLE Main ------------------
async def ble_main():
    """Descubre el periférico por nombre, se conecta y se suscribe a Notify."""
    print("🔎 Buscando periféricos BLE…")
    device = None
    devices = await BleakScanner.discover(timeout=5.0)
    for d in devices:
        if d.name == BLE_DEVICE_NAME:
            device = d
            break

    if device is None:
        print(f"❌ No se encontró un periférico con nombre '{BLE_DEVICE_NAME}'.")
        print("    Verifica que el ESP32 esté anunciando y que el nombre coincida.")
        return

    print(f"📶 Conectando a {device.name} ({device.address})…")
    async with BleakClient(device) as client:
        if not client.is_connected:
            print("❌ No se pudo conectar.")
            return
        print("✅ Conectado. Suscribiéndose a NOTIFY…")

        def handle_notify(_, data: bytearray):
            try:
                chunk = data.decode("utf-8", errors="ignore")
            except Exception:
                return
            process_records(chunk)

        await client.start_notify(TX_CHAR_UUID, handle_notify)
        print("🟢 Suscrito. Pulsa 's' para graficar. 'd' inicia Top-3 continuo. 'x' para salir.")

        while not program_exit:
            await asyncio.sleep(0.1)

        try:
            await client.stop_notify(TX_CHAR_UUID)
        except Exception:
            pass
    print("🔌 Desconectado.")

# ------------------ Main ------------------
def main():
    global is_logging, plotting_active, program_exit
    global csv_file, csv_writer, csv_path
    global model_pipe, label_names, model_win_sec, model_hop_frac
    global diag_active

    # Cargar modelo
    try:
        md = load(MODEL_PATH)
        model_pipe   = md["pipe"]
        label_names  = md["label_names"]
        model_win_sec= float(md["cfg"]["win_sec"])
        model_hop_frac = float(md["cfg"]["hop_frac"])
        print(f"Modelo cargado. Ventana={model_win_sec:.1f}s, hop={int(model_hop_frac*100)}%")
    except Exception as e:
        print(f"⚠️  No pude cargar el modelo en {MODEL_PATH}: {e}")
        print("    El modo 'd' no funcionará hasta que esté el modelo.")

    # CSV
    csv_path = Path(CSV_NAME).resolve()
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(HEADERS)

    # Hilos: teclado + plot
    t_cmd = threading.Thread(target=user_input_thread, daemon=True)
    t_cmd.start()

    t_plot = threading.Thread(target=plotter_loop, daemon=True)
    t_plot.start()

    print(f"CSV: {csv_path.name}")
    print("Esperando comandos: 's' (start), 'q' (pausa), 'd' (Top-3 cada 1s), 'x' (salir)…\n")

    # Ejecuta BLE asyncio en este hilo
    try:
        asyncio.run(ble_main())
    finally:
        # Detener diagnóstico si está activo
        diag_active = False
        # Cierre ordenado
        try:
            if csv_file:
                csv_file.flush()
                csv_file.close()
                print(f"✅ CSV guardado en: {csv_path}")
        except Exception:
            pass

        start_plot_event.set()
        time.sleep(0.2)
        print("Finalizado (cierra las ventanas cuando quieras).")

if __name__ == "__main__":
    main()
