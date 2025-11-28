# gym_live_dashboard.py
# Versión optimizada y corregida del sistema de recepción BLE + gráficas + inferencia
# Diseñado para macOS (usa Bleak + asyncio en hilo separado).
# Controles (en la terminal): 's' = start/resume, 'q' = pause, 'd' = toggle diagnóstico Top-3, 'x' = salir

import time
import threading
import asyncio
from pathlib import Path
from collections import deque
from threading import Lock
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from joblib import load

from bleak import BleakScanner, BleakClient

# Opcionales para DSP (si no las necesitas, el código sigue funcionando sin crash)
from scipy.signal import butter, filtfilt, welch, lfilter
from scipy.integrate import trapezoid

# ===================== CONFIG =====================
BLE_DEVICE_NAME = "ESP32-MPU-Mac"
TX_CHAR_UUID = "abcdef12-3456-7890-abcd-ef1234567890"
MODEL_PATH = Path(__file__).with_name("modelo_multiclase.joblib")

WINDOW_SEC = 10             # ventana visible en gráfica (segundos)
SAMPLE_HZ_GUESS = 50.0
SAMPLE_HZ = 50.0           # valor por defecto si no se puede estimar
MODEL_MIN_SAMPLES_SEC = 1.5

# Buffers
MAX_POINTS = int(60 * SAMPLE_HZ)  # almacena hasta 60s por defecto
buf_lock = Lock()
t_buf   = deque(maxlen=MAX_POINTS)
ax_buf  = deque(maxlen=MAX_POINTS)
ay_buf  = deque(maxlen=MAX_POINTS)
az_buf  = deque(maxlen=MAX_POINTS)
gx_buf  = deque(maxlen=MAX_POINTS)
gy_buf  = deque(maxlen=MAX_POINTS)
gz_buf  = deque(maxlen=MAX_POINTS)

# RX text buffer (paquetes terminados con ';')
rx_text_buffer = ""
rx_lock = Lock()

# Estado
is_logging = False
plotting_active = False
program_exit = False
start_plot_event = threading.Event()

diag_active = False
diag_thread = None

t0_millis = None

# CSV
csv_file = None
csv_writer = None
csv_path = None

# Modelo
model_pipe = None
label_names = None
model_win_sec = 2.0
model_hop_frac = 0.5

# Constantes
G = 9.80665
RAD = np.pi / 180.0

HEADERS = ["timestamp_ms","ax_g","ay_g","az_g","gx_dps","gy_dps","gz_dps"]

# ===================== UTILIDADES DSP / FEATURES =====================
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
        # lfilter es más seguro si no hay suficiente longitud
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
        # si falla filtfilt, devolver datos sin filtrar
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

# ===================== PREDICCIÓN Top-K =====================
def predict_topk_now(k_top=3):
    """Toma la última ventana del buffer y devuelve Top-k (label, prob)."""
    global model_pipe, label_names, model_win_sec

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

    if t_vals.size < 3:
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

    try:
        proba = model_pipe.predict_proba(X)[0]
    except Exception as e:
        return None, None, f"⚠️ Error en predict_proba: {e}"

    idxs = np.argsort(proba)[::-1][:k_top]
    top = [(label_names[int(i)], float(proba[int(i)])) for i in idxs]
    t0 = float(seg["t"].iloc[0])
    return t0, top, None

# Diagnóstico contínuo
def diag_loop():
    print("🟡 Diagnóstico continuo: Top-3 cada 1s (sal con 'x').")
    global diag_active
    while diag_active and not program_exit:
        t0, top, err = predict_topk_now(k_top=3)
        if err:
            print(err)
        else:
            pretty = " | ".join([f"{i+1}) {lbl} {prob:.2f}" for i,(lbl,prob) in enumerate(top)])
            print(f"[{t0:7.2f}s] {pretty}")
        time.sleep(1.0)
    print("🔵 Diagnóstico continuo detenido.")

# ===================== RX / PARSING =====================
def process_records(text_chunk: str):
    """Acumula texto de notificaciones, separa por ';', parsea y actualiza buffers/CSV."""
    global rx_text_buffer, t0_millis, csv_writer, is_logging

    with rx_lock:
        rx_text_buffer += text_chunk
        parts = rx_text_buffer.split(';')
        rx_text_buffer = parts[-1]  # última parte posible incompleta

    for rec in parts[:-1]:
        rec = rec.strip().replace('\r','').replace('\n','')
        if not rec:
            continue
        cols = rec.split(',')
        if len(cols) != 7:
            continue

        # escribir CSV si está activo
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

# ===================== BLE =====================
async def ble_main():
    """Busca por nombre y se suscribe a NOTIFY (Bleak)."""
    print("🔎 Buscando periféricos BLE… (timeout 5s)")
    try:
        devices = await BleakScanner.discover(timeout=5.0)
    except Exception as e:
        print(f"❌ Error en discovery BLE: {e}")
        return

    device = next((d for d in devices if d.name == BLE_DEVICE_NAME), None)
    if device is None:
        print(f"❌ No se encontró '{BLE_DEVICE_NAME}'. Asegúrate de que esté anunciando.")
        return

    print(f"📶 Conectando a {device.name} ({device.address})…")
    try:
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
            # Mantener conexión hasta que se pida salir
            while not program_exit:
                await asyncio.sleep(0.2)
            try:
                await client.stop_notify(TX_CHAR_UUID)
            except Exception:
                pass
    except Exception as e:
        print(f"❌ Error en conexión BLE: {e}")

    print("🔌 BLE: desconectado / terminado.")

# ===================== GRAFICADO (FuncAnimation) =====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle("GymEdge Live Monitor", fontsize=16, fontweight="bold")

line_ax, = ax1.plot([], [], lw=1.2, label="ax")
line_ay, = ax1.plot([], [], lw=1.2, label="ay")
line_az, = ax1.plot([], [], lw=1.2, label="az")
ax1.set_ylabel("Aceleración (m/s²)")
ax1.set_ylim(-30, 30)
ax1.legend(); ax1.grid(True, alpha=0.3)

pred_text = ax2.text(0.02, 0.6, "Esperando datos...", fontsize=20, fontweight="bold", transform=ax2.transAxes, color="gray")
conf_text = ax2.text(0.02, 0.35, "", fontsize=14, transform=ax2.transAxes, color="black")
ax2.axis("off")

def get_current_prediction_simple():
    """Versión simple: usa predict_topk_now y devuelve etiqueta y confianza."""
    t0, top, err = predict_topk_now(k_top=1)
    if err:
        return None, 0.0, err
    if not top:
        return None, 0.0, "Sin top"
    lbl, prob = top[0]
    return lbl, prob, None

def update_plot(frame):
    global plotting_active
    with buf_lock:
        if len(t_buf) == 0:
            return line_ax, line_ay, line_az, pred_text, conf_text
        t = np.array(t_buf)
        ax = np.array(ax_buf)
        ay = np.array(ay_buf)
        az = np.array(az_buf)

    # ventana móvil
    t_last = float(t[-1])
    t_min = max(0.0, t_last - WINDOW_SEC)
    mask = t >= t_min

    if mask.any():
        line_ax.set_data(t[mask], ax[mask])
        line_ay.set_data(t[mask], ay[mask])
        line_az.set_data(t[mask], az[mask])
        ax1.set_xlim(t_min, t_last)
        ax1.set_title(f"Tiempo: {t_last:.1f}s | Muestras: {len(t_buf)}")
    else:
        line_ax.set_data([], []); line_ay.set_data([], []); line_az.set_data([], [])

    # predicción
    lbl, prob, err = get_current_prediction_simple()
    if err:
        pred_text.set_text(err)
        pred_text.set_color("gray")
        conf_text.set_text("")
    else:
        pred_text.set_text(lbl.upper())
        color = "green" if prob*100 > 70 else "orange" if prob*100 > 40 else "gray"
        pred_text.set_color(color)
        conf_text.set_text(f"{prob*100:.1f}% confianza")

    return line_ax, line_ay, line_az, pred_text, conf_text

ani = FuncAnimation(fig, update_plot, interval=200, blit=False, cache_frame_data=False)

# ===================== ENTRADA POR TECLADO (hilo) =====================
def init_user_folder():
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

def user_input_thread():
    global is_logging, plotting_active, program_exit, diag_active, diag_thread, csv_file, csv_writer
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
                print("▶️ REANUDADO: CSV + gráficas.")
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
                diag_active = False
                print("⏹️  Diagnóstico continuo OFF.")
            else:
                diag_active = True
                if (diag_thread is None) or (not diag_thread.is_alive()):
                    diag_thread = threading.Thread(target=diag_loop, daemon=True)
                    diag_thread.start()
                print("▶️  Diagnóstico continuo ON (Top-3 cada 1s).")
        elif cmd == "x":
            print("👋 Saliendo...")
            program_exit_set()
        else:
            print("Comando no reconocido. Usa 's', 'q', 'd' o 'x'.")

def program_exit_set():
    global program_exit, diag_active
    program_exit = True
    diag_active = False
    start_plot_event.set()  # en caso que plotter esté esperando

# ===================== MAIN =====================
def main():
    global csv_file, csv_writer, csv_path, model_pipe, label_names, model_win_sec, model_hop_frac

    # 1) Cargar modelo (si existe)
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
            print(f"Modelo cargado. Ventana={model_win_sec:.1f}s, hop={int(model_hop_frac*100)}%")
        except Exception as e:
            print(f"⚠️ No pude cargar el modelo: {e}")
            model_pipe = None
    else:
        print("⚠️ NO se encontró modelo_multiclase.joblib → correrás solo gráficas.")

    # 2) Carpeta usuario + CSV
    print("=== SISTEMA DE RECEPCIÓN ===")
    user_name, user_dir, csv_path = init_user_folder()
    try:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        import csv as _csv
        csv_writer = _csv.writer(csv_file)
        csv_writer.writerow(HEADERS)
        csv_file.flush()
        print(f"CSV: {csv_path}")
    except Exception as e:
        csv_file = None
        csv_writer = None
        print(f"⚠️ No se pudo abrir CSV: {e}")

    # 3) Hilo plotter (usa FuncAnimation para actualizar)
    t_plot = threading.Thread(target=_plt_show_thread, daemon=True)
    t_plot.start()

    # 4) Hilo teclado
    threading.Thread(target=user_input_thread, daemon=True).start()

    # 5) Hilo BLE (asyncio.run en hilo)
    threading.Thread(target=lambda: asyncio.run(ble_main()), daemon=True).start()

    # 6) loop principal: esperar salida
    try:
        while not program_exit:
            time.sleep(0.2)
    except KeyboardInterrupt:
        program_exit_set()

    # limpiar
    print("Cerrando recursos...")
    try:
        if csv_file:
            csv_file.flush()
            csv_file.close()
    except Exception:
        pass
    print("Listo. Adiós.")

def _plt_show_thread():
    """Thread separado para mostrar la ventana matplotlib.
    Espera a que el usuario pulse 's' (start) para mostrar si se desea."""
    start_plot_event.wait()
    # Mostrar ventana de matplotlib (bloqueante)
    plt.show()

if __name__ == "__main__":
    main()
