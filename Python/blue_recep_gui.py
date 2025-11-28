# gym_live_monitor.py
# Ventana simple con gráficas en tiempo real + predicción del ejercicio

import asyncio
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from joblib import load

from bleak import BleakScanner, BleakClient

# ===================== CONFIGURACIÓN =====================
BLE_DEVICE_NAME = "ESP32-MPU-Mac"          # Cambia si tu ESP32 tiene otro nombre
TX_CHAR_UUID = "abcdef12-3456-7890-abcd-ef1234567890"

MODEL_PATH =  Path(__file__).with_name("modelo_multiclase.joblib")  # Pon tu modelo aquí

# Ventana de tiempo visible (segundos)
WINDOW_SEC = 10

# Buffers circulares
MAX_POINTS = 1000
t_buf = deque(maxlen=MAX_POINTS)
ax_buf = deque(maxlen=MAX_POINTS)
ay_buf = deque(maxlen=MAX_POINTS)
az_buf = deque(maxlen=MAX_POINTS)

# Modelo
model = None
label_names = []
t0 = None

# ===================== CARGAR MODELO =====================
if MODEL_PATH.exists():
    try:
        data = load(MODEL_PATH)
        model = data["pipe"]
        label_names = data["label_names"]
        print(f"Modelo cargado → {len(label_names)} ejercicios: {label_names}")
    except Exception as e:
        print(f"No se pudo cargar el modelo: {e}")
else:
    print("No se encontró modelo_multiclase.joblib → solo gráficas")

# ===================== PROCESAMIENTO DE DATOS =====================
G = 9.80665
RAD_TO_DEG = 180.0 / np.pi

def process_packet(text):
    global t0
    lines = text.strip().replace('\r', '').split(';')
    for line in lines:
        line = line.strip()
        if not line: continue
        cols = line.split(',')
        if len(cols) != 7: continue

        try:
            ts_ms = int(cols[0])
            ax_g = float(cols[1])
            ay_g = float(cols[2])
            az_g = float(cols[3])
            gx_dps = float(cols[4])
            gy_dps = float(cols[5])
            gz_dps = float(cols[6])

            if t0 is None:
                t0 = ts_ms / 1000.0
            t_sec = ts_ms / 1000.0 - t0

            # Convertir a unidades físicas
            ax = ax_g * G
            ay = ay_g * G
            az = az_g * G

            t_buf.append(t_sec)
            ax_buf.append(ax)
            ay_buf.append(ay)
            az_buf.append(az)
        except:
            pass

# ===================== PREDICCIÓN SIMPLE =====================
def get_current_prediction():
    if model is None or len(t_buf) < 50:
        return "Esperando datos...", 0.0

    # Últimos 2 segundos
    t = np.array(t_buf)
    if len(t) == 0: return "Sin datos", 0.0
    t_recent = t[-int(2 * 50):]  # ~2 segundos a ~50 Hz
    if len(t_recent) < 50:
        return "Calentando...", 0.0

    idx = np.searchsorted(t, t_recent[0])
    ax = np.array(ax_buf)[idx:idx+len(t_recent)]
    ay = np.array(ay_buf)[idx:idx+len(t_recent)]
    az = np.array(az_buf)[idx:idx+len(t_recent)]

    # Magnitud del acelerómetro
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    X = np.column_stack([ax, ay, az, mag]).reshape(1, -1)

    try:
        proba = model.predict_proba(X)[0]
        idx = np.argmax(proba)
        label = label_names[idx]
        conf = proba[idx]
        return f"{label.upper()}", conf * 100
    except:
        return "Error en predicción", 0.0

# ===================== BLE =====================
async def ble_connect():
    print("Buscando tu ESP32...")
    devices = await BleakScanner.discover(timeout=10)
    device = next((d for d in devices if d.name == BLE_DEVICE_NAME), None)
    if not device:
        print(f"No se encontró '{BLE_DEVICE_NAME}'. Asegúrate de que esté encendido.")
        return

    print(f"Conectado a {device.name}")

    async with BleakClient(device) as client:
        def handler(_, data):
            try:
                process_packet(data.decode("utf-8", errors="ignore"))
            except:
                pass

        await client.start_notify(TX_CHAR_UUID, handler)
        print("Datos llegando → Mira la ventana gráfica!")
        while True:
            await asyncio.sleep(1)

# ===================== GRÁFICA =====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("GymEdge Live Monitor", fontsize=16, fontweight="bold")

# Gráfica de aceleración
line_ax, = ax1.plot([], [], label="X (rojo)", color="red", lw=1.5)
line_ay, = ax1.plot([], [], label="Y (verde)", color="green", lw=1.5)
line_az, = ax1.plot([], [], label="Z (azul)", color="blue", lw=1.5)
ax1.set_ylabel("Aceleración (m/s²)")
ax1.set_ylim(-30, 30)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Texto de predicción
pred_text = ax2.text(0.05, 0.6, "Esperando datos...", fontsize=24, fontweight="bold",
                     transform=ax2.transAxes, color="gray")
conf_text = ax2.text(0.05, 0.4, "", fontsize=18, transform=ax2.transAxes)
ax2.axis("off")

def update_plot(frame):
    if len(t_buf) == 0:
        return line_ax, line_ay, line_az

    t = np.array(t_buf)
    ax = np.array(ax_buf)
    ay = np.array(ay_buf)
    az = np.array(az_buf)

    # Ventana móvil
    t_min = max(0, t[-1] - WINDOW_SEC) if len(t) > 0 else 0
    mask = t >= t_min

    line_ax.set_data(t[mask], ax[mask])
    line_ay.set_data(t[mask], ay[mask])
    line_az.set_data(t[mask], az[mask])

    ax1.set_xlim(t_min, t[-1] if len(t) > 0 else 1)
    ax1.set_title(f"Tiempo: {t[-1]:.1f}s | Muestras: {len(t_buf)}")

    # Actualizar predicción
    label, conf = get_current_prediction()
    pred_text.set_text(label)
    pred_text.set_color("green" if conf > 70 else "orange" if conf > 40 else "gray")
    conf_text.set_text(f"{conf:.1f}% confianza" if conf > 0 else "")

    return line_ax, line_ay, line_az

ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

# ===================== INICIO =====================
def start_ble():
    asyncio.run(ble_connect())

if __name__ == "__main__":
    print("Iniciando GymEdge Live Monitor...")
    print("Cierra la ventana para salir.\n")

    # BLE en hilo aparte
    threading.Thread(target=start_ble, daemon=True).start()

    plt.show()