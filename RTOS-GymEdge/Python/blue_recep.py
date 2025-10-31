# plot_spp_realtime_mpu.py
import sys
import time
import csv
from collections import deque
from math import sqrt
import serial
from serial.tools import list_ports

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================= CONFIG =================
SERIAL_PORT = "COM5"   # <-- CAMBIA AQUÍ (p.ej., COM8 en Windows, /dev/rfcomm0 en Linux)
BAUDRATE    = 115200
WINDOW_SEC  = 30       # ventana visible en segundos
SAVE_CSV    = True
CSV_PATH    = "log_mpu_spp.csv"

MAX_PRINT_SAMPLES = 6  # cuántas muestras imprimir por paquete en la terminal
READ_INTERVAL_MS  = 150  # intervalo de refresco del animador (ms)
# ==========================================

# Buffers (ventana deslizante)
t_ms = deque(maxlen=5000)
temp = deque(maxlen=5000)
gyro = deque(maxlen=5000)  # |ω| en deg/s
accm = deque(maxlen=5000)  # |acc| en g

# CSV opcional
csv_file = None
csv_writer = None
if SAVE_CSV:
    csv_file = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp_ms", "tempC", "gyro_mag_dps", "ax", "ay", "az", "acc_mag_g"])
    print(f"📝 Guardando datos en: {CSV_PATH}")

# Abrir puerto serie
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"✅ Puerto abierto: {SERIAL_PORT} @ {BAUDRATE} baudios")
except Exception as e:
    print(f"❌ Error abriendo {SERIAL_PORT}: {e}")
    sys.exit(1)

# Figura y ejes
plt.figure(figsize=(10, 6))
ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2)
ax3 = plt.subplot(3,1,3)

line1, = ax1.plot([], [], label="Temp (°C)")
line2, = ax2.plot([], [], label="|ω| Gyro (deg/s)")
line3, = ax3.plot([], [], label="|acc| (g)")

for ax in (ax1, ax2, ax3):
    ax.grid(True)
    ax.legend(loc="upper left")

ax1.set_ylabel("°C")
ax2.set_ylabel("deg/s")
ax3.set_ylabel("g")
ax3.set_xlabel("Tiempo (s)")

t0 = None  # para base de tiempo relativa

def parse_packet_line(raw_line: str):
    """
    raw_line es el paquete ENTERO en una sola línea, con muestras separadas por ';'
    y cada muestra con 6 campos CSV: ts_ms,tempC,gyro_mag_dps,ax,ay,az
    Retorna lista de tuplas parseadas.
    """
    samples = []
    for s in raw_line.split(";"):
        s = s.strip()
        if not s or "," not in s:
            continue
        parts = s.split(",")
        if len(parts) != 6:
            continue
        try:
            ts = float(parts[0])
            tc = float(parts[1])
            gy = float(parts[2])  # deg/s (magnitud ya calculada en el ESP32)
            ax = float(parts[3])
            ay = float(parts[4])
            az = float(parts[5])
            acc = sqrt(ax*ax + ay*ay + az*az)  # magnitud en g
        except ValueError:
            continue
        samples.append((ts, tc, gy, ax, ay, az, acc))
    return samples

def update(_):
    global t0
    updated = False

    # Leemos una línea (un paquete) si está disponible
    if ser.in_waiting:
        raw = ser.readline().decode(errors="ignore").strip()
        if raw:
            parsed = parse_packet_line(raw)
            if parsed:
                # Imprimir un resumen del paquete
                print(f"\n📦 Paquete recibido con {len(parsed)} muestras")
                for i, (ts, tc, gy, ax_v, ay_v, az_v, acc_v) in enumerate(parsed[:MAX_PRINT_SAMPLES]):
                    print(f"  🟢 {ts:.0f} ms | T={tc:.2f}°C | |ω|={gy:.2f} deg/s | |acc|={acc_v:.3f} g")
                if len(parsed) > MAX_PRINT_SAMPLES:
                    print(f"  … (+{len(parsed)-MAX_PRINT_SAMPLES} muestras)")

                # Añadir a buffers
                for (ts, tc, gy_v, ax_v, ay_v, az_v, acc_v) in parsed:
                    t_ms.append(ts)
                    temp.append(tc)
                    gyro.append(gy_v)
                    accm.append(acc_v)
                    # Guardar en CSV
                    if csv_writer:
                        csv_writer.writerow([ts, tc, gy_v, ax_v, ay_v, az_v, acc_v])

                updated = True

    if not updated:
        return line1, line2, line3

    # Base de tiempo relativa a la primera muestra
    if t0 is None and t_ms:
        t0 = t_ms[0]

    t_sec = [(ts - t0) / 1000.0 for ts in t_ms]

    # Actualizar datos de líneas
    line1.set_data(t_sec, list(temp))
    line2.set_data(t_sec, list(gyro))
    line3.set_data(t_sec, list(accm))

    # Autoscale y ventana deslizante
    for ax in (ax1, ax2, ax3):
        ax.relim()
        ax.autoscale_view()

    if t_sec:
        xmax = t_sec[-1]
        xmin = max(0, xmax - WINDOW_SEC)
        ax1.set_xlim(xmin, xmax + 0.05)
        ax2.set_xlim(xmin, xmax + 0.05)
        ax3.set_xlim(xmin, xmax + 0.05)

    return line1, line2, line3

ani = FuncAnimation(plt.gcf(), update, interval=READ_INTERVAL_MS, blit=False)
print("📈 Graficando en tiempo real. Cierra la ventana para finalizar.")

try:
    plt.tight_layout()
    plt.show()
finally:
    ser.close()
    if csv_file: csv_file.close()
    print("🔚 Finalizado.")