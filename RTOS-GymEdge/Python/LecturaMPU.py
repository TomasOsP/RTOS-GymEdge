import serial
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# -------- CONFIGURACIÓN DEL PUERTO SERIAL --------
PORT = '/dev/ttyUSB0'   # Cambia según tu sistema: COM3 en Windows, /dev/ttyUSB0 en Linux/Mac
BAUDRATE = 115200

# -------- CONFIGURACIÓN DE GRÁFICAS --------
WINDOW_SIZE = 200  # Cantidad de puntos visibles en la gráfica

# Inicializa buffers circulares para cada eje
accel_x, accel_y, accel_z = deque(maxlen=WINDOW_SIZE), deque(maxlen=WINDOW_SIZE), deque(maxlen=WINDOW_SIZE)
gyro_x, gyro_y, gyro_z = deque(maxlen=WINDOW_SIZE), deque(maxlen=WINDOW_SIZE)
temp_data = deque(maxlen=WINDOW_SIZE)

# -------- CONFIGURACIÓN MATPLOTLIB --------
plt.ion()
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Acelerómetro
axs[0].set_title('Acelerómetro (m/s²)')
axs[0].set_ylabel('Aceleración')
axs[0].set_ylim(-20, 20)
line_ax, = axs[0].plot([], [], label='Ax')
line_ay, = axs[0].plot([], [], label='Ay')
line_az, = axs[0].plot([], [], label='Az')
axs[0].legend(loc='upper right')

# Giroscopio
axs[1].set_title('Giroscopio (rad/s)')
axs[1].set_ylabel('Vel. Angular')
axs[1].set_ylim(-5, 5)
line_gx, = axs[1].plot([], [], label='Gx')
line_gy, = axs[1].plot([], [], label='Gy')
line_gz, = axs[1].plot([], [], label='Gz')
axs[1].legend(loc='upper right')

plt.tight_layout()

# -------- LECTURA SERIAL Y ACTUALIZACIÓN DE GRÁFICA --------
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"📡 Leyendo datos desde {PORT} a {BAUDRATE} baud...")
    time.sleep(2)  # Espera para que el puerto se estabilice

    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                values = list(map(float, line.split(';')))
                if len(values) == 7:
                    ax, ay, az, gx, gy, gz, t = values
                    accel_x.append(ax)
                    accel_y.append(ay)
                    accel_z.append(az)
                    gyro_x.append(gx)
                    gyro_y.append(gy)
                    gyro_z.append(gz)
                    temp_data.append(t)

                    # Actualiza las líneas de la gráfica
                    x_range = np.arange(len(accel_x))
                    line_ax.set_data(x_range, accel_x)
                    line_ay.set_data(x_range, accel_y)
                    line_az.set_data(x_range, accel_z)

                    line_gx.set_data(x_range, gyro_x)
                    line_gy.set_data(x_range, gyro_y)
                    line_gz.set_data(x_range, gyro_z)

                    for ax_plot in axs:
                        ax_plot.relim()
                        ax_plot.autoscale_view()

                    plt.pause(0.001)
            except ValueError:
                pass

except KeyboardInterrupt:
    print("\n🛑 Lectura interrumpida por el usuario.")
except serial.SerialException:
    print("❌ Error: no se pudo abrir el puerto serial.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
    plt.ioff()
    plt.show()
