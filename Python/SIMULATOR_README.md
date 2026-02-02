# 🎬 Simulador de Sensor IMU para GIF del Dashboard

Este simulador genera datos aleatorios realistas del sensor IMU para que puedas grabar un GIF del dashboard sin necesidad del hardware.

## 📋 Requisitos

```bash
pip install aiohttp aiofiles numpy
```

## 🚀 Uso

### 1. Iniciar el servidor simulador

```bash
cd /Users/tomasospina/Documents/GitHub/RTOS-GymEdge/Python
python simulate_sensor.py
```

El servidor estará disponible en: `http://127.0.0.1:5000/dashboard.html`

### 2. Abrir el dashboard en el navegador

- Abre `http://127.0.0.1:5000/dashboard.html` en tu navegador
- El dashboard mostrará datos simulados en vivo

### 3. Cambiar ejercicios (en otra terminal)

**Cambiar a un ejercicio específico:**
```bash
python change_exercise.py bicep          # Cambiar a bíceps
python change_exercise.py circular       # Cambiar a movimiento circular
python change_exercise.py elevacion_lateral
python change_exercise.py nado
python change_exercise.py remo
python change_exercise.py neutro         # Sin movimiento
```

**Rotar automáticamente entre ejercicios:**
```bash
python change_exercise.py auto           # Rota cada 10 segundos
python change_exercise.py auto 15        # Rota cada 15 segundos
```

## 📊 Características del simulador

El simulador incluye:

- ✅ **Acelerómetro (ax, ay, az)** con valores realistas según el ejercicio
- ✅ **Giroscopio (gx, gy, gz)** con rangos apropiados
- ✅ **Magnitudes calculadas** (|a|, |g|)
- ✅ **Ruido realista** en los datos
- ✅ **Predicción de ejercicios** con probabilidades top-3
- ✅ **Estadísticas del sistema** (FPS, latencia, pérdida de paquetes)
- ✅ **Contador de repeticiones** (incrementa ocasionalmente)

## 🎞️ Capturar GIF

### Con ScreenFlow (macOS):
1. Abre ScreenFlow
2. Selecciona la ventana del navegador
3. Inicia la grabación
4. Ejecuta: `python change_exercise.py auto 5` para rotar ejercicios
5. Detén la grabación cuando tengas suficiente contenido
6. Exporta como GIF

### Con ffmpeg:
```bash
# Grabar pantalla (macOS)
ffmpeg -f avfoundation -i "1" -t 60 recording.mp4

# Convertir a GIF
ffmpeg -i recording.mp4 -vf "fps=10,scale=800:-1:flags=lanczos" dashboard.gif
```

### Con Gifski (más rápido):
```bash
# Primero grabar con ffmpeg o ScreenFlow
# Luego convertir con Gifski (available en macOS)
# O usar online: https://gifski.app/
```

## 🔧 Personalizar rangos de datos

Edita `simulate_sensor.py` en la sección `EXERCISES`:

```python
EXERCISES = {
    "mi_ejercicio": {
        "ax_range": (-2.0, 2.0),
        "ay_range": (-2.0, 2.0),
        "az_range": (8.0, 10.0),
        "gx_range": (-250, 250),
        "gy_range": (-250, 250),
        "gz_range": (-150, 150),
        "frequency": 1.2,      # Frecuencia del movimiento (Hz)
        "noise_level": 0.35,   # Nivel de ruido
    },
}
```

## 📊 Dashboard muestra

El dashboard simulado muestra:

- **Gráfica de Acelerómetro**: ax, ay, az, |a| en tiempo real
- **Gráfica de Giroscopio**: gx, gy, gz, |g| en tiempo real
- **Predicción**: Ejercicio detectado con probabilidad
- **Top-3**: Las 3 predicciones más probables
- **Datos en bruto**: Últimas muestras de acelerómetro y giroscopio
- **RMS**: Root Mean Square de cada eje
- **Frecuencia de muestreo**: Hz en tiempo real
- **Contador de repeticiones**: Incrementa al detectar repeticiones
- **Estado IMU**: FPS, latencia, pérdida de paquetes
- **GIF del ejercicio**: Imagen del ejercicio detectado

## ⚙️ Configuración avanzada

### Cambiar puerto:
```python
PORT = 8000  # En simulate_sensor.py
```

### Cambiar frecuencia de muestreo:
```python
SAMPLE_RATE = 100  # Hz (por defecto 50)
```

### Modificar predicción:
```python
# En SensorSimulator.generate_prediction()
# Aumentar/disminuir probabilidad del ejercicio actual
```

## 🐛 Solución de problemas

**Error: "Cannot connect to server"**
- Asegúrate de que `simulate_sensor.py` está ejecutándose
- Verifica que el puerto 5000 no está ocupado: `lsof -i :5000`

**El dashboard muestra "Esperando datos..."**
- Espera unos segundos a que los datos lleguen
- Comprueba la consola del servidor para errores

**Los GIF no se cargan**
- Verifica que existen en `/Python/static/gifs/`
- Los nombres deben coincidir con las claves en `gifMap`

## 📝 Notas

- Los datos generados son aleatorios pero realistas
- Cada ejercicio tiene sus propios rangos y características
- La predicción favorece al ejercicio actual activamente
- Los datos cambian suavemente (interpolación con senos)

¡Listo para grabar tu GIF! 🎬
