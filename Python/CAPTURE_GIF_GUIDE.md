# 🎬 Guía para Capturar GIF del Dashboard

## ✅ Ya configurado

He creado dos scripts para simular datos del sensor IMU:

1. **`simulate_sensor.py`** - Servidor que genera datos aleatorios realistas
2. **`change_exercise.py`** - Utilidad para cambiar ejercicios

## 🚀 Paso 1: Iniciar el servidor

Abre una terminal y ejecuta:

```bash
cd /Users/tomasospina/Documents/GitHub/RTOS-GymEdge/Python
python3 simulate_sensor.py
```

Deberías ver:
```
🚀 Servidor iniciado en http://127.0.0.1:5000
📊 Abre http://127.0.0.1:5000/dashboard.html en el navegador

📋 Ejercicios disponibles:
   - bicep
   - circular
   - elevacion_lateral
   - nado
   - remo
   - neutro
```

## 🌐 Paso 2: Abrir el dashboard

Abre tu navegador y ve a: `http://127.0.0.1:5000/dashboard.html`

Verás en tiempo real:
- ✅ Gráficas de acelerómetro (ax, ay, az, |a|)
- ✅ Gráficas de giroscopio (gx, gy, gz, |g|)
- ✅ Predicción de ejercicio con probabilidad
- ✅ Top-3 de predicciones
- ✅ Contador de repeticiones
- ✅ Estadísticas (FPS, latencia, pérdida)
- ✅ GIF animado del ejercicio

## 🎞️ Paso 3: Cambiar ejercicios (para el GIF)

En OTRA terminal, ejecuta para cambiar ejercicios:

**Opción A: Cambiar manualmente**
```bash
cd /Users/tomasospina/Documents/GitHub/RTOS-GymEdge/Python
python3 change_exercise.py bicep               # Bíceps
python3 change_exercise.py circular            # Movimiento circular
python3 change_exercise.py elevacion_lateral   # Elevación lateral
python3 change_exercise.py nado                # Nado
python3 change_exercise.py remo                # Remo
python3 change_exercise.py neutro              # Reposo
```

**Opción B: Rotar automáticamente**
```bash
python3 change_exercise.py auto 8              # Cambia cada 8 segundos
```

## 📹 Paso 4: Capturar el GIF

### Opción 1: Con ScreenFlow (recomendado para macOS)

1. Abre **ScreenFlow** (Applications → ScreenFlow)
2. Haz clic en **"Start Recording"**
3. Selecciona la ventana del navegador con el dashboard
4. En la terminal, ejecuta:
   ```bash
   python3 change_exercise.py auto 5
   ```
5. Deja grabar durante 1-2 minutos
6. Detén la grabación
7. Exporta como GIF:
   - File → Export
   - Format: GIF
   - Quality: High
   - FPS: 10-15

### Opción 2: Con QuickTime + ffmpeg

```bash
# 1. Abrir QuickTime y grabar pantalla
# File → New Screen Recording
# Grabar durante 1-2 minutos con el simulador corriendo

# 2. Convertir a GIF con ffmpeg
ffmpeg -i "recording.mov" -vf "fps=10,scale=900:-1" dashboard.gif
```

### Opción 3: Con Gifski (en línea)

1. Graba con QuickTime o ScreenFlow en MP4
2. Sube a https://gifski.app/
3. Convierte a GIF de alta calidad

### Opción 4: Con byzanz (Linux)

```bash
byzanz-record --duration=60 --x=0 --y=0 --width=1200 --height=800 dashboard.gif
```

## 🎨 Recomendaciones para un buen GIF

1. **Duración**: 30-90 segundos es ideal
2. **Resolución**: 900-1200 px de ancho
3. **FPS**: 8-12 fps para archivo más pequeño
4. **Fondo**: El dashboard tiene fondo claro, se ve bien
5. **Cambios**: Rota entre 2-3 ejercicios diferentes

## 📊 Datos que mostrará el GIF

- Acelerómetro en movimiento (3 ejes)
- Giroscopio detectando rotaciones
- Predicción cambiando según el ejercicio
- Gráficas actualizándose en tiempo real
- Contador de repeticiones incrementando
- Estadísticas de sistema actualizando

## 🔧 Personalizar datos simulados

Si quieres ajustar los datos (rangos, ruido, frecuencia), edita `simulate_sensor.py`:

```python
EXERCISES = {
    "bicep": {
        "ax_range": (0.5, 2.5),          # Rango de aceleración X
        "ay_range": (-1.5, 1.5),
        "az_range": (8.5, 10.5),
        "gx_range": (-200, 200),         # Rango de rotación X
        "gy_range": (-300, 300),
        "gz_range": (-100, 100),
        "frequency": 1.5,                # Velocidad del movimiento (Hz)
        "noise_level": 0.3,              # Ruido en los datos
    },
}
```

## 🐛 Solución de problemas

| Problema | Solución |
|----------|----------|
| "Cannot connect" | Asegúrate que el servidor está corriendo |
| Puerto ocupado | `lsof -i :5000` y mata el proceso |
| Dashboard muestra "Esperando datos..." | Espera 5 segundos, recarga página |
| GIF muy lento/rápido | Ajusta FPS al exportar |
| Cambio de ejercicio no se ve | Actualiza la página del navegador |

## 📁 Archivos creados

- `simulate_sensor.py` - Servidor WebSocket con datos simulados
- `change_exercise.py` - CLI para cambiar ejercicios
- `SIMULATOR_README.md` - Documentación técnica
- `CAPTURE_GIF_GUIDE.md` - Esta guía

## ✨ Resultado final

Tendrás un GIF mostrando:
- Dashboard en vivo con datos realistas
- Múltiples ejercicios de forma secuencial
- Gráficas actualizándose suavemente
- Predicciones precisas por ejercicio
- Interfaz limpia y profesional

¡A grabar! 🎬
