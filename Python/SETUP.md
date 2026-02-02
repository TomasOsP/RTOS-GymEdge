# 🎬 Resumen: Simulador de Datos IMU para GIF del Dashboard

## ¿Qué se ha hecho?

He creado un sistema completo para simular datos del sensor IMU sin necesidad del hardware físico. Esto te permite **grabar un GIF funcional del dashboard** con datos realistas.

## 📦 Archivos creados

### Scripts principales:
1. **`simulate_sensor.py`** (11.7 KB)
   - Servidor WebSocket que genera datos simulados
   - Emula sensor IMU con 6 ejercicios diferentes
   - Predice ejercicios y calcula estadísticas
   - Puerto: 5000

2. **`change_exercise.py`** (1.9 KB)
   - Cambia ejercicio activo en el simulador
   - Modo manual o automático
   - Útil para variar el contenido del GIF

3. **`start_gif_capture.py`** (1.4 KB)
   - Script todo-en-uno para iniciar el simulador
   - Incluye rotación automática de ejercicios
   - Ideal para grabar el GIF directamente

### Documentación:
4. **`CAPTURE_GIF_GUIDE.md`** - Guía paso a paso completa
5. **`SIMULATOR_README.md`** - Documentación técnica
6. **`SETUP.md`** - Este archivo

## 🚀 Cómo usar (Quick Start)

### Opción A: Ejecución simple (recomendado)
```bash
cd ~/Documents/GitHub/RTOS-GymEdge/Python
python3 start_gif_capture.py
```

Esto automáticamente:
- ✅ Inicia el servidor simulador
- ✅ Comienza a rotar ejercicios cada 8 segundos
- ✅ Te muestra la URL del dashboard

### Opción B: Control manual
```bash
# Terminal 1: Iniciar servidor
cd ~/Documents/GitHub/RTOS-GymEdge/Python
python3 simulate_sensor.py

# Terminal 2: Cambiar ejercicios
cd ~/Documents/GitHub/RTOS-GymEdge/Python
python3 change_exercise.py auto 5
```

### Opción C: Cambios manuales
```bash
# Terminal 1
python3 simulate_sensor.py

# Terminal 2 (en diferentes momentos)
python3 change_exercise.py bicep
python3 change_exercise.py circular
python3 change_exercise.py nado
# etc...
```

## 📊 Características del simulador

| Característica | Detalles |
|---|---|
| **Acelerómetro** | 3 ejes (X, Y, Z) + magnitud |
| **Giroscopio** | 3 ejes (X, Y, Z) + magnitud |
| **Ejercicios** | bicep, circular, elevacion_lateral, nado, remo, neutro |
| **Ruido realista** | Simulado con distribución normal |
| **Frecuencia variable** | Cada ejercicio tiene su propia frecuencia |
| **Predicción ML** | Top-3 probabilidades por ejercicio |
| **Repeticiones** | Contador que incrementa ocasionalmente |
| **Estadísticas** | FPS, latencia, pérdida de paquetes |
| **Frecuencia muestreo** | 50 Hz (configurable) |

## 🎯 Datos de cada ejercicio

```
bicep:
  - Movimiento lento y controlado
  - Máxima aceleración en Y (brazo arriba/abajo)
  - Giroscopio moderado
  - Ruido bajo

circular:
  - Movimiento rotatorio suave
  - Aceleración multidireccional
  - Giroscopio elevado en X y Y
  - Ruido medio

elevacion_lateral:
  - Aceleración en Y dominante
  - Movimiento lento
  - Giroscopio en Y muy alto
  - Ruido bajo

nado:
  - Movimiento rápido (1.8 Hz)
  - Aceleración multidireccional
  - Giroscopio muy alto
  - Ruido moderado

remo:
  - Movimiento controlado con tracción
  - Aceleración en Y moderada-alta
  - Giroscopio equilibrado
  - Ruido bajo

neutro:
  - Sin movimiento
  - Valores cercanos a (0, 0, 10) en acelerómetro
  - Giroscopio casi cero
  - Mínimo ruido
```

## 🎬 Capturar el GIF (resumen rápido)

1. **Inicia el simulador**: `python3 start_gif_capture.py`
2. **Abre navegador**: http://127.0.0.1:5000/dashboard.html
3. **Abre ScreenFlow** (o similar)
4. **Graba durante 1-2 minutos**
5. **Exporta como GIF** (10-12 FPS, 900px ancho)

Verás en el GIF:
- ✨ Gráficas actualizando en tiempo real
- 📊 Predicción cambiando según ejercicio
- 🔄 Contador de repeticiones incrementando
- 📈 Datos de acelerómetro y giroscopio
- 🎨 Interfaz limpia y profesional

## 🔧 Personalización

### Cambiar puerto:
```python
# En simulate_sensor.py, línea ~10
PORT = 8000  # en lugar de 5000
```

### Modificar rangos de datos:
```python
# En simulate_sensor.py, sección EXERCISES
EXERCISES = {
    "mi_ejercicio": {
        "ax_range": (-2.0, 2.0),
        "frequency": 1.5,
        "noise_level": 0.3,
        ...
    }
}
```

### Cambiar frecuencia de muestreo:
```python
# En simulate_sensor.py, línea ~14
SAMPLE_RATE = 100  # Hz (default 50)
```

## 🐛 Troubleshooting

| Error | Solución |
|-------|----------|
| `ModuleNotFoundError: No module named 'aiohttp'` | `python3 -m pip install aiohttp aiofiles numpy` |
| `OSError: [Errno 48] Address already in use` | Puerto ocupado: `lsof -i :5000` y mata proceso |
| Dashboard: "Esperando datos..." | Espera 5 segundos y recarga la página |
| Cambio de ejercicio no se ve | Actualiza navegador (Cmd+R) |
| GIF muy grande | Reduce FPS (a 8) o resolución (a 800px) |

## 📦 Dependencias instaladas

```
aiohttp==3.13.3
aiofiles==25.1.0
numpy==2.0.2
```

Todas ya están instaladas en tu sistema.

## 💾 Espacio requerido

- Scripts: ~25 KB
- No hay bases de datos ni archivos grandes
- GIF final: 2-10 MB (depende duración)

## 🎓 Cómo funciona técnicamente

```
Usuario abre navegador
        ↓
  Dashboard.html
        ↓
WebSocket a 127.0.0.1:5000
        ↓
  Servidor aiohttp
        ↓
  SensorSimulator genera datos
        ↓
  Datos pseudoaleatorios pero realistas
        ↓
  Envía JSON al navegador
        ↓
  Chart.js dibuja gráficas
  Predicción se actualiza
  Contador de reps incrementa
        ↓
  Todo visible en tiempo real
```

## 🎯 Casos de uso

✅ **Demostración** del dashboard a stakeholders  
✅ **Documentación** del proyecto  
✅ **README** animado en GitHub  
✅ **Presentación** en conferencias  
✅ **Testing** sin sensor físico  
✅ **Desarrollo** sin Hardware  

## 📝 Notas finales

- Los datos son **pseudoaleatorios pero realistas**
- Cada ejercicio tiene **características únicas**
- Las predicciones **favorecen el ejercicio activo**
- El sistema es **totalmente reproducible**
- Puedes **personalizar cada aspecto**

## 🚀 Próximos pasos

1. Ejecuta: `python3 start_gif_capture.py`
2. Abre: http://127.0.0.1:5000/dashboard.html
3. Graba con ScreenFlow o similar
4. Exporta como GIF
5. ¡Disfruta del resultado! 🎉

---

**Creado el**: 2025-02-02  
**Scripts**: 3 (simulate_sensor.py, change_exercise.py, start_gif_capture.py)  
**Documentación**: 3 archivos .md  
**Estado**: ✅ Listo para usar
