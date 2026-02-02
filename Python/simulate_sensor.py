#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador de sensor IMU para generar GIF del dashboard
Genera datos aleatorios realistas para diferentes ejercicios
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from aiohttp import web
import aiofiles

# =================== CONFIGURACIÓN ===================
SAMPLE_RATE = 50  # Hz
PORT = 5000
DEBUG = True

# Modelos de ejercicios con sus características
EXERCISES = {
    "bicep": {
        "ax_range": (0.5, 2.5),
        "ay_range": (-1.5, 1.5),
        "az_range": (8.5, 10.5),
        "gx_range": (-200, 200),
        "gy_range": (-300, 300),
        "gz_range": (-100, 100),
        "frequency": 1.5,  # Hz - frecuencia del movimiento
        "noise_level": 0.3,
    },
    "circular": {
        "ax_range": (-2.0, 2.0),
        "ay_range": (-2.0, 2.0),
        "az_range": (8.0, 10.0),
        "gx_range": (-250, 250),
        "gy_range": (-250, 250),
        "gz_range": (-150, 150),
        "frequency": 1.2,
        "noise_level": 0.35,
    },
    "elevacion_lateral": {
        "ax_range": (-2.5, 2.5),
        "ay_range": (0.5, 2.5),
        "az_range": (8.0, 10.0),
        "gx_range": (-200, 200),
        "gy_range": (-350, 350),
        "gz_range": (-100, 100),
        "frequency": 1.0,
        "noise_level": 0.3,
    },
    "nado": {
        "ax_range": (-3.0, 3.0),
        "ay_range": (-3.0, 3.0),
        "az_range": (7.0, 10.0),
        "gx_range": (-400, 400),
        "gy_range": (-200, 200),
        "gz_range": (-300, 300),
        "frequency": 1.8,
        "noise_level": 0.4,
    },
    "remo": {
        "ax_range": (-2.0, 2.0),
        "ay_range": (0.0, 3.0),
        "az_range": (8.0, 10.0),
        "gx_range": (-250, 250),
        "gy_range": (-300, 300),
        "gz_range": (-100, 100),
        "frequency": 1.3,
        "noise_level": 0.32,
    },
    "neutro": {
        "ax_range": (-0.3, 0.3),
        "ay_range": (-0.3, 0.3),
        "az_range": (9.5, 10.0),
        "gx_range": (-10, 10),
        "gy_range": (-10, 10),
        "gz_range": (-10, 10),
        "frequency": 0.1,
        "noise_level": 0.1,
    },
}

# =================== GENERADOR DE DATOS ===================

class SensorSimulator:
    def __init__(self, exercise="bicep"):
        self.exercise = exercise
        self.config = EXERCISES.get(exercise, EXERCISES["neutro"])
        self.t_start = time.time()
        self.sample_count = 0
        self.last_prediction_time = 0
        self.rep_count = 0
        
    def set_exercise(self, exercise):
        """Cambiar ejercicio activo"""
        self.exercise = exercise
        self.config = EXERCISES.get(exercise, EXERCISES["neutro"])
        
    def _generate_signal(self, t, range_vals, frequency):
        """Generar señal oscilatoria realista"""
        center = (range_vals[0] + range_vals[1]) / 2
        amplitude = (range_vals[1] - range_vals[0]) / 2
        
        # Combinación de senos para movimiento más natural
        signal = center + amplitude * (
            0.6 * np.sin(2 * np.pi * frequency * t) +
            0.3 * np.sin(4 * np.pi * frequency * t) * 0.5 +
            0.1 * np.sin(6 * np.pi * frequency * t) * 0.3
        )
        
        # Añadir ruido
        noise = np.random.normal(0, self.config["noise_level"])
        return signal + noise
        
    def generate_sample(self):
        """Generar un sample de sensor"""
        t = time.time() - self.t_start
        freq = self.config["frequency"]
        
        ax = self._generate_signal(t, self.config["ax_range"], freq)
        ay = self._generate_signal(t, self.config["ay_range"], freq)
        az = self._generate_signal(t, self.config["az_range"], freq * 0.5)
        
        gx = self._generate_signal(t, self.config["gx_range"], freq)
        gy = self._generate_signal(t, self.config["gy_range"], freq)
        gz = self._generate_signal(t, self.config["gz_range"], freq)
        
        # Calcular magnitudes
        a_mag = np.sqrt(ax**2 + ay**2 + az**2)
        g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        
        self.sample_count += 1
        
        return {
            "type": "sample",
            "t": t,
            "ax": float(ax),
            "ay": float(ay),
            "az": float(az),
            "gx": float(gx),
            "gy": float(gy),
            "gz": float(gz),
            "a_mag": float(a_mag),
            "g_mag": float(g_mag),
        }
    
    def generate_prediction(self):
        """Generar predicción aleatoria con probabilidades"""
        current_time = time.time()
        
        # Predicción cada ~2 segundos
        if current_time - self.last_prediction_time < 2.0:
            return None
            
        self.last_prediction_time = current_time
        
        # Probabilidad alta para el ejercicio actual, baja para otros
        exercises = list(EXERCISES.keys())
        probs = []
        
        for exc in exercises:
            if exc == self.exercise:
                prob = np.random.uniform(0.5, 0.95)
            else:
                prob = np.random.uniform(0.0, 0.2)
            probs.append(prob)
        
        # Normalizar probabilidades
        total = sum(probs)
        probs = [p / total for p in probs]
        
        # Crear top-k
        sorted_probs = sorted(zip(exercises, probs), key=lambda x: x[1], reverse=True)
        topk = sorted_probs[:3]
        
        best_label = sorted_probs[0][0]
        best_prob = sorted_probs[0][1]
        
        # Aumentar contador de repeticiones ocasionalmente
        if np.random.random() < 0.05:  # 5% de probabilidad
            self.rep_count += 1
        
        return {
            "type": "pred",
            "best_label": best_label,
            "best_prob": float(best_prob),
            "topk": [(label, float(prob)) for label, prob in topk],
        }
    
    def generate_stats(self):
        """Generar estadísticas del sistema"""
        return {
            "type": "stats",
            "fps": SAMPLE_RATE + np.random.normal(0, 0.5),
            "latency_ms": np.random.uniform(5, 15),
            "packet_loss": np.random.uniform(0.0, 0.02),
        }

# =================== WEBSOCKET Y SERVIDOR ===================

class WebSocketHandler:
    def __init__(self):
        self.simulator = SensorSimulator("bicep")
        self.clients = set()
        self.running = True
        
    async def handle_ws(self, request):
        """Manejar conexión WebSocket"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.clients.add(ws)
        print(f"[WS] Cliente conectado. Total: {len(self.clients)}")
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get("action") == "change_exercise":
                            exercise = data.get("exercise", "bicep")
                            if exercise in EXERCISES:
                                self.simulator.set_exercise(exercise)
                                print(f"[WS] Ejercicio cambiado a: {exercise}")
                    except:
                        pass
        finally:
            self.clients.discard(ws)
            print(f"[WS] Cliente desconectado. Total: {len(self.clients)}")
        
        return ws
    
    async def broadcast_data(self):
        """Enviar datos continuamente a todos los clientes"""
        tick = 0
        while self.running:
            # Generar sample
            sample = self.simulator.generate_sample()
            
            # Enviar a todos los clientes
            for ws in self.clients:
                if not ws.closed:
                    try:
                        await ws.send_json(sample)
                    except:
                        pass
            
            # Cada 50 samples, enviar predicción
            if self.simulator.sample_count % 50 == 0:
                pred = self.simulator.generate_prediction()
                if pred:
                    for ws in self.clients:
                        if not ws.closed:
                            try:
                                await ws.send_json(pred)
                            except:
                                pass
            
            # Cada 100 samples, enviar estadísticas
            if self.simulator.sample_count % 100 == 0:
                stats = self.simulator.generate_stats()
                for ws in self.clients:
                    if not ws.closed:
                        try:
                            await ws.send_json(stats)
                        except:
                            pass
            
            # Mantener frecuencia de muestreo
            await asyncio.sleep(1.0 / SAMPLE_RATE)

# =================== RUTAS HTTP ===================

async def serve_dashboard(request):
    """Servir dashboard.html"""
    dashboard_path = Path(__file__).parent / "templates" / "dashboard.html"
    if dashboard_path.exists():
        async with aiofiles.open(dashboard_path, mode="r", encoding="utf-8") as f:
            content = await f.read()
        return web.Response(text=content, content_type="text/html")
    return web.Response(text="Dashboard not found", status=404)

async def serve_index(request):
    """Servir index.html (login)"""
    index_path = Path(__file__).parent / "templates" / "index.html"
    if index_path.exists():
        async with aiofiles.open(index_path, mode="r", encoding="utf-8") as f:
            content = await f.read()
        return web.Response(text=content, content_type="text/html")
    return web.Response(text="Index not found", status=404)

async def api_exercise_change(request):
    """API para cambiar ejercicio"""
    data = await request.json()
    exercise = data.get("exercise", "bicep")
    
    # Acceder al handler global
    if exercise in EXERCISES:
        request.app["ws_handler"].simulator.set_exercise(exercise)
        return web.json_response({"status": "ok", "exercise": exercise})
    
    return web.json_response({"status": "error", "message": "Unknown exercise"}, status=400)

# =================== MAIN ===================

async def main():
    """Iniciar servidor"""
    app = web.Application()
    ws_handler = WebSocketHandler()
    app["ws_handler"] = ws_handler
    
    # Rutas
    app.router.add_get("/", serve_dashboard)
    app.router.add_get("/index.html", serve_index)
    app.router.add_get("/dashboard.html", serve_dashboard)
    app.router.add_get("/ws", ws_handler.handle_ws)
    app.router.add_post("/api/change-exercise", api_exercise_change)
    
    # Servir archivos estáticos
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.router.add_static("/static/", path=static_path, name="static")
    
    # Iniciar servidor
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", PORT)
    await site.start()
    
    print(f"🚀 Servidor iniciado en http://127.0.0.1:{PORT}")
    print(f"📊 Abre http://127.0.0.1:{PORT}/dashboard.html en el navegador")
    print(f"\n📋 Ejercicios disponibles:")
    for exc in EXERCISES.keys():
        print(f"   - {exc}")
    
    # Corrutina de broadcast
    broadcast_task = asyncio.create_task(ws_handler.broadcast_data())
    
    try:
        await broadcast_task
    except KeyboardInterrupt:
        print("\n⚠️  Servidor cerrado")
        ws_handler.running = False
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Simulador finalizado")
