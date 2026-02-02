#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script todo-en-uno para iniciar el simulador con rotación automática de ejercicios
Útil para grabar el GIF de forma rápida
"""

import subprocess
import time
import sys
import os
import signal

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    simulator_script = os.path.join(script_dir, "simulate_sensor.py")
    rotate_script = os.path.join(script_dir, "change_exercise.py")
    
    print("🎬 Iniciando simulador de GIF...")
    print("=" * 60)
    
    # Iniciar servidor simulador
    print("1️⃣  Iniciando servidor simulador...")
    simulator_process = subprocess.Popen(
        [sys.executable, simulator_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Esperar a que el servidor esté listo
    time.sleep(2)
    
    print("✅ Servidor iniciado")
    print("\n📊 Dashboard disponible en: http://127.0.0.1:5000/dashboard.html")
    print("\n2️⃣  Iniciando rotación automática de ejercicios...")
    print("   (Cambiará cada 8 segundos)")
    print("\n⏱️  Abre el navegador y empieza a grabar el GIF")
    print("=" * 60)
    print("\n💡 Tip: Usa Cmd+Tab para cambiar entre ventanas rápidamente")
    print("💾 Exporta el GIF en formato GIF con 10-12 FPS\n")
    
    try:
        # Iniciar rotación de ejercicios
        subprocess.run(
            [sys.executable, rotate_script, "auto", "8"],
            check=False
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Deteniendo...")
    finally:
        # Limpiar procesos
        simulator_process.terminate()
        try:
            simulator_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            simulator_process.kill()
        
        print("✅ Simulador detenido")

if __name__ == "__main__":
    main()
