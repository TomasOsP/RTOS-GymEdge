#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidad para cambiar ejercicios en el simulador
Permite rotar entre ejercicios automáticamente o manualmente
"""

import asyncio
import aiohttp
import time
import sys

API_URL = "http://127.0.0.1:5000/api/change-exercise"

EXERCISES = ["bicep", "circular", "elevacion_lateral", "nado", "remo", "neutro"]

async def change_exercise(exercise):
    """Cambiar ejercicio activo"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, json={"exercise": exercise}) as resp:
                result = await resp.json()
                if resp.status == 200:
                    print(f"✅ Ejercicio cambiado a: {exercise}")
                    return True
                else:
                    print(f"❌ Error: {result.get('message')}")
                    return False
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

async def auto_rotate(interval=10):
    """Rotar automáticamente entre ejercicios"""
    idx = 0
    print(f"🔄 Rotación automática cada {interval} segundos")
    try:
        while True:
            exercise = EXERCISES[idx % len(EXERCISES)]
            print(f"\n[{time.strftime('%H:%M:%S')}] Cambiando a: {exercise}")
            await change_exercise(exercise)
            idx += 1
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\n✅ Rotación detenida")

async def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "auto":
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            await auto_rotate(interval)
        else:
            exercise = sys.argv[1]
            if exercise in EXERCISES:
                await change_exercise(exercise)
            else:
                print(f"❌ Ejercicio no válido: {exercise}")
                print(f"Disponibles: {', '.join(EXERCISES)}")
    else:
        print("Uso:")
        print(f"  python {sys.argv[0]} <ejercicio>          # Cambiar a ejercicio específico")
        print(f"  python {sys.argv[0]} auto [intervalo]     # Rotar automáticamente")
        print(f"\nEjercicios disponibles: {', '.join(EXERCISES)}")

if __name__ == "__main__":
    asyncio.run(main())
