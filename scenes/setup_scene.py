"""
CoppeliaSim Scene Verifier & Joint Diagnostic Script
Laboratorio 3: CNN para Reconocimiento de Gestos (0 a 4 dedos)
Universidad Militar Nueva Granada - Inteligencia Artificial

Uso:
    python scenes/setup_scene.py

Qué hace este script:
    1. Intenta conectarse a CoppeliaSim via ZeroMQ (puerto 23000).
    2. Si la conexión es exitosa, descubre y muestra los handles de los 3 joints
       y el gripper del robot activo (según config/config.yaml -> active_robot).
    3. Si CoppeliaSim no está corriendo, muestra instrucciones de cómo preparar
       la escena manualmente.
"""

import sys
import os
import math
import time
from typing import Optional

# Ensure project root is in sys.path regardless of where the script is run from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.coppelia_client import CoppeliaSimClient
except ImportError:
    from coppelia_client import CoppeliaSimClient


def create_or_verify_coppelia_scene():
    """
    Connects to CoppeliaSim via ZeroMQ Remote API and verifies joint handles
    for the active robot profile defined in config/config.yaml.

    NOTA: El archivo .ttt de la escena NO existe en este repositorio porque
    es un archivo binario de CoppeliaSim que debes construir manualmente desde
    su GUI (Model Browser). Ver instrucciones abajo.
    """
    print("=" * 70)
    print("VERIFICADOR DE ESCENA COPPELIASIM")
    print("Robot activo: segun config/config.yaml -> coppeliasim.active_robot")
    print("=" * 70)

    client = CoppeliaSimClient()
    connected = client.connect()

    if not connected:
        print()
        print("[Aviso] CoppeliaSim no esta en ejecucion (puerto 23000 no disponible).")
        print()
        print("PASOS PARA PREPARAR LA ESCENA:")
        print("-" * 50)
        print(" 1. Abre CoppeliaSim Edu.")
        print(" 2. En el panel izquierdo, abre el 'Model browser'.")
        print("    Navega a: robots -> non-mobile")
        print("    Arrastra 'uArm with gripper' a la escena.")
        print(" 3. (Opcional) Agrega objetos: Add -> Primitive Shape ->")
        print("    Cylinder, Cube, Sphere. Colocalos frente al uArm.")
        print(" 4. Presiona el boton PLAY (triangulo verde) para iniciar")
        print("    la simulacion fisica.")
        print(" 5. Con la simulacion corriendo, ejecuta:")
        print("      python scenes/setup_scene.py   <- verifica joints")
        print("      python src/main_app.py          <- inicia la aplicacion")
        print()
        print(" TIP: Guarda la escena con File -> Save scene as...")
        print("      Ruta recomendada: scenes/uarm_scene.ttt")
        print("-" * 50)
        return

    sim = client.sim
    print("[Exito] Conexion establecida con CoppeliaSim.")
    print()

    # Discover all joints in the scene and show full diagnostic
    print("DIAGNOSTICO DE ARTICULACIONES:")
    print("-" * 50)

    joint_names = ["Joint1", "Joint2", "Joint3"]
    all_ok = True
    for jname in joint_names:
        handle = client.get_joint_handle(jname)
        is_real = handle is not None and not str(handle).startswith("EMU_")
        if is_real:
            try:
                pos_rad = sim.getJointPosition(handle)
                pos_deg = math.degrees(pos_rad)
                alias = sim.getObjectAlias(handle)
                status = f"OK  alias='{alias}' handle={handle}  pos={pos_deg:.1f} deg"
            except Exception as e:
                status = f"OK  handle={handle}  (no se pudo leer posicion: {e})"
        else:
            status = "NO ENCONTRADO EN ESCENA (emulado)"
            all_ok = False
        symbol = "+" if is_real else "!"
        print(f" [{symbol}] {jname}: {status}")

    print()
    if all_ok:
        print("[OK] Todos los joints detectados. La escena esta lista.")
        print("     Ejecuta: python src/main_app.py")
    else:
        print("[!] Algunos joints no fueron encontrados.")
        print("    Revisa que el modelo del robot este en la escena y")
        print("    que la simulacion este en estado PLAY.")
        print()
        print("    Si los aliases son diferentes, verifica el nombre del")
        print("    joint en CoppeliaSim (clic derecho -> Object properties -> Alias)")
        print("    y actualiza config/config.yaml -> coppeliasim.robots.uarm.joints")
    print("-" * 50)


if __name__ == "__main__":
    create_or_verify_coppelia_scene()
