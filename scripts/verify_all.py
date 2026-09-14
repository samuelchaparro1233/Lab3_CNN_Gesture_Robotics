"""
Automated Verification Suite for Lab 3 (CNN Gesture Recognition + CoppeliaSim)
Laboratorio 3: CNN para Reconocimiento de Gestos
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import os
import sys
import json
import yaml

def verify_all():
    print("=" * 70)
    print("VERIFICADOR INTEGRAL DE PROYECTO - LABORATORIO 3")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 0

    # 1. Check directories
    required_dirs = ["config", "dataset", "models", "results", "results/plots", "results/metrics", "src", "scenes", "docs"]
    for d in required_dirs:
        total_checks += 1
        if os.path.isdir(d):
            print(f" [PASS] Directorio: {d}")
            checks_passed += 1
        else:
            print(f" [FAIL] Directorio faltante: {d}")

    # 2. Check key code files
    required_files = [
        "config/config.yaml",
        "src/__init__.py",
        "src/model.py",
        "src/dataset.py",
        "src/generate_dataset.py",
        "src/train.py",
        "src/command_filter.py",
        "src/coppelia_client.py",
        "src/robot_adapter.py",
        "src/cnn_inference.py",
        "src/main_app.py",
        "src/benchmark.py",
        "scenes/setup_scene.py",
        "README.md",
        "docs/INFORME_LAB3_CNN_IEEE.md",
        "docs/PLANTILLA_ABET_LAB03.md"
    ]
    for f in required_files:
        total_checks += 1
        if os.path.isfile(f) and os.path.getsize(f) > 0:
            print(f" [PASS] Archivo: {f} ({os.path.getsize(f):,} bytes)")
            checks_passed += 1
        else:
            print(f" [FAIL] Archivo faltante o vacío: {f}")

    # 3. Check plots
    required_plots = [
        "results/plots/learning_curves.png",
        "results/plots/confusion_matrix.png",
        "results/plots/latency_distribution.png",
        "results/plots/live_trials_performance.png",
        "results/plots/decoupled_layers_diagnostic.png",
        "results/plots/robustness_degradation.png"
    ]
    for p in required_plots:
        total_checks += 1
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            print(f" [PASS] Gráfica de Resultados: {p}")
            checks_passed += 1
        else:
            print(f" [FAIL] Gráfica faltante: {p}")

    # 4. Check metrics JSON
    metrics_files = [
        "results/metrics/test_metrics.json",
        "results/metrics/benchmark_protocol_results.json",
        "results/metrics/robustness_metrics.json"
    ]
    for mf in metrics_files:
        total_checks += 1
        if os.path.isfile(mf):
            print(f" [PASS] Archivo de Métricas: {mf}")
            checks_passed += 1
        else:
            print(f" [FAIL] Archivo de métricas no encontrado: {mf}")

    print("=" * 70)
    print(f"RESULTADO DE VERIFICACIÓN: {checks_passed}/{total_checks} comprobaciones superadas exitosamente ({(checks_passed/total_checks)*100:.1f}%).")
    print("=" * 70)
    return checks_passed == total_checks

if __name__ == "__main__":
    success = verify_all()
    sys.exit(0 if success else 1)
