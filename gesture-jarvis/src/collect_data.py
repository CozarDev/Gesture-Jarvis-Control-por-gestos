"""
collect_data.py — Recolección de muestras de gestos con la webcam.

Uso:
    python src/collect_data.py

Controles:
    1  → seleccionar etiqueta THUMBS_UP
    2  → seleccionar etiqueta THUMBS_DOWN
    3  → seleccionar etiqueta INDEX_POINT
    4  → seleccionar etiqueta PINCH
    5  → seleccionar etiqueta UNKNOWN (gestos aleatorios)
    SPACE  → guardar muestra actual (landmarks + label) en CSV
    Q  → salir
"""

import os
import sys
import csv
import time
import cv2
import numpy as np


# Añadir raíz del proyecto al path para imports relativos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    init_mediapipe,
    detect_hand,
    extract_landmarks,
    normalize_landmarks,
    draw_landmarks,
    put_text,
    get_csv_columns,
    TOTAL_FEATURES,
)

# ─── Configuración ─────────────────────────────────────────────────────────────
GESTURES = {
    ord("1"): "THUMBS_UP",
    ord("2"): "THUMBS_DOWN",
    ord("3"): "INDEX_POINT",
    ord("4"): "PINCH",
    ord("5"): "UNKNOWN",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "raw_samples.csv")


def main() -> None:
    # Asegurar que existe el directorio de datos
    os.makedirs(DATA_DIR, exist_ok=True)

    # Inicializar CSV si no existe (escribir cabeceras)
    csv_exists = os.path.isfile(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    csv_file = open(CSV_PATH, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow(get_csv_columns())
        csv_file.flush()

    # MediaPipe Tasks API
    landmarker = init_mediapipe()

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la webcam.")
        return

    current_label = "THUMBS_UP"
    sample_count = 0

    # Contar muestras existentes
    if csv_exists:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            sample_count = max(0, sum(1 for _ in f) - 1)  # descontar cabecera

    print("=" * 60)
    print("  GestureJarvis — Recolección de datos")
    print("=" * 60)
    print("  1 = THUMBS_UP | 2 = THUMBS_DOWN")
    print("  3 = INDEX_POINT | 4 = PINCH")
    print("  5 = UNKNOWN (gestos aleatorios / mano abierta / etc.)")
    print("  SPACE = guardar muestra | Q = salir")
    print(f"  Muestras almacenadas: {sample_count}")
    print("=" * 60)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convertir BGR → RGB para MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Timestamp creciente para modo VIDEO
        frame_idx += 1
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= 0:
            timestamp_ms = frame_idx * 33  # ~30 fps fallback

        result = detect_hand(landmarker, rgb, timestamp_ms)

        landmarks_ready = False
        norm_landmarks = None

        if result is not None and result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]  # primera mano
            draw_landmarks(frame, hand_lms)
            raw = extract_landmarks(hand_lms)
            norm_landmarks = normalize_landmarks(raw)
            landmarks_ready = True

        # Overlay informativo
        color_label = (0, 255, 255)
        put_text(frame, f"Gesto: {current_label}", (10, 30), 0.8, color_label)
        put_text(frame, f"Muestras: {sample_count}", (10, 65), 0.7, (255, 255, 255))

        status = "Mano detectada" if landmarks_ready else "Sin mano"
        status_color = (0, 255, 0) if landmarks_ready else (0, 0, 255)
        put_text(frame, status, (10, 100), 0.7, status_color)

        put_text(
            frame,
            "SPACE=guardar | 1-5=gesto | Q=salir",
            (10, h - 20),
            0.5,
            (200, 200, 200),
        )

        cv2.imshow("GestureJarvis - Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF

        # Cambiar etiqueta
        if key in GESTURES:
            current_label = GESTURES[key]
            print(f"  [LABEL] → {current_label}")

        # Guardar muestra
        elif key == ord(" "):
            if landmarks_ready and norm_landmarks is not None:
                row = norm_landmarks.tolist() + [current_label]
                writer.writerow(row)
                csv_file.flush()
                sample_count += 1
                print(
                    f"  [SAVED] muestra #{sample_count} — {current_label}"
                )
            else:
                print("  [WARN] No se detecta mano. Muestra NO guardada.")

        # Salir
        elif key == ord("q") or key == ord("Q"):
            break

    # Limpieza
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\n  Total muestras guardadas: {sample_count}")
    print(f"  Archivo: {os.path.abspath(CSV_PATH)}")


if __name__ == "__main__":
    main()
