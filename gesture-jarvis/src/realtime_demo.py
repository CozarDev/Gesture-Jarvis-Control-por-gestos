"""
realtime_demo.py — Demo en tiempo real de GestureJarvis.

Uso:
    python src/realtime_demo.py

Abre la webcam, detecta la mano, predice el gesto con el modelo entrenado
y ejecuta acciones del sistema:
    THUMBS_UP   → subir volumen
    THUMBS_DOWN → bajar volumen
    INDEX_POINT → modo ratón (mover cursor con el dedo índice)
    PINCH       → click izquierdo (transición de no-pinch → pinch)

Tecla Q para salir.
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import joblib
import pyautogui

import torch
import torch.nn as nn

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    init_mediapipe,
    detect_hand,
    extract_landmarks,
    normalize_landmarks,
    draw_landmarks,
    put_text,
    TOTAL_FEATURES,
)
from src.actions import volume_up, volume_down, mouse_move, left_click
from src.train_model import GestureMLP

# ─── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ─── Configuración ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80   # confianza mínima para ejecutar acción
MOUSE_SMOOTH_WINDOW = 5       # frames para moving average del ratón
MOUSE_ACTIVATION_DELAY = 0.5  # segundos que debe mantenerse INDEX_POINT
INDEX_TIP = 8                 # landmark del índice (punta)
RECOVERY_TIMEOUT = 8.0        # segundos sin mano antes de reiniciar landmarker
MAX_RECOVERY = 3              # máximo de reinicios antes de dejar de intentar

# Zona activa de la cámara para mapear al 100% de la pantalla.
# Si tu mano ocupa el rango ~10%-90% de la cámara, ese rango se
# estira para cubrir 0%-100% del monitor. Ajusta si necesitas más/menos margen.
CAM_MARGIN_X = 0.12           # margen horizontal (12% por cada lado)
CAM_MARGIN_Y = 0.12           # margen vertical   (12% por cada lado)

# ─── Smoothing para el ratón ──────────────────────────────────────────────────
class MouseSmoother:
    """Moving average para suavizar la posición del ratón."""

    def __init__(self, window_size: int = MOUSE_SMOOTH_WINDOW):
        self.window_size = window_size
        self.x_history: deque[int] = deque(maxlen=window_size)
        self.y_history: deque[int] = deque(maxlen=window_size)

    def update(self, x: int, y: int) -> tuple[int, int]:
        self.x_history.append(x)
        self.y_history.append(y)
        smooth_x = int(np.mean(self.x_history))
        smooth_y = int(np.mean(self.y_history))
        return smooth_x, smooth_y

    def reset(self):
        self.x_history.clear()
        self.y_history.clear()


# ─── Demo principal ───────────────────────────────────────────────────────────
def main() -> None:
    # ── Verificar artefactos ──
    for path, name in [
        (MODEL_PATH, "Modelo"),
        (SCALER_PATH, "Scaler"),
        (ENCODER_PATH, "LabelEncoder"),
    ]:
        if not os.path.isfile(path):
            print(f"[ERROR] {name} no encontrado: {path}")
            print("       Ejecuta primero: python src/train_model.py")
            sys.exit(1)

    # ── Cargar modelo ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = GestureMLP(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    classes = checkpoint["classes"]

    print(f"Modelo cargado. Clases: {classes}")

    # ── MediaPipe Tasks API ──
    landmarker = init_mediapipe()

    # ── Webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la webcam.")
        return

    # ── Estado ──
    smoother = MouseSmoother()
    mouse_mode = False
    index_point_start_time = 0.0     # cuando se empezó a detectar INDEX_POINT
    prev_pinch = False               # para detectar transición PINCH
    screen_w, screen_h = pyautogui.size()
    last_hand_time = time.time()     # último momento con mano detectada
    recovery_count = 0               # número de veces que se ha reiniciado

    print("=" * 60)
    print("  GestureJarvis — Demo en tiempo real")
    print("  Pulsa Q para salir")
    print("=" * 60)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Timestamp creciente para modo VIDEO
        frame_idx += 1
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= 0:
            timestamp_ms = frame_idx * 33  # ~30 fps fallback

        result = detect_hand(landmarker, rgb, timestamp_ms)

        gesture_text = "---"
        confidence = 0.0
        predicted_label = None

        if result is not None and result.hand_landmarks:
            last_hand_time = time.time()  # resetear timer de recovery
            recovery_count = 0            # mano encontrada → resetear intentos
            hand_lms = result.hand_landmarks[0]  # primera mano
            # Dibujar landmarks
            draw_landmarks(frame, hand_lms)

            # Extraer y normalizar
            raw = extract_landmarks(hand_lms)
            norm = normalize_landmarks(raw)

            # Escalar con StandardScaler
            scaled = scaler.transform(norm.reshape(1, -1)).astype(np.float32)

            # Predecir
            with torch.no_grad():
                tensor = torch.tensor(scaled, dtype=torch.float32).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, idx = probs.max(dim=1)
                confidence = conf.item()
                predicted_idx = idx.item()

            predicted_label = le.inverse_transform([predicted_idx])[0]
            gesture_text = f"{predicted_label} ({confidence:.2f})"

            # ── Ejecutar acciones con umbral de confianza ──
            if confidence >= CONFIDENCE_THRESHOLD and predicted_label != "UNKNOWN":
                _execute_action(
                    predicted_label,
                    hand_lms,
                    w, h,
                    screen_w, screen_h,
                    smoother,
                )
            else:
                # Si la confianza baja, no mantener modo ratón
                predicted_label = None

        else:
            # Sin mano: desactivar modo ratón y resetear
            mouse_mode = False
            index_point_start_time = 0.0
            smoother.reset()
            prev_pinch = False

            # Auto-recovery: si no detecta mano durante RECOVERY_TIMEOUT,
            # reiniciar el landmarker para evitar que se quede "pillado".
            # Se limita a MAX_RECOVERY intentos para no entrar en bucle.
            elapsed_no_hand = time.time() - last_hand_time
            if elapsed_no_hand > RECOVERY_TIMEOUT and recovery_count < MAX_RECOVERY:
                try:
                    landmarker.close()
                    landmarker = init_mediapipe()
                    last_hand_time = time.time()  # resetear timer
                    recovery_count += 1
                    frame_idx = 0  # resetear timestamps
                    print(f"  [RECOVERY] Landmarker reiniciado (x{recovery_count})")
                except Exception as e:
                    print(f"  [ERROR] Recovery falló: {e}")

        # ── Gestionar modo ratón (requiere 0.5s continuos) ──
        mouse_mode = _update_mouse_mode(
            predicted_label, mouse_mode, smoother
        )

        # ── Overlay ──
        put_text(frame, f"Gesto: {gesture_text}", (10, 30), 0.8, (0, 255, 0))

        mouse_str = "ON" if mouse_mode else "OFF"
        mouse_color = (0, 255, 0) if mouse_mode else (0, 0, 255)
        put_text(frame, f"Raton: {mouse_str}", (10, 65), 0.7, mouse_color)
        put_text(frame, "Q = salir", (10, h - 20), 0.5, (200, 200, 200))

        cv2.imshow("GestureJarvis - Demo", frame)

        if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
            break

    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Demo finalizada.")


# ─── Variables de estado (module-level para persistencia entre frames) ─────────
_index_point_start: float = 0.0
_mouse_mode_active: bool = False
_prev_pinch: bool = False
_smoother_ref: MouseSmoother | None = None


def _update_mouse_mode(
    predicted_label: str | None,
    current_mode: bool,
    smoother: MouseSmoother,
) -> bool:
    """Actualiza el estado del modo ratón.

    El modo ratón se activa si INDEX_POINT se mantiene al menos
    MOUSE_ACTIVATION_DELAY segundos. Se desactiva cuando cambia de gesto.
    """
    global _index_point_start, _mouse_mode_active

    if predicted_label == "INDEX_POINT":
        if _index_point_start == 0.0:
            _index_point_start = time.time()

        elapsed = time.time() - _index_point_start
        if elapsed >= MOUSE_ACTIVATION_DELAY:
            _mouse_mode_active = True
    else:
        _index_point_start = 0.0
        if _mouse_mode_active:
            _mouse_mode_active = False
            smoother.reset()

    return _mouse_mode_active


def _execute_action(
    label: str,
    hand_lms,
    frame_w: int,
    frame_h: int,
    screen_w: int,
    screen_h: int,
    smoother: MouseSmoother,
) -> None:
    """Ejecuta la acción correspondiente al gesto detectado."""
    global _prev_pinch

    if label == "THUMBS_UP":
        volume_up()

    elif label == "THUMBS_DOWN":
        volume_down()

    elif label == "INDEX_POINT":
        # El modo ratón se controla por _update_mouse_mode
        if _mouse_mode_active:
            # Landmark 8 = punta del dedo índice
            lm = hand_lms[INDEX_TIP]
            # Mapear zona activa de la cámara (con márgenes) → pantalla completa
            # Esto permite que sin llevar la mano al borde de la cámara
            # el cursor pueda alcanzar los bordes del monitor.
            norm_x = (lm.x - CAM_MARGIN_X) / (1.0 - 2 * CAM_MARGIN_X)
            norm_y = (lm.y - CAM_MARGIN_Y) / (1.0 - 2 * CAM_MARGIN_Y)
            # Clamp a [0, 1] para no salirse de la pantalla
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            raw_x = int(norm_x * screen_w)
            raw_y = int(norm_y * screen_h)
            smooth_x, smooth_y = smoother.update(raw_x, raw_y)
            mouse_move(smooth_x, smooth_y)

    elif label == "PINCH":
        # Solo click en transición: no-pinch → pinch
        if not _prev_pinch:
            left_click()
        _prev_pinch = True
        return  # no resetear _prev_pinch abajo

    # Resetear estado de pinch si no es PINCH
    if label != "PINCH":
        _prev_pinch = False


if __name__ == "__main__":
    main()
