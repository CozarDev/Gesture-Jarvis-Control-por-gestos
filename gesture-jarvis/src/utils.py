"""
utils.py — Utilidades para GestureJarvis.

Funciones de inicialización de MediaPipe Tasks API (>=0.10.14),
extracción y normalización de landmarks, y dibujado sobre frames de OpenCV.
"""

import os
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python as mp_tasks


# ─── Constantes ────────────────────────────────────────────────────────────────
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3          # x, y, z
TOTAL_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63

# Índices de landmarks de referencia para normalización
WRIST = 0
MIDDLE_MCP = 9

# Ruta al modelo hand_landmarker.task (en gesture-jarvis/models/)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
)

# Conexiones entre landmarks para dibujar el esqueleto de la mano
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # índice
    (0, 9), (9, 10), (10, 11), (11, 12),   # medio
    (0, 13), (13, 14), (14, 15), (15, 16), # anular
    (0, 17), (17, 18), (18, 19), (19, 20), # meñique
    (5, 9), (9, 13), (13, 17),             # nudillos
]


# ─── Inicialización de MediaPipe ───────────────────────────────────────────────
def init_mediapipe(
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.5,
    model_path: str | None = None,
):
    """Crea un HandLandmarker con la MediaPipe Tasks API.

    Parameters
    ----------
    max_num_hands : int
    min_detection_confidence : float
    min_tracking_confidence : float
    model_path : str | None
        Ruta al fichero .task. Por defecto usa models/hand_landmarker.task.

    Returns
    -------
    hand_landmarker : vision.HandLandmarker
        Detector configurado en modo VIDEO.
    """
    if model_path is None:
        model_path = _DEFAULT_MODEL_PATH

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            "Descárgalo de: https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


# ─── Detección de landmarks en un frame ───────────────────────────────────────
def detect_hand(landmarker, frame_rgb: np.ndarray, timestamp_ms: int):
    """Detecta manos en un frame y devuelve el resultado.

    Parameters
    ----------
    landmarker : vision.HandLandmarker
    frame_rgb : np.ndarray
        Frame en formato RGB.
    timestamp_ms : int
        Timestamp en milisegundos (debe ser creciente para VIDEO mode).

    Returns
    -------
    result : HandLandmarkerResult  o  None si no hay mano.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if result.hand_landmarks:
        return result
    return None


# ─── Extracción de landmarks ──────────────────────────────────────────────────
def extract_landmarks(hand_landmarks_list: list) -> np.ndarray:
    """Extrae las 63 features (21 landmarks × 3 coordenadas) en un array 1-D.

    Parameters
    ----------
    hand_landmarks_list : list[NormalizedLandmark]
        Lista de 21 NormalizedLandmark de MediaPipe Tasks API.

    Returns
    -------
    np.ndarray de shape (63,) con [x0, y0, z0, x1, y1, z1, ...].
    """
    coords = []
    for lm in hand_landmarks_list:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


# ─── Normalización de landmarks ───────────────────────────────────────────────
def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """Normaliza landmarks para invarianza a posición y escala.

    Pasos:
        1. Resta la muñeca (landmark 0) como origen.
        2. Escala dividiendo por la distancia muñeca → MIDDLE_MCP
           para que el tamaño de la mano no afecte.

    Parameters
    ----------
    raw : np.ndarray (63,)
        Landmarks crudos.

    Returns
    -------
    np.ndarray (63,) normalizado.
    """
    pts = raw.copy().reshape(NUM_LANDMARKS, FEATURES_PER_LANDMARK)

    # Origen en la muñeca
    wrist = pts[WRIST].copy()
    pts -= wrist

    # Distancia de referencia (muñeca → MIDDLE_MCP)
    ref_dist = np.linalg.norm(pts[MIDDLE_MCP])
    if ref_dist < 1e-6:
        ref_dist = 1e-6  # evitar división por cero

    pts /= ref_dist
    return pts.flatten()


# ─── Dibujado de landmarks ────────────────────────────────────────────────────
def draw_landmarks(frame: np.ndarray, hand_landmarks_list: list) -> None:
    """Dibuja los landmarks y las conexiones sobre el frame.

    Implementación manual ya que mp.solutions.drawing_utils no existe
    en mediapipe >= 0.10.14.

    Parameters
    ----------
    frame : np.ndarray
        Imagen BGR de OpenCV (se modifica in-place).
    hand_landmarks_list : list[NormalizedLandmark]
        Lista de 21 NormalizedLandmark de la Tasks API.
    """
    h, w, _ = frame.shape

    # Convertir a coordenadas de píxel
    points = []
    for lm in hand_landmarks_list:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Dibujar conexiones
    for i, j in HAND_CONNECTIONS:
        cv2.line(frame, points[i], points[j], (255, 255, 255), 2)

    # Dibujar puntos
    for px, py in points:
        cv2.circle(frame, (px, py), 4, (0, 255, 0), cv2.FILLED)


# ─── Overlay de texto ─────────────────────────────────────────────────────────
def put_text(
    frame,
    text: str,
    position: tuple = (10, 30),
    font_scale: float = 0.8,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
):
    """Escribe texto sobre un frame con fondo semitransparente.

    Parameters
    ----------
    frame : np.ndarray
        Imagen BGR.
    text : str
    position : tuple (x, y)
    font_scale : float
    color : tuple BGR
    thickness : int
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # Fondo oscuro para legibilidad
    cv2.rectangle(
        frame,
        (x - 2, y - th - 4),
        (x + tw + 4, y + baseline + 4),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(frame, text, position, font, font_scale, color, thickness)


# ─── Nombres de columns para CSV ──────────────────────────────────────────────
def get_csv_columns() -> list[str]:
    """Genera los nombres de columnas del CSV: x0, y0, z0, ..., x20, y20, z20, label."""
    cols = []
    for i in range(NUM_LANDMARKS):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])
    cols.append("label")
    return cols
