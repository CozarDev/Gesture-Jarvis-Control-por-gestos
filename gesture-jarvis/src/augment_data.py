"""
augment_data.py — Aumentación sintética de landmarks para GestureJarvis.

Genera variantes artificiales a partir de las muestras recolectadas,
multiplicando el dataset sin necesidad de volver a grabar.

Técnicas aplicadas:
    1. Ruido gaussiano          — simula imprecisión de detección
    2. Escalado aleatorio       — simula cambio de distancia a la cámara
    3. Rotación 2D (x,y)       — simula inclinación de la mano
    4. Espejado horizontal      — simula mano izquierda/derecha
    5. Jitter por landmark      — ruido independiente por punto

Uso:
    python src/augment_data.py
    python src/augment_data.py --factor 5       (x5 aumento, default x3)
    python src/augment_data.py --no-mirror       (sin espejado)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import NUM_LANDMARKS, FEATURES_PER_LANDMARK, TOTAL_FEATURES

# ─── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV = os.path.join(BASE_DIR, "data", "raw_samples.csv")
AUG_CSV = os.path.join(BASE_DIR, "data", "augmented_samples.csv")


# ─── Funciones de aumentación ─────────────────────────────────────────────────
def add_gaussian_noise(landmarks: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    """Añade ruido gaussiano a todos los landmarks.

    Simula la imprecisión natural de MediaPipe entre frames.

    Parameters
    ----------
    landmarks : np.ndarray (63,)
        Landmarks normalizados.
    sigma : float
        Desviación estándar del ruido. Valores típicos: 0.01-0.03.

    Returns
    -------
    np.ndarray (63,) con ruido añadido.
    """
    noise = np.random.normal(0, sigma, landmarks.shape)
    return landmarks + noise


def random_scale(landmarks: np.ndarray, scale_range: tuple = (0.85, 1.15)) -> np.ndarray:
    """Escala todos los landmarks por un factor aleatorio.

    Simula variaciones en la distancia entre la mano y la cámara.
    Como los landmarks ya están normalizados (divididos por dist muñeca→middle_mcp),
    este escalado simula pequeñas diferencias residuales.

    Parameters
    ----------
    landmarks : np.ndarray (63,)
    scale_range : tuple (min, max)
        Rango del factor de escala.

    Returns
    -------
    np.ndarray (63,) escalado.
    """
    factor = np.random.uniform(*scale_range)
    return landmarks * factor


def rotate_2d(landmarks: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
    """Rota los landmarks en el plano XY alrededor del origen (muñeca).

    Simula la inclinación natural de la mano al hacer el gesto.
    Solo afecta a las coordenadas x,y; z se mantiene.

    Parameters
    ----------
    landmarks : np.ndarray (63,)
    max_angle_deg : float
        Ángulo máximo de rotación en grados (±).

    Returns
    -------
    np.ndarray (63,) rotado.
    """
    pts = landmarks.copy().reshape(NUM_LANDMARKS, FEATURES_PER_LANDMARK)
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    pts[:, 0] = cos_a * x - sin_a * y
    pts[:, 1] = sin_a * x + cos_a * y

    return pts.flatten()


def mirror_horizontal(landmarks: np.ndarray) -> np.ndarray:
    """Espeja los landmarks en el eje X.

    Simula la misma pose pero con la mano opuesta.
    Invierte la coordenada x de todos los puntos.

    Parameters
    ----------
    landmarks : np.ndarray (63,)

    Returns
    -------
    np.ndarray (63,) espejado.
    """
    pts = landmarks.copy().reshape(NUM_LANDMARKS, FEATURES_PER_LANDMARK)
    pts[:, 0] = -pts[:, 0]  # invertir x
    return pts.flatten()


def landmark_jitter(landmarks: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Aplica ruido independiente por landmark (más sutil que el global).

    Simula el micro-movimiento de cada dedo individualmente.

    Parameters
    ----------
    landmarks : np.ndarray (63,)
    sigma : float
        Desviación estándar del jitter por punto.

    Returns
    -------
    np.ndarray (63,) con jitter.
    """
    pts = landmarks.copy().reshape(NUM_LANDMARKS, FEATURES_PER_LANDMARK)
    # Más ruido en las puntas de los dedos (landmarks 4, 8, 12, 16, 20)
    # ya que son las que más se mueven naturalmente
    fingertips = [4, 8, 12, 16, 20]
    for i in range(NUM_LANDMARKS):
        s = sigma * 2.0 if i in fingertips else sigma
        pts[i] += np.random.normal(0, s, FEATURES_PER_LANDMARK)
    return pts.flatten()


def augment_sample(landmarks: np.ndarray, use_mirror: bool = True) -> list[np.ndarray]:
    """Genera múltiples variantes de una muestra.

    Aplica combinaciones aleatorias de las técnicas de aumentación.

    Parameters
    ----------
    landmarks : np.ndarray (63,)
    use_mirror : bool
        Si True, incluye una variante espejada.

    Returns
    -------
    list[np.ndarray] — lista de variantes generadas.
    """
    variants = []

    # Variante 1: ruido + rotación leve
    v = add_gaussian_noise(landmarks, sigma=0.015)
    v = rotate_2d(v, max_angle_deg=10.0)
    variants.append(v)

    # Variante 2: escala + jitter
    v = random_scale(landmarks, (0.9, 1.1))
    v = landmark_jitter(v, sigma=0.012)
    variants.append(v)

    # Variante 3: ruido fuerte + rotación mayor
    v = add_gaussian_noise(landmarks, sigma=0.025)
    v = rotate_2d(v, max_angle_deg=15.0)
    v = random_scale(v, (0.85, 1.15))
    variants.append(v)

    # Variante 4: jitter solo (sutil)
    v = landmark_jitter(landmarks, sigma=0.015)
    variants.append(v)

    # Variante 5: espejado + ruido
    if use_mirror:
        v = mirror_horizontal(landmarks)
        v = add_gaussian_noise(v, sigma=0.01)
        variants.append(v)

    return variants


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Aumentación sintética de landmarks")
    parser.add_argument(
        "--factor", type=int, default=3,
        help="Factor de aumento: cuántas variantes por muestra original (default: 3)"
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="No generar variantes espejadas"
    )
    parser.add_argument(
        "--input", type=str, default=RAW_CSV,
        help=f"CSV de entrada (default: {RAW_CSV})"
    )
    parser.add_argument(
        "--output", type=str, default=AUG_CSV,
        help=f"CSV de salida (default: {AUG_CSV})"
    )
    args = parser.parse_args()

    # Cargar datos originales
    if not os.path.isfile(args.input):
        print(f"[ERROR] No se encontró el dataset: {args.input}")
        print("        Ejecuta primero: python src/collect_data.py")
        sys.exit(1)

    df = pd.read_csv(args.input)
    original_count = len(df)

    print("=" * 60)
    print("  GestureJarvis — Aumentación Sintética de Datos")
    print("=" * 60)
    print(f"  Dataset original: {original_count} muestras")
    print(f"  Distribución original:")
    for label, count in df["label"].value_counts().items():
        print(f"    {label}: {count}")
    print(f"  Factor de aumento: x{args.factor}")
    print(f"  Espejado: {'NO' if args.no_mirror else 'SÍ'}")
    print("=" * 60)

    use_mirror = not args.no_mirror
    columns = df.columns.tolist()
    feature_cols = columns[:-1]

    augmented_rows = []

    for idx, row in df.iterrows():
        landmarks = row[feature_cols].values.astype(np.float32)
        label = row["label"]

        # Generar variantes
        all_variants = augment_sample(landmarks, use_mirror=use_mirror)

        # Tomar solo las que nos pide el factor
        selected = all_variants[:args.factor]

        for variant in selected:
            new_row = variant.tolist() + [label]
            augmented_rows.append(new_row)

    # Combinar original + aumentado
    aug_df = pd.DataFrame(augmented_rows, columns=columns)
    combined_df = pd.concat([df, aug_df], ignore_index=True)

    # Guardar
    combined_df.to_csv(args.output, index=False)

    print(f"\n  Muestras originales: {original_count}")
    print(f"  Muestras generadas:  {len(aug_df)}")
    print(f"  Total combinado:     {len(combined_df)}")
    print(f"\n  Distribución final:")
    for label, count in combined_df["label"].value_counts().items():
        print(f"    {label}: {count}")
    print(f"\n  Guardado en: {os.path.abspath(args.output)}")
    print("\n  Ahora entrena con:")
    print(f"    python src/train_model.py --data {os.path.relpath(args.output, BASE_DIR)}")
    print("\n  ¡Aumentación completada!")


if __name__ == "__main__":
    main()
