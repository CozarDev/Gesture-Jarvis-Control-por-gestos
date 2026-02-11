# GestureJarvis

**Reconocimiento de gestos con webcam + acciones del sistema** — Proyecto con OpenCV, MediaPipe y PyTorch.

GestureJarvis captura vídeo en tiempo real, detecta la mano, clasifica gestos mediante un modelo MLP entrenado con PyTorch, y ejecuta acciones del sistema operativo (volumen, ratón, click).

---

## Gestos soportados

| Gesto | Acción |
|-------|--------|
| 👍 `THUMBS_UP` | Subir volumen del sistema |
| 👎 `THUMBS_DOWN` | Bajar volumen del sistema |
| ☝️ `INDEX_POINT` | Activar modo ratón (mover cursor con el dedo índice) |
| 🤏 `PINCH` | Click izquierdo (pellizco pulgar + índice) |

---

## Pipeline completo

```
Webcam → MediaPipe Hands → 21 landmarks (63 features)
    → Normalización (origen en muñeca + escala)
        → StandardScaler → MLP (PyTorch) → Predicción + Confianza
            → Si confianza > 80%: ejecutar acción del sistema
```

---

## Estructura del proyecto

```
gesture-jarvis/
├── data/
│   └── raw_samples.csv          # Dataset recolectado
├── models/
│   ├── gesture_model.pt         # Modelo PyTorch entrenado
│   ├── scaler.pkl               # StandardScaler (sklearn)
│   └── label_encoder.pkl        # LabelEncoder (sklearn)
├── src/
│   ├── __init__.py
│   ├── utils.py                 # MediaPipe, landmarks, normalización
│   ├── collect_data.py          # Recolección de muestras con webcam
│   ├── train_model.py           # Entrenamiento del MLP
│   ├── realtime_demo.py         # Demo en tiempo real
│   └── actions.py               # Acciones del sistema (volumen, ratón)
├── requirements.txt
└── README.md
```

---

## Requisitos previos

- **Windows 10/11**
- **Python 3.11**
- **Webcam** funcional
- **Conda** instalado (Anaconda o Miniconda)

---

## Instalación

### 1. Crear entorno conda

```bash
conda create -n gesturejarvis python=3.11 -y
conda activate gesturejarvis
```

### 2. Instalar dependencias

```bash
cd gesture-jarvis
pip install -r requirements.txt
```

> **Nota GPU:** El proyecto funciona **sin GPU**. Si tienes una GPU NVIDIA compatible, instala la versión CUDA de PyTorch:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Verificar instalación

```bash
python -c "import cv2, mediapipe, torch, pyautogui; print('Todo OK'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Uso

El flujo de trabajo tiene 3 pasos: **recolectar datos → entrenar → ejecutar demo**.

### Paso 1: Recolectar dataset

```bash
python src/collect_data.py
```

**Controles:**
| Tecla | Acción |
|-------|--------|
| `1` | Seleccionar etiqueta `THUMBS_UP` |
| `2` | Seleccionar etiqueta `THUMBS_DOWN` |
| `3` | Seleccionar etiqueta `INDEX_POINT` |
| `4` | Seleccionar etiqueta `PINCH` |
| `ESPACIO` | Guardar muestra actual |
| `Q` | Salir |

**Recomendaciones para el dataset:**

- **Mínimo 200 muestras por gesto** (800 total).
- Lo ideal son **300-500 por gesto**.
- Varía la posición de la mano (centro, izquierda, derecha, arriba, abajo).
- Varía la distancia a la cámara (cerca, medio, lejos).
- Varía ligeramente la orientación de la mano.
- Graba con la iluminación habitual de tu entorno.
- Usa ambas manos si quieres soporte para las dos.
- Las muestras se acumulan en `data/raw_samples.csv`; puedes ejecutar el script múltiples veces.

### Paso 2: Entrenar el modelo

```bash
python src/train_model.py
```

Salida esperada:
- **Accuracy** en el conjunto de test.
- **Classification report** con precision/recall por gesto.
- Artefactos guardados en `models/`.

Si el accuracy es bajo (<90%), recolecta más muestras o revisa que los gestos sean suficientemente diferentes entre sí.

### Paso 3: Ejecutar la demo en tiempo real

```bash
python src/realtime_demo.py
```

La ventana mostrará:
- Los landmarks de la mano dibujados.
- El gesto predicho y su confianza.
- Si el modo ratón está `ON` o `OFF`.

**Pulsa `Q` para salir.**

---

## Detalles técnicos

### Normalización de landmarks

Para que el modelo sea **robusto a posición y escala**:

1. Se resta la posición de la **muñeca** (landmark 0) como origen → elimina dependencia de la posición en el frame.
2. Se divide por la **distancia muñeca → MIDDLE_MCP** (landmark 9) → elimina dependencia del tamaño de la mano / distancia a la cámara.

### Modelo MLP

```
Input (63) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
           → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
           → Linear(num_classes)
```

Se entrena con **Adam** (lr=0.001), **CrossEntropyLoss**, durante **50 epochs**.

### Modo ratón

- Se activa solo si `INDEX_POINT` se mantiene **≥ 0.5 segundos** continuos.
- La posición del landmark 8 (punta del índice) se mapea a coordenadas de pantalla.
- Se aplica un **moving average** (ventana de 5 frames) para suavizar el movimiento.

### Click con PINCH

- Solo se ejecuta en la **transición** de "no pinch" → "pinch".
- Mientras se mantiene el gesto, **no se repite** el click.
- Tiene un **cooldown de 0.5 segundos**.

### Control de volumen

- Usa **pycaw** (COM API de Windows) para controlar el volumen del sistema.
- Cada acción cambia un **5%** del volumen.
- Cooldown de **0.3 segundos** para evitar cambios bruscos.

---

## Solución de problemas

| Problema | Solución |
|----------|----------|
| `No se pudo abrir la webcam` | Verifica que la webcam está conectada y no la usa otra app |
| `Modelo no encontrado` | Ejecuta `python src/train_model.py` primero |
| `Accuracy muy baja` | Recolecta más muestras (≥200/gesto) con variedad |
| `El ratón se mueve erráticamente` | Aumenta `MOUSE_SMOOTH_WINDOW` en `realtime_demo.py` |
| `El volumen no cambia` | Ejecuta como administrador si los permisos COM fallan |
| `Error de importación pycaw` | Asegúrate de tener `comtypes` instalado: `pip install comtypes` |

---

## Licencia

Proyecto educativo — Curso EOI. Pedro Manuel Cózar Ortiz.
