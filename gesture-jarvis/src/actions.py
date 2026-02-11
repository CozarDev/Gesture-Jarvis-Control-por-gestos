"""
actions.py — Acciones del sistema controladas por gestos.

Incluye control de volumen (pycaw), movimiento de ratón y click (pyautogui).
Todas las acciones tienen cooldown para evitar disparos repetitivos.
"""

import time
import pyautogui
from pycaw.pycaw import AudioUtilities


# ─── Configuración de PyAutoGUI ────────────────────────────────────────────────
pyautogui.FAILSAFE = True   # mover el ratón a (0,0) aborta
pyautogui.PAUSE = 0.0       # sin pausa entre llamadas


# ─── Control de volumen (pycaw / COM) ─────────────────────────────────────────
def _get_volume_interface():
    """Obtiene la interfaz de volumen del endpoint de audio por defecto."""
    devices = AudioUtilities.GetSpeakers()
    return devices.EndpointVolume


# Interfaz de volumen (se inicializa una sola vez)
try:
    _volume = _get_volume_interface()
    print("[INFO] Control de volumen inicializado correctamente")
except Exception as e:
    print(f"[ERROR] No se pudo inicializar control de volumen: {e}")
    _volume = None


# ─── Cooldown genérico ────────────────────────────────────────────────────────
_last_action_time: dict[str, float] = {}


def _check_cooldown(action_name: str, cooldown_s: float = 0.3) -> bool:
    """Devuelve True si ha pasado suficiente tiempo desde la última ejecución.

    Parameters
    ----------
    action_name : str
        Identificador de la acción (e.g. "volume_up").
    cooldown_s : float
        Segundos mínimos entre ejecuciones.

    Returns
    -------
    bool  True → se puede ejecutar; False → todavía en cooldown.
    """
    now = time.time()
    last = _last_action_time.get(action_name, 0.0)
    if now - last >= cooldown_s:
        _last_action_time[action_name] = now
        return True
    return False


# ─── Acciones ──────────────────────────────────────────────────────────────────
VOLUME_STEP = 0.05  # 5 % por paso


def volume_up() -> None:
    """Sube el volumen del sistema un paso (~5 %).

    Respeta un cooldown de 0.3 s para no dispararse.
    """
    if _volume is None:
        print("[WARN] volume_up: interfaz de volumen no disponible")
        return
    if not _check_cooldown("volume_up", 0.3):
        return
    try:
        current = _volume.GetMasterVolumeLevelScalar()
        new_vol = min(1.0, current + VOLUME_STEP)
        _volume.SetMasterVolumeLevelScalar(new_vol, None)
        print(f"[VOL↑] {current:.0%} → {new_vol:.0%}")
    except Exception as e:
        print(f"[ERROR] volume_up falló: {e}")


def volume_down() -> None:
    """Baja el volumen del sistema un paso (~5 %).

    Respeta un cooldown de 0.3 s.
    """
    if _volume is None:
        print("[WARN] volume_down: interfaz de volumen no disponible")
        return
    if not _check_cooldown("volume_down", 0.3):
        return
    try:
        current = _volume.GetMasterVolumeLevelScalar()
        new_vol = max(0.0, current - VOLUME_STEP)
        _volume.SetMasterVolumeLevelScalar(new_vol, None)
        print(f"[VOL↓] {current:.0%} → {new_vol:.0%}")
    except Exception as e:
        print(f"[ERROR] volume_down falló: {e}")


def mouse_move(x: int, y: int) -> None:
    """Mueve el cursor del ratón a la posición (x, y) de pantalla.

    No tiene cooldown: se llama cada frame para un movimiento fluido.

    Parameters
    ----------
    x, y : int
        Coordenadas de pantalla en píxeles.
    """
    screen_w, screen_h = pyautogui.size()
    # Clampar dentro de la pantalla
    x = max(0, min(x, screen_w - 1))
    y = max(0, min(y, screen_h - 1))
    pyautogui.moveTo(x, y, _pause=False)


def left_click() -> None:
    """Ejecuta un click izquierdo.

    Respeta un cooldown de 0.5 s para no repetir clicks.
    """
    if not _check_cooldown("left_click", 0.5):
        return
    pyautogui.click()
