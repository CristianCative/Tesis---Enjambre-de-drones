# =============================================================================
# config3d.py — Q-Learning 3D con obstáculos esféricos
# Domo esférico ±3.5m en X, Y, Z
# =============================================================================
from dataclasses import dataclass

@dataclass
class Config3D:
    # ── Entorno ───────────────────────────────────────────────────────────────
    domo              : float = 3.5    # radio del domo esférico [m]
    dt                : float = 0.025
    # ── Enjambre ──────────────────────────────────────────────────────────────
    n_drones          : int   = 9
    separacion        : float = 0.6
    max_pasos_enjambre: int   = 300
    dist_colision     : float = 0.3
    dist_llegada      : float = 0.20
    # ── Formaciones (en plano Z=0, horizontal) ────────────────────────────────
    z_formacion       : float = 0.0   # altura fija de las formaciones
    # ── Obstáculos en caja 3D ─────────────────────────────────────────────────
    n_obstaculos      : int   = 4
    obs_min           : float = 0.3    # tamaño mínimo por lado [m]
    obs_max           : float = 1.0    # tamaño máximo por lado [m]
    r_obstaculo       : float = -15.0
    # ── Q-Learning ────────────────────────────────────────────────────────────
    alpha             : float = 0.2
    gamma             : float = 0.99
    epsilon_ini       : float = 1.0
    epsilon_fin       : float = 0.1
    epsilon_decay     : float = 0.995
    # ── Discretización 3D (bins por eje) ──────────────────────────────────────
    n_bins            : int   = 13     # igual que 2D, el eje Z ya agrega más estados
    obs_bins          : int   = 5
    limite_error      : float = 2.5
    # ── Entrenamiento ─────────────────────────────────────────────────────────
    n_episodios       : int   = 5000
    seed              : int   = 42
    log_cada          : int   = 100
    ventana_conv      : int   = 100

CFG = Config3D()
