# =============================================================================
# config.py — Q-Learning 2D con obstáculo rectangular
# =============================================================================
from dataclasses import dataclass

@dataclass
class Config:
    # ── Entorno ───────────────────────────────────────────────────────────────
    domo              : float = 3.5
    dt                : float = 0.025
    # ── Enjambre ──────────────────────────────────────────────────────────────
    n_drones          : int   = 9
    separacion        : float = 0.8
    max_pasos_enjambre: int   = 300
    dist_colision     : float = 0.3
    dist_llegada      : float = 0.20
    # ── Obstáculo rectangular ─────────────────────────────────────────────────
    obs_w             : float = 0.6    # ancho [m]
    obs_h             : float = 0.6    # alto  [m]
    obs_bins          : int   = 5      # bins por eje distancia al obstáculo
    obs_limite        : float = 3.0    # rango máx a discretizar
    r_obstaculo       : float = -15.0  # penalización colisión obstáculo
    # ── Q-Learning ────────────────────────────────────────────────────────────
    alpha             : float = 0.2
    gamma             : float = 0.99
    epsilon_ini       : float = 1.0
    epsilon_fin       : float = 0.1
    epsilon_decay     : float = 0.995
    # ── Discretización ────────────────────────────────────────────────────────
    n_bins            : int   = 13
    limite_error      : float = 2.5
    # ── Entrenamiento ─────────────────────────────────────────────────────────
    n_episodios       : int   = 5000
    seed              : int   = 42
    log_cada          : int   = 100
    ventana_conv      : int   = 100

CFG = Config()
