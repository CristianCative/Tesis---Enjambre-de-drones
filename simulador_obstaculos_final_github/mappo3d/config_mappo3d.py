# =============================================================================
# config_mappo3d.py — MAPPO 3D, domo esférico ±3.5m
# =============================================================================
from dataclasses import dataclass, field
from typing import List

@dataclass
class ConfigMAPPO3D:
    # ── Entorno ───────────────────────────────────────────────────────────────
    n_drones        : int   = 9
    domo            : float = 3.5
    separacion      : float = 0.6
    dist_llegada    : float = 0.20
    dist_colision   : float = 0.30
    max_pasos       : int   = 300
    seed            : int   = 42
    z_formacion     : float = 0.0
    # ── Obstáculos en caja 3D ─────────────────────────────────────────────────
    n_obstaculos    : int   = 4
    obs_min         : float = 0.3
    obs_max         : float = 1.0
    r_obstaculo     : float = -15.0
    # ── Entrenamiento ─────────────────────────────────────────────────────────
    n_episodios     : int   = 5000
    formaciones     : List[str] = field(default_factory=lambda: ['linea','v','circulo'])
    # ── Red neuronal ──────────────────────────────────────────────────────────
    # Obs 3D: err(3) + vel(3) + 3 vecinos(9) + dist_min(1) + id(1) + obs_rel(3) = 20
    dim_obs         : int   = 20
    dim_accion      : int   = 15   # 15 acciones 3D
    capas_actor     : List[int] = field(default_factory=lambda: [64, 64])
    capas_critico   : List[int] = field(default_factory=lambda: [64, 64])
    # ── Hiperparámetros MAPPO ─────────────────────────────────────────────────
    lr_actor        : float = 3e-4
    lr_critico      : float = 1e-3
    gamma           : float = 0.99
    lam             : float = 0.95
    clip_eps        : float = 0.2
    epochs_ppo      : int   = 4
    tam_minibatch   : int   = 64
    coef_entropia   : float = 0.01
    coef_valor      : float = 0.5
    grad_max        : float = 0.5
    # ── Recompensa ────────────────────────────────────────────────────────────
    r_llegada       : float = 50.0
    r_progreso      : float = 10.0
    r_distancia     : float = 0.5
    r_colision      : float = -20.0
    r_tiempo        : float = -0.5
    r_proximidad    : float = 5.0
    # ── Logging ───────────────────────────────────────────────────────────────
    log_cada        : int   = 100
    ventana_conv    : int   = 100
    n_vecinos       : int   = 3

CFG = ConfigMAPPO3D()
