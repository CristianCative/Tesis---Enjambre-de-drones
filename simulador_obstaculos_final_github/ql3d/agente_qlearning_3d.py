# =============================================================================
# agente_qlearning_3d.py — Q-Learning tabular 3D
# Estado: (bin_ex, bin_ey, bin_ez, bin_obs) — 9³ × 4 = 2916 estados/dron
# =============================================================================
import numpy as np
from collections import defaultdict
from config3d import CFG
from entorno_enjambre_3d import N_ACCIONES

class AgenteQLearning3D:

    def __init__(self):
        self.Q            = defaultdict(lambda: np.zeros(N_ACCIONES, dtype=np.float32))
        self.epsilon      = CFG.epsilon_ini
        self.n_estados    = 0
        self.actualizaciones = 0

    # ── Discretización ────────────────────────────────────────────────────────
    def _discretizar(self, obs, obstaculos, pos_i):
        """obs: vector 9D → estado discreto (4-tuple)."""
        # Error propio: obs[0:3] ya normalizado por domo*2
        ex, ey, ez = obs[0], obs[1], obs[2]
        bx = int(np.clip((ex + 0.5) / 1.0 * CFG.n_bins, 0, CFG.n_bins - 1))
        by = int(np.clip((ey + 0.5) / 1.0 * CFG.n_bins, 0, CFG.n_bins - 1))
        bz = int(np.clip((ez + 0.5) / 1.0 * CFG.n_bins, 0, CFG.n_bins - 1))
        # Distancia al centro del obstáculo más cercano
        min_d = np.inf
        for (cx, cy, cz, w, h, d) in obstaculos:
            dist = float(np.linalg.norm(pos_i - np.array([cx, cy, cz])))
            min_d = min(min_d, dist)
        b_obs = int(np.clip(min_d / CFG.domo * CFG.obs_bins, 0, CFG.obs_bins - 1))
        return (bx, by, bz, b_obs)

    # ── Selección de acción ───────────────────────────────────────────────────
    def seleccionar_acciones_enjambre(self, obs_list, obstaculos, pos,
                                       determinista=False):
        acciones = []
        for i, obs in enumerate(obs_list):
            s = self._discretizar(obs, obstaculos, pos[i])
            if not determinista and np.random.random() < self.epsilon:
                acciones.append(np.random.randint(N_ACCIONES))
            else:
                acciones.append(int(np.argmax(self.Q[s])))
        return acciones

    # ── Actualización Q ───────────────────────────────────────────────────────
    def actualizar_enjambre(self, obs_list, obstaculos, pos,
                             acciones, recompensas,
                             obs_sig_list, pos_sig, done):
        td_total = 0.0
        for i in range(CFG.n_drones):
            s      = self._discretizar(obs_list[i], obstaculos, pos[i])
            s_sig  = self._discretizar(obs_sig_list[i], obstaculos, pos_sig[i])
            if s not in self.Q:
                self.n_estados += 1
            Q_max  = 0.0 if done else float(np.max(self.Q[s_sig]))
            td     = recompensas[i] + CFG.gamma * Q_max - self.Q[s][acciones[i]]
            self.Q[s][acciones[i]] += CFG.alpha * td
            td_total += abs(td)
            self.actualizaciones += 1
        return td_total / CFG.n_drones

    def decaer_epsilon(self):
        self.epsilon = max(CFG.epsilon_fin,
                           self.epsilon * CFG.epsilon_decay)

    @property
    def n_estados(self):
        return self._n_estados

    @n_estados.setter
    def n_estados(self, v):
        self._n_estados = v

# Fix para que funcione el defaultdict con el contador
AgenteQLearning3D._n_estados = 0
