# =============================================================================
# agente_qlearning_2d.py — Q-Learning con estado extendido (obstáculo)
# Estado: (id_dron, bin_ex, bin_ey, bin_ox, bin_oy)
# Espacio: 9 × 13² × 5² = 38,025 estados posibles
# =============================================================================
import numpy as np
import pickle
from config import CFG
from entorno_enjambre_2d import N_ACCIONES_2D


def discretizar(obs, id_dron, obstaculos, pos_dron):
    # Error de posición (2 bins)
    e    = np.clip(obs[0:2], -CFG.limite_error, CFG.limite_error)
    norm = (e + CFG.limite_error) / (2.0 * CFG.limite_error)
    bins_e = np.clip(np.floor(norm * CFG.n_bins).astype(int), 0, CFG.n_bins-1)

    # Obstáculo más cercano al dron
    min_d  = np.inf
    mejor  = np.array([obstaculos[0][0], obstaculos[0][1]], dtype=np.float32)
    for (cx, cy, w, h) in obstaculos:
        d = np.sqrt((pos_dron[0]-cx)**2 + (pos_dron[1]-cy)**2)
        if d < min_d:
            min_d = d; mejor = np.array([cx, cy], dtype=np.float32)

    delta  = np.clip(mejor - pos_dron, -CFG.obs_limite, CFG.obs_limite)
    norm_o = (delta + CFG.obs_limite) / (2.0 * CFG.obs_limite)
    bins_o = np.clip(np.floor(norm_o * CFG.obs_bins).astype(int), 0, CFG.obs_bins-1)

    return (id_dron, int(bins_e[0]), int(bins_e[1]),
            int(bins_o[0]), int(bins_o[1]))


class AgenteQLearning2D:

    def __init__(self):
        self._Q      = {}
        self.alpha   = CFG.alpha
        self.gamma   = CFG.gamma
        self.epsilon = CFG.epsilon_ini
        self._rng    = np.random.default_rng(CFG.seed)
        self.episodios = 0
        self.actualizaciones = 0

        estados = CFG.n_drones * (CFG.n_bins**2) * (CFG.obs_bins**2)
        print('='*57)
        print('  Agente Q-Learning 2D — Enjambre cooperativo + Obstáculo')
        print(f'  α={CFG.alpha}  γ={CFG.gamma}')
        print(f'  ε: {CFG.epsilon_ini} → {CFG.epsilon_fin}  (decay={CFG.epsilon_decay})')
        print(f'  Estado: (id_dron, bin_ex, bin_ey, bin_ox, bin_oy)')
        print(f'  Bins error: {CFG.n_bins}/dim  |  Bins obstáculo: {CFG.obs_bins}/dim')
        print(f'  Estados posibles: {estados:,}')
        print(f'  Acciones: {N_ACCIONES_2D} direccionales 2D')
        print(f'  Política: COMPARTIDA ({CFG.n_drones} drones, 1 tabla Q)')
        print('='*57)

    def _q(self, estado):
        if estado not in self._Q:
            self._Q[estado] = np.zeros(N_ACCIONES_2D, dtype=np.float64)
        return self._Q[estado]

    def seleccionar_accion(self, obs, id_dron, obstaculos, pos_dron,
                           determinista=False):
        estado = discretizar(obs, id_dron, obstaculos, pos_dron)
        if not determinista and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, N_ACCIONES_2D))
        return int(np.argmax(self._q(estado)))

    def seleccionar_acciones_enjambre(self, obs_lista, obstaculos,
                                       pos_drones, determinista=False):
        return [
            self.seleccionar_accion(obs_lista[i], i, obstaculos,
                                    pos_drones[i], determinista)
            for i in range(len(obs_lista))
        ]

    def actualizar(self, obs, id_dron, obstaculos, pos_dron, accion,
                   recompensa, obs_sig, obstaculos_sig, pos_dron_sig,
                   terminado):
        s  = discretizar(obs,     id_dron, obstaculos,     pos_dron)
        s_ = discretizar(obs_sig, id_dron, obstaculos_sig, pos_dron_sig)
        q_sa  = self._q(s)[accion]
        q_max = 0.0 if terminado else float(np.max(self._q(s_)))
        td = recompensa + self.gamma * q_max - q_sa
        self._q(s)[accion] += self.alpha * td
        self.actualizaciones += 1
        return abs(td)

    def actualizar_enjambre(self, obs_lista, obstaculos, pos_drones,
                             acciones, recompensas,
                             obs_sig_lista, pos_drones_sig, terminado):
        tds = []
        for i in range(len(obs_lista)):
            td = self.actualizar(
                obs_lista[i], i, obstaculos, pos_drones[i],
                acciones[i], recompensas[i],
                obs_sig_lista[i], obstaculos, pos_drones_sig[i],
                terminado)
            tds.append(td)
        return float(np.mean(tds))

    def decaer_epsilon(self):
        self.epsilon = max(CFG.epsilon_fin, self.epsilon * CFG.epsilon_decay)
        self.episodios += 1

    @property
    def n_estados(self):
        return len(self._Q)
