# =============================================================================
# entorno_enjambre_3d.py — Q-Learning 3D, domo esférico ±3.5m
# Formaciones horizontales (Z fijo), obstáculos esféricos
# =============================================================================
import numpy as np
from config3d import CFG

# ── Formaciones en plano Z=0 ──────────────────────────────────────────────────
def _formacion_linea(n, sep, z=0.0):
    # Diagonal 3D: distribuida por igual en X, Y y Z
    mitad = (n - 1) / 2.0
    return np.array([
        [(i - mitad)*sep,
         (i - mitad)*sep,
         (i - mitad)*sep * 0.3]
        for i in range(n)], dtype=np.float32)

def _formacion_v(n, sep, z=0.0):
    # V invertida simetrica: centro arriba, ramas bajan en Y y Z por igual
    pos = [(0.0, 0.0, sep * 1.0)]
    for k in range(1, (n + 1) // 2):
        pos.append((-k*sep, -k*sep, sep*(1.0 - k*0.5)))
        if len(pos) < n:
            pos.append(( k*sep, -k*sep, sep*(1.0 - k*0.5)))
    arr = np.array(pos[:n], dtype=np.float32)
    arr[:, 1] -= arr[:, 1].mean()
    return arr

def _formacion_circulo(n, sep, z=0.0):
    # Circulo inclinado 45 grados en el espacio (plano X-Z)
    r = 2.0 / (2.0 * np.sin(np.pi / n))
    inc = np.pi / 4.0
    return np.array([
        [r * np.cos(2*np.pi*i/n - np.pi/2),
         r * np.sin(2*np.pi*i/n - np.pi/2) * np.cos(inc),
         r * np.sin(2*np.pi*i/n - np.pi/2) * np.sin(inc)]
        for i in range(n)], dtype=np.float32)

FORMACIONES = {
    'linea'  : _formacion_linea,
    'v'      : _formacion_v,
    'circulo': _formacion_circulo,
}

# ── Acciones 3D: 6 ejes + 8 diagonales + quieto = 15 acciones ────────────────
_s = 1.0 / np.sqrt(2)
_t = 1.0 / np.sqrt(3)
ACCIONES_3D = np.array([
    # Ejes principales
    [ 1., 0., 0.], [-1., 0., 0.],
    [ 0., 1., 0.], [ 0.,-1., 0.],
    [ 0., 0., 1.], [ 0., 0.,-1.],
    # Diagonales horizontales
    [ _s,  _s, 0.], [ _s, -_s, 0.],
    [-_s,  _s, 0.], [-_s, -_s, 0.],
    # Diagonales con Z
    [ _s, 0.,  _s], [ _s, 0., -_s],
    [-_s, 0.,  _s], [-_s, 0., -_s],
    # Quieto
    [ 0., 0., 0.],
], dtype=np.float32)
N_ACCIONES = len(ACCIONES_3D)   # 15

# ── Obstáculos en caja 3D (cx, cy, cz, w, h, d) ──────────────────────────────
OBS_MIN = 0.3
OBS_MAX = 1.0

def _generar_obstaculos(rng, formacion, pos_drones):
    """4 cajas 3D en zona central del domo. No solapan targets."""
    obstaculos = []   # lista de (cx, cy, cz, w, h, d)
    fallbacks = [
        ( CFG.domo*0.35,  CFG.domo*0.35, 0.0, 0.4, 0.4, 0.4),
        (-CFG.domo*0.35,  CFG.domo*0.35, 0.0, 0.4, 0.4, 0.4),
        ( CFG.domo*0.35, -CFG.domo*0.35, 0.0, 0.4, 0.4, 0.4),
        (-CFG.domo*0.35, -CFG.domo*0.35, 0.0, 0.4, 0.4, 0.4),
    ]
    for k in range(CFG.n_obstaculos):
        w = float(rng.uniform(OBS_MIN, OBS_MAX))
        h = float(rng.uniform(OBS_MIN, OBS_MAX))
        d = float(rng.uniform(OBS_MIN, OBS_MAX))
        margen = max(w, h, d) / 2.0 + 0.2
        limite = min(CFG.domo * 0.45, CFG.domo - margen)
        colocado = False
        for _ in range(300):
            cx = float(rng.uniform(-limite, limite))
            cy = float(rng.uniform(-limite, limite))
            cz = float(rng.uniform(-limite*0.4, limite*0.4))  # más plano en Z
            libre = True
            # No solapar targets
            for t in formacion:
                if (abs(cx-t[0]) < w/2+0.4 and
                    abs(cy-t[1]) < h/2+0.4 and
                    abs(cz-t[2]) < d/2+0.4):
                    libre = False; break
            if not libre: continue
            # No solapar otros obstáculos
            for (ox,oy,oz,ow,oh,od) in obstaculos:
                if (abs(cx-ox) < (w+ow)/2+0.2 and
                    abs(cy-oy) < (h+oh)/2+0.2 and
                    abs(cz-oz) < (d+od)/2+0.2):
                    libre = False; break
            if libre:
                obstaculos.append((cx, cy, cz, w, h, d))
                colocado = True; break
        if not colocado:
            obstaculos.append(fallbacks[k])
    return obstaculos   # lista de 4 tuplas (cx, cy, cz, w, h, d)

def _colision_obstaculos(pos, obstaculos):
    """True si el dron está dentro de alguna caja 3D."""
    for (cx, cy, cz, w, h, d) in obstaculos:
        if (abs(pos[0]-cx) < w/2 and
            abs(pos[1]-cy) < h/2 and
            abs(pos[2]-cz) < d/2):
            return True
    return False

def _dentro_domo(pos):
    """True si la posición está dentro del domo esférico."""
    return float(np.linalg.norm(pos)) <= CFG.domo

# ── Entorno 3D ────────────────────────────────────────────────────────────────
class EntornoEnjambre3D:

    def __init__(self, forma='linea'):
        assert forma in FORMACIONES
        self.forma      = forma
        self.n          = CFG.n_drones
        self.formacion  = FORMACIONES[forma](self.n, CFG.separacion, CFG.z_formacion)
        self.pos        = np.zeros((self.n, 3), dtype=np.float32)
        self.vel        = np.zeros((self.n, 3), dtype=np.float32)
        self._paso      = 0
        self._prev_dist = np.zeros(self.n, dtype=np.float32)
        self.colisiones = 0
        self.dist_minima = np.inf
        self.paso_estabilizacion = None
        self.obstaculos = []

    def reset(self, ep=0):
        rng = np.random.default_rng(CFG.seed + ep)
        DIST_MIN_TARGET = 1.5
        pos = np.zeros((self.n, 3), dtype=np.float32)
        for i in range(self.n):
            for _ in range(500):
                # Spawn dentro del domo esférico
                p = rng.uniform(-CFG.domo*0.8, CFG.domo*0.8, 3).astype(np.float32)
                if (np.linalg.norm(p) <= CFG.domo * 0.8 and
                        np.linalg.norm(p - self.formacion[i]) >= DIST_MIN_TARGET):
                    pos[i] = p
                    break
            else:
                angle  = np.arctan2(-self.formacion[i,1], -self.formacion[i,0])
                pos[i] = np.array([np.cos(angle)*CFG.domo*0.7,
                                   np.sin(angle)*CFG.domo*0.7,
                                   0.0], dtype=np.float32)
        self.pos = pos
        self.vel = np.zeros((self.n, 3), dtype=np.float32)
        self._paso = 0
        self.colisiones = 0
        self.dist_minima = np.inf
        self.paso_estabilizacion = None
        self.obstaculos = _generar_obstaculos(rng, self.formacion, self.pos)
        # Expulsar drones que spawnen dentro de obstáculos
        for i in range(self.n):
            if _colision_obstaculos(self.pos[i], self.obstaculos):
                for (cx, cy, cz, w, h, d) in self.obstaculos:
                    if (abs(self.pos[i,0]-cx) < w/2 and
                        abs(self.pos[i,1]-cy) < h/2 and
                        abs(self.pos[i,2]-cz) < d/2):
                        dx = self.pos[i,0]-cx
                        dy = self.pos[i,1]-cy
                        dz = self.pos[i,2]-cz
                        # Empujar por el eje con menor penetración
                        px = w/2 - abs(dx)
                        py = h/2 - abs(dy)
                        pz = d/2 - abs(dz)
                        if px <= py and px <= pz:
                            self.pos[i,0] = cx + np.sign(dx)*(w/2+0.05)
                        elif py <= px and py <= pz:
                            self.pos[i,1] = cy + np.sign(dy)*(h/2+0.05)
                        else:
                            self.pos[i,2] = cz + np.sign(dz)*(d/2+0.05)
                        break
        self._prev_dist = np.array([
            np.linalg.norm(self.formacion[i] - self.pos[i])
            for i in range(self.n)], dtype=np.float32)
        return [self._obs(i) for i in range(self.n)]

    def step(self, acciones):
        self._paso += 1
        velocidad = 5.0
        for i in range(self.n):
            delta    = ACCIONES_3D[acciones[i]] * velocidad * CFG.dt
            nueva_vel = self.vel[i] * 0.2 + delta
            nueva_pos = self.pos[i] + nueva_vel

            # Contener dentro del domo esférico
            dist_centro = float(np.linalg.norm(nueva_pos))
            if dist_centro > CFG.domo:
                nueva_pos = (nueva_pos / dist_centro * CFG.domo).astype(np.float32)
                nueva_vel = nueva_vel * 0.0  # detener al chocar con el domo

            # Bloqueo físico por obstáculo esférico
            if _colision_obstaculos(nueva_pos, self.obstaculos):
                # Intentar componentes individuales
                pos_x = np.array([nueva_pos[0], self.pos[i][1], self.pos[i][2]], dtype=np.float32)
                pos_y = np.array([self.pos[i][0], nueva_pos[1], self.pos[i][2]], dtype=np.float32)
                pos_z = np.array([self.pos[i][0], self.pos[i][1], nueva_pos[2]], dtype=np.float32)
                if not _colision_obstaculos(pos_x, self.obstaculos):
                    nueva_pos = pos_x; nueva_vel = np.array([nueva_vel[0], 0., 0.], dtype=np.float32)
                elif not _colision_obstaculos(pos_y, self.obstaculos):
                    nueva_pos = pos_y; nueva_vel = np.array([0., nueva_vel[1], 0.], dtype=np.float32)
                elif not _colision_obstaculos(pos_z, self.obstaculos):
                    nueva_pos = pos_z; nueva_vel = np.array([0., 0., nueva_vel[2]], dtype=np.float32)
                else:
                    nueva_pos = self.pos[i].copy(); nueva_vel = np.zeros(3, dtype=np.float32)

            self.vel[i] = nueva_vel
            self.pos[i] = nueva_pos

        # Colisiones entre drones
        col_paso = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                d = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                self.dist_minima = min(self.dist_minima, d)
                if d < CFG.dist_colision:
                    col_paso += 1
        self.colisiones += col_paso

        errores = []
        recompensas = []
        for i in range(self.n):
            dist = float(np.linalg.norm(self.formacion[i] - self.pos[i]))
            errores.append(dist)
            recompensas.append(self._recompensa(i, dist))
            self._prev_dist[i] = dist

        todos = all(e < CFG.dist_llegada for e in errores)
        if todos and self.paso_estabilizacion is None:
            self.paso_estabilizacion = self._paso

        terminado = self._paso >= CFG.max_pasos_enjambre
        info = {
            'errores'             : errores,
            'error_promedio'      : float(np.mean(errores)),
            'colisiones_paso'     : col_paso,
            'colisiones_total'    : self.colisiones,
            'dist_minima'         : self.dist_minima,
            'estabilidad_formacion': float(np.std(errores)),
            'paso_estabilizacion' : self.paso_estabilizacion,
            'todos_en_formacion'  : todos,
            'drones_en_formacion' : sum(1 for e in errores if e < CFG.dist_llegada),
        }
        return [self._obs(i) for i in range(self.n)], recompensas, terminado, info

    def _obs(self, i):
        """Obs 9D: error_propio(3) + vec_vecino(3) + vec_obstaculo(3)."""
        error_propio = (self.formacion[i] - self.pos[i]) / (CFG.domo * 2)
        # Vecino más cercano
        dist_min, vec_vecino = np.inf, np.zeros(3, dtype=np.float32)
        for j in range(self.n):
            if j == i: continue
            d = float(np.linalg.norm(self.pos[j] - self.pos[i]))
            if d < dist_min:
                dist_min = d
                vec_vecino = (self.pos[j] - self.pos[i]) / (CFG.domo * 2)
        # Obstáculo más cercano (distancia al centro de la caja)
        min_d_obs = np.inf
        vec_obs   = np.zeros(3, dtype=np.float32)
        for (cx, cy, cz, w, h, d) in self.obstaculos:
            centro = np.array([cx, cy, cz])
            dist = float(np.linalg.norm(self.pos[i] - centro))
            if dist < min_d_obs:
                min_d_obs = dist
                vec_obs   = (centro - self.pos[i]) / (CFG.domo * 2)
        return np.concatenate([error_propio, vec_vecino, vec_obs]).astype(np.float32)

    def _recompensa(self, i, dist):
        r  = (self._prev_dist[i] - dist) * 10.0
        r -= dist * 0.5
        r += 20.0 if dist < CFG.dist_llegada else 0.0
        r += 5.0  if dist < 0.5 else 0.0
        for j in range(self.n):
            if j != i:
                d_ij = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                if d_ij < CFG.dist_colision * 2:
                    r -= (CFG.dist_colision * 2 - d_ij) * 10.0
        if _colision_obstaculos(self.pos[i], self.obstaculos):
            r += CFG.r_obstaculo
        return r
