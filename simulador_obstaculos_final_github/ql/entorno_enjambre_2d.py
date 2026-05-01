# =============================================================================
# entorno_enjambre_2d.py — Q-Learning 2D con obstáculo rectangular
# =============================================================================
import numpy as np
from config import CFG

# ── Formaciones ───────────────────────────────────────────────────────────────
def _formacion_linea(n, sep):
    mitad = (n - 1) / 2.0
    return np.array([[(i - mitad)*sep, 0.0] for i in range(n)], dtype=np.float32)

def _formacion_v(n, sep):
    # Punta abajo: drone central en la punta (y más alto positivo),
    # brazos se abren simétricamente hacia arriba (y decrece).
    pos = [(0.0, 0.0)]                       # punta (centro-inferior)
    for k in range(1, (n + 1) // 2):
        pos.append((-k * sep, -k * sep))     # brazo izquierdo sube
        if len(pos) < n:
            pos.append(( k * sep, -k * sep)) # brazo derecho sube
    arr = np.array(pos[:n], dtype=np.float32)
    arr[:, 1] -= arr[:, 1].mean()            # centrar verticalmente
    return arr

def _formacion_circulo(n, sep):
    r = 2.0 / (2.0 * np.sin(np.pi / n))
    return np.array([
        [r*np.cos(2*np.pi*i/n - np.pi/2),
         r*np.sin(2*np.pi*i/n - np.pi/2)]
        for i in range(n)], dtype=np.float32)

FORMACIONES = {'linea': _formacion_linea, 'v': _formacion_v, 'circulo': _formacion_circulo}

ACCIONES_2D = np.array([
    [ 1.0,  0.0], [-1.0,  0.0], [ 0.0,  1.0], [ 0.0, -1.0],
    [ 0.7,  0.7], [ 0.7, -0.7], [-0.7,  0.7], [-0.7, -0.7],
    [ 0.0,  0.0],
], dtype=np.float32)
N_ACCIONES_2D = len(ACCIONES_2D)

# ── Obstáculos (4 rectángulos con tamaño aleatorio) ──────────────────────────
N_OBSTACULOS = 4
OBS_MIN      = 0.3   # tamaño mínimo por lado [m]
OBS_MAX      = 1.0   # tamaño máximo por lado [m]

def _generar_obstaculos(rng, formacion, pos_drones):
    """Genera 4 obstáculos con tamaño y posición aleatoria.
    - Se concentran más en el centro (zona ±domo*0.45).
    - No solapan targets (formación final).
    - SÍ pueden solapar posiciones iniciales de drones → generan cruce real.
    """
    obstaculos = []   # lista de (cx, cy, w, h)
    # Fallbacks también en zona central
    fallbacks  = [
        ( CFG.domo*0.35,  CFG.domo*0.35),
        (-CFG.domo*0.35,  CFG.domo*0.35),
        ( CFG.domo*0.35, -CFG.domo*0.35),
        (-CFG.domo*0.35, -CFG.domo*0.35),
    ]
    for k in range(N_OBSTACULOS):
        w = float(rng.uniform(OBS_MIN, OBS_MAX))
        h = float(rng.uniform(OBS_MIN, OBS_MAX))
        margen = max(w, h) / 2.0 + 0.2
        # Zona central: los obstáculos se generan dentro del 45 % del domo
        limite = min(CFG.domo * 0.45, CFG.domo - margen)
        colocado = False
        for _ in range(300):
            cx = float(rng.uniform(-limite, limite))
            cy = float(rng.uniform(-limite, limite))
            libre = True
            # No solapar targets (formación deseada)
            for t in formacion:
                if abs(cx-t[0]) < w/2+0.4 and abs(cy-t[1]) < h/2+0.4:
                    libre = False; break
            if not libre: continue
            # No solapar otros obstáculos ya colocados
            for (ox, oy, ow, oh) in obstaculos:
                if abs(cx-ox) < (w+ow)/2+0.2 and abs(cy-oy) < (h+oh)/2+0.2:
                    libre = False; break
            # NOTA: se elimina la restricción sobre pos_drones iniciales a propósito
            # para que los drones partan desde/a través de obstáculos → cruce real.
            if libre:
                obstaculos.append((cx, cy, w, h))
                colocado = True; break
        if not colocado:
            fx, fy = fallbacks[k]
            obstaculos.append((fx, fy, 0.4, 0.4))
    return obstaculos   # lista de 4 tuplas (cx, cy, w, h)

def _colision_obstaculos(pos, obstaculos):
    """True si el dron colisiona con alguno de los obstáculos."""
    for (cx, cy, w, h) in obstaculos:
        if abs(pos[0]-cx) < w/2 and abs(pos[1]-cy) < h/2:
            return True
    return False

# ── Entorno ───────────────────────────────────────────────────────────────────
class EntornoEnjambre2D:

    def __init__(self, forma='linea'):
        assert forma in FORMACIONES
        self.forma     = forma
        self.n         = CFG.n_drones
        self.formacion = FORMACIONES[forma](self.n, CFG.separacion)
        self.pos       = np.zeros((self.n, 2), dtype=np.float32)
        self.vel       = np.zeros((self.n, 2), dtype=np.float32)
        self._paso     = 0
        self._prev_dist = np.zeros(self.n, dtype=np.float32)
        self.colisiones = 0
        self.dist_minima = np.inf
        self.paso_estabilizacion = None
        self.obstaculos = []   # lista de (cx, cy, w, h)

    def reset(self, ep=0):
        rng = np.random.default_rng(CFG.seed + ep)
        # Generación de drones: posición aleatoria con distancia mínima al target
        DIST_MIN_TARGET = 1.5   # drones arrancan al menos a 1.5 m de su target
        pos = np.zeros((self.n, 2), dtype=np.float32)
        for i in range(self.n):
            for _ in range(500):
                p = rng.uniform(-CFG.domo*0.8, CFG.domo*0.8, 2).astype(np.float32)
                if np.linalg.norm(p - self.formacion[i]) >= DIST_MIN_TARGET:
                    pos[i] = p
                    break
            else:
                # fallback: lado opuesto al target
                angle = np.arctan2(-self.formacion[i, 1], -self.formacion[i, 0])
                pos[i] = np.clip(
                    np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * CFG.domo * 0.7,
                    -CFG.domo, CFG.domo)
        self.pos = pos
        self.vel = np.zeros((self.n, 2), dtype=np.float32)
        self._paso = 0
        self.colisiones = 0
        self.dist_minima = np.inf
        self.paso_estabilizacion = None
        self.obstaculos = _generar_obstaculos(rng, self.formacion, self.pos)
        # Expulsar drones que hayan quedado dentro de un obstáculo al inicio
        for i in range(self.n):
            if _colision_obstaculos(self.pos[i], self.obstaculos):
                for (cx, cy, w, h) in self.obstaculos:
                    if abs(self.pos[i,0]-cx) < w/2 and abs(self.pos[i,1]-cy) < h/2:
                        dx = self.pos[i,0] - cx
                        dy = self.pos[i,1] - cy
                        if abs(dx) >= abs(dy):
                            self.pos[i,0] = cx + np.sign(dx) * (w/2 + 0.05)
                        else:
                            self.pos[i,1] = cy + np.sign(dy) * (h/2 + 0.05)
                        break
        self._prev_dist = np.array([
            np.linalg.norm(self.formacion[i] - self.pos[i])
            for i in range(self.n)], dtype=np.float32)
        return [self._obs(i) for i in range(self.n)]

    def step(self, acciones):
        self._paso += 1
        velocidad = 5.0
        for i in range(self.n):
            dx_dy = ACCIONES_2D[acciones[i]] * velocidad * CFG.dt
            nueva_vel = self.vel[i] * 0.2 + dx_dy
            nueva_pos = np.clip(self.pos[i] + nueva_vel, -CFG.domo, CFG.domo)
            # ── Bloqueo físico: si la nueva posición penetra un obstáculo,
            #    se intenta deslizar por cada eje individualmente.
            #    Si ambos ejes están bloqueados, el dron se detiene.
            if _colision_obstaculos(nueva_pos, self.obstaculos):
                # Intentar deslizamiento eje X
                pos_x = np.array([nueva_pos[0], self.pos[i][1]], dtype=np.float32)
                pos_y = np.array([self.pos[i][0], nueva_pos[1]], dtype=np.float32)
                if not _colision_obstaculos(pos_x, self.obstaculos):
                    nueva_pos = pos_x
                    nueva_vel = np.array([nueva_vel[0], 0.0], dtype=np.float32)
                elif not _colision_obstaculos(pos_y, self.obstaculos):
                    nueva_pos = pos_y
                    nueva_vel = np.array([0.0, nueva_vel[1]], dtype=np.float32)
                else:
                    nueva_pos = self.pos[i].copy()
                    nueva_vel = np.zeros(2, dtype=np.float32)
            self.vel[i] = nueva_vel
            self.pos[i] = nueva_pos

        col_paso = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                d = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                self.dist_minima = min(self.dist_minima, d)
                if d < CFG.dist_colision:
                    col_paso += 1
        self.colisiones += col_paso

        recompensas, errores = [], []
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
            'errores': errores,
            'error_promedio': float(np.mean(errores)),
            'colisiones_paso': col_paso,
            'colisiones_total': self.colisiones,
            'dist_minima': self.dist_minima,
            'estabilidad_formacion': float(np.std(errores)),
            'paso_estabilizacion': self.paso_estabilizacion,
            'todos_en_formacion': todos,
            'drones_en_formacion': sum(1 for e in errores if e < CFG.dist_llegada),
        }
        return [self._obs(i) for i in range(self.n)], recompensas, terminado, info

    def _obs(self, i):
        """Obs 4D: error propio (2) + vector vecino más cercano (2)."""
        error_propio = self.formacion[i] - self.pos[i]
        dist_min, vec_vecino = np.inf, np.zeros(2, dtype=np.float32)
        for j in range(self.n):
            if j == i: continue
            d = float(np.linalg.norm(self.pos[j] - self.pos[i]))
            if d < dist_min:
                dist_min = d
                vec_vecino = self.pos[j] - self.pos[i]
        obs = np.zeros(4, dtype=np.float32)
        obs[0:2] = error_propio
        obs[2:4] = vec_vecino
        return obs

    def _recompensa(self, i, dist):
        r  = (self._prev_dist[i] - dist) * 10.0
        r -= dist * 0.5
        r += 20.0 if dist < CFG.dist_llegada else 0.0
        r += 5.0  if dist < 0.5 else 0.0
        r += CFG.r_tiempo
        for j in range(self.n):
            if j != i:
                d_ij = float(np.linalg.norm(self.pos[i] - self.pos[j]))
                if d_ij < CFG.dist_colision * 2:
                    r -= (CFG.dist_colision * 2 - d_ij) * 10.0
    
        if _colision_obstaculos(self.pos[i], self.obstaculos):
            r += CFG.r_obstaculo
    
        return r
