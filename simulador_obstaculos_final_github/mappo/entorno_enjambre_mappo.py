# =============================================================================
# entorno_enjambre_mappo.py — MAPPO 2D con obstáculo rectangular
# Formaciones, dinámica y obstáculo idénticos al Q-Learning 2D
# =============================================================================
import numpy as np
from config_mappo import CFG

# ── Formaciones (idénticas al Q-Learning) ────────────────────────────────────
def _formacion_linea(n, sep):
    mitad = (n-1)/2.0
    return np.array([[(i-mitad)*sep, 0.0] for i in range(n)], dtype=np.float32)

def _formacion_v(n, sep):
    pos = [(0.0, 0.0)]
    for k in range(1, (n+1)//2):
        pos.append((-k*sep, -k*sep))
        if len(pos) < n: pos.append((k*sep, -k*sep))
    arr = np.array(pos[:n], dtype=np.float32)
    arr[:,1] -= arr[:,1].mean()
    return arr

def _formacion_circulo(n, sep):
    r = 2.0 / (2.0 * np.sin(np.pi/n))
    return np.array([
        [r*np.cos(2*np.pi*i/n - np.pi/2),
         r*np.sin(2*np.pi*i/n - np.pi/2)]
        for i in range(n)], dtype=np.float32)

FORMACIONES = {'linea': _formacion_linea, 'v': _formacion_v, 'circulo': _formacion_circulo}

ACCIONES_2D = np.array([
    [ 1.0, 0.0],[-1.0, 0.0],[ 0.0, 1.0],[ 0.0,-1.0],
    [ 0.7, 0.7],[ 0.7,-0.7],[-0.7, 0.7],[-0.7,-0.7],
    [ 0.0, 0.0]], dtype=np.float32)
VEL_ESCALAR = 5.0
DT          = 0.025

# ── Obstáculos (4 rectángulos con tamaño aleatorio, idéntico al Q-Learning) ──
N_OBSTACULOS = 4
OBS_MIN      = 0.3
OBS_MAX      = 1.0

def _generar_obstaculos(rng, formacion, pos_drones):
    """Genera 4 obstáculos en zona central, sin evitar posiciones iniciales de drones
    para que exista cruce real durante el entrenamiento."""
    obstaculos = []
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
        margen = max(w, h)/2.0 + 0.2
        # Zona central: ±45 % del domo
        limite = min(CFG.domo * 0.45, CFG.domo - margen)
        colocado = False
        for _ in range(300):
            cx = float(rng.uniform(-limite, limite))
            cy = float(rng.uniform(-limite, limite))
            libre = True
            # No solapar targets
            for t in formacion:
                if abs(cx-t[0]) < w/2+0.4 and abs(cy-t[1]) < h/2+0.4:
                    libre = False; break
            if not libre: continue
            # No solapar otros obstáculos
            for (ox, oy, ow, oh) in obstaculos:
                if abs(cx-ox) < (w+ow)/2+0.2 and abs(cy-oy) < (h+oh)/2+0.2:
                    libre = False; break
            # SÍ pueden solapar drones iniciales → cruce realista
            if libre:
                obstaculos.append((cx, cy, w, h)); colocado = True; break
        if not colocado:
            fx, fy = fallbacks[k]
            obstaculos.append((fx, fy, 0.4, 0.4))
    return obstaculos

def _colision_obstaculos(pos, obstaculos):
    for (cx, cy, w, h) in obstaculos:
        if abs(pos[0]-cx) < w/2 and abs(pos[1]-cy) < h/2:
            return True
    return False

# ── Entorno ───────────────────────────────────────────────────────────────────
class EntornoEnjambreMAPPO:

    def __init__(self, forma='linea'):
        assert forma in FORMACIONES
        self.forma      = forma
        self.n          = CFG.n_drones
        self.formacion  = FORMACIONES[forma](self.n, CFG.separacion)
        self.pos        = np.zeros((self.n, 2), dtype=np.float32)
        self.vel        = np.zeros((self.n, 2), dtype=np.float32)
        self._paso      = 0
        self._prev_dist = np.zeros(self.n, dtype=np.float32)
        self.colisiones = 0
        self.dist_minima = np.inf
        self.paso_estabilizacion = None
        self.obstaculos = []   # lista de (cx, cy, w, h)

    def reset(self, ep=0):
        rng = np.random.default_rng(CFG.seed + ep)   # misma semilla que QL
        # Drones: posición aleatoria con distancia mínima a su target asignado
        DIST_MIN_TARGET = 1.5   # drones arrancan al menos a 1.5 m de su target
        pos = np.zeros((self.n, 2), dtype=np.float32)
        for i in range(self.n):
            for _ in range(500):
                p = rng.uniform(-CFG.domo*0.8, CFG.domo*0.8, 2).astype(np.float32)
                if np.linalg.norm(p - self.formacion[i]) >= DIST_MIN_TARGET:
                    pos[i] = p
                    break
            else:
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
        self._prev_dist = np.linalg.norm(
            self.pos - self.formacion, axis=1).astype(np.float32)
        return self._obs_todos()

    def step(self, acciones):
        self._paso += 1
        for i in range(self.n):
            dx_dy = ACCIONES_2D[acciones[i]] * VEL_ESCALAR * DT
            nueva_vel = self.vel[i]*0.2 + dx_dy
            nueva_pos = np.clip(self.pos[i] + nueva_vel, -CFG.domo, CFG.domo)
            # ── Bloqueo físico con deslizamiento lateral
            if _colision_obstaculos(nueva_pos, self.obstaculos):
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
                d = float(np.linalg.norm(self.pos[i]-self.pos[j]))
                self.dist_minima = min(self.dist_minima, d)
                if d < CFG.dist_colision: col_paso += 1
        self.colisiones += col_paso

        errores = np.linalg.norm(self.pos - self.formacion, axis=1)
        recompensas = np.array([self._recompensa(i, errores[i])
                                for i in range(self.n)], dtype=np.float32)

        todos = bool(np.all(errores < CFG.dist_llegada))
        if todos and self.paso_estabilizacion is None:
            self.paso_estabilizacion = self._paso

        self._prev_dist = errores.astype(np.float32)
        terminado = self._paso >= CFG.max_pasos

        info = {
            'errores': errores,
            'error_promedio': float(np.mean(errores)),
            'colisiones_paso': col_paso,
            'colisiones_total': self.colisiones,
            'dist_minima': float(self.dist_minima),
            'estabilidad_formacion': float(np.std(errores)),
            'paso_estabilizacion': self.paso_estabilizacion,
            'todos_en_formacion': todos,
            'drones_en_formacion': int(np.sum(errores < CFG.dist_llegada)),
        }
        return self._obs_todos(), recompensas, terminado, info

    def _obs_todos(self):
        obs = np.zeros((self.n, CFG.dim_obs), dtype=np.float32)
        for i in range(self.n):
            err   = (self.formacion[i] - self.pos[i]) / (CFG.domo*2)
            vel_n = self.vel[i] / (VEL_ESCALAR*DT + 1e-8)
            dists = np.linalg.norm(self.pos - self.pos[i], axis=1)
            dists[i] = np.inf
            vecinos = np.argsort(dists)[:CFG.n_vecinos]
            rel = np.concatenate([(self.pos[j]-self.pos[i])/(CFG.domo*2)
                                   for j in vecinos])
            dist_min = dists[vecinos[0]] / (CFG.domo*2)
            id_n     = np.float32(i / (self.n-1))
            # Obstáculo más cercano → dirección relativa normalizada (2 dims)
            min_d, mejor_c = np.inf, np.array([self.obstaculos[0][0],
                                                self.obstaculos[0][1]], dtype=np.float32)
            for (cx, cy, w, h) in self.obstaculos:
                d = np.sqrt((self.pos[i,0]-cx)**2 + (self.pos[i,1]-cy)**2)
                if d < min_d:
                    min_d = d; mejor_c = np.array([cx, cy], dtype=np.float32)
            obs_rel = (mejor_c - self.pos[i]) / (CFG.domo*2)
            obs[i]  = np.concatenate([err, vel_n, rel, [dist_min], [id_n], obs_rel])
        return obs

    def _recompensa(self, i, dist):
        r  = (self._prev_dist[i] - dist) * CFG.r_progreso
        r -= dist * CFG.r_distancia
        r += CFG.r_llegada    if dist < CFG.dist_llegada    else 0.0
        r += CFG.r_proximidad if dist < CFG.dist_llegada*2  else 0.0
        r += CFG.r_tiempo
        for j in range(self.n):
            if j != i:
                if np.linalg.norm(self.pos[i]-self.pos[j]) < CFG.dist_colision:
                    r += CFG.r_colision
        # Colisión con cualquiera de los 4 obstáculos
        if _colision_obstaculos(self.pos[i], self.obstaculos):
            r += CFG.r_obstaculo
        return r
