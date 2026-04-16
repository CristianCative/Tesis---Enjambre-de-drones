# =============================================================================
# entorno_enjambre_mappo3d.py — MAPPO 3D, domo esférico ±3.5m
# Formaciones horizontales (Z fijo), obstáculos esféricos
# =============================================================================
import numpy as np
from config_mappo3d import CFG

# ── Formaciones 3D reales (distribuidas en X, Y y Z) ─────────────────────────
def _formacion_linea(n, sep, z=0.0):
    # Diagonal 3D: distribuida por igual en X, Y y Z
    mitad = (n-1)/2.0
    return np.array([
        [(i-mitad)*sep,
         (i-mitad)*sep,
         (i-mitad)*sep * 0.3]
        for i in range(n)], dtype=np.float32)

def _formacion_v(n, sep, z=0.0):
    # V invertida simetrica: centro arriba, ramas bajan en Y y Z por igual
    pos = [(0.0, 0.0, sep * 1.0)]
    for k in range(1, (n+1)//2):
        pos.append((-k*sep, -k*sep, sep*(1.0 - k*0.5)))
        if len(pos) < n: pos.append((k*sep, -k*sep, sep*(1.0 - k*0.5)))
    arr = np.array(pos[:n], dtype=np.float32)
    arr[:,1] -= arr[:,1].mean()
    return arr

def _formacion_circulo(n, sep, z=0.0):
    # Circulo inclinado 45 grados en el espacio (plano X-Z)
    r = 2.0 / (2.0 * np.sin(np.pi/n))
    inc = np.pi / 4.0
    return np.array([
        [r * np.cos(2*np.pi*i/n - np.pi/2),
         r * np.sin(2*np.pi*i/n - np.pi/2) * np.cos(inc),
         r * np.sin(2*np.pi*i/n - np.pi/2) * np.sin(inc)]
        for i in range(n)], dtype=np.float32)

FORMACIONES = {'linea': _formacion_linea, 'v': _formacion_v, 'circulo': _formacion_circulo}

# ── Acciones 3D (15 acciones) ─────────────────────────────────────────────────
_s = 1.0/np.sqrt(2)
ACCIONES_3D = np.array([
    [ 1., 0., 0.], [-1., 0., 0.],
    [ 0., 1., 0.], [ 0.,-1., 0.],
    [ 0., 0., 1.], [ 0., 0.,-1.],
    [ _s, _s, 0.], [ _s,-_s, 0.],
    [-_s, _s, 0.], [-_s,-_s, 0.],
    [ _s, 0., _s], [ _s, 0.,-_s],
    [-_s, 0., _s], [-_s, 0.,-_s],
    [ 0., 0., 0.],
], dtype=np.float32)
VEL_ESCALAR = 5.0
DT          = 0.025

# ── Obstáculos en caja 3D (cx, cy, cz, w, h, d) ──────────────────────────────
OBS_MIN = 0.3
OBS_MAX = 1.0

def _generar_obstaculos(rng, formacion, pos_drones):
    obstaculos = []
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
            cz = float(rng.uniform(-limite*0.4, limite*0.4))
            libre = True
            for t in formacion:
                if (abs(cx-t[0]) < w/2+0.4 and
                    abs(cy-t[1]) < h/2+0.4 and
                    abs(cz-t[2]) < d/2+0.4):
                    libre = False; break
            if not libre: continue
            for (ox,oy,oz,ow,oh,od) in obstaculos:
                if (abs(cx-ox) < (w+ow)/2+0.2 and
                    abs(cy-oy) < (h+oh)/2+0.2 and
                    abs(cz-oz) < (d+od)/2+0.2):
                    libre = False; break
            if libre:
                obstaculos.append((cx,cy,cz,w,h,d)); colocado=True; break
        if not colocado:
            obstaculos.append(fallbacks[k])
    return obstaculos

def _colision_obstaculos(pos, obstaculos):
    for (cx,cy,cz,w,h,d) in obstaculos:
        if (abs(pos[0]-cx) < w/2 and
            abs(pos[1]-cy) < h/2 and
            abs(pos[2]-cz) < d/2):
            return True
    return False

# ── Entorno MAPPO 3D ──────────────────────────────────────────────────────────
class EntornoEnjambreMAPPO3D:

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
                p = rng.uniform(-CFG.domo*0.8, CFG.domo*0.8, 3).astype(np.float32)
                if (np.linalg.norm(p) <= CFG.domo*0.8 and
                        np.linalg.norm(p - self.formacion[i]) >= DIST_MIN_TARGET):
                    pos[i] = p; break
            else:
                angle  = np.arctan2(-self.formacion[i,1], -self.formacion[i,0])
                pos[i] = np.array([np.cos(angle)*CFG.domo*0.7,
                                   np.sin(angle)*CFG.domo*0.7, 0.0], dtype=np.float32)
        self.pos = pos
        self.vel = np.zeros((self.n, 3), dtype=np.float32)
        self._paso = 0; self.colisiones = 0
        self.dist_minima = np.inf; self.paso_estabilizacion = None
        self.obstaculos = _generar_obstaculos(rng, self.formacion, self.pos)
        # Expulsar drones dentro de obstáculos
        for i in range(self.n):
            if _colision_obstaculos(self.pos[i], self.obstaculos):
                for (cx,cy,cz,w,h,d) in self.obstaculos:
                    if (abs(self.pos[i,0]-cx)<w/2 and
                        abs(self.pos[i,1]-cy)<h/2 and
                        abs(self.pos[i,2]-cz)<d/2):
                        dx=self.pos[i,0]-cx; dy=self.pos[i,1]-cy; dz=self.pos[i,2]-cz
                        px=w/2-abs(dx); py=h/2-abs(dy); pz=d/2-abs(dz)
                        if px<=py and px<=pz:
                            self.pos[i,0]=cx+np.sign(dx)*(w/2+0.05)
                        elif py<=px and py<=pz:
                            self.pos[i,1]=cy+np.sign(dy)*(h/2+0.05)
                        else:
                            self.pos[i,2]=cz+np.sign(dz)*(d/2+0.05)
                        break
        self._prev_dist = np.linalg.norm(self.pos - self.formacion, axis=1).astype(np.float32)
        return self._obs_todos()

    def step(self, acciones):
        self._paso += 1
        for i in range(self.n):
            delta     = ACCIONES_3D[acciones[i]] * VEL_ESCALAR * DT
            nueva_vel = self.vel[i]*0.2 + delta
            nueva_pos = self.pos[i] + nueva_vel
            # Contener en domo esférico
            d = float(np.linalg.norm(nueva_pos))
            if d > CFG.domo:
                nueva_pos = (nueva_pos/d*CFG.domo).astype(np.float32)
                nueva_vel = np.zeros(3, dtype=np.float32)
            # Bloqueo obstáculos
            if _colision_obstaculos(nueva_pos, self.obstaculos):
                px = np.array([nueva_pos[0], self.pos[i][1], self.pos[i][2]], dtype=np.float32)
                py = np.array([self.pos[i][0], nueva_pos[1], self.pos[i][2]], dtype=np.float32)
                pz = np.array([self.pos[i][0], self.pos[i][1], nueva_pos[2]], dtype=np.float32)
                if   not _colision_obstaculos(px, self.obstaculos):
                    nueva_pos=px; nueva_vel=np.array([nueva_vel[0],0.,0.],dtype=np.float32)
                elif not _colision_obstaculos(py, self.obstaculos):
                    nueva_pos=py; nueva_vel=np.array([0.,nueva_vel[1],0.],dtype=np.float32)
                elif not _colision_obstaculos(pz, self.obstaculos):
                    nueva_pos=pz; nueva_vel=np.array([0.,0.,nueva_vel[2]],dtype=np.float32)
                else:
                    nueva_pos=self.pos[i].copy(); nueva_vel=np.zeros(3,dtype=np.float32)
            self.vel[i]=nueva_vel; self.pos[i]=nueva_pos

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
            'errores'              : errores,
            'error_promedio'       : float(np.mean(errores)),
            'colisiones_paso'      : col_paso,
            'colisiones_total'     : self.colisiones,
            'dist_minima'          : float(self.dist_minima),
            'estabilidad_formacion': float(np.std(errores)),
            'paso_estabilizacion'  : self.paso_estabilizacion,
            'todos_en_formacion'   : todos,
            'drones_en_formacion'  : int(np.sum(errores < CFG.dist_llegada)),
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
            id_n     = np.float32(i/(self.n-1))
            # Obstáculo más cercano (distancia al centro de la caja)
            min_d = np.inf
            vec_o = np.zeros(3, dtype=np.float32)
            for (cx,cy,cz,w,h,d) in self.obstaculos:
                centro = np.array([cx,cy,cz])
                dist = float(np.linalg.norm(self.pos[i]-centro))
                if dist < min_d:
                    min_d=dist; vec_o=(centro-self.pos[i])/(CFG.domo*2)
            obs[i] = np.concatenate([err, vel_n, rel, [dist_min], [id_n], vec_o])
        return obs

    def _recompensa(self, i, dist):
        r  = (self._prev_dist[i]-dist) * CFG.r_progreso
        r -= dist * CFG.r_distancia
        r += CFG.r_llegada    if dist < CFG.dist_llegada   else 0.0
        r += CFG.r_proximidad if dist < CFG.dist_llegada*2 else 0.0
        r += CFG.r_tiempo
        for j in range(self.n):
            if j != i and np.linalg.norm(self.pos[i]-self.pos[j]) < CFG.dist_colision:
                r += CFG.r_colision
        if _colision_obstaculos(self.pos[i], self.obstaculos):
            r += CFG.r_obstaculo
        return r
