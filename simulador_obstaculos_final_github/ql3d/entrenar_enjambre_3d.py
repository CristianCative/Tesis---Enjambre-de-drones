# =============================================================================
# entrenar_enjambre_3d.py — Q-Learning 3D con obstáculos esféricos
# Visualización: HTML interactivo Three.js + PNG matplotlib 3D (resumen métricas)
# =============================================================================
import os, time, json
import numpy as np
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from config3d import CFG
from entorno_enjambre_3d import EntornoEnjambre3D
from agente_qlearning_3d import AgenteQLearning3D

COLORES      = ['#1f77b4','#ff7f0e','#d62728','#8c564b','#e377c2',
                '#2ca02c','#17becf','#9467bd','#bcbd22']
NOMBRE_FORMA = {'linea': 'Línea', 'v': 'V', 'circulo': 'Círculo'}
FORMAS       = ['linea', 'v', 'circulo']

os.makedirs('evidencias', exist_ok=True)


# =============================================================================
# ENTRENAMIENTO
# =============================================================================
def entrenar(forma, n_ep, timestamp):
    entorno = EntornoEnjambre3D(forma=forma)
    agente  = AgenteQLearning3D()
    hist    = {k: [] for k in ['recompensa','error','colisiones','td_error',
                                'dist_minima','estabilidad','drones_form','paso_estab']}
    t0 = time.time()

    with mlflow.start_run(run_name=f"QL3D_{forma}_{timestamp}"):
        estados_posibles = CFG.n_drones * (CFG.n_bins**3) * CFG.obs_bins
        mlflow.log_params({
            'algoritmo'        : 'Q-Learning-3D',
            'formacion'        : forma,
            'n_drones'         : CFG.n_drones,
            'n_episodios'      : n_ep,
            'domo'             : CFG.domo,
            'separacion'       : CFG.separacion,
            'max_pasos'        : CFG.max_pasos_enjambre,
            'dist_colision'    : CFG.dist_colision,
            'dist_llegada'     : CFG.dist_llegada,
            'alpha'            : CFG.alpha,
            'gamma'            : CFG.gamma,
            'epsilon_ini'      : CFG.epsilon_ini,
            'epsilon_fin'      : CFG.epsilon_fin,
            'epsilon_decay'    : CFG.epsilon_decay,
            'n_bins'           : CFG.n_bins,
            'obs_bins'         : CFG.obs_bins,
            'n_obstaculos'     : CFG.n_obstaculos,
            'estados_posibles' : estados_posibles,
        })

        print(f"\n{'='*60}")
        print(f"  Q-Learning 3D — Formación {forma.upper()}")
        print(f"  α={CFG.alpha}  γ={CFG.gamma}  ε={CFG.epsilon_ini}→{CFG.epsilon_fin}")
        print(f"  Estados posibles: {estados_posibles:,}  ({CFG.n_bins}³ bins × {CFG.obs_bins} obs × {CFG.n_drones} drones)")
        print(f"{'='*60}")

        for ep in range(1, n_ep+1):
            obs  = entorno.reset(ep=ep)
            done = False
            R_ep = td_ep = 0.0

            while not done:
                acc = agente.seleccionar_acciones_enjambre(
                    obs, entorno.obstaculos, entorno.pos)
                obs_sig, rews, done, info = entorno.step(acc)
                td = agente.actualizar_enjambre(
                    obs, entorno.obstaculos, entorno.pos,
                    acc, rews, obs_sig, entorno.pos, done)
                R_ep  += sum(rews)
                td_ep += td
                obs    = obs_sig

            agente.decaer_epsilon()
            hist['recompensa'].append(R_ep)
            hist['error'].append(info['error_promedio'])
            hist['colisiones'].append(info['colisiones_total'])
            hist['td_error'].append(td_ep)
            hist['dist_minima'].append(info['dist_minima'] if info['dist_minima'] < np.inf else 0.0)
            hist['estabilidad'].append(info['estabilidad_formacion'])
            hist['drones_form'].append(info['drones_en_formacion'])
            hist['paso_estab'].append(info['paso_estabilizacion'] or CFG.max_pasos_enjambre)

            cobertura = agente.n_estados / estados_posibles * 100
            mlflow.log_metrics({
                'recompensa'            : R_ep,
                'error_posicion'        : info['error_promedio'],
                'colisiones'            : info['colisiones_total'],
                'convergencia'          : td_ep,
                'dist_minima'           : hist['dist_minima'][-1],
                'estabilidad_formacion' : info['estabilidad_formacion'],
                'drones_formacion'      : info['drones_en_formacion'],
                'tiempo_estabilizacion' : info['paso_estabilizacion'] or CFG.max_pasos_enjambre,
            }, step=ep)

            if ep % CFG.log_cada == 0:
                w = CFG.ventana_conv
                print(f'  Ep {ep:5d}/{n_ep}  |  R={np.mean(hist["recompensa"][-w:]):9.1f}'
                      f'  |  Ep={np.mean(hist["error"][-w:]):.3f}m'
                      f'  |  Col={np.mean(hist["colisiones"][-w:]):.1f}'
                      f'  |  ε={agente.epsilon:.3f}'
                      f'  |  Q={agente.n_estados:,}/{estados_posibles:,} ({cobertura:.1f}%)'
                      f'  |  t={time.time()-t0:.0f}s')

        np.save(f'evidencias/hist_ql3d_{forma}_{n_ep}ep_{timestamp}.npy', hist)

        # ── Resumen tabla Q ───────────────────────────────────────────────────
        cobertura = agente.n_estados / estados_posibles * 100
        print(f'\n  ── Tabla Q 3D [{forma.upper()}] ──────────────────────────────')
        print(f'  Estados posibles  : {estados_posibles:,}')
        print(f'  Estados visitados : {agente.n_estados:,}  ({cobertura:.1f}%)')
        print(f'  Actualizaciones   : {agente.actualizaciones:,}')
        print(f'  ────────────────────────────────────────────────────────────')

        mlflow.log_metrics({
            'final_recompensa_media'      : float(np.mean(hist['recompensa'][-100:])),
            'final_error_posicion'        : float(np.mean(hist['error'][-100:])),
            'final_colisiones_media'      : float(np.mean(hist['colisiones'][-100:])),
            'final_drones_formacion'      : float(np.mean(hist['drones_form'][-100:])),
            'final_estabilidad_formacion' : float(np.mean(hist['estabilidad'][-100:])),
            'final_tiempo_estabilizacion' : float(np.mean(hist['paso_estab'][-100:])),
            'final_dist_minima'           : float(np.mean(hist['dist_minima'][-100:])),
            'tiempo_entrenamiento_s'      : time.time() - t0,
        })

        _graficar_metricas(hist, forma, n_ep, timestamp)
        _exportar_threejs(entorno, agente, forma, n_ep, timestamp)

        mlflow.log_artifact(f'evidencias/metricas_ql3d_{forma}_{timestamp}.png')
        mlflow.log_artifact(f'evidencias/visualizacion_ql3d_{forma}_{timestamp}.html')

    return hist, agente


# =============================================================================
# GRÁFICAS PNG (métricas 2D — igual que versión 2D)
# =============================================================================
def _graficar_metricas(hist, forma, n_ep, ts):
    def suav(arr, w=100):
        a = np.array(arr, dtype=float)
        return np.convolve(a, np.ones(w)/w, mode='valid')

    ep_s = np.arange(50, n_ep-49)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f'Q-Learning 3D — Formación {NOMBRE_FORMA[forma]}\n'
                 f'{n_ep} episodios', fontsize=13, fontweight='bold')

    paneles = [
        ('recompensa',  'M7: Recompensa acumulada',              None,              None),
        ('error',       'M3: Error posición $E_p$ [m]',          CFG.dist_llegada,  f'$d_{{llegada}}$={CFG.dist_llegada}m'),
        ('colisiones',  'M1: Colisiones/episodio',               None,              None),
        ('drones_form', 'M8: Drones en formación',               9,                 'Objetivo 9/9'),
        ('td_error',    'M6: TD-Error (convergencia)',           None,              None),
        ('dist_minima', 'M4: Distancia mínima entre drones [m]', CFG.dist_colision, f'$d_{{col}}$={CFG.dist_colision}m'),
        ('estabilidad', 'M5: Estabilidad σ [m]',                 None,              None),
        ('paso_estab',  'M2: Tiempo estabilización [pasos]',     None,              None),
    ]

    for ax, (key, titulo, umbral, ulabel) in zip(axes.flat, paneles):
        arr = np.array(hist[key], dtype=float)
        sav = suav(arr)
        n   = min(len(ep_s), len(sav))
        ax.plot(ep_s[:n], sav[:n], color='#1f77b4', lw=2.0)
        if umbral:
            ax.axhline(umbral, color='forestgreen', lw=1.5, ls='--', label=ulabel)
            ax.legend(fontsize=8)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel('Episodio'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'evidencias/metricas_ql3d_{forma}_{ts}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [PNG] evidencias/metricas_ql3d_{forma}_{ts}.png')


# =============================================================================
# VISUALIZACIÓN THREE.JS
# =============================================================================
def _exportar_threejs(entorno, agente, forma, n_ep, ts):
    """Ejecuta el mejor episodio, graba trayectorias y genera HTML Three.js."""

    # Buscar episodio representativo: moda de drones_form en los 5000 episodios
    from scipy import stats as _stats
    hist_tmp = []
    for ep_test in range(1, n_ep + 1):
        obs = entorno.reset(ep=ep_test)
        for _ in range(CFG.max_pasos_enjambre):
            acc = agente.seleccionar_acciones_enjambre(
                obs, entorno.obstaculos, entorno.pos, determinista=True)
            obs, _, done, info = entorno.step(acc)
            if done: break
        hist_tmp.append((ep_test, info['drones_en_formacion']))
    moda_val = int(_stats.mode([v for _, v in hist_tmp], keepdims=True).mode[0])
    candidatos = [ep for ep, v in hist_tmp if v == moda_val]
    mejor_ep = candidatos[len(candidatos) // 2]

    # Grabar trayectorias
    obs = entorno.reset(ep=mejor_ep)
    tray = [[p.tolist()] for p in entorno.pos]
    obs_guardados = list(entorno.obstaculos)
    formacion_pts = [f.tolist() for f in entorno.formacion]

    for _ in range(CFG.max_pasos_enjambre):
        acc = agente.seleccionar_acciones_enjambre(
            obs, entorno.obstaculos, entorno.pos, determinista=True)
        obs, _, done, info = entorno.step(acc)
        for i in range(entorno.n):
            tray[i].append(entorno.pos[i].tolist())
        if done: break

    datos = {
        'algoritmo'  : 'Q-Learning 3D',
        'formacion'  : forma,
        'domo'       : CFG.domo,
        'colores'    : COLORES,
        'trayectorias': tray,
        'targets'    : formacion_pts,
        'obstaculos' : [{'cx':cx,'cy':cy,'cz':cz,'w':w,'h':h,'d':d}
                        for (cx,cy,cz,w,h,d) in obs_guardados],
        'nombre_forma': NOMBRE_FORMA[forma],
    }

    html = _generar_html(datos)
    path = f'evidencias/visualizacion_ql3d_{forma}_{ts}.html'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  [HTML] {path}')


def _generar_html(d):
    datos_json = json.dumps(d, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Enjambre {d['algoritmo']} — Formación {d['nombre_forma']}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#ffffff; color:#222; font-family:sans-serif; overflow:hidden; }}
  #canvas {{ display:block; width:100vw; height:100vh; }}
  #info {{
    position:absolute; top:14px; left:14px;
    background:rgba(255,255,255,0.92); padding:12px 16px;
    border-radius:8px; font-size:13px; line-height:1.8;
    border:1px solid rgba(0,0,0,0.13);
  }}
  #info h3 {{ font-size:15px; margin-bottom:6px; color:#1a6fbf; }}
  #controles {{
    position:absolute; bottom:14px; left:50%;
    transform:translateX(-50%);
    background:rgba(255,255,255,0.92); padding:8px 20px;
    border-radius:20px; font-size:12px; color:#444;
    border:1px solid rgba(0,0,0,0.13);
  }}
  .dot {{ display:inline-block; width:10px; height:10px;
          border-radius:50%; margin-right:5px; vertical-align:middle; }}
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="info">
  <h3>🚁 {d['algoritmo']} — {d['nombre_forma']}</h3>
  <div id="legend"></div>
  <div style="margin-top:6px;font-size:11px;color:#888;">
    Zona ±{d['domo']}m &nbsp;|&nbsp; {len(d['trayectorias'])} drones
  </div>
</div>
<div id="controles">🖱 Arrastrar: rotar &nbsp;|&nbsp; Scroll: zoom &nbsp;|&nbsp; Click derecho: pan</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const DATA = {datos_json};

// ── Escena ────────────────────────────────────────────────────────────────────
const W = window.innerWidth, H = window.innerHeight;
const renderer = new THREE.WebGLRenderer({{canvas:document.getElementById('canvas'),antialias:true}});
renderer.setSize(W, H);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(60, W/H, 0.1, 100);
camera.position.set(8, 6, 8);
camera.lookAt(0, 0, 0);

// ── Luces ─────────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
dirLight.position.set(5, 10, 5);
scene.add(dirLight);

// ── Grid central ──────────────────────────────────────────────────────────────
const grid = new THREE.GridHelper(DATA.domo*2, 14, 0xbbbbcc, 0xddddee);
grid.material.transparent = true; grid.material.opacity = 0.5;
scene.add(grid);

// ── Obstáculos — cajas 3D ─────────────────────────────────────────────────────
DATA.obstaculos.forEach((o) => {{
  const geo = new THREE.BoxGeometry(o.w, o.d, o.h);
  const mat = new THREE.MeshPhongMaterial({{
    color:0xcc8844, transparent:true, opacity:0.55, emissive:0x553311}});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(o.cx, o.cz, o.cy);
  scene.add(mesh);
  const wfmat = new THREE.MeshBasicMaterial({{
    color:0xaa5522, wireframe:true, transparent:true, opacity:0.7}});
  const wf = new THREE.Mesh(geo.clone(), wfmat);
  wf.position.set(o.cx, o.cz, o.cy);
  scene.add(wf);
}});

// ── Targets — solo anillo ──────────────────────────────────────────────────────
DATA.targets.forEach((t, i) => {{
  const col = new THREE.Color(DATA.colores[i]);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.18, 0.018, 10, 30),
    new THREE.MeshBasicMaterial({{color:col, transparent:true, opacity:0.85}}));
  ring.position.set(t[0], t[2], t[1]);
  ring.rotation.x = Math.PI/2;
  scene.add(ring);
}});

// ── Trayectorias (líneas más visibles) ────────────────────────────────────────
DATA.trayectorias.forEach((tray, i) => {{
  const col = new THREE.Color(DATA.colores[i]);
  const pts = tray.map(p => new THREE.Vector3(p[0], p[2], p[1]));
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({{color:col, transparent:false, opacity:1.0, linewidth:2}});
  scene.add(new THREE.Line(geo, mat));
}});

// ── Drones — tamaño reducido ──────────────────────────────────────────────────
function crearDron(color) {{
  const grupo = new THREE.Group();
  const col   = new THREE.Color(color);

  // Cuerpo central pequeño
  const cuerpoGeo = new THREE.BoxGeometry(0.09, 0.03, 0.09);
  const cuerpoMat = new THREE.MeshPhongMaterial({{color:col, emissive:col, emissiveIntensity:0.4}});
  grupo.add(new THREE.Mesh(cuerpoGeo, cuerpoMat));

  // 4 brazos diagonales
  const brazMat = new THREE.MeshPhongMaterial({{color:0x555555}});
  [[-1,-1],[1,-1],[-1,1],[1,1]].forEach(([sx,sz]) => {{
    const geo = new THREE.CylinderGeometry(0.008, 0.008, 0.14, 6);
    const m   = new THREE.Mesh(geo, brazMat);
    m.rotation.z = Math.PI/2;
    m.rotation.y = Math.atan2(sz, sx);
    m.position.set(sx*0.07, 0, sz*0.07);
    grupo.add(m);
  }});

  // 4 rotores pequeños
  const rotMat = new THREE.MeshPhongMaterial({{
    color:0x999999, transparent:true, opacity:0.7, side:THREE.DoubleSide}});
  [[-1,-1],[1,-1],[-1,1],[1,1]].forEach(([sx,sz]) => {{
    const geo = new THREE.CylinderGeometry(0.055, 0.055, 0.005, 14);
    const m   = new THREE.Mesh(geo, rotMat);
    m.position.set(sx*0.11, 0, sz*0.11);
    grupo.add(m);
  }});

  return grupo;
}}

const drones = DATA.trayectorias.map((tray, i) => {{
  const d = crearDron(DATA.colores[i]);
  const p = tray[tray.length-1];
  d.position.set(p[0], p[2], p[1]);
  scene.add(d);
  return d;
}});

// ── Leyenda ───────────────────────────────────────────────────────────────────
const leg = document.getElementById('legend');
DATA.colores.slice(0, DATA.trayectorias.length).forEach((c, i) => {{
  leg.innerHTML += `<span class="dot" style="background:${{c}}"></span>Dron ${{i}}<br>`;
}});

// ── Controles de órbita ───────────────────────────────────────────────────────
let isDragging=false, isRightDrag=false;
let lastX=0, lastY=0;
let theta=0.8, phi=0.6, radius=8;
let panX=0, panY=0;

function updateCamera() {{
  camera.position.set(
    panX + radius*Math.sin(phi)*Math.sin(theta),
    panY + radius*Math.cos(phi),
    panX + radius*Math.sin(phi)*Math.cos(theta));
  camera.lookAt(panX, panY, 0);
}}
updateCamera();

renderer.domElement.addEventListener('mousedown', e => {{
  isDragging=true; isRightDrag=(e.button===2);
  lastX=e.clientX; lastY=e.clientY;
}});
window.addEventListener('mouseup', () => isDragging=false);
window.addEventListener('mousemove', e => {{
  if(!isDragging) return;
  const dx=e.clientX-lastX, dy=e.clientY-lastY;
  if(isRightDrag) {{ panX -= dx*0.01; panY += dy*0.01; }}
  else {{ theta -= dx*0.008; phi = Math.max(0.1,Math.min(Math.PI-0.1,phi+dy*0.008)); }}
  lastX=e.clientX; lastY=e.clientY;
  updateCamera();
}});
renderer.domElement.addEventListener('wheel', e => {{
  radius = Math.max(2, Math.min(30, radius+e.deltaY*0.02));
  updateCamera();
}});
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ── Animación: rotores girando ────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  const t = Date.now()*0.003;
  drones.forEach(d => {{
    d.children.filter((_,i) => i>4).forEach((r,j) => {{
      r.rotation.y = t * (j%2===0 ? 1 : -1) * 3;
    }});
  }});
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""



# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    import time as _time
    timestamp = _time.strftime('%Y%m%d_%H%M%S')

    mlflow.set_tracking_uri('../mlruns')
    mlflow.set_experiment('Enjambre_Drones_QL_vs_MAPPO')

    print(f'\n  Sesión Q-Learning 3D + Obstáculos: {timestamp}')
    resumen = []
    for forma in FORMAS:
        print(f'\n>>> Formación: {forma.upper()} <<<\n')
        hist, agente = entrenar(forma, CFG.n_episodios, timestamp)
        resumen.append((forma, agente, hist))

    estados_posibles = CFG.n_drones * (CFG.n_bins**3) * CFG.obs_bins
    print('\n' + '='*66)
    print('  RESUMEN GLOBAL — Q-Learning 3D')
    print('='*66)
    print(f'  {"Forma":8s}  |  {"Visitados":>12s}  |  {"Cobertura":>10s}  |  {"Updates":>10s}')
    print(f'  {"-"*8}  |  {"-"*12}  |  {"-"*10}  |  {"-"*10}')
    for forma, ag, hist in resumen:
        cob = ag.n_estados / estados_posibles * 100
        barra = '█' * int(cob/5) + '░' * (20 - int(cob/5))
        print(f'  {forma.upper():8s}  |  {ag.n_estados:>12,}  |  {cob:>9.1f}%  |  {ag.actualizaciones:>10,}')
        print(f'            [{barra}] {cob:.1f}%')
    print('='*66)
    print('\n  Q-Learning 3D completado.')
