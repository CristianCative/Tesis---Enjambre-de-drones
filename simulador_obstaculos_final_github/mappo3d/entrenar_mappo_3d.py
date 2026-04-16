# =============================================================================
# entrenar_mappo_3d.py — MAPPO 3D con obstáculos esféricos
# Visualización: HTML interactivo Three.js + PNG métricas
# =============================================================================
import os, time, json
import numpy as np
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from config_mappo3d import CFG
from entorno_enjambre_mappo3d import EntornoEnjambreMAPPO3D

COLORES      = ['#1f77b4','#ff7f0e','#d62728','#8c564b','#e377c2',
                '#2ca02c','#17becf','#9467bd','#bcbd22']
NOMBRE_FORMA = {'linea': 'Línea', 'v': 'V', 'circulo': 'Círculo'}

os.makedirs('evidencias', exist_ok=True)


# =============================================================================
# ENTRENAMIENTO
# =============================================================================
def entrenar(forma, n_ep, timestamp, AgenteMAPPO3D):
    entorno = EntornoEnjambreMAPPO3D(forma=forma)
    agente  = AgenteMAPPO3D()
    hist    = {k: [] for k in ['recompensa','error','colisiones','loss_actor',
                                'dist_minima','estabilidad','drones_form',
                                'paso_estab','entropia']}
    t0 = time.time()
    total_pasos = 0

    print(f"\n{'='*60}")
    print(f"  MAPPO 3D — Formación {forma.upper()}")
    print(f"  γ={CFG.gamma}  λ={CFG.lam}  lr_actor={CFG.lr_actor}")
    print(f"  Obs: {CFG.dim_obs} dims  |  Acciones: {CFG.dim_accion}")
    print(f"  Política: COMPARTIDA (9 drones, 1 red)")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=f"MAPPO3D_{forma}_{timestamp}"):
        import torch
        params_red = sum(p.numel() for p in agente.red.parameters())
        mlflow.log_params({
            'algoritmo'     : 'MAPPO-3D',
            'formacion'     : forma,
            'n_drones'      : CFG.n_drones,
            'n_episodios'   : n_ep,
            'domo'          : CFG.domo,
            'separacion'    : CFG.separacion,
            'max_pasos'     : CFG.max_pasos,
            'dist_colision' : CFG.dist_colision,
            'dist_llegada'  : CFG.dist_llegada,
            'gamma'         : CFG.gamma,
            'lam'           : CFG.lam,
            'lr_actor'      : CFG.lr_actor,
            'lr_critico'    : CFG.lr_critico,
            'clip_eps'      : CFG.clip_eps,
            'epochs_ppo'    : CFG.epochs_ppo,
            'tam_minibatch' : CFG.tam_minibatch,
            'coef_entropia' : CFG.coef_entropia,
            'dim_obs'       : CFG.dim_obs,
            'dim_accion'    : CFG.dim_accion,
            'n_obstaculos'  : CFG.n_obstaculos,
            'capas_actor'   : str(CFG.capas_actor),
            'params_red'    : params_red,
        })

        for ep in range(1, n_ep+1):
            obs  = entorno.reset(ep=ep)
            done = False
            R_ep = 0.0

            while not done:
                acc, logp, val = agente.seleccionar_acciones(obs)
                obs_sig, rews, done, info = entorno.step(acc)
                agente.buffer.guardar(obs, acc, logp, rews, val, done)
                R_ep += float(np.sum(rews))
                total_pasos += 1
                obs = obs_sig

            _, _, ultimo_v = agente.seleccionar_acciones(obs)
            loss_a, _, entr = agente.actualizar(ultimo_v)

            hist['recompensa'].append(R_ep)
            hist['error'].append(info['error_promedio'])
            hist['colisiones'].append(info['colisiones_total'])
            hist['loss_actor'].append(loss_a)
            hist['entropia'].append(entr)
            hist['dist_minima'].append(info['dist_minima'] if info['dist_minima'] < np.inf else 0.0)
            hist['estabilidad'].append(info['estabilidad_formacion'])
            hist['drones_form'].append(info['drones_en_formacion'])
            hist['paso_estab'].append(info['paso_estabilizacion'] or CFG.max_pasos)

            mlflow.log_metrics({
                'recompensa'            : R_ep,
                'error_posicion'        : info['error_promedio'],
                'colisiones'            : info['colisiones_total'],
                'convergencia'          : loss_a,
                'dist_minima'           : hist['dist_minima'][-1],
                'estabilidad_formacion' : info['estabilidad_formacion'],
                'drones_formacion'      : info['drones_en_formacion'],
                'tiempo_estabilizacion' : info['paso_estabilizacion'] or CFG.max_pasos,
            }, step=ep)

            if ep % CFG.log_cada == 0:
                w = CFG.ventana_conv
                entr_m = np.mean(hist['entropia'][-w:])
                print(f'  Ep {ep:5d}/{n_ep}  |  R={np.mean(hist["recompensa"][-w:]):9.1f}'
                      f'  |  Ep={np.mean(hist["error"][-w:]):.3f}m'
                      f'  |  Col={np.mean(hist["colisiones"][-w:]):.1f}'
                      f'  |  H={entr_m:.3f}'
                      f'  |  t={time.time()-t0:.0f}s')

        np.save(f'evidencias/hist_mappo3d_{forma}_{n_ep}ep_{timestamp}.npy', hist)

        entr_ini = np.mean(hist['entropia'][:10])
        entr_fin = np.mean(hist['entropia'][-10:])
        print(f'\n  ── Red MAPPO 3D [{forma.upper()}] ───────────────────────────')
        print(f'  Parámetros de la red  : {params_red:,}')
        print(f'  Transiciones totales  : {total_pasos*CFG.n_drones:,}  ({total_pasos:,} pasos × {CFG.n_drones} drones)')
        print(f'  Actualizaciones PPO   : {agente.n_updates:,}')
        print(f'  Entropía política     : {entr_ini:.3f} (inicio) → {entr_fin:.3f} (final)')
        conv = "convergió ✓" if entr_fin < entr_ini*0.6 else "aún explorando"
        print(f'  Estado de convergencia: {conv}')
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

        mlflow.log_artifact(f'evidencias/metricas_mappo3d_{forma}_{timestamp}.png')
        mlflow.log_artifact(f'evidencias/visualizacion_mappo3d_{forma}_{timestamp}.html')

    return hist, agente, total_pasos


# =============================================================================
# GRÁFICAS PNG
# =============================================================================
def _graficar_metricas(hist, forma, n_ep, ts):
    def suav(arr, w=100):
        a = np.array(arr, dtype=float)
        return np.convolve(a, np.ones(w)/w, mode='valid')

    ep_s = np.arange(50, n_ep-49)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f'MAPPO 3D — Formación {NOMBRE_FORMA[forma]}\n'
                 f'{n_ep} episodios', fontsize=13, fontweight='bold')

    paneles = [
        ('recompensa',  'M7: Recompensa acumulada',              None,              None),
        ('error',       'M3: Error posición $E_p$ [m]',          CFG.dist_llegada,  f'$d_{{llegada}}$={CFG.dist_llegada}m'),
        ('colisiones',  'M1: Colisiones/episodio',               None,              None),
        ('drones_form', 'M8: Drones en formación',               9,                 'Objetivo 9/9'),
        ('loss_actor',  'M6: Loss actor (convergencia)',         None,              None),
        ('dist_minima', 'M4: Distancia mínima entre drones [m]', CFG.dist_colision, f'$d_{{col}}$={CFG.dist_colision}m'),
        ('estabilidad', 'M5: Estabilidad σ [m]',                 None,              None),
        ('entropia',    'Entropía política H',                   None,              None),
    ]

    for ax, (key, titulo, umbral, ulabel) in zip(axes.flat, paneles):
        arr = np.array(hist[key], dtype=float)
        sav = suav(arr)
        n   = min(len(ep_s), len(sav))
        ax.plot(ep_s[:n], sav[:n], color='#ff7f0e', lw=2.0)
        if umbral:
            ax.axhline(umbral, color='forestgreen', lw=1.5, ls='--', label=ulabel)
            ax.legend(fontsize=8)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel('Episodio'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'evidencias/metricas_mappo3d_{forma}_{ts}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [PNG] evidencias/metricas_mappo3d_{forma}_{ts}.png')


# =============================================================================
# VISUALIZACIÓN THREE.JS (misma función que QL3D, adaptada para MAPPO)
# =============================================================================
def _exportar_threejs(entorno, agente, forma, n_ep, ts):
    mejor_ep, mejor_n, mejor_err = 1, -1, np.inf
    for ep_test in range(1, min(n_ep, 300)+1):
        obs = entorno.reset(ep=ep_test)
        done = False
        while not done:
            acc, _, _ = agente.seleccionar_acciones(obs, determinista=True)
            obs, _, done, info = entorno.step(acc)
        n_ll = info['drones_en_formacion']
        if n_ll > mejor_n or (n_ll == mejor_n and info['error_promedio'] < mejor_err):
            mejor_n=n_ll; mejor_err=info['error_promedio']; mejor_ep=ep_test

    obs = entorno.reset(ep=mejor_ep)
    tray = [[p.tolist()] for p in entorno.pos]
    obs_guardados = list(entorno.obstaculos)
    formacion_pts = [f.tolist() for f in entorno.formacion]
    done = False
    while not done:
        acc, _, _ = agente.seleccionar_acciones(obs, determinista=True)
        obs, _, done, info = entorno.step(acc)
        for i in range(entorno.n):
            tray[i].append(entorno.pos[i].tolist())

    datos = {
        'algoritmo'   : 'MAPPO 3D',
        'formacion'   : forma,
        'domo'        : CFG.domo,
        'colores'     : COLORES,
        'trayectorias': tray,
        'targets'     : formacion_pts,
        'obstaculos'  : [{'cx':cx,'cy':cy,'cz':cz,'w':w,'h':h,'d':d}
                         for (cx,cy,cz,w,h,d) in obs_guardados],
        'nombre_forma': NOMBRE_FORMA[forma],
    }
    html = _generar_html(datos)
    path = f'evidencias/visualizacion_mappo3d_{forma}_{ts}.html'
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
  #info h3 {{ font-size:15px; margin-bottom:6px; color:#bf6a1a; }}
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

const renderer = new THREE.WebGLRenderer({{canvas:document.getElementById('canvas'),antialias:true}});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 100);
camera.position.set(8, 6, 8);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(5,10,5); scene.add(dir);

// Grid
const grid = new THREE.GridHelper(DATA.domo*2, 14, 0xbbbbcc, 0xddddee);
grid.material.transparent=true; grid.material.opacity=0.5; scene.add(grid);

// Obstáculos — cajas 3D
DATA.obstaculos.forEach(o => {{
  const geo = new THREE.BoxGeometry(o.w, o.d, o.h);
  const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({{
    color:0xcc8844, transparent:true, opacity:0.55, emissive:0x553311}}));
  mesh.position.set(o.cx, o.cz, o.cy); scene.add(mesh);
  const wf = new THREE.Mesh(geo.clone(), new THREE.MeshBasicMaterial({{
    color:0xaa5522, wireframe:true, transparent:true, opacity:0.7}}));
  wf.position.set(o.cx, o.cz, o.cy); scene.add(wf);
}});

// Targets — solo anillo
DATA.targets.forEach((t,i) => {{
  const col = new THREE.Color(DATA.colores[i]);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.18,0.018,10,30),
    new THREE.MeshBasicMaterial({{color:col, transparent:true, opacity:0.85}}));
  ring.position.set(t[0],t[2],t[1]); ring.rotation.x=Math.PI/2; scene.add(ring);
}});

// Trayectorias más visibles
DATA.trayectorias.forEach((tray,i) => {{
  const pts = tray.map(p => new THREE.Vector3(p[0],p[2],p[1]));
  scene.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({{color:new THREE.Color(DATA.colores[i]),transparent:false,opacity:1.0,linewidth:2}})));
}});

// Dron pequeño
function crearDron(color) {{
  const g = new THREE.Group();
  const col = new THREE.Color(color);
  g.add(new THREE.Mesh(new THREE.BoxGeometry(0.09,0.03,0.09),
    new THREE.MeshPhongMaterial({{color:col,emissive:col,emissiveIntensity:0.4}})));
  const bMat = new THREE.MeshPhongMaterial({{color:0x555555}});
  [[-1,-1],[1,-1],[-1,1],[1,1]].forEach(([sx,sz]) => {{
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.008,0.008,0.14,6), bMat);
    m.rotation.z=Math.PI/2; m.rotation.y=Math.atan2(sz,sx);
    m.position.set(sx*0.07,0,sz*0.07); g.add(m);
  }});
  const rMat = new THREE.MeshPhongMaterial({{color:0x999999,transparent:true,opacity:0.7,side:THREE.DoubleSide}});
  [[-1,-1],[1,-1],[-1,1],[1,1]].forEach(([sx,sz]) => {{
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.055,0.055,0.005,14), rMat);
    m.position.set(sx*0.11,0,sz*0.11); g.add(m);
  }});
  return g;
}}

const drones = DATA.trayectorias.map((tray,i) => {{
  const dr = crearDron(DATA.colores[i]);
  const p  = tray[tray.length-1];
  dr.position.set(p[0],p[2],p[1]); scene.add(dr); return dr;
}});

// Leyenda
const leg = document.getElementById('legend');
DATA.colores.slice(0,DATA.trayectorias.length).forEach((c,i) => {{
  leg.innerHTML += `<span class="dot" style="background:${{c}}"></span>Dron ${{i}}<br>`;
}});

// Controles órbita
let isDrag=false,isRight=false,lastX=0,lastY=0,theta=0.8,phi=0.6,radius=8,panX=0,panY=0;
function updCam() {{
  camera.position.set(
    panX+radius*Math.sin(phi)*Math.sin(theta),
    panY+radius*Math.cos(phi),
    panX+radius*Math.sin(phi)*Math.cos(theta));
  camera.lookAt(panX,panY,0);
}}
updCam();
renderer.domElement.addEventListener('mousedown',e=>{{isDrag=true;isRight=e.button===2;lastX=e.clientX;lastY=e.clientY;}});
window.addEventListener('mouseup',()=>isDrag=false);
window.addEventListener('mousemove',e=>{{
  if(!isDrag) return;
  const dx=e.clientX-lastX,dy=e.clientY-lastY;
  if(isRight){{panX-=dx*0.01;panY+=dy*0.01;}}
  else{{theta-=dx*0.008;phi=Math.max(0.1,Math.min(Math.PI-0.1,phi+dy*0.008));}}
  lastX=e.clientX;lastY=e.clientY;updCam();
}});
renderer.domElement.addEventListener('wheel',e=>{{radius=Math.max(2,Math.min(30,radius+e.deltaY*0.02));updCam();}});
renderer.domElement.addEventListener('contextmenu',e=>e.preventDefault());
window.addEventListener('resize',()=>{{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  const t=Date.now()*0.003;
  drones.forEach(d=>{{
    d.children.filter((_,i)=>i>4).forEach((r,j)=>{{r.rotation.y=t*(j%2===0?1:-1)*3;}});
  }});
  renderer.render(scene,camera);
}}
animate();
</script>
</body>
</html>"""



# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    from agente_mappo3d import AgenteMAPPO as AgenteMAPPO3D
    import torch
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    mlflow.set_tracking_uri('../mlruns')
    mlflow.set_experiment('Enjambre_Drones_QL_vs_MAPPO')

    print(f'\n  Sesión MAPPO 3D + Obstáculos: {timestamp}')
    resumen = []
    for forma in CFG.formaciones:
        print(f'\n>>> Formación: {forma.upper()} <<<\n')
        hist, agente, pasos = entrenar(forma, CFG.n_episodios, timestamp, AgenteMAPPO3D)
        resumen.append((forma, agente, pasos, hist))

    params = sum(p.numel() for p in resumen[0][1].red.parameters())
    print('\n' + '='*66)
    print('  RESUMEN GLOBAL — MAPPO 3D')
    print('='*66)
    print(f'  Parámetros de la red (compartida): {params:,}')
    print()
    for forma, ag, pasos, hist in resumen:
        entr_fin = np.mean(hist['entropia'][-10:])
        print(f'  {forma.upper():8s}  |  Trans: {pasos*CFG.n_drones:>10,}'
              f'  |  Updates PPO: {ag.n_updates:>5,}'
              f'  |  H final: {entr_fin:.3f}')
    print('='*66)
    print('\n  MAPPO 3D completado.')
