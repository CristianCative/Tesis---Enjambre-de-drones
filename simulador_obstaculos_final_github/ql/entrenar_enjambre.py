# =============================================================================
# entrenar_enjambre.py — Q-Learning 2D con obstáculo rectangular
# =============================================================================
import os, time
import numpy as np
import mlflow
import mlflow.artifacts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from config import CFG
from entorno_enjambre_2d import EntornoEnjambre2D
from agente_qlearning_2d import AgenteQLearning2D

COLORES      = ['#1f77b4','#ff7f0e','#d62728','#8c564b','#e377c2',
                '#2ca02c','#17becf','#9467bd','#bcbd22']
NOMBRE_FORMA = {'linea': 'Línea', 'v': 'V', 'circulo': 'Círculo'}
FORMAS       = ['linea', 'v', 'circulo']

os.makedirs('evidencias', exist_ok=True)


def entrenar(forma, n_ep, timestamp):
    entorno = EntornoEnjambre2D(forma=forma)
    agente  = AgenteQLearning2D()
    hist    = {k: [] for k in ['recompensa','error','colisiones','td_error',
                                'dist_minima','estabilidad','drones_form','paso_estab']}
    t0 = time.time()

    # ── MLflow: iniciar corrida ───────────────────────────────────────────────
    with mlflow.start_run(run_name=f"QL_{forma}_{timestamp}"):

        # Registrar todos los hiperparámetros de CFG
        mlflow.log_params({
            'algoritmo'        : 'Q-Learning',
            'formacion'        : forma,
            'n_drones'         : CFG.n_drones,
            'n_episodios'      : n_ep,
            'domo'             : CFG.domo,
            'separacion'       : CFG.separacion,
            'dt'               : CFG.dt,
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
            'r_obstaculo'      : CFG.r_obstaculo,
            'n_obstaculos'     : 4,
            'estados_posibles' : CFG.n_drones * (CFG.n_bins**2) * (CFG.obs_bins**2),
        })

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

            # ── MLflow: métricas por episodio ─────────────────────────────────
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
                      f'  |  t={time.time()-t0:.0f}s')

        np.save(f'evidencias/hist_ql_{forma}_{n_ep}ep_{timestamp}.npy', hist)

        # ── Resumen de cobertura de la tabla Q al finalizar ───────────────────
        estados_posibles = CFG.n_drones * (CFG.n_bins**2) * (CFG.obs_bins**2)
        cobertura = agente.n_estados / estados_posibles * 100
        print(f'\n  ── Tabla Q [{forma.upper()}] ──────────────────────────────────')
        print(f'  Estados posibles  : {estados_posibles:,}')
        print(f'  Estados visitados : {agente.n_estados:,}  ({cobertura:.1f}% de la tabla)')
        print(f'  Actualizaciones   : {agente.actualizaciones:,}')
        print(f'  Actualizaciones/estado visitado: {agente.actualizaciones/max(agente.n_estados,1):.1f}')
        print(f'  ────────────────────────────────────────────────────────────────')

        # ── MLflow: métricas resumen finales ──────────────────────────────────
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
        _visualizar(entorno, agente, forma, n_ep, timestamp)

        # ── MLflow: subir imágenes generadas como artefactos ──────────────────
        mlflow.log_artifact(f'evidencias/metricas_ql_{forma}_{timestamp}.png')
        mlflow.log_artifact(f'evidencias/visualizacion_ql_{forma}_{timestamp}.png')

    return hist, agente


def _graficar_metricas(hist, forma, n_ep, ts):
    def suav(arr, w=100):
        a = np.array(arr, dtype=float)
        return np.convolve(a, np.ones(w)/w, mode='valid')

    ep_s = np.arange(50, n_ep-49)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f'Q-Learning 2D + Obstáculos — Formación {NOMBRE_FORMA[forma]}\n'
                 f'{n_ep} episodios', fontsize=13, fontweight='bold')

    # Fila 1: métricas principales | Fila 2: métricas secundarias
    paneles = [
        # Fila 1
        ('recompensa',  'M7: Recompensa acumulada',              None,              None),
        ('error',       'M3: Error posición $E_p$ [m]',          CFG.dist_llegada,  f'$d_{{llegada}}$={CFG.dist_llegada}m'),
        ('colisiones',  'M1: Colisiones/episodio',               None,              None),
        ('drones_form', 'M8: Drones en formación',               9,                 'Objetivo 9/9'),
        # Fila 2
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
        ax.set_xlabel('Episodio', fontsize=9)
        ax.set_title(titulo, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'evidencias/metricas_ql_{forma}_{ts}.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def _dibujar_dron(ax, x, y, color, escala=0.18, zorder=6):
    """Dibuja un dron estilizado (cuerpo + 4 brazos + 4 rotores) centrado en (x, y).
    La física no cambia — (x,y) es el punto de masa del agente."""
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms
    e = escala
    t = ax.transData

    # 4 brazos diagonales
    for ang in [45, 135, 225, 315]:
        rad = np.deg2rad(ang)
        x1, y1 = x + e*0.22*np.cos(rad), y + e*0.22*np.sin(rad)
        x2, y2 = x + e*0.85*np.cos(rad), y + e*0.85*np.sin(rad)
        ax.plot([x1, x2], [y1, y2], color='#444444', lw=2.2,
                solid_capstyle='round', zorder=zorder)

    # 4 rotores (elipses aplanadas en punta de cada brazo)
    for ang in [45, 135, 225, 315]:
        rad = np.deg2rad(ang)
        rx, ry = x + e*np.cos(rad), y + e*np.sin(rad)
        rotor = mpatches.Ellipse((rx, ry), width=e*0.72, height=e*0.22,
                                  angle=ang, facecolor='#888888',
                                  edgecolor='#555555', lw=0.8,
                                  alpha=0.75, zorder=zorder)
        ax.add_patch(rotor)

    # Cuerpo central cuadrado redondeado
    cuerpo = mpatches.FancyBboxPatch(
        (x - e*0.28, y - e*0.28), e*0.56, e*0.56,
        boxstyle='round,pad=0.03',
        facecolor=color, edgecolor='white',
        linewidth=1.2, zorder=zorder+1)
    ax.add_patch(cuerpo)

    # LED central (punto brillante)
    ax.plot(x, y, 'o', color='white', markersize=3.5,
            markeredgewidth=0, zorder=zorder+2)


def _visualizar(entorno, agente, forma, n_ep, ts):
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
    tray = [[p.copy()] for p in entorno.pos]
    obs_guardados = list(entorno.obstaculos)   # guardar obstáculos del episodio
    for _ in range(CFG.max_pasos_enjambre):
        acc = agente.seleccionar_acciones_enjambre(
            obs, entorno.obstaculos, entorno.pos, determinista=True)
        obs, _, done, info = entorno.step(acc)
        for i in range(entorno.n):
            tray[i].append(entorno.pos[i].copy())
        if done: break

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'Enjambre Q-Learning 2D — Política aprendida\n'
                 f'Formación {NOMBRE_FORMA[forma]}', fontsize=12, fontweight='bold')

    # Trayectorias
    for i in range(entorno.n):
        pts = np.array(tray[i])
        ax.plot(pts[:,0], pts[:,1], color=COLORES[i], alpha=0.45, lw=1.2)

    # Targets — círculo punteado sin fondo, radio visual reducido
    RADIO_TARGET_VISUAL = 0.10   # más chico visualmente (antes = dist_llegada 0.20)
    for i, f in enumerate(entorno.formacion):
        circ = plt.Circle(f, RADIO_TARGET_VISUAL, color=COLORES[i],
                          fill=False, linestyle='--', lw=1.4, alpha=0.85)
        ax.add_patch(circ)
        ax.annotate(f'F{i}', f, textcoords='offset points',
                    xytext=(5, 5), fontsize=8, color=COLORES[i])

    # Posición final drones — forma de dron geométrico
    for i in range(entorno.n):
        p = entorno.pos[i]
        _dibujar_dron(ax, p[0], p[1], color=COLORES[i])
        ax.annotate(str(i), p, textcoords='offset points',
                    xytext=(10, 8), fontsize=9, fontweight='bold', color=COLORES[i])

    # 4 Obstáculos — muy tenues para no tapar la formación
    for k, (cx, cy, w, h) in enumerate(obs_guardados):
        rect = Rectangle((cx-w/2, cy-h/2), w, h,
                          linewidth=1.2, edgecolor='#555555',
                          facecolor='dimgray', alpha=0.20, zorder=4)
        ax.add_patch(rect)
        ax.text(cx, cy, f'O{k+1}', ha='center', va='center',
                fontsize=7, color='#555555', fontweight='normal', zorder=5)

    ax.set_xlim(-CFG.domo-0.5, CFG.domo+0.5)
    ax.set_ylim(-CFG.domo-0.5, CFG.domo+0.5)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    # Leyenda con ícono de dron simulado
    from matplotlib.patches import FancyBboxPatch
    drone_patch = FancyBboxPatch((0,0), 1, 1, boxstyle='round,pad=0.1',
                                  facecolor='#1f77b4', edgecolor='white', lw=1)
    ax.legend(handles=[
        drone_patch,
        Line2D([0],[0], ls='--', color='gray', label='Target (posición objetivo)'),
        plt.Rectangle((0,0),1,1, fc='dimgray', alpha=0.20, ec='#555555', lw=1.2, label='Obstáculos')],
        labels=['Dron (posición final)', 'Target (posición objetivo)', 'Obstáculos'],
        loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'evidencias/visualizacion_ql_{forma}_{ts}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Guardado] evidencias/visualizacion_ql_{forma}_{ts}.png')


if __name__ == '__main__':
    import time as _time
    timestamp = _time.strftime('%Y%m%d_%H%M%S')

    # ── MLflow: apunta al mismo directorio para QL y MAPPO ───────────────────
    mlflow.set_tracking_uri('../mlruns')
    mlflow.set_experiment('Enjambre_Drones_QL_vs_MAPPO')

    print(f'\n  Sesión Q-Learning + Obstáculo: {timestamp}')
    estados_posibles = CFG.n_drones * (CFG.n_bins**2) * (CFG.obs_bins**2)
    resumen = []
    for forma in FORMAS:
        print(f'\n>>> Formación: {forma.upper()} <<<\n')
        hist, agente = entrenar(forma, CFG.n_episodios, timestamp)
        resumen.append((forma, agente))

    print('\n' + '='*66)
    print('  RESUMEN GLOBAL — Cobertura tabla Q por formación')
    print('='*66)
    for forma, ag in resumen:
        cob = ag.n_estados / estados_posibles * 100
        barra = '█' * int(cob / 2) + '░' * (50 - int(cob / 2))
        print(f'  {forma.upper():8s}  {barra}  {cob:5.1f}%  ({ag.n_estados:,}/{estados_posibles:,})')
    print('='*66)
    print('\n  Q-Learning completado.')
