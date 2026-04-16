# =============================================================================
# entrenar_mappo.py — MAPPO 2D con obstáculo rectangular
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

from config_mappo import CFG
from entorno_enjambre_mappo import EntornoEnjambreMAPPO

COLORES      = ['#1f77b4','#ff7f0e','#d62728','#8c564b','#e377c2',
                '#2ca02c','#17becf','#9467bd','#bcbd22']
NOMBRE_FORMA = {'linea': 'Línea', 'v': 'V', 'circulo': 'Círculo'}

os.makedirs('evidencias', exist_ok=True)


def entrenar(forma, n_ep, timestamp, AgenteMAPPO):
    entorno = EntornoEnjambreMAPPO(forma=forma)
    agente  = AgenteMAPPO()
    hist    = {k: [] for k in ['recompensa','error','colisiones','loss_actor',
                                'dist_minima','estabilidad','drones_form','paso_estab',
                                'entropia']}
    t0 = time.time()
    total_pasos = 0

    print(f"\n{'='*60}")
    print(f"  MAPPO 2D + Obstáculo — Formación {forma.upper()}")
    print(f"  γ={CFG.gamma}  λ={CFG.lam}  lr_actor={CFG.lr_actor}")
    print(f"  Obs: {CFG.dim_obs} dims  |  Acciones: {CFG.dim_accion}")
    print(f"  Política: COMPARTIDA (9 drones, 1 red)")
    print(f"{'='*60}")

    # ── MLflow: iniciar corrida ───────────────────────────────────────────────
    with mlflow.start_run(run_name=f"MAPPO_{forma}_{timestamp}"):

        mlflow.log_params({
            'algoritmo'      : 'MAPPO',
            'formacion'      : forma,
            'n_drones'       : CFG.n_drones,
            'n_episodios'    : n_ep,
            'domo'           : CFG.domo,
            'separacion'     : CFG.separacion,
            'max_pasos'      : CFG.max_pasos,
            'dist_colision'  : CFG.dist_colision,
            'dist_llegada'   : CFG.dist_llegada,
            'gamma'          : CFG.gamma,
            'lam'            : CFG.lam,
            'lr_actor'       : CFG.lr_actor,
            'lr_critico'     : CFG.lr_critico,
            'clip_eps'       : CFG.clip_eps,
            'epochs_ppo'     : CFG.epochs_ppo,
            'tam_minibatch'  : CFG.tam_minibatch,
            'coef_entropia'  : CFG.coef_entropia,
            'dim_obs'        : CFG.dim_obs,
            'dim_accion'     : CFG.dim_accion,
            'r_obstaculo'    : CFG.r_obstaculo,
            'n_obstaculos'   : 4,
            'capas_actor'    : str(CFG.capas_actor),
            'capas_critico'  : str(CFG.capas_critico),
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
                obs   = obs_sig

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

            # ── MLflow: métricas por episodio ─────────────────────────────────
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
                entr_media = np.mean(hist['entropia'][-w:])
                print(f'  Ep {ep:5d}/{n_ep}  |  R={np.mean(hist["recompensa"][-w:]):9.1f}'
                      f'  |  Ep={np.mean(hist["error"][-w:]):.3f}m'
                      f'  |  Col={np.mean(hist["colisiones"][-w:]):.1f}'
                      f'  |  H={entr_media:.3f}'
                      f'  |  t={time.time()-t0:.0f}s')

        np.save(f'evidencias/hist_mappo_{forma}_{n_ep}ep_{timestamp}.npy', hist)

        # ── Resumen de experiencia al finalizar ───────────────────────────────
        params   = sum(p.numel() for p in agente.red.parameters())
        entr_ini = np.mean(hist['entropia'][:10])
        entr_fin = np.mean(hist['entropia'][-10:])
        print(f'\n  ── Red MAPPO [{forma.upper()}] ──────────────────────────────────')
        print(f'  Parámetros de la red  : {params:,}')
        print(f'  Transiciones totales  : {total_pasos * CFG.n_drones:,}  ({total_pasos:,} pasos × {CFG.n_drones} drones)')
        print(f'  Actualizaciones PPO   : {agente.n_updates:,}')
        print(f'  Entropía política     : {entr_ini:.3f} (inicio) → {entr_fin:.3f} (final)')
        convergencia = "convergió ✓" if entr_fin < entr_ini * 0.6 else "aún explorando"
        print(f'  Estado de convergencia: {convergencia}')
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

        # ── MLflow: subir imágenes como artefactos ────────────────────────────
        mlflow.log_artifact(f'evidencias/metricas_mappo_{forma}_{timestamp}.png')
        mlflow.log_artifact(f'evidencias/visualizacion_mappo_{forma}_{timestamp}.png')

    return hist, agente, total_pasos


def _graficar_metricas(hist, forma, n_ep, ts):
    def suav(arr, w=100):
        a = np.array(arr, dtype=float)
        return np.convolve(a, np.ones(w)/w, mode='valid')

    ep_s = np.arange(50, n_ep-49)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f'MAPPO 2D + Obstáculo — Formación {NOMBRE_FORMA[forma]}\n'
                 f'{n_ep} episodios', fontsize=13, fontweight='bold')

    # Fila 1: métricas principales | Fila 2: métricas secundarias
    paneles = [
        # Fila 1
        ('recompensa',  'M7: Recompensa acumulada',               None,              None),
        ('error',       'M3: Error posición $E_p$ [m]',           CFG.dist_llegada,  f'$d_{{llegada}}$={CFG.dist_llegada}m'),
        ('colisiones',  'M1: Colisiones/episodio',                None,              None),
        ('drones_form', 'M8: Drones en formación',                9,                 'Objetivo 9/9'),
        # Fila 2
        ('loss_actor',  'M6: Loss Actor (convergencia)',          None,              None),
        ('dist_minima', 'M4: Distancia mínima entre drones [m]',  CFG.dist_colision, f'$d_{{col}}$={CFG.dist_colision}m'),
        ('estabilidad', 'M5: Estabilidad σ [m]',                  None,              None),
        ('paso_estab',  'M2: Tiempo estabilización [pasos]',      None,              None),
    ]

    for ax, (key, titulo, umbral, ulabel) in zip(axes.flat, paneles):
        arr = np.array(hist[key], dtype=float)
        sav = suav(arr)
        n   = min(len(ep_s), len(sav))
        ax.plot(ep_s[:n], sav[:n], color='#d62728', lw=2.0)
        if umbral:
            ax.axhline(umbral, color='forestgreen', lw=1.5, ls='--', label=ulabel)
            ax.legend(fontsize=8)
        ax.set_xlabel('Episodio', fontsize=9)
        ax.set_title(titulo, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'evidencias/metricas_mappo_{forma}_{ts}.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def _dibujar_dron(ax, x, y, color, escala=0.18, zorder=6):
    """Dibuja un dron estilizado (cuerpo + 4 brazos + 4 rotores) centrado en (x, y).
    La física no cambia — (x,y) es el punto de masa del agente."""
    import matplotlib.patches as mpatches
    e = escala

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

    # LED central
    ax.plot(x, y, 'o', color='white', markersize=3.5,
            markeredgewidth=0, zorder=zorder+2)


def _visualizar(entorno, agente, forma, n_ep, ts):
    # Buscar episodio representativo: moda de drones_form en los 5000 episodios
    from scipy import stats as _stats
    hist_tmp = []
    for ep_test in range(1, n_ep + 1):
        obs = entorno.reset(ep=ep_test)
        done = False
        while not done:
            acc, _, _ = agente.seleccionar_acciones(obs)
            obs, _, done, info = entorno.step(acc)
        hist_tmp.append((ep_test, info['drones_en_formacion']))
    moda_val = int(_stats.mode([v for _, v in hist_tmp], keepdims=True).mode[0])
    candidatos = [ep for ep, v in hist_tmp if v == moda_val]
    mejor_ep = candidatos[len(candidatos) // 2]

    # Grabar trayectorias
    obs = entorno.reset(ep=mejor_ep)
    tray = [[p.copy()] for p in entorno.pos]
    obs_guardados = list(entorno.obstaculos)   # guardar obstáculos del episodio
    done = False
    while not done:
        acc, _, _ = agente.seleccionar_acciones(obs)
        obs, _, done, info = entorno.step(acc)
        for i in range(entorno.n):
            tray[i].append(entorno.pos[i].copy())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f'Enjambre MAPPO 2D — Política aprendida\n'
                 f'Formación {NOMBRE_FORMA[forma]}', fontsize=12, fontweight='bold')

    for i in range(entorno.n):
        pts = np.array(tray[i])
        ax.plot(pts[:,0], pts[:,1], color=COLORES[i], alpha=0.45, lw=1.2)

    # Targets — círculo punteado sin fondo, radio visual reducido
    RADIO_TARGET_VISUAL = 0.10   # más chico visualmente
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
    plt.savefig(f'evidencias/visualizacion_mappo_{forma}_{ts}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Guardado] evidencias/visualizacion_mappo_{forma}_{ts}.png')


if __name__ == '__main__':
    from agente_mappo import AgenteMAPPO
    import torch
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # ── MLflow: mismo experimento que QL para comparar en la misma UI ─────────
    mlflow.set_tracking_uri('../mlruns')
    mlflow.set_experiment('Enjambre_Drones_QL_vs_MAPPO')

    print(f'\n  Sesión MAPPO + Obstáculo: {timestamp}')
    resumen = []
    for forma in CFG.formaciones:
        print(f'\n>>> Formación: {forma.upper()} <<<\n')
        hist, agente, pasos = entrenar(forma, CFG.n_episodios, timestamp, AgenteMAPPO)
        resumen.append((forma, agente, pasos, hist))

    print('\n' + '='*66)
    print('  RESUMEN GLOBAL — Experiencia acumulada MAPPO')
    print('='*66)
    params = sum(p.numel() for p in resumen[0][1].red.parameters())
    print(f'  Parámetros de la red (compartida): {params:,}')
    print()
    for forma, ag, pasos, hist in resumen:
        trans = pasos * CFG.n_drones
        entr_fin = np.mean(hist['entropia'][-10:])
        print(f'  {forma.upper():8s}  |  Transiciones: {trans:>10,}'
              f'  |  Updates PPO: {ag.n_updates:>5,}'
              f'  |  H final: {entr_fin:.3f}')
    print('='*66)
    print('\n  MAPPO completado.')
