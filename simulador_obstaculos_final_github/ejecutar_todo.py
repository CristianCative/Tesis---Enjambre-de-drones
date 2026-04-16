# =============================================================================
# ejecutar_todo.py — Orquestador completo
#
# Ejecuta secuencialmente los 4 entornos:
#   1. Q-Learning  2D  (linea, v, circulo)
#   2. MAPPO       2D  (linea, v, circulo)
#   3. Q-Learning  3D  (linea, v, circulo)
#   4. MAPPO       3D  (linea, v, circulo)
#
# Todo el output de terminal se guarda en:
#   logs/entrenamiento_YYYYMMDD_HHMMSS.log
#
# Uso:
#   python ejecutar_todo.py
# =============================================================================
import os, sys, time, runpy
from datetime import datetime

RAIZ      = os.path.dirname(os.path.abspath(__file__))
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR   = os.path.join(RAIZ, 'logs')
LOG_PATH  = os.path.join(LOG_DIR, f'entrenamiento_{TIMESTAMP}.log')
os.makedirs(LOG_DIR, exist_ok=True)


# ── TeeLogger: escribe en terminal Y en archivo simultáneamente ───────────────
class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, 'w', encoding='utf-8', buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def sep(texto, ancho=66):
    print(f'\n{"="*ancho}\n  {texto}\n{"="*ancho}\n')


def fmt_t(s):
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    return f'{h}h {m}m {s}s' if h else (f'{m}m {s}s' if m else f'{s}s')


def limpiar_sys_path_y_modulos():
    """Elimina del path y del cache todos los módulos de los 4 entornos."""
    carpetas = ['ql', 'mappo', 'ql3d', 'mappo3d']
    for c in carpetas:
        p = os.path.join(RAIZ, c)
        if p in sys.path:
            sys.path.remove(p)
    prefijos = [
        'config', 'entorno_enjambre', 'agente_', 'entrenar_',
        'config_mappo', 'config3d', 'config_mappo3d',
    ]
    for nombre in list(sys.modules.keys()):
        for p in prefijos:
            if nombre.startswith(p):
                del sys.modules[nombre]
                break


def ejecutar_entorno(nombre, carpeta, script_py):
    sep(f'INICIANDO {nombre}')
    t0 = time.time()
    limpiar_sys_path_y_modulos()
    carpeta_abs = os.path.join(RAIZ, carpeta)
    sys.path.insert(0, carpeta_abs)
    os.chdir(carpeta_abs)
    try:
        runpy.run_path(os.path.join(carpeta_abs, script_py),
                       run_name='__main__')
    except SystemExit:
        pass   # algunos scripts llaman sys.exit(0) al terminar
    except Exception as e:
        import traceback
        print(f'\n  [ERROR en {nombre}]')
        traceback.print_exc()
    duracion = time.time() - t0
    sep(f'{nombre} COMPLETADO  ({fmt_t(duracion)})')
    return duracion


# =============================================================================
if __name__ == '__main__':

    logger = TeeLogger(LOG_PATH)
    sys.stdout = logger

    print('=' * 66)
    print('  SIMULADOR ENJAMBRE DE DRONES — EJECUCIÓN COMPLETA')
    print(f'  Inicio : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Log    : logs/entrenamiento_{TIMESTAMP}.log')
    print('=' * 66)

    tiempos = {}
    tiempos['QL 2D']    = ejecutar_entorno('Q-LEARNING 2D',  'ql',     'entrenar_enjambre.py')
    tiempos['MAPPO 2D'] = ejecutar_entorno('MAPPO 2D',       'mappo',  'entrenar_mappo.py')
    tiempos['QL 3D']    = ejecutar_entorno('Q-LEARNING 3D',  'ql3d',   'entrenar_enjambre_3d.py')
    tiempos['MAPPO 3D'] = ejecutar_entorno('MAPPO 3D',       'mappo3d','entrenar_mappo_3d.py')

    total = sum(tiempos.values())

    print('\n' + '=' * 66)
    print('  RESUMEN FINAL')
    print('=' * 66)
    print(f'  {"Entorno":<15}  {"Tiempo":>12}')
    print(f'  {"-"*15}  {"-"*12}')
    for nombre, t in tiempos.items():
        print(f'  {nombre:<15}  {fmt_t(t):>12}')
    print(f'  {"-"*15}  {"-"*12}')
    print(f'  {"TOTAL":<15}  {fmt_t(total):>12}')
    print('=' * 66)
    print(f'\n  Log completo : logs/entrenamiento_{TIMESTAMP}.log')
    print(f'  MLflow UI    : mlflow ui --port 3000')
    print(f'                 http://localhost:3000')
    print('\n' + '=' * 66)

    sys.stdout = logger.terminal
    logger.close()
    print(f'\n  Log guardado en: logs/entrenamiento_{TIMESTAMP}.log')
