# MAPPO 3D — Módulo `mappo3d/`

Extensión de MAPPO al entorno tridimensional. La arquitectura de la red y el proceso de optimización son idénticos al módulo `mappo/`. Los cambios se concentran en la dimensión del vector de observación y en la dinámica del entorno.

---

## Archivos

### `config_mappo3d.py`
Extiende `config_mappo.py` con los cambios necesarios para el entorno 3D.

| Parámetro | Valor 2D | Valor 3D | Cambio |
|---|---|---|---|
| `dim_obs` | 14 | 20 | +6 dimensiones (eje Z) |
| `dim_accion` | 9 | 15 | +6 acciones con componente Z |
| `capas_actor` | [64, 64] | [64, 64] | Sin cambio |
| `capas_critico` | [64, 64] | [64, 64] | Sin cambio |

**Vector de observación 3D (20 dimensiones):**
Añade al vector 2D los componentes Z de: error al target (`ez`), velocidad propia (`vz`), cohesión (`Σez_vec`), velocidad relativa del vecino (`vz_rel`), errores de los dos vecinos (`ez_v1`, `ez_v2`) y vector al obstáculo (`oz`).

**Acciones 3D (15 acciones):** las 9 del plano XY más 6 adicionales con componente Z (±Z puro, y 4 combinaciones ±Z con diagonales XY).

---

### `entorno_enjambre_mappo3d.py`
Extiende el entorno MAPPO 2D a tres dimensiones:

- **Formación Línea 3D:** drones distribuidos en diagonal XYZ con separación uniforme
- **Formación V 3D:** estructura en V con profundidad en Z, brazos a ±45° en los tres planos
- **Formación Círculo 3D:** círculo inclinado 45° respecto al plano horizontal, distribución uniforme en el espacio
- **Obstáculos 3D:** generados como paralelepípedos (extensión del obstáculo 2D con altura aleatoria). El vector de observación incluye `(ox, oy, oz)` como desplazamiento tridimensional al centro del obstáculo más cercano.

---

### `agente_mappo3d.py`
La arquitectura es idéntica a `agente_mappo.py`. Solo cambia la dimensión de entrada de la red (`dim_obs = 20`) y la dimensión de salida del actor (`dim_accion = 15`). El proceso de entrenamiento PPO con GAE no requiere modificaciones.

---

### `entrenar_mappo_3d.py`
Idéntico a `entrenar_mappo.py` en estructura. Genera visualizaciones interactivas en HTML con Three.js para las trayectorias 3D, guardadas en `evidencias/`.

**Ejecución directa:**
```bash
cd mappo3d/
python entrenar_mappo_3d.py
```

---

## Resultados (últimos 100 episodios)

| Formación | Colisiones/ep | T. estab. (pasos) | Error Ep (m) | Drones en formación |
|---|---|---|---|---|
| Línea | 4.0 | 53 | 0.118 | 8.67/9 |
| V | 4.8 | 60 | 0.124 | 8.42/9 |
| Círculo | 0.8 | 57 | 0.119 | 8.50/9 |

> MAPPO mantiene un desempeño prácticamente equivalente al entorno 2D al escalar a 3D, demostrando que la representación continua de 20 dimensiones generaliza eficientemente sin degradación significativa.
