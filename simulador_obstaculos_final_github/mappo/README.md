# MAPPO 2D — Módulo `mappo/`

Implementación de MAPPO (Multi-Agent Proximal Policy Optimization) con política compartida para el control cooperativo de 9 drones en entorno bidimensional con obstáculos. Utiliza entrenamiento centralizado con ejecución descentralizada (CTDE).

---

## Archivos

### `config_mappo.py`
Define la clase `ConfigMAPPO` con todos los hiperparámetros del modelo y el entorno.

| Parámetro | Valor | Descripción |
|---|---|---|
| `dim_obs` | 14 | Dimensión del vector de observación por dron |
| `dim_accion` | 9 | Número de acciones discretas |
| `capas_actor` | [64, 64] | Neuronas por capa oculta del actor |
| `capas_critico` | [64, 64] | Neuronas por capa oculta del crítico |
| `lr_actor` | 3e-4 | Tasa de aprendizaje del actor |
| `lr_critico` | 1e-3 | Tasa de aprendizaje del crítico |
| `gamma` | 0.99 | Factor de descuento |
| `lam` | 0.95 | Factor GAE (Generalized Advantage Estimation) |
| `clip_eps` | 0.2 | Parámetro de recorte PPO |
| `epochs_ppo` | 4 | Épocas de optimización por rollout |
| `tam_minibatch` | 64 | Tamaño del minibatch |
| `coef_entropia` | 0.01 | Coeficiente de bonus de entropía |
| `n_episodios` | 5000 | Episodios de entrenamiento |

**Vector de observación (14 dimensiones):**
`[ex, ey, vx, vy, Σex_vec, Σey_vec, vx_rel, vy_rel, ex_v1, ey_v1, ex_v2, ey_v2, ox, oy]`

Donde `ex/ey` es el error al target, `vx/vy` la velocidad propia, `Σex_vec/Σey_vec` el vector de cohesión hacia los 3 vecinos más cercanos, `vx_rel/vy_rel` la velocidad relativa del vecino más cercano, `ex_v1/ey_v1` y `ex_v2/ey_v2` los errores de los dos vecinos más cercanos, y `ox/oy` el vector al obstáculo más cercano.

---

### `agente_mappo.py`
Contiene la clase `RedActorCritico` y la clase `AgenteMAPPO`.

**`RedActorCritico`**
Red neuronal con dos cabezas independientes que comparten el mismo vector de entrada:
- **Actor:** produce logits sobre las 9 acciones discretas
- **Crítico:** produce el valor escalar V(s)

Inicialización ortogonal de pesos con ganancia √2. Activaciones Tanh en capas ocultas.

**`AgenteMAPPO`**
- Recolecta rollouts completos por episodio (transiciones estado-acción-recompensa)
- Calcula ventajas con GAE: `A_t = Σ (γλ)^k · δ_{t+k}`
- Optimiza con pérdida clipeada: `L = L_CLIP − c_H · H(π) + c_V · L_V`
- La política es compartida: todos los drones pasan sus observaciones por la misma red
- Compatible con CPU y GPU (auto-detección de dispositivo)

---

### `entorno_enjambre_mappo.py`
Entorno idéntico al Q-Learning 2D en cuanto a dinámica, formaciones y generación de obstáculos. La diferencia está en la función de observación: devuelve vectores continuos de 14 dimensiones en lugar de estados discretizados.

La velocidad máxima de los drones es `vel_max = 0.125 m/paso`, escalada a partir de las mismas 9 direcciones discretas del Q-Learning pero con control de magnitud continua.

---

### `entrenar_mappo.py`
Bucle de entrenamiento principal. En cada episodio:

1. Recolecta un rollout completo para los 9 drones
2. Calcula retornos y ventajas GAE al final del episodio
3. Optimiza la red durante `epochs_ppo` épocas con minibatches de tamaño `tam_minibatch`
4. Registra métricas en MLflow
5. Al finalizar, guarda figuras y el historial `.npy`

**Ejecución directa:**
```bash
cd mappo/
python entrenar_mappo.py
```

---

## Resultados (últimos 100 episodios)

| Formación | Colisiones/ep | T. estab. (pasos) | Error Ep (m) | Drones en formación |
|---|---|---|---|---|
| Línea | 25.2 | 48 | 0.094 | 8.80/9 |
| V | 12.7 | 70 | 0.098 | 8.71/9 |
| Círculo | 8.9 | 79 | 0.112 | 8.59/9 |
