# Q-Learning 2D — Módulo `ql/`

Implementación de Q-Learning tabular con política compartida para el control cooperativo de un enjambre de 9 drones en un entorno bidimensional con 4 obstáculos rectangulares por episodio.

---

## Archivos

### `config.py`
Define la clase `Config` con todos los hiperparámetros del experimento. Se instancia como `CFG` y es importada por los demás módulos.

| Parámetro | Valor | Descripción |
|---|---|---|
| `n_drones` | 9 | Número de agentes en el enjambre |
| `domo` | 3.5 m | Radio del área de vuelo |
| `n_bins` | 13 | Bins de discretización por eje de error |
| `obs_bins` | 5 | Bins por eje de distancia al obstáculo |
| `alpha` | 0.20 | Tasa de aprendizaje |
| `gamma` | 0.99 | Factor de descuento |
| `epsilon_ini` | 1.0 | Exploración inicial |
| `epsilon_fin` | 0.1 | Exploración mínima |
| `epsilon_decay` | 0.995 | Decaimiento de exploración por episodio |
| `n_episodios` | 5000 | Episodios de entrenamiento |
| `dist_llegada` | 0.20 m | Radio de llegada al target |
| `dist_colision` | 0.30 m | Umbral de colisión entre drones |
| `r_obstaculo` | -15.0 | Penalización por colisión con obstáculo |

El espacio de estados resultante es: `9 × 13² × 5² = 38,025 estados`.

---

### `entorno_enjambre_2d.py`
Simula la dinámica del enjambre en el plano XY. Sus responsabilidades principales son:

- Generar posiciones iniciales aleatorias para los 9 drones dentro del domo
- Definir los targets de cada formación (línea, V, círculo) centrados en el origen
- Generar 4 obstáculos rectangulares aleatorios por episodio, concentrados en la zona central
- Calcular la función de recompensa individual para cada dron en cada paso
- Detectar colisiones entre drones y entre drones y obstáculos
- Devolver observaciones de 12 dimensiones base por dron (error de posición, velocidad, posiciones relativas de vecinos, velocidad relativa del vecino más cercano)

**Acciones disponibles:** 9 acciones discretas (4 cardinales + 4 diagonales + quieto).

---

### `agente_qlearning_2d.py`
Contiene la función `discretizar()` y la clase `AgenteQLearning2D`.

**`discretizar(obs, id_dron, obstaculos, pos_dron)`**
Convierte la observación continua en una tupla de estado discreta:
`(id_dron, bin_ex, bin_ey, bin_ox, bin_oy)`

- `bin_ex`, `bin_ey`: bins del error de posición en X e Y (rango ±2.5 m)
- `bin_ox`, `bin_oy`: bins del vector hacia el obstáculo más cercano (rango ±3.0 m)

**`AgenteQLearning2D`**
- Mantiene la tabla Q como diccionario Python (solo almacena estados visitados)
- Implementa política ε-greedy con decaimiento exponencial por episodio
- Actualiza la tabla usando la regla TD: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]`
- Comparte la misma tabla entre los 9 agentes (política compartida)

---

### `entrenar_enjambre.py`
Punto de entrada para el entrenamiento de las tres formaciones en secuencia. Para cada formación:

1. Instancia el entorno y el agente
2. Ejecuta el bucle de entrenamiento episodio a episodio
3. Registra métricas en MLflow en cada episodio
4. Genera y guarda las figuras de métricas y trayectorias al finalizar
5. Guarda el historial de entrenamiento en formato `.npy` dentro de `evidencias/`

**Ejecución directa:**
```bash
cd ql/
python entrenar_enjambre.py
```

---

## Resultados (últimos 100 episodios)

| Formación | Colisiones/ep | T. estab. (pasos) | Error Ep (m) | Drones en formación |
|---|---|---|---|---|
| Línea | 42.1 | 296 | 0.239 | 4.30/9 |
| V | 16.4 | 295 | 0.248 | 3.82/9 |
| Círculo | 12.0 | 299 | 0.271 | 3.70/9 |
