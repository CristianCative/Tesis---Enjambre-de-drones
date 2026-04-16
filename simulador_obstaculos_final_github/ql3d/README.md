# Q-Learning 3D — Módulo `ql3d/`

Extensión del módulo Q-Learning al entorno tridimensional. Mantiene la misma lógica y hiperparámetros que `ql/`, añadiendo el eje Z en el espacio de estados y en las acciones.

---

## Archivos

### `config3d.py`
Extiende la configuración 2D con parámetros específicos del entorno 3D. El cambio más significativo es la representación del obstáculo: en lugar de dos bins por eje (ox, oy), se usa un único bin escalar de distancia (`obs_bins = 5`) para mantener el espacio de estados manejable.

| Parámetro | Valor | Diferencia respecto a 2D |
|---|---|---|
| `n_bins` | 13 | Igual, ahora aplicado a 3 ejes |
| `obs_bins` | 5 | Un único bin escalar (distancia al obstáculo) |
| `estados_posibles` | 98,865 | `9 × 13³ × 5 = 98,865` |

El aumento de 38,025 a 98,865 estados (+160 %) es la causa directa del deterioro del desempeño: con 5,000 episodios, el agente solo puede visitar una fracción pequeña del espacio.

---

### `entorno_enjambre_3d.py`
Extiende el entorno 2D al espacio XYZ. Las diferencias principales con respecto a `entorno_enjambre_2d.py` son:

- **Formación Línea 3D:** diagonal en los tres ejes (no plana en Y=0)
- **Formación V 3D:** brazos con componente Z variable
- **Formación Círculo 3D:** inclinada 45° respecto al plano XY
- **Observación:** 18 dimensiones base (añade errores en Z, posiciones Z de vecinos)
- **Obstáculo:** se representa como distancia escalar al centro del obstáculo más cercano, en lugar de vector 2D

---

### `agente_qlearning_3d.py`
Adapta la función `discretizar()` al estado tridimensional:
`(id_dron, bin_ex, bin_ey, bin_ez, bin_obs)`

- `bin_ez`: bin del error en el eje Z
- `bin_obs`: bin escalar de distancia al obstáculo más cercano

El resto de la lógica (política ε-greedy, actualización TD, tabla compartida) es idéntica al agente 2D.

---

### `entrenar_enjambre_3d.py`
Igual estructura que `entrenar_enjambre.py` del módulo 2D. Genera visualizaciones interactivas en formato HTML usando Three.js en lugar de imágenes estáticas PNG, dado que las trayectorias 3D requieren rotación para ser interpretadas correctamente.

**Ejecución directa:**
```bash
cd ql3d/
python entrenar_enjambre_3d.py
```

---

## Resultados (últimos 100 episodios)

| Formación | Colisiones/ep | T. estab. (pasos) | Error Ep (m) | Drones en formación |
|---|---|---|---|---|
| Línea | 15.5 | 300 | 0.343 | 0.89/9 |
| V | 14.7 | 300 | 0.355 | 0.95/9 |
| Círculo | 2.2 | 300 | 0.395 | 0.71/9 |

> El tiempo de estabilización de 300 pasos en los tres casos indica que el enjambre nunca completa la formación dentro del límite del episodio, confirmando la limitación del método tabular en entornos tridimensionales.
