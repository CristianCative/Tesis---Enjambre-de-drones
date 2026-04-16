# Orquestador — `ejecutar_todo.py`

Script principal que lanza los cuatro módulos de entrenamiento en secuencia, garantizando que todos los experimentos se ejecuten bajo las mismas condiciones y con semillas compartidas para reproducibilidad.

---

## Orden de ejecución

```
1. Q-Learning  2D  →  ql/entrenar_enjambre.py      (línea, V, círculo)
2. MAPPO       2D  →  mappo/entrenar_mappo.py       (línea, V, círculo)
3. Q-Learning  3D  →  ql3d/entrenar_enjambre_3d.py  (línea, V, círculo)
4. MAPPO       3D  →  mappo3d/entrenar_mappo_3d.py  (línea, V, círculo)
```

En total se ejecutan **12 entrenamientos** de 5,000 episodios cada uno.

---

## Características

- **TeeLogger:** todo el output de terminal se guarda simultáneamente en `logs/entrenamiento_YYYYMMDD_HHMMSS.log`, sin perder ninguna línea aunque el proceso dure horas.
- **Semilla compartida:** todos los módulos usan `seed = 42`, garantizando que los episodios sean comparables entre algoritmos.
- **Timestamp único:** cada ejecución genera un timestamp al inicio que se propaga a todos los módulos, manteniendo consistencia en los nombres de archivos de evidencias.
- **Registro MLflow:** todos los runs se registran automáticamente en el experimento `Enjambre_Drones_QL_vs_MAPPO` dentro del directorio `mlruns/`.

---

## Uso

Desde la carpeta raíz del proyecto:

```bash
python ejecutar_todo.py
```

Tiempo estimado total: entre 6 y 8 horas dependiendo del hardware (CPU).

---

## Visualización de resultados

Una vez completado el entrenamiento:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Abrir `http://localhost:5000` para acceder al panel comparativo de los 12 experimentos.
