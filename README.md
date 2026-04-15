
# Simulación de enjambre de drones — Q-Learning vs MAPPO

Repositorio del trabajo de grado: **"Simulación de un enjambre de drones para la coordinación y formación mediante algoritmos multi-agente"**

Autor: Cristian Cative  
Institución: Grupo de Investigación GED — Robótica y Sistemas Autónomos  
Año: 2026

---

## Descripción general

Este repositorio contiene la implementación completa de dos algoritmos de aprendizaje por refuerzo multi-agente aplicados al control cooperativo de un enjambre de 9 drones en entornos 2D y 3D con obstáculos. Los algoritmos evaluados son:

- **Q-Learning tabular** con política compartida y estado discretizado
- **MAPPO** (Multi-Agent Proximal Policy Optimization) con red actor-crítico compartida

Cada algoritmo se entrena sobre tres formaciones geométricas (línea, V y círculo) en entornos bidimensionales y tridimensionales, totalizando 12 experimentos independientes registrados con MLflow.

---

## Estructura del repositorio

```
simulador_obstaculos_final/
│
├── ejecutar_todo.py        # Orquestador: lanza los 4 módulos secuencialmente
│
├── ql/                     # Q-Learning 2D
│   ├── config.py
│   ├── entorno_enjambre_2d.py
│   ├── agente_qlearning_2d.py
│   ├── entrenar_enjambre.py
│   └── evidencias/
│
├── ql3d/                   # Q-Learning 3D
│   ├── config3d.py
│   ├── entorno_enjambre_3d.py
│   ├── agente_qlearning_3d.py
│   ├── entrenar_enjambre_3d.py
│   └── evidencias/
│
├── mappo/                  # MAPPO 2D
│   ├── config_mappo.py
│   ├── entorno_enjambre_mappo.py
│   ├── agente_mappo.py
│   ├── entrenar_mappo.py
│   └── evidencias/
│
├── mappo3d/                # MAPPO 3D
│   ├── config_mappo3d.py
│   ├── entorno_enjambre_mappo3d.py
│   ├── agente_mappo3d.py
│   ├── entrenar_mappo_3d.py
│   └── evidencias/
│
├── mlruns/                 # Experimentos MLflow
└── logs/                   # Logs de entrenamiento
```

---

## Requisitos

```bash
pip install numpy torch matplotlib mlflow
```

Python 3.10 o superior recomendado.

---

## Uso

Para ejecutar todos los experimentos en secuencia:

```bash
python ejecutar_todo.py
```

Para visualizar los resultados en MLflow:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Luego abrir `http://localhost:5000` en el navegador.

---

## Licencia y uso

> ⚠️ **Aviso de uso y citación**
>
> Este repositorio es de libre acceso con fines académicos, educativos y de investigación. Si utilizas este código, total o parcialmente, en un trabajo propio —incluso con modificaciones— debes citarlo de la siguiente manera:
>
> **Cative, C. (2026).** *Simulación de un enjambre de drones para la coordinación y formación mediante algoritmos multi-agente* [Trabajo de grado]. Grupo de Investigación GED, Colombia.
>
> El uso comercial sin autorización expresa del autor no está permitido.

---

## Contacto

Para preguntas relacionadas con el código o la investigación, puedes abrir un *issue* en este repositorio.
