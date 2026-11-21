# Control de calidad de manzanas: segmentación por instancia y estado (sana / deteriorada)

Proyecto de Aprendizaje Computacional (Ing. de Sistemas). Prototipo de inspección visual de manzanas: en una foto con varias manzanas se segmenta cada una, se clasifica como **sana** o **deteriorada** y se cuenta el total, las sanas y las deterioradas.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EMatCM/proyectoAprendizaje/blob/main/notebooks/baseline.ipynb)

## Estado del proyecto

| Etapa | Estado |
|---|---|
| Baseline de clasificación (una manzana por imagen) | Hecho |
| Experimentos controlados (una modificación por experimento) | Hecho (rama `experimento`) |
| Segmentación por instancia (varias manzanas por imagen) | Hecho |
| Resumen por imagen (total, sanas, deterioradas) | Hecho |
| Interfaz (Colab y Gradio) | Hecho |

## Estructura del repositorio

Este repositorio tiene dos ramas con propósitos distintos:

| Rama | Contenido |
|---|---|
| `main` | Solo lo necesario para usar el prototipo: el modelo final (`modelos/finalbaseline.pt`), el notebook de inferencia (`notebooks/baseline.ipynb`), `requirements.txt` y `resultados/baseline_test.json`. |
| `experimento` | Todo el trabajo de experimentación: los modelos de cada experimento y sus semillas (`modelos/`), las métricas guardadas (`resultados/`) y el notebook de entrenamiento y comparación (`notebooks/experimentos.ipynb`). |

## Cómo usarlo

1. Abre `notebooks/baseline.ipynb` en Colab con el botón de arriba.
2. En la sección 3 del notebook, elige la interfaz con dos interruptores independientes:

   | Interruptor | `True` hace |
   |---|---|
   | `USAR_COLAB` | Subes una foto en la celda y ves el resultado debajo. |
   | `USAR_GRADIO` | Abre la interfaz web Gradio (enlace público temporal mientras la sesión esté activa). |

3. Ejecuta las celdas en orden. El notebook clona este repositorio, carga `modelos/finalbaseline.pt` y el detector, y analiza la foto.

No hace falta descargar el dataset ni entrenar nada para usar el prototipo.

## Cómo funciona

Flujo para cada foto:

1. **Detector (Mask R-CNN ResNet50 FPN v2, pesos COCO, no se entrena).** Encuentra cada fruta y su máscara. Se aceptan las clases COCO `apple`, `orange`, `donut` y `cake`, porque el detector llama así a muchas manzanas muy podridas. La clase `bowl` no se acepta, porque daría falsos positivos con recipientes.
2. **Filtrado de detecciones.** Se descartan grupos (una máscara que contiene a varias manzanas), duplicados (máscaras con IoU alto) y detecciones dudosas que se pegan a otra ya aceptada. Las detecciones con confianza entre 0.35 y 0.5 solo se rescatan si no se superponen con otra manzana.
3. **Recorte.** Cada manzana se recorta con su máscara y fondo blanco (margen del 8 %).
4. **Clasificador (ResNet18).** Decide sana o deteriorada para cada recorte. Umbral para "deteriorada": 0.5. Si la confianza es menor de 0.6, el resultado se marca como incierto para revisar a mano.
5. **Resumen.** Total, sanas y deterioradas de la foto.

Componentes externos: ResNet18 con pesos de ImageNet y Mask R-CNN con pesos de COCO, ambos vía torchvision. Lo propio: el entrenamiento de la capa final, los experimentos, el filtrado de detecciones, las métricas y el análisis.

## Modelo base (baseline)

- Arquitectura: ResNet18 preentrenada en ImageNet (torchvision, `IMAGENET1K_V1`).
- Estrategia: transferencia de aprendizaje. Backbone congelado; solo se entrena la capa final (`fc`) de 2 clases.
- Entrada: imágenes redimensionadas a 224x224 y normalizadas con media y desviación de ImageNet.
- Sin aumento de datos en el baseline.
- Hiperparámetros: 5 épocas, batch 32, Adam, lr = 1e-3, semilla 42.
- Selección del mejor modelo (mejor época): F1 de la clase deteriorada sobre validación.
- Clase positiva en las métricas: `rottenapples` (el error más costoso es dejar pasar una manzana deteriorada, por eso se vigila su recall).

Resultados en `test` (996 imágenes, una evaluación por modelo):

| Modelo | Exactitud | Precisión (deteriorada) | Recall (deteriorada) | F1 (deteriorada) | TP / FP / FN / TN |
|---|---|---|---|---|---|
| `baseline_v1` | 0.979 | 0.997 | 0.968 | 0.982 | 582 / 2 / 19 / 393 |
| `baseline_v2` (reentrenado con `val`) | 0.985 | 0.993 | 0.982 | 0.987 | 590 / 4 / 11 / 391 |

## Datos

- **Origen del dataset base:** dataset público de Kaggle, "Fruits fresh and rotten for classification" (autor: sriramr). https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
- **Dataset de trabajo (`apples_dataset`):** versión derivada del original, con solo las clases de manzana (`freshapples`, `rottenapples`).
- **Número de imágenes:**

  | Clase | Total | Train | Test |
  |---|---|---|---|
  | `freshapples` | 2.088 | 1.693 | 395 |
  | `rottenapples` | 2.943 | 2.342 | 601 |

- **Anotación:** etiqueta por imagen, heredada del original. No incluye máscaras de instancia.
- **Particiones:** el original trae `train` y `test`. La validación (`val`) se separó de `train` (15 %, semilla 42), así que `test` no interviene en ninguna decisión.
- **Dataset casero usado en los experimentos:** además del dataset base, se usó un conjunto de imágenes armado por el equipo, que combina imágenes de fondo blanco con otras de fondos más complejos o con varias manzanas en la misma foto. Este conjunto no se encuentra disponible para descarga.
- El dataset base (`apples_dataset`) no se incluye en este repositorio. Solo se puede descargar mediante el notebook de experimentos (`notebooks/experimentos.ipynb`, rama `experimento`), que lo enlaza con el Drive donde está guardado.

## Experimentos

Cada experimento cambia **una sola cosa** respecto al modelo con el que se compara. El detalle, el código y las salidas están en `notebooks/experimentos.ipynb` (rama `experimento`), y las métricas por época en `resultados/` de esa rama.

| Experimento | Qué cambia |
|---|---|
| `exp1_aumento` | Aumento de datos sobre el baseline |
| `exp5_sanas_y_malas` | Imágenes adicionales de 100x100 (sanas y deterioradas) |
| `exp6_recortes` | Entrena con recortes de manzana con fondo blanco (lo que ve el clasificador en uso) |
| `exp7_recortes_color` | exp6 con aumento de color |
| `exp8_recortes_color_ext` | exp7 con fotos de otra fuente (`Defectos_Manzanas`) |
| `exp9_recortes_color_ext_rojas` | exp8 con las imágenes adicionales de exp5 (sanas rojas y deterioradas de 100x100) |

Validación (mejor época, semilla 42). **Los experimentos 1 y 5 se miden sobre fotos completas (604 imágenes de validación) y los experimentos 6 a 9 sobre recortes (567), así que no son directamente comparables entre los dos grupos.** La validación tiene la misma distribución que el entrenamiento y casi no distingue entre modelos.

| Modelo | Recall (deteriorada) | Precisión | F1 | Imágenes de validación |
|---|---|---|---|---|
| `baseline_v2` | 0.997 | 0.986 | 0.992 | 604 |
| `exp1_aumento` | 0.994 | 0.983 | 0.989 | 604 |
| `exp5_sanas_y_malas` | 0.983 | 0.986 | 0.984 | 604 |
| `exp6_recortes` | 0.985 | 0.981 | 0.983 | 567 |
| `exp7_recortes_color` | 0.981 | 0.988 | 0.984 | 567 |
| `exp8_recortes_color_ext` | 0.988 | 0.982 | 0.985 | 567 |
| `exp9_recortes_color_ext_rojas` | 0.985 | 0.981 | 0.983 | 567 |

Conjunto externo `ext_test` (54 recortes de la fuente `Defectos_Manzanas` apartados antes de entrenar, semilla 42; salida de `experimentos.ipynb`). Con tan pocas imágenes, uno o dos errores mueven mucho las métricas:

| Modelo | Falsos positivos (sana marcada como deteriorada) | Falsos negativos (deteriorada marcada como sana) |
|---|---|---|
| `baseline_v2` | 18 | 1 |
| `exp5_sanas_y_malas` | 13 | 0 |
| `exp6_recortes` | 9 | 1 |
| `exp7_recortes_color` | 7 | 0 |
| `exp8_recortes_color_ext` | 4 | 0 |
| `exp9_recortes_color_ext_rojas` | 0 | 2 |

Los experimentos 6 a 9 se repitieron con las semillas 43 y 44. Los resultados están en `experimentos.ipynb`.

## Modelo final

`modelos/finalbaseline.pt` es el modelo del experimento 9 (`exp9_recortes_color_ext_rojas`), entrenado con la **semilla 44**. Sus pesos son idénticos a `exp9_recortes_color_ext_rojas_s44.pt` de la rama `experimento`.

- **Por qué se eligió:** identifica mejor las manzanas sanas que los demás modelos, a costa de sacrificar algo de recall en la clase deteriorada. En `ext_test` (54 recortes), con la semilla 42 no marca ninguna sana como deteriorada (0 falsos positivos, contra 4 de `exp8`) y deja pasar 2 deterioradas (2 falsos negativos, contra 0 de `exp8`). Las tres semillas de exp9 dan entre 0 y 1 falsos positivos y entre 1 y 2 falsos negativos, aunque en las pruebas individuales no se detectó un falso negativo.
- **Práctica manual con fotos propias** (cualitativa, no una evaluación formal; las fotos se fueron eligiendo durante las pruebas): frente a los demás modelos, exp9 corrigió varias manzanas sanas muy rojas, rosadas, naranjas o veteadas que se marcaban como deterioradas. Aun así sigue teniendo dificultades con algunas de ese tipo (una manzana rosada clara y varias manzanas naranjas en una foto con muchas apiladas), pero menos que los demás modelos. En la carpeta de 45 manzanas sanas de prueba, exp9 marcó 17 como deterioradas, contra 20 de `baseline_v2`, 19 de `exp5` y 24 de `exp8`; una sola foto con muchas manzanas apiladas concentra la mayor parte de esos errores.

## Limitaciones conocidas

Observadas en pruebas con fotos propias, **no medidas con una evaluación formal**:

- **Manzanas tapadas o muy apiladas:** el detector no las encuentra todas, y las máscaras de manzanas medio tapadas salen cortadas, lo que empeora la clasificación de ese recorte.
- **Manzanas muy podridas:** pueden no ser detectadas como `apple`. Hay detecciones que llegan como `bowl` y no se aceptan.
- **Colores poco comunes o manzanas veteadas entre las sanas** (rosa claro, naranja, amarillo, rojas con vetas): el clasificador puede marcarlas como deterioradas. *Hipótesis, no comprobada con un experimento:* las manzanas deterioradas suelen tener zonas veteadas o manchadas que generan un alto contraste, y el modelo podría asociar ese contraste con deterioro; los colores claros podrían confundirse con moho por el mismo motivo, el contraste de colores.
- **Confianza baja:** varios aciertos tienen confianza cercana a 0.5. Se avisa en el resultado para revisar a mano.

## Reproducibilidad

- Semilla fija (42) en todos los entrenamientos, y semillas 43 y 44 para medir variación.
- Dependencias en `requirements.txt` (para Gradio: `pip install gradio`).
- Los entrenamientos y comparaciones se rehacen con `notebooks/experimentos.ipynb` (rama `experimento`), que descarga el dataset y guarda modelos y métricas.

## Equipo y uso de IA

- Integrantes: Mateo Martinez Colon, Norelis Contreras y Camilo Diaz
- Declaración del uso de herramientas de IA: se consultó IA generativa para explicar errores, revisar código en los experimentos y como ayuda para el manejo de repositorio desde Colab. Además, se usó IA generativa para escribir la interfaz Gradio.
