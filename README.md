%%writefile README.md
# Control de calidad de manzanas: clasificación sana / deteriorada

Proyecto de Aprendizaje Computacional (Ing. de Sistemas). Prototipo de inspección
visual de manzanas. **Esta primera entrega es el modelo base (baseline)**, que
clasifica una imagen con una sola manzana como sana o deteriorada. Las siguientes
etapas añadirán segmentación por instancia, conteo por imagen e interfaz en Gradio.

> La inspección es solo visual: no se afirma inocuidad alimentaria ni se detectan
> defectos internos no visibles.

## Estado del proyecto

| Etapa | Estado |
|---|---|
| Baseline de clasificación (una manzana por imagen) | Hecho |
| Experimento controlado (una modificación sobre el baseline) | Pendiente |
| Segmentación por instancia (varias manzanas por imagen) | Pendiente |
| Resumen por imagen (total, sanas, deterioradas) | Pendiente |
| Interfaz Gradio | Pendiente |

## Modelo base

- Arquitectura: ResNet18 preentrenada en ImageNet (torchvision, `IMAGENET1K_V1`).
- Estrategia: transferencia de aprendizaje. Backbone congelado; solo se entrena la
  capa final (`fc`) de 2 clases.
- Entrada: imágenes redimensionadas a 224x224 y normalizadas con media/desviación de ImageNet.
- Sin aumento de datos en el baseline.
- Hiperparámetros: 5 épocas, batch 32, Adam, lr = 1e-3, semilla 42.
- Umbral de decisión para "deteriorada": 0.5.
- Selección del mejor modelo: F1 de la clase deteriorada sobre validación.
- Clase positiva en las métricas: `rottenapples` (el error más costoso es dejar
  pasar una manzana deteriorada, por eso se vigila su recall).

## Datos

- **Origen:** dataset público de Kaggle, "Fruits fresh and rotten for classification"
  (autor: sriramr).
  https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
- **Dataset de trabajo (`apples_dataset`):** versión derivada y depurada por el equipo
  a partir del original. Se conservaron solo las clases de manzana (`freshapples`,
  `rottenapples`) y se descartaron las demás frutas. 
- **Número de imágenes:**
  freshapples: 2,088 imágenes en total
  
        |Train (entrenamiento): 1,693 imágenes
        |Test (prueba): 395 imágenes
    -rottenapples: 2,943 imágenes en total
  
        |Train (entrenamiento): 2,342 imágenes
        |Test (prueba): 601 imágenes
  
- **Tipo de anotación:** etiqueta por imagen (freshapples / rottenapples), heredada del original.
  No incluye máscaras de instancia; las máscaras para la segmentación se obtendrán
  aparte.
- **Particiones:** el original trae `train` y `test`. La validación (`val`) se separó
  de `train` (15 %, semilla 42), por lo que `test` no interviene en ninguna decisión.
- El dataset no se incluye en este repositorio. Para reproducirlo, se proporcionará un link con el
  repositorio para descragar.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── modelos/
│   └── baseline_v1.pt          # pesos del baseline
├── resultados/
│   └── baseline_test.json      # métricas sobre test
├── notebooks/                  # notebook de entrenamiento y evaluación
└── docs/                       # ficha de datos y bitácora
```


## Instalación y uso

```bash
pip install -r requirements.txt
```

Organiza los datos así (los nombres de carpeta definen las clases):

```
apples_dataset/
├── train/  freshapples/  rottenapples/
├── val/    freshapples/  rottenapples/
└── test/   freshapples/  rottenapples/
```

Entrenamiento y evaluación: ejecuta el notebook en `notebooks/`. Para clasificar una
imagen con el modelo guardado, carga `modelos/baseline_v1.pt`  y aplica las mismas transformaciones de evaluación.

## Resultados del baseline

Métricas sobre `test`, tomando `rottenapples` como clase positiva. Detalle en
`resultados/baseline_test.json`.

| Métrica | Valor |
|---|---|
| Accuracy | [0.97] |
| Precisión (deteriorada) | [0.96] |
| Recall (deteriorada) | [0.96] |
| F1 (deteriorada) | [0.98] |

Matriz de confusión: [completar TP / FP / FN / TN].

Estos números corresponden a clasificación de **imágenes con una sola manzana**. No
representan el desempeño del sistema final con varias manzanas por foto.

## Limitaciones conocidas

- El baseline no segmenta ni cuenta: asume una manzana por imagen.
- Posible dependencia de fondo e iluminación del dataset original; no se ha probado
  con fotos propias ni con manzanas superpuestas.
- No distingue tipos de daño, solo sana / deteriorada.

## Componentes externos y propios

- **Externos:** ResNet18 con pesos de ImageNet (torchvision/PyTorch); dataset de Kaggle.
- **Propios:** código de entrenamiento y evaluación, particiones, métricas y análisis.

## Uso de IA

Se consultó IA generativa para explicar errores, revisar código y como ayuda para el manejo de 
repositorio desde colab.
