# Clasificación de manzanas (buena/mala) - Baseline

Baseline: ResNet18 preentrenada (ImageNet), backbone congelado, solo se entrena la capa final.
Sin aumento de datos, umbral 0.5, 5 épocas.

## Datos
Dataset de manzanas freshapples/rottenapples. Origen y licencia: (completar).
El dataset no se incluye en el repositorio.

## Uso
pip install -r requirements.txt
Ver notebooks/baseline.ipynb
