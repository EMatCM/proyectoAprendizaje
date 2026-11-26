"""Interfaz web (Streamlit) del control de calidad de manzanas.

Reutiliza el pipeline de app.py (detector Mask R-CNN + clasificador final), sin duplicar la lógica.
Despliegue en Streamlit Community Cloud: archivo principal = streamlit_app.py
"""
import os

# En Streamlit Cloud hay más memoria que en Railway: se usan los valores originales del detector.
os.environ.setdefault("MAX_DET", "100")
os.environ.setdefault("DET_MIN_SIZE", "800")

import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Control de calidad de manzanas", page_icon="🍎", layout="wide")
st.title("Control de calidad de manzanas")
st.caption("Segmenta cada manzana y la clasifica como sana o deteriorada. "
           "Inspección solo visual: revisa a mano los resultados con confianza baja.")
st.warning("Además, evalúa manualmente la imagen: el modelo tiende a dar problemas con las manzanas "
           "veteadas y amarillas, por lo que su resultado en esos casos debe confirmarse a ojo.")


@st.cache_resource(show_spinner="Cargando los modelos (la primera vez tarda un par de minutos)...")
def cargar_pipeline():
    import app as nucleo      # carga el clasificador final y el detector una sola vez
    return nucleo


nucleo = cargar_pipeline()

with st.form("formulario"):
    archivo = st.file_uploader("Foto con varias manzanas", type=["jpg", "jpeg", "png", "webp"])
    c1, c2 = st.columns(2)
    umbral_mala = c1.slider("Umbral para 'deteriorada'", 0.1, 0.9, 0.5, 0.05)
    umbral_det = c2.slider("Umbral de detección", 0.1, 0.9, 0.5, 0.05)
    enviar = st.form_submit_button("Analizar", type="primary")

if enviar:
    if archivo is None:
        st.warning("Sube una foto.")
    else:
        img = ImageOps.exif_transpose(Image.open(archivo)).convert("RGB")
        with st.spinner("Analizando..."):
            res = nucleo.analizar(img, umbral_mala, umbral_det)
            resultado = nucleo.dibujar(res)
            resumen = nucleo.resumen_texto(res)
        izq, der = st.columns(2)
        izq.image(img, caption="Foto original", use_container_width=True)
        der.image(resultado, caption="Resultado", use_container_width=True)
        st.subheader("Resumen")
        st.text(resumen)
