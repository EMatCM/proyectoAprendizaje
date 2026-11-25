"""Control de calidad de manzanas: interfaz web (Gradio).

Segmenta cada manzana de una foto (Mask R-CNN, COCO), la clasifica como sana o deteriorada
con el modelo final y cuenta el total, las sanas y las deterioradas.

Uso local o en Colab:   python app.py          (GRADIO_SHARE=1 para un enlace público temporal)
Uso en Hugging Face Spaces: sube este archivo, requirements.txt y modelos/finalbaseline.pt
"""
import os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms.functional import to_tensor

# ---- Parámetros ----
TAM = 224
MEDIA, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASE_MALA = "rottenapples"   # clase positiva (la que se cuenta como deteriorada)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# El modelo se busca junto a este archivo (Spaces) o en la carpeta modelos/ del repositorio.
AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
CANDIDATOS = [AQUI / "modelos" / "finalbaseline.pt", AQUI.parent / "modelos" / "finalbaseline.pt"]
MODELO_FINAL = next((p for p in CANDIDATOS if p.exists()), CANDIDATOS[0])


def tf_eval():
    return transforms.Compose([
        transforms.Resize((TAM, TAM)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, STD),
    ])


def crear_modelo(n_clases=2, congelar=True):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if congelar:
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_clases)
    return m



# ---- Parámetros de la segmentación ----
UMBRAL_DET = 0.5        # confianza para aceptar una manzana sin más condiciones
UMBRAL_RESCATE = 0.35   # entre este valor y UMBRAL_DET solo se acepta si NO se pega a otra manzana (rescata podridas dudosas)
UMBRAL_MASCARA = 0.5    # umbral para volver binaria la máscara suave del detector
UMBRAL_MALA = 0.5       # probabilidad mínima para decir "deteriorada"
CONF_BAJA = 0.6         # por debajo de esta confianza se avisa que el resultado es incierto
MARGEN = 0.08           # margen extra alrededor de cada recorte (8 %)
MAX_LADO = 1024         # las fotos más grandes se reducen a este lado máximo
MIN_AREA_FRAC = 0.002   # descarta detecciones menores al 0,2 % de la imagen
FRAC_CONTENIDA = 0.8    # una máscara 'contiene' a otra si cubre al menos este % del área de la otra
MAX_CONTENIDAS = 2      # una detección que contiene a 2 o más manzanas es un grupo, no una manzana
MAX_AREA_REL = 4.0      # con 6 o más manzanas seguras, una detección más grande que 4 veces la mediana es un grupo (None = desactivar)
IOU_DUPLICADA = 0.6     # dos máscaras con IoU >= esto son la misma manzana: se queda la de mayor puntaje
SOLAPE_RESCATE = 0.3    # una detección dudosa se descarta si más de este % de su área cae sobre otra ya aceptada
COLOR = {"sana": (40, 170, 60), "deteriorada": (220, 40, 40)}

# ---- Clasificador de estado (modelo final) ----
if not MODELO_FINAL.exists():
    raise FileNotFoundError(f"No existe {MODELO_FINAL}. Pon finalbaseline.pt en modelos/ junto a este archivo.")
ck_clf = torch.load(MODELO_FINAL, map_location=dev)
clases_clf = ck_clf.get("clases", ["freshapples", "rottenapples"])
assert CLASE_MALA in clases_clf, f"Clases del modelo: {clases_clf}"
clasificador = crear_modelo(len(clases_clf), congelar=False)
clasificador.load_state_dict(ck_clf["state_dict"])
clasificador = clasificador.to(dev).eval()

# ---- Detector / segmentador (Mask R-CNN preentrenado en COCO, no se entrena) ----
pesos_det = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
detector = maskrcnn_resnet50_fpn_v2(weights=pesos_det).to(dev).eval()
ID_MANZANA = pesos_det.meta["categories"].index("apple")
# El detector llama 'donut', 'orange' o 'cake' a muchas manzanas muy podridas; se aceptan como posible manzana.
CLASES_FRUTA = ["apple", "orange", "donut", "cake"]
IDS_FRUTA = torch.tensor([pesos_det.meta["categories"].index(c) for c in CLASES_FRUTA], device=dev)

def filtrar_instancias(puntajes, mascaras, umbral_det):
    """Índices (por puntaje decreciente) de las detecciones que son UNA manzana.

    Trabaja con las máscaras, no con las cajas: una caja grande puede tocar muchas manzanas sin ser un grupo.
    1) quita grupos: máscaras que contienen a 2 o más manzanas seguras, o que son enormes frente a las demás;
    2) quita duplicados: de dos máscaras casi iguales deja la de mayor puntaje;
    3) las de puntaje dudoso (entre UMBRAL_RESCATE y umbral_det) solo se aceptan si no se montan sobre otra.
    """
    n = len(puntajes)
    if n == 0:
        return np.zeros(0, dtype=int)
    M = np.asarray(mascaras).reshape(n, -1).astype(np.float32)
    area = M.sum(1)
    inter = M @ M.T                                   # píxeles en común entre cada par de máscaras
    seguras = np.asarray(puntajes) >= umbral_det
    vivos = np.ones(n, dtype=bool)
    mediana = np.median(area[seguras]) if seguras.sum() >= 6 else None
    for i in range(n):
        dentro = sum(1 for j in range(n)
                     if j != i and seguras[j] and area[j] < 0.6 * area[i] and inter[i, j] >= FRAC_CONTENIDA * area[j])
        enorme = MAX_AREA_REL is not None and mediana is not None and area[i] > MAX_AREA_REL * mediana
        if dentro >= MAX_CONTENIDAS or enorme:
            vivos[i] = False
    kept = []
    for i in np.argsort(-np.asarray(puntajes), kind="stable"):
        if not vivos[i]:
            continue
        if kept:
            k = np.array(kept)
            iou = inter[i, k] / (area[i] + area[k] - inter[i, k])
            if (iou >= IOU_DUPLICADA).any():
                continue
            if puntajes[i] < umbral_det and (inter[i, k] / area[i] > SOLAPE_RESCATE).any():
                continue
        kept.append(int(i))
    return np.array(kept, dtype=int)


@torch.no_grad()
def segmentar(img, umbral_det=None):
    """Devuelve cajas, puntajes y máscaras booleanas de cada manzana detectada."""
    umbral_det = UMBRAL_DET if umbral_det is None else umbral_det
    piso = min(UMBRAL_RESCATE, umbral_det)            # el detector entrega también las dudosas; luego se filtran
    out = detector([to_tensor(img).to(dev)])[0]
    ok = torch.isin(out["labels"], IDS_FRUTA) & (out["scores"] >= piso)
    cajas = out["boxes"][ok].cpu().numpy()
    puntajes = out["scores"][ok].cpu().numpy()
    mascaras = (out["masks"][ok, 0] > UMBRAL_MASCARA).cpu().numpy()
    area_min = MIN_AREA_FRAC * img.width * img.height
    keep = np.array([i for i, m in enumerate(mascaras) if m.sum() >= area_min], dtype=int)
    cajas, puntajes, mascaras = cajas[keep], puntajes[keep], mascaras[keep]
    keep = filtrar_instancias(puntajes, mascaras, umbral_det)
    return cajas[keep], puntajes[keep], mascaras[keep]


def recortar(arr, caja, mascara, margen=MARGEN):
    """Recorta una manzana y pone fondo blanco fuera de su máscara."""
    x1, y1, x2, y2 = caja
    mx, my = margen * (x2 - x1), margen * (y2 - y1)
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(arr.shape[1], int(x2 + mx)), min(arr.shape[0], int(y2 + my))
    sub = arr[y1:y2, x1:x2].copy()
    sub[~mascara[y1:y2, x1:x2]] = 255
    return Image.fromarray(sub)


@torch.no_grad()
def prob_deteriorada(recortes, lote=32):
    """Probabilidad de 'deteriorada' para cada recorte (el mismo clasificador del baseline)."""
    if not recortes:
        return []
    idx = clases_clf.index(CLASE_MALA)
    probs = []
    for i in range(0, len(recortes), lote):
        x = torch.stack([tf_eval()(r) for r in recortes[i:i + lote]]).to(dev)
        probs += torch.softmax(clasificador(x), dim=1)[:, idx].cpu().tolist()
    return probs

def analizar(img, umbral_mala=None, umbral_det=None):
    """Segmenta, clasifica cada manzana y arma el resumen de la imagen."""
    umbral_mala = UMBRAL_MALA if umbral_mala is None else umbral_mala
    t0 = time.time()
    img = img.convert("RGB")
    factor = 1.0
    if max(img.size) > MAX_LADO:                      # fotos de celular muy grandes
        factor = MAX_LADO / max(img.size)
        img = img.resize((round(img.width * factor), round(img.height * factor)))

    cajas, puntajes, mascaras = segmentar(img, umbral_det)
    arr = np.asarray(img)
    recortes = [recortar(arr, c, m) for c, m in zip(cajas, mascaras)]
    probs = prob_deteriorada(recortes)

    instancias = []
    for i, (c, s, m, p) in enumerate(zip(cajas, puntajes, mascaras, probs), 1):
        mala = p >= umbral_mala
        instancias.append({
            "id": i,
            "caja": [float(v) for v in c],
            "puntaje_det": float(s),
            "prob_deteriorada": float(p),
            "estado": "deteriorada" if mala else "sana",
            "confianza": float(p if mala else 1 - p),
            "area": int(m.sum()),
            "mascara": m,
        })

    n_malas = sum(d["estado"] == "deteriorada" for d in instancias)
    avisos = []
    if not instancias:
        avisos.append("No se detectó ninguna manzana. Prueba bajar el umbral de detección.")
    dudosas = [d["id"] for d in instancias if d["confianza"] < CONF_BAJA]
    if dudosas:
        avisos.append(f"Confianza baja en las manzanas {dudosas}: revisar a mano.")
    return {"imagen": img, "factor": factor, "instancias": instancias,
            "total": len(instancias), "sanas": len(instancias) - n_malas,
            "deterioradas": n_malas, "avisos": avisos, "segundos": time.time() - t0}


def resumen_texto(res):
    lineas = [f"Manzanas: {res['total']} | sanas: {res['sanas']} | deterioradas: {res['deterioradas']} "
              f"| {res['segundos']:.1f} s"]
    for d in res["instancias"]:
        lineas.append(f"  #{d['id']}: {d['estado']} (confianza {d['confianza']:.0%}, "
                      f"detección {d['puntaje_det']:.0%})")
    lineas += [f"Aviso: {a}" for a in res["avisos"]]
    return "\n".join(lineas)

def _borde(m, grosor=3):
    """Contorno de una máscara booleana, sin necesidad de OpenCV."""
    interior = m.copy()
    for _ in range(grosor):
        p = np.pad(interior, 1, constant_values=False)
        interior = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~interior


def _fuente(tam):
    for ruta in ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(ruta, tam)
        except OSError:
            pass
    return ImageFont.load_default()


def dibujar(res, alpha=0.40):
    """Pinta cada manzana (verde = sana, rojo = deteriorada) con contorno y etiqueta."""
    arr = np.asarray(res["imagen"]).astype(np.float32)
    for d in res["instancias"]:
        color = COLOR[d["estado"]]
        m = d["mascara"]
        arr[m] = (1 - alpha) * arr[m] + alpha * np.array(color, dtype=np.float32)
        arr[_borde(m)] = color
    out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(out)
    fuente = _fuente(max(14, out.width // 45))
    for d in res["instancias"]:
        color = COLOR[d["estado"]]
        texto = f"#{d['id']} {d['estado']} {d['confianza']:.0%}"
        x1, y1 = d["caja"][:2]
        bb = dr.textbbox((0, 0), texto, font=fuente)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        px, py = x1, max(0, y1 - h - 10)
        dr.rectangle([px, py, px + w + 10, py + h + 10], fill=color)
        dr.text((px + 5, py + 3), texto, fill="white", font=fuente)
    return out

# ---- Interfaz web ----
import gradio as gr


def interfaz(img, umbral_mala, umbral_det):
    if img is None:
        return None, "Sube una foto."
    res = analizar(img, umbral_mala, umbral_det)
    return dibujar(res), resumen_texto(res)


demo = gr.Interface(
    fn=interfaz,
    inputs=[gr.Image(type="pil", label="Foto con varias manzanas"),
            gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Umbral para 'deteriorada'"),
            gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Umbral de detección")],
    outputs=[gr.Image(label="Resultado"), gr.Textbox(label="Resumen", lines=8)],
    title="Control de calidad de manzanas",
    description="Segmenta cada manzana y la clasifica como sana o deteriorada. Inspección solo visual: "
                "revisa a mano los resultados con confianza baja.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
