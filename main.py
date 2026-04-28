import cv2
import os

from procesamiento.ruido import quitar_ruido
from procesamiento.segmentacion import segmentar
from procesamiento.contornos import detectar_contornos

# ── Ruta de la imagen ───────────────────────────────────────
ruta_imagen = "dataset/Onion/Oni_040.jpg"
#ruta_imagen = "dataset/Tomates/Tre_040.jpg"

# ── Cargar imagen ───────────────────────────────────────────
imagen = cv2.imread(ruta_imagen)
if imagen is None:
    print("Error: no se pudo cargar la imagen:", ruta_imagen)
    exit()

# ── Preprocesamiento ────────────────────────────────────────
imagen_sin_ruido = quitar_ruido(imagen)

# ── Segmentación ────────────────────────────────────────────
# segmentar() recibe BGR directamente; la conversión a gris
# ocurre internamente para que watershed use la imagen a color.
umbral = segmentar(imagen_sin_ruido)

# Guardar versión en grises solo para visualización
gris = cv2.cvtColor(imagen_sin_ruido, cv2.COLOR_BGR2GRAY)

# ── Contornos + caracterización ─────────────────────────────
imagen_contornos, contador = detectar_contornos(imagen, umbral)

print("Objetos detectados:", contador)

# ── Texto sobre la imagen resultado ─────────────────────────
cv2.putText(
    imagen_contornos,
    f"Objetos: {contador}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)

# ── Guardar resultados ───────────────────────────────────────
os.makedirs("resultados", exist_ok=True)
cv2.imwrite("resultados/sin_ruido.jpg",  imagen_sin_ruido)
cv2.imwrite("resultados/gris.jpg",       gris)
cv2.imwrite("resultados/segmentada.jpg", umbral)
cv2.imwrite("resultados/contornos.jpg",  imagen_contornos)

# ── Mostrar ──────────────────────────────────────────────────
cv2.imshow("Original",    imagen)
cv2.imshow("Gris",        gris)
cv2.imshow("Segmentada",  umbral)
cv2.imshow("Contornos",   imagen_contornos)
cv2.waitKey(0)
cv2.destroyAllWindows()