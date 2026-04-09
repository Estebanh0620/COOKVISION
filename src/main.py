import cv2
import os

from procesamiento.ruido import quitar_ruido
from procesamiento.segmentacion import segmentar
from procesamiento.contornos import detectar_contornos

# Ruta de la imagen
ruta_imagen = "dataset/Onion/Oni_040.jpg" 
#ruta_imagen = "dataset/Tomates/Tre_045.jpg" 

# Cargar imagen
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("No se pudo cargar la imagen")
    exit()

#preprocesamiento
imagen_sin_ruido = quitar_ruido(imagen)

#convertir a escala de grises (FALTABA)
gris = cv2.cvtColor(imagen_sin_ruido, cv2.COLOR_BGR2GRAY)

#segmentación
umbral = segmentar(gris)

#contornos + caracterización
imagen_contornos, contador = detectar_contornos(imagen, umbral)

# Mostrar resultado en consola
print("Objetos detectados:", contador)

# Texto en imagen
cv2.putText(
    imagen_contornos,
    f"Objetos: {contador}",
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0,255,0),
    2
)

# Crear carpeta resultados
os.makedirs("resultados", exist_ok=True)

# Guardar resultados
cv2.imwrite("resultados/sin_ruido.jpg", imagen_sin_ruido)
cv2.imwrite("resultados/gris.jpg", gris)
cv2.imwrite("resultados/segmentada.jpg", umbral)
cv2.imwrite("resultados/contornos.jpg", imagen_contornos)

# Mostrar imágenes
cv2.imshow("Original", imagen)
cv2.imshow("Gris", gris)
cv2.imshow("Segmentada", umbral)
cv2.imshow("Contornos", imagen_contornos)

cv2.waitKey(0)
cv2.destroyAllWindows()