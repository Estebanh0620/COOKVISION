import cv2
import os

# Ruta de la imagen de prueba
ruta_imagen = "dataset/Tomates/Tre_001.jpg"

# Cargar imagen
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("No se pudo cargar la imagen")
    exit()

# Quitar ruido (filtro gaussiano)
imagen_sin_ruido = cv2.GaussianBlur(imagen, (5,5), 0)

#Convertir a escala de grises
gris = cv2.cvtColor(imagen_sin_ruido, cv2.COLOR_BGR2GRAY)

#Segmentación (umbral)
_, umbral = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY_INV)

#Detectar contornos
bordes = cv2.Canny(gris, 50, 150)

contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos
imagen_contornos = imagen.copy()

areas = []

for c in contornos:
    areas.append(cv2.contourArea(c))

area_max = max(areas)

for c in contornos:
    area = cv2.contourArea(c)

    if 300 < area < 5000:
        cv2.drawContours(imagen_contornos, [c], -1, (0,255,0), 2)


# Crear carpeta resultados si no existe
os.makedirs("resultados", exist_ok=True)

# Guardar resultados
cv2.imwrite("resultados/sin_ruido.jpg", imagen_sin_ruido)
cv2.imwrite("resultados/gris.jpg", gris)
cv2.imwrite("resultados/segmentada.jpg", umbral)
cv2.imwrite("resultados/contornos.jpg", imagen_contornos)

# Mostrar cantidad de objetos detectados
print("Objetos detectados:", len(contornos))

# Mostrar imágenes
cv2.imshow("Original", imagen)
cv2.imshow("Segmentada", umbral)
cv2.imshow("Contornos", imagen_contornos)

cv2.waitKey(0)
cv2.destroyAllWindows()