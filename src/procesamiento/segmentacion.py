import cv2
import numpy as np

def segmentar(gris):

    # 1. Threshold (binarización automática con Otsu)
    _, umbral = cv2.threshold(
        gris, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. Eliminar ruido (apertura: erosión + dilatación)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel, iterations=2)

    # Separar objetos pegados (erosión)
    separada = cv2.erode(opening, kernel, iterations=2)

    # Reconstruir tamaño (dilatación)
    separada = cv2.dilate(separada, kernel, iterations=2)

    # 3. Fondo seguro
    sure_bg = cv2.dilate(separada, kernel, iterations=3)

    # 4. Objetos seguros (foreground)
    dist = cv2.distanceTransform(separada, cv2.DIST_L2, 5)

    _, thresh_dist = cv2.threshold(
        dist, 0.3 * dist.max(), 255, 0
    )
    sure_fg = np.uint8(thresh_dist)

    # 5. Zona desconocida
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Etiquetado
    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed necesita imagen en color
    imagen_color = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(imagen_color, markers)

    # 7. Crear máscara final
    mascara = np.zeros(gris.shape, dtype=np.uint8)
    mascara[markers > 1] = 255

    return mascara