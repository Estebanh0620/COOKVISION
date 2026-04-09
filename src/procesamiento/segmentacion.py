import cv2
import numpy as np


def segmentar(imagen_bgr):
    """
    Recibe la imagen BGR original (sin ruido).
    Devuelve la máscara binaria segmentada.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Umbralización automática con Otsu
    _, umbral = cv2.threshold(
        gris, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. Apertura: eliminar ruido pequeño
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Separar objetos pegados (erosión) y reconstruir (dilatación)
    separada = cv2.erode(opening, kernel, iterations=2)
    separada = cv2.dilate(separada, kernel, iterations=2)

    # 4. Fondo seguro
    sure_bg = cv2.dilate(separada, kernel, iterations=3)

    # 5. Foreground seguro por transformada de distancia
    dist = cv2.distanceTransform(separada, cv2.DIST_L2, 5)
    _, thresh_dist = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    sure_fg = np.uint8(thresh_dist)

    # 6. Zona desconocida
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Etiquetado + watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # watershed necesita imagen BGR (no gris convertida)
    markers = cv2.watershed(imagen_bgr, markers)

    # 8. Máscara final
    mascara = np.zeros(gris.shape, dtype=np.uint8)
    mascara[markers > 1] = 255

    return mascara