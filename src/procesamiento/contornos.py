import cv2

def detectar_contornos(imagen, umbral):

    contornos, _ = cv2.findContours(
        umbral,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    imagen_contornos = imagen.copy()
    contador = 0

    for c in contornos:
        area = cv2.contourArea(c)

        if 1000 < area < 20000:

            x, y, w, h = cv2.boundingRect(c)

            cv2.drawContours(imagen_contornos, [c], -1, (0,255,0), 2)
            cv2.rectangle(imagen_contornos, (x,y), (x+w,y+h), (255,0,0), 2)

            contador += 1

    return imagen_contornos, contador