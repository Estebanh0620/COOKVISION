def segmentar_color(imagen):

    import cv2
    import numpy as np

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # rango MÁS AMPLIO (más tolerante)
    lower = np.array([0, 30, 30])
    upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask