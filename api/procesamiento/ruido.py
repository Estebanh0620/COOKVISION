import cv2

def quitar_ruido(imagen):
    return cv2.GaussianBlur(imagen, (5,5), 0)