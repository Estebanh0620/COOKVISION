import cv2
import numpy as np


def detectar_contornos(imagen, umbral):

    contornos, _ = cv2.findContours(
        umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    imagen_contornos = imagen.copy()
    contador = 0

    # convertir a HSV una sola vez
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    for c in contornos:

        area = cv2.contourArea(c)
        if area < 500:
            continue

        # ---------------- geometría ----------------
        perimetro = cv2.arcLength(c, True)

        circularidad = 0
        if perimetro != 0:
            circularidad = (4 * np.pi * area) / (perimetro ** 2)

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h

        # ---------------- máscara del objeto ----------------
        mask = np.zeros(umbral.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        # ---------------- media RGB ----------------
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        pixels_rgb = rgb[mask > 0]
        media_RGB = np.mean(pixels_rgb, axis=0)

        # ---------------- media HSV ----------------
        pixels_hsv = hsv[mask > 0]
        media_HSV = np.mean(pixels_hsv, axis=0)

        # =====================================================
        # ----------- PROCESAMIENTO PARA CLASIFICACIÓN ---------
        # =====================================================

        objeto_hsv = hsv[mask > 0]

        H = objeto_hsv[:, 0]
        S = objeto_hsv[:, 1]
        V = objeto_hsv[:, 2]

        # eliminar brillos/blancos
        mask_color = (S > 40) & (V < 230)

        H = H[mask_color]
        S = S[mask_color]
        V = V[mask_color]

        # evitar error si no hay píxeles válidos
        if len(H) == 0:
            continue

        # ---------------- métricas ----------------
        H_prom = np.mean(H)
        S_prom = np.mean(S)

        std_H = np.std(H)
        std_S = np.std(S)

        # ---- debug (puedes comentarlo luego) ----
        print(f"H limpio: {H_prom:.2f}")
        print(f"S limpio: {S_prom:.2f}")
        print(f"Std H: {std_H:.2f}")
        print(f"Std S: {std_S:.2f}")

# ---------------- CLASIFICACIÓN FINAL ----------------

       # prioridad: detectar tomate primero
    if S_prom > 90 and std_S < 25:
        tipo = "Tomate"
        color_texto = (0, 0, 255)

# cebolla: más variación o menor saturación
    elif std_S > 20 or S_prom < 80:
        tipo = "Cebolla"
        color_texto = (255, 0, 255)

    else:
        tipo = "Desconocido"
        color_texto = (0, 255, 255)

        # ---------------- imprimir ----------------
        print("\nRESUMEN DE CARACTERIZACIÓN")
        print(f"Area: {area:.0f} px")
        print(f"Perimetro: {perimetro:.1f}")
        print(f"Circularidad: {circularidad:.4f}")
        print(f"Aspect Ratio: {aspect_ratio:.3f}")
        print(f"Media RGB: R={media_RGB[0]:.1f} G={media_RGB[1]:.1f} B={media_RGB[2]:.1f}")
        print(f"Media HSV: H={media_HSV[0]:.1f} S={media_HSV[1]:.1f} V={media_HSV[2]:.1f}")
        print(f"Clasificación: {tipo}")

        # ---------------- dibujar ----------------
        cv2.drawContours(imagen_contornos, [c], -1, (0, 255, 0), 2)
        cv2.rectangle(imagen_contornos, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(
            imagen_contornos,
            tipo,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_texto,
            2,
        )

        contador += 1

    return imagen_contornos, contador