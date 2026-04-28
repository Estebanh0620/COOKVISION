import cv2
import numpy as np


def detectar_contornos(imagen, umbral):
    """
    Detecta, caracteriza y clasifica los contornos del umbral sobre la imagen BGR.
    """
    print(">>> VERSION CORRECTA CARGADA <<<")

    contornos, _ = cv2.findContours(
        umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    imagen_contornos = imagen.copy()
    contador = 0

    # Convertir espacios de color una sola vez, fuera del loop
    hsv      = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    rgb      = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    gris_obj = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    for c in contornos:

        print("─" * 50)

        area = cv2.contourArea(c)
        if area < 500:
            continue

        # ── Geometría ──────────────────────────────────────────────────────
        perimetro = cv2.arcLength(c, True)

        circularidad = 0.0
        if perimetro > 0:
            circularidad = (4 * np.pi * area) / (perimetro ** 2)

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h

        # ── Máscara del objeto ─────────────────────────────────────────────
        mask = np.zeros(umbral.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        # ── Media RGB ──────────────────────────────────────────────────────
        pixels_rgb = rgb[mask > 0]
        media_RGB  = np.mean(pixels_rgb, axis=0)

        # ── Media HSV ──────────────────────────────────────────────────────
        pixels_hsv = hsv[mask > 0]
        media_HSV  = np.mean(pixels_hsv, axis=0)

        # ── Textura (varianza del Laplacian sobre la máscara) ──────────────
        laplacian   = cv2.Laplacian(gris_obj, cv2.CV_64F)
        textura     = laplacian[mask > 0]
        var_textura = np.var(textura)

        # ── Extracción HSV filtrada para clasificación ─────────────────────
        objeto_hsv = hsv[mask > 0]
        H_raw = objeto_hsv[:, 0]
        S_raw = objeto_hsv[:, 1]
        V_raw = objeto_hsv[:, 2]

        # Filtrar reflejos, píxeles muy oscuros y poco saturados
        mascara_color = (S_raw > 40) & (V_raw < 230) & (V_raw > 30)
        H = H_raw[mascara_color]
        S = S_raw[mascara_color]

        if len(H) == 0:
            print(f"[AVISO] Contorno sin píxeles válidos (área {area:.0f}), omitido.")
            continue

        # ── Métricas de clasificación ──────────────────────────────────────
        S_prom = np.mean(S)
        std_S  = np.std(S)
        H_prom = np.mean(H)
        H_med  = np.median(H)
        std_H  = np.std(H)

        print(f"  H_prom={H_prom:.1f}  H_med={H_med:.1f}  S_prom={S_prom:.1f}  "
              f"std_H={std_H:.1f}  std_S={std_S:.1f}  var_textura={var_textura:.1f}")

        # ── Clasificación ──────────────────────────────────────────────────
        #
        # ES_TOMATE_H:
        #   H en rango rojo (< 20 o > 160) con variación moderada (std_H < 55)
        #
        # ES_TOMATE_TEXTURA:
        #   Se cumple si var_textura < 350  (superficie lisa)
        #   O si std_H < 15                (tono muy concentrado = tomate uniforme)
        #   → basta con que UNO se cumpla
        #   → la cebolla rojiza falla AMBOS: var_textura alta Y std_H alto

        ES_TOMATE_H       = ((H_med < 20) or (H_med > 160)) and (std_H < 55)
        ES_TOMATE_S       = S_prom > 50
        ES_TOMATE_STD     = std_S < 60
        ES_TOMATE_TEXTURA = (var_textura < 350) or (std_H < 15)

        print(f"  ES_TOMATE_H={ES_TOMATE_H}  ES_TOMATE_S={ES_TOMATE_S}  "
              f"ES_TOMATE_STD={ES_TOMATE_STD}  ES_TOMATE_TEXTURA={ES_TOMATE_TEXTURA}")

        if ES_TOMATE_H and ES_TOMATE_S and ES_TOMATE_STD and ES_TOMATE_TEXTURA:
            tipo = "Tomate"
            color_texto = (0, 0, 255)       # rojo en BGR

        elif not ES_TOMATE_H:
            tipo = "Cebolla"
            color_texto = (255, 0, 255)     # magenta en BGR

        elif not ES_TOMATE_TEXTURA:
            # Tono rojizo pero textura rugosa y tono disperso → cebolla rojiza
            tipo = "Cebolla"
            color_texto = (255, 0, 255)     # magenta en BGR

        elif S_prom < 50 or std_S > 45:
            # Saturación baja o muy variable → cebolla blanca/amarilla
            tipo = "Cebolla"
            color_texto = (255, 0, 255)     # magenta en BGR

        else:
            tipo = "Desconocido"
            color_texto = (0, 255, 255)     # amarillo en BGR

        # ── Resumen en consola ─────────────────────────────────────────────
        print("\n  RESUMEN DE CARACTERIZACIÓN")
        print(f"  Área:          {area:.0f} px²")
        print(f"  Perímetro:     {perimetro:.1f}")
        print(f"  Circularidad:  {circularidad:.4f}")
        print(f"  Aspect ratio:  {aspect_ratio:.3f}")
        print(f"  Media RGB:     R={media_RGB[0]:.1f}  "
              f"G={media_RGB[1]:.1f}  B={media_RGB[2]:.1f}")
        print(f"  Media HSV:     H={media_HSV[0]:.1f}  "
              f"S={media_HSV[1]:.1f}  V={media_HSV[2]:.1f}")
        print(f"  var_textura:   {var_textura:.1f}")
        print(f"  Clasificación: {tipo}")
        print()

        # ── Dibujar sobre la imagen ────────────────────────────────────────
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