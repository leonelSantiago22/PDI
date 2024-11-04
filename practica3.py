#sudo apt install python3-opencv
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Parámetros globales utilizados para las transformaciones de imágenes.
gamma_imagenes = 1.5  # Factor gamma para la transformación gamma.
c_transformacion_logaritmica = 1  # Constante para la transformación logarítmica.
rebanada_intensidad_nivel = 100  # Nivel de intensidad para la rebanada.
rebanada_intensidad_ancho = 50  # Ancho de la rebanada de intensidad.

# Funciones de transformación
def negativo(img):
     
    img = img.astype(np.float32) # Convertir la imagen a flotante
    negativo = 255 - img # Aplicar la transformación negativa L-1-r  255 - img 
    negativo = np.uint8(negativo) # Convertir la imagen a entero
    
    return negativo

# Función para obtener los planos de bits de una imagen en escala de grises.
# Descompone cada píxel de la imagen en sus 8 bits (de menor a mayor significancia).
def rebanada_plano_bit(img):
    planos_bits = []  # Lista para almacenar cada plano de bits.
    for bit in range(8):
        # Extrae el bit correspondiente usando una operación AND bit a bit.
        plano_bit = np.bitwise_and(img, 1 << bit) >> bit
        # Almacena el plano de bits escalado a 255 para visualizar.
        planos_bits.append(plano_bit * 255)
    return planos_bits

# Función para aplicar transformación gamma a una imagen.
# Normaliza la imagen dividiendo por 255, aplica la transformación gamma y escala de nuevo a 0-255.
def transformacion_gamma(img, gamma):
    inversa_de_gamma = 1.0 / gamma  # Calcula el inverso de gamma.
    # Aplica la fórmula de transformación gamma: s = r^γ, donde r es el valor normalizado.
    return np.array(255 * (img / 255) ** inversa_de_gamma, dtype='uint8')
    # Divide cada elemento de img por 255 para normalizarlo a un rango entre 0 
    # y 1, luego eleva cada elemento a la potencia de invergaDeGamma 

# Función para aplicar la transformación logarítmica a una imagen.
# Es útil para expandir los valores oscuros de una imagen.
def transformacion_logaritmica(img, c):
    # Aplica la transformación logarítmica: s = c * log(1 + r), donde r es el valor del píxel.
    s = c * np.log(1 + img)
    # Clipa los valores entre 0 y 255 para asegurar que están dentro del rango de valores de la imagen.
    s = np.clip(s, 0, 255)
    return np.array(s, dtype=np.uint8)


# El estiramiento de contraste es un proceso que expande el rango de niveles de intensidad en una imagen, tal que este abaraca el rango de intensidad completo.
# Función para aplicar estiramiento de contraste.
# Mejora el contraste de una imagen ajustando sus valores mínimos y máximos.
def estiramiento_contraste(img):
    a, b = np.min(img), np.max(img)  # Encuentra los valores mínimo (a) y máximo (b) de la imagen.
    # Aplica la fórmula del estiramiento de contraste: s = (r - a) * (255 / (b - a))
    return 255 * (img - a) / (b - a)

# Produce una imagen binaria, ilumina(u oscurese) el rango de intensidades deseado pero mantiene los demas 
#niveles de intensidad sin cambios.
def rebanada_nivel_intensidad(img, nivel, ancho, valor_resaltado=255, preservar_fuera=True):
    # Crear una copia de la imagen para trabajar
    img_nueva = np.zeros_like(img)

    # Crear la máscara para los valores dentro del rango [nivel, nivel + ancho]
    mascara = (img >= nivel) & (img <= (nivel + ancho))

    if preservar_fuera:
        # Preservar los valores fuera del rango y resaltar los valores dentro del rango
        img_nueva = np.where(mascara, valor_resaltado, img)
    else:
        # Resaltar los valores dentro del rango y los valores fuera del rango se establecen a 0
        img_nueva[mascara] = valor_resaltado

    return img_nueva



# Función para cargar y procesar una imagen
def procesar_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen,  cv2.IMREAD_GRAYSCALE)  # Cargar imagen en escala de grises

    img_negativo = negativo(img)
    img_gamma = transformacion_gamma(img, gamma_imagenes)
    img_log = transformacion_logaritmica(img, c_transformacion_logaritmica)
    img_contraste = estiramiento_contraste(img).astype(np.uint8)
    img_rebanada_intensidad = rebanada_nivel_intensidad(img, rebanada_intensidad_nivel, rebanada_intensidad_ancho)
    img_rebanadas_bit = rebanada_plano_bit(img)

    return img, img_negativo, img_gamma, img_log, img_contraste, img_rebanada_intensidad, img_rebanadas_bit

# Función para graficar los histogramas
def graficar_histogramas(img, img_contraste, titulo):
    # Calcular histogramas
    hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_contraste = cv2.calcHist([img_contraste], [0], None, [256], [0, 256])

    # Graficar histogramas
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_original, color='blue')
    plt.title(f'Histograma Original - {titulo}')
    plt.xlim([0, 256])

    plt.subplot(1, 2, 2)
    plt.plot(hist_contraste, color='red')
    plt.title(f'Histograma Estiramiento de Contraste - {titulo}')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

# Función para graficar las imágenes transformadas con títulos personalizados
def graficar_transformaciones(img, img_negativo, img_gamma, img_log, img_contraste, img_rebanada_intensidad, titulo):
    fig, axs1 = plt.subplots(2, 3, figsize=(12, 8))

    axs1[0, 0].imshow(img, cmap='gray')
    axs1[0, 0].set_title(f'Original - {titulo}')

    axs1[0, 1].imshow(img_negativo, cmap='gray')
    axs1[0, 1].set_title('Negativo')

    axs1[0, 2].imshow(img_gamma, cmap='gray')
    axs1[0, 2].set_title('Transformación Gamma')

    axs1[1, 0].imshow(img_log, cmap='gray')
    axs1[1, 0].set_title('Transformación Logarítmica')

    axs1[1, 1].imshow(img_contraste, cmap='gray')
    axs1[1, 1].set_title('Estiramiento de Contraste')

    axs1[1, 2].imshow(img_rebanada_intensidad, cmap='gray')
    axs1[1, 2].set_title('Rebanada Nivel Intensidad')

    for ax in axs1.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Función para graficar los planos de bits
def graficar_rebanadas_bits(img_rebanadas_bit, titulo):
    fig, axs2 = plt.subplots(2, 4, figsize=(12, 6))

    for i in range(8):
        axs2[i // 4, i % 4].imshow(img_rebanadas_bit[i], cmap='gray')
        axs2[i // 4, i % 4].set_title(f'{titulo} - Plano de bit {i}')

    for ax in axs2.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Función principal para manejar múltiples imágenes con títulos personalizados
def procesar_y_graficar_imagenes(lista_imagenes, lista_titulos):
    for ruta_imagen, titulo in zip(lista_imagenes, lista_titulos):
        img, img_negativo, img_gamma, img_log, img_contraste, img_rebanada_intensidad, img_rebanadas_bit = procesar_imagen(ruta_imagen)
        print(f"Procesando imagen: {ruta_imagen} con título: {titulo}")
        graficar_transformaciones(img, img_negativo, img_gamma, img_log, img_contraste, img_rebanada_intensidad, titulo)
        graficar_histogramas(img, img_contraste, titulo)
        graficar_rebanadas_bits(img_rebanadas_bit, titulo)
        

# Lista de imágenes y títulos para procesar
lista_imagenes = ["./bajo_contraste.jpeg", "./altoContraste.jpg", "./poca_iluminacion.webp"]
lista_titulos = ["Bajo Contraste", "Alto Contraste", "Poca Iluminación"]

# Ejecutar el procesamiento
procesar_y_graficar_imagenes(lista_imagenes, lista_titulos)
