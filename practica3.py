#sudo apt install python3-opencv
import cv2
import numpy as np
import matplotlib.pyplot as plt


gamma_imagenes = 1.5
c_transformacion_logaritmica = 1
rebanada_intensidad_nivel = 100
rebanada_intensidad_ancho = 50

# Funciones de transformación
def negativo(img):
    return 255 - img

def rebanada_plano_bit(img):
    planos_bits = []
    for bit in range(8):
        plano_bit = np.bitwise_and(img, 1 << bit) >> bit
        planos_bits.append(plano_bit * 255)
    return planos_bits

def transformacion_gamma(img, gamma):
    inversa_de_gamma = 1.0 / gamma
    return np.array(255 * (img / 255) ** inversa_de_gamma, dtype='uint8')
    # Divide cada elemento de img por 255 para normalizarlo a un rango entre 0 
    # y 1, luego eleva cada elemento a la potencia de invergaDeGamma
def transformacion_logaritmica(img, c):
    # El factor c controla la velocidad de este crecimiento exponencial.
    s = c * np.log(1 + img) # Calculo de la transformacion
    s = np.clip(s, 0, 255)
    return np.array(s, dtype=np.uint8)

def estiramiento_contraste(img):
    a, b = np.min(img), np.max(img)
    return 255 * (img - a) / (b - a)

def rebanada_nivel_intensidad(img, nivel, ancho):
    img_nueva = np.zeros(img.shape, dtype=np.uint8)
    img_nueva[(img >= nivel) & (img <= nivel + ancho)] = 255
    return img_nueva



# Función para cargar y procesar una imagen
def procesar_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen, 0)  # Cargar imagen en escala de grises

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
