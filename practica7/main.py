
'''
    Utilizando una herramienta de PDI, realice lo siguiente:
        1. Descargue las imagenes binarias de escenas, deportiva, edificio, gente. Las imagenes 
        deberan contener al menos un itsmo o golfo o puestes delgados

        2. Implementar las operaciones morfologicas basicas utulizando diferentes 
        elementos estructurales: erosion, dilatacion, apertura, cierre. Tambien, implemente
        los algoritmos morfologicos: extraccion de limites, rellenando hoyos y extraccion de componentes conectados 

        3. Pruebe las operaciones y algoritmos morfologicos sobre las imagenes descargadas en el punto 1. 
        Debera recomendar un elemento estructurante, asi como una operaciones o algoritmo morfologico para cada imagen 

'''


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def leer_imagenes(ruta_img):
    """Leer una imagen en escala de grises."""
    imagen = cv.imread(ruta_img, cv.IMREAD_GRAYSCALE)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en la ruta: {ruta_img}")
    return imagen

def mostrar_imagen(titulo, imagen):
    """Mostrar una imagen utilizando matplotlib."""
    plt.figure(figsize=(6, 6))
    plt.title(titulo)
    plt.axis('off')
    plt.imshow(imagen, cmap='gray')
    plt.show()

def operacion_morfologica(imagen, operacion, elem_estruc):
    """Aplicar una operación morfológica."""
    return cv.morphologyEx(imagen, operacion, elem_estruc)

def extraccion_limites(imagen, kernel):
    """Extraer los límites de los objetos en la imagen."""
    erosion = cv.erode(imagen, kernel)
    limites = cv.subtract(imagen, erosion)
    return limites

def rellenar_hoyos(imagen):
    """Rellenar los hoyos dentro de los objetos binarios."""
    h, w = imagen.shape
    mascara = np.zeros((h + 2, w + 2), np.uint8)
    imagen_filled = imagen.copy()
    cv.floodFill(imagen_filled, mascara, (0, 0), 255)
    return cv.bitwise_or(imagen, cv.bitwise_not(imagen_filled))

def extraer_componentes_conectados(imagen):
    """Extraer componentes conectados de la imagen."""
    num_labels, labels_im = cv.connectedComponents(imagen)
    return num_labels, labels_im

def main():
    # Ruta de imágenes (actualiza estas rutas según tus archivos locales)
    rutas = ["escena.jpg", "deportiva.jpg", "edificio.jpg", "gente.jpg"]

    # Definición de elementos estructurantes
    kernel_cuadrado = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel_cruz = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    kernel_elipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    kernels = {
        "Cuadrado": kernel_cuadrado,
        "Cruz": kernel_cruz,
        "Elipse": kernel_elipse,
    }

    for ruta in rutas:
        try:
            imagen = leer_imagenes(ruta)
            mostrar_imagen(f"Imagen original: {ruta}", imagen)

            # Operaciones morfológicas
            for nombre, kernel in kernels.items():
                dilatada = cv.dilate(imagen, kernel)
                erosionada = cv.erode(imagen, kernel)
                apertura = operacion_morfologica(imagen, cv.MORPH_OPEN, kernel)
                cierre = operacion_morfologica(imagen, cv.MORPH_CLOSE, kernel)

                mostrar_imagen(f"Dilatación ({nombre})", dilatada)
                mostrar_imagen(f"Erosión ({nombre})", erosionada)
                mostrar_imagen(f"Apertura ({nombre})", apertura)
                mostrar_imagen(f"Cierre ({nombre})", cierre)

            # Extracción de límites
            limites = extraccion_limites(imagen, kernel_cuadrado)
            mostrar_imagen(f"Extracción de límites: {ruta}", limites)

            # Rellenar hoyos
            imagen_hoyos = rellenar_hoyos(imagen)
            mostrar_imagen(f"Relleno de hoyos: {ruta}", imagen_hoyos)

            # Componentes conectados
            num_labels, labels_im = extraer_componentes_conectados(imagen)
            mostrar_imagen(f"Componentes conectados: {ruta} (Etiquetas: {num_labels})", labels_im)

        except FileNotFoundError as e:
            print(e)

if __name__ == "__main__":
    main()
