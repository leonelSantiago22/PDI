
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
    """Leer una imagen en escala de grises y convertirla a binaria."""
    imagen = cv.imread(ruta_img, cv.IMREAD_GRAYSCALE)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en la ruta: {ruta_img}")
    
    # Convertir la imagen a binaria
    _, imagen_binaria = cv.threshold(imagen, 127, 255, cv.THRESH_BINARY)
    return imagen_binaria

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

def menu_seleccion_imagen(rutas):
    """Mostrar un menú para seleccionar una imagen."""
    print("Seleccione una imagen para procesar:")
    for i, ruta in enumerate(rutas, 1):
        print(f"{i}. {ruta}")
    opcion = int(input("Ingrese el número de la imagen: "))
    if 1 <= opcion <= len(rutas):
        return rutas[opcion - 1]
    else:
        print("Opción inválida. Intente de nuevo.")
        return menu_seleccion_imagen(rutas)

def menu_operaciones():
    """Mostrar un menú para seleccionar una operación a realizar."""
    print("Seleccione una operación para aplicar:")
    print("1. Dilatación")
    print("2. Erosión")
    print("3. Apertura")
    print("4. Cierre")
    print("5. Extracción de límites")
    print("6. Relleno de hoyos")
    print("7. Extracción de componentes conectados")
    print("8. Salir")
    opcion = int(input("Ingrese el número de la operación: "))
    if 1 <= opcion <= 8:
        return opcion
    else:
        print("Opción inválida. Intente de nuevo.")
        return menu_operaciones()

def main():
    # Ruta de imágenes (actualiza estas rutas según tus archivos locales)
    rutas = ["deporte.webp", "edificio.webp", "gente.webp"]

    # Definición de elementos estructurantes
    kernel_cuadrado = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    kernel_cruz = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    kernel_elipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    kernels = {
        "Cuadrado": kernel_cuadrado,
        "Cruz": kernel_cruz,
        "Elipse": kernel_elipse,
    }

    ruta_seleccionada = menu_seleccion_imagen(rutas)

    try:
        imagen = leer_imagenes(ruta_seleccionada)
        mostrar_imagen(f"Imagen original: {ruta_seleccionada}", imagen)

        while True:
            opcion = menu_operaciones()

            if opcion == 8:
                print("Saliendo del programa.")
                break

            if opcion in [1, 2, 3, 4]:
                for nombre, kernel in kernels.items():
                    if opcion == 1:
                        resultado = cv.dilate(imagen, kernel)
                        titulo = f"Dilatación ({nombre})"
                    elif opcion == 2:
                        resultado = cv.erode(imagen, kernel)
                        titulo = f"Erosión ({nombre})"
                    elif opcion == 3:
                        resultado = operacion_morfologica(imagen, cv.MORPH_OPEN, kernel)
                        titulo = f"Apertura ({nombre})"
                    elif opcion == 4:
                        resultado = operacion_morfologica(imagen, cv.MORPH_CLOSE, kernel)
                        titulo = f"Cierre ({nombre})"

                    mostrar_imagen(titulo, resultado)

            elif opcion == 5:
                limites = extraccion_limites(imagen, kernel_cuadrado)
                mostrar_imagen("Extracción de límites", limites)

            elif opcion == 6:
                imagen_hoyos = rellenar_hoyos(imagen)
                mostrar_imagen("Relleno de hoyos", imagen_hoyos)

            elif opcion == 7:
                num_labels, labels_im = extraer_componentes_conectados(imagen)
                mostrar_imagen(f"Componentes conectados (Etiquetas: {num_labels})", labels_im)

    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
