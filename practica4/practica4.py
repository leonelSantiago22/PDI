'''
Utilizando la herramienta de PDI, realice lo siguiente

* Descargue cuatro imagenes: alto contraste y bajo contraste, y alta y baja iluminacion
* Implemente las tecnicas basadas en el procesamiento del histograma: ecualizacion de histograma global 
    y local, calculo de media y variaza y local.
* Pruebe las tecnicas basas en el procesamiento de histograma sobre las imagenes descargadas en el punto 1. De acuerdo a la media y variaza 
    (global y local) debera recomendar y aplicar un procesamiento local o global de cada imagen.

Realice un informde de la practica el 14 de noviembre
'''
import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np

def cargar_imagen(ruta):
    # Cargar imagen en escala de grises
    return cv.imread(ruta, cv.IMREAD_GRAYSCALE)

def mostrar_imagen(titulo, img, descripcion_imagen):
    plt.figure()
    plt.title(f"{titulo} - {descripcion_imagen}")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def mostrar_todas_operaciones(imagen, descripcion_imagen):
    # Generar los resultados de todas las operaciones
    eq_global = ecualizacion_histograma_global(imagen)
    eq_local = ecualizacion_histograma_local(imagen)
    media_global, varianza_global = calcular_media_varianza(imagen)
    media_local, varianza_local = calcular_media_varianza_local(imagen)
    
    # Crear una figura de 2x3
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Resultados de todas las operaciones - {descripcion_imagen}")

    # Mostrar la imagen original
    axs[0, 0].imshow(imagen, cmap='gray')
    axs[0, 0].set_title("Imagen Original")
    axs[0, 0].axis('off')

    # Mostrar cada resultado en un subplot
    axs[0, 1].imshow(eq_global, cmap='gray')
    axs[0, 1].set_title("Ecualización de Histograma Global")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(eq_local, cmap='gray')
    axs[0, 2].set_title("Ecualización de Histograma Local")
    axs[0, 2].axis('off')

    axs[1, 0].imshow(media_local, cmap='gray')
    axs[1, 0].set_title(f"Media Local (Media global: {media_global:.2f})")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(varianza_local, cmap='gray')
    axs[1, 1].set_title(f"Varianza Local (Varianza global: {varianza_global:.2f})")
    axs[1, 1].axis('off')

    # Espacio vacío para la figura de 2x3
    axs[1, 2].axis('off')
    
    plt.show()

def ecualizacion_histograma_global(img):
    # Ecualización de histograma global
    return cv.equalizeHist(img)

def ecualizacion_histograma_local(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Ecualización de histograma local (CLAHE)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def calcular_media_varianza(img):
    # Calcular la media y varianza global de la imagen
    media = np.mean(img)
    varianza = np.var(img)
    return media, varianza

def calcular_media_varianza_local(img, kernel_size=5):
    # Calcular media y varianza local usando un filtro de media y desviación estándar
    media_local = cv.blur(img, (kernel_size, kernel_size))
    varianza_local = cv.blur((img - media_local) ** 2, (kernel_size, kernel_size))
    return media_local, varianza_local

def menu_opciones_operaciones():
    print("Opciones:")
    print("1. Mostrar imagen original")
    print("2. Ecualización de histograma global")
    print("3. Ecualización de histograma local")
    print("4. Calcular media y varianza global")
    print("5. Calcular media y varianza local")
    print("6. Aplicar todas las operaciones a la vez")
    print("0. Volver al menú de selección de imágenes")
    opcion = int(input("Elige una opción: "))
    return opcion

def menu_imagen():
    print("Selecciona la imagen con la que deseas trabajar:")
    print("1. Imagen de alto contraste")
    print("2. Imagen de bajo contraste")
    print("3. Imagen de alta iluminación")
    print("4. Imagen de baja iluminación")
    print("0. Salir")
    opcion = int(input("Elige una opción: "))
    
    if opcion == 1:
        return cargar_imagen("alto_contraste_img.jpg"), "Imagen de alto contraste"
    elif opcion == 2:
        return cargar_imagen("bajo_contraste.jpeg"), "Imagen de bajo contraste"
    elif opcion == 3:
        return cargar_imagen("alta_iluminacion.jpg"), "Imagen de alta iluminación"
    elif opcion == 4:
        return cargar_imagen("baja_iluminacion.jpg"), "Imagen de baja iluminación"
    elif opcion == 0:
        return None, None
    else:
        print("Opción no válida. Intente de nuevo.")
        return None, None

def main():
    while True:
        imagen, descripcion_imagen = menu_imagen()
        if imagen is None:
            print("Saliendo del programa...")
            break

        while True:
            opcion = menu_opciones_operaciones()
            
            if opcion == 1:
                mostrar_imagen("Imagen Original", imagen, descripcion_imagen)
            
            elif opcion == 2:
                resultado = ecualizacion_histograma_global(imagen)
                mostrar_imagen("Ecualización de Histograma Global", resultado, descripcion_imagen)
            
            elif opcion == 3:
                resultado = ecualizacion_histograma_local(imagen)
                mostrar_imagen("Ecualización de Histograma Local", resultado, descripcion_imagen)
            
            elif opcion == 4:
                media, varianza = calcular_media_varianza(imagen)
                print(f"Media global: {media}, Varianza global: {varianza}")
            
            elif opcion == 5:
                media_local, varianza_local = calcular_media_varianza_local(imagen)
                mostrar_imagen("Media Local", media_local, descripcion_imagen)
                mostrar_imagen("Varianza Local", varianza_local, descripcion_imagen)
            
            elif opcion == 6:
                mostrar_todas_operaciones(imagen, descripcion_imagen)
            
            elif opcion == 0:
                print("Volviendo al menú de selección de imágenes...")
                break
            
            else:
                print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()
