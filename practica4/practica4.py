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

#Mostrar las imagenes y su historial
def mostrar_imagen(titulo, img, descripcion_imagen):
    # Crear una figura con dos subplots: uno para la imagen y otro para el histograma
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{titulo} - {descripcion_imagen}")

    # Mostrar la imagen en el primer subplot
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Imagen")
    axs[0].axis('off')

    # Calcular y mostrar el histograma en el segundo subplot
    histograma = cv.calcHist([img], [0], None, [256], [0, 256])
    axs[1].plot(histograma, color='black')
    axs[1].set_title("Histograma")
    axs[1].set_xlim([0, 256])

    plt.show()

def mostrar_todas_operaciones(imagen, descripcion_imagen):
    # Generar los resultados de todas las operaciones
    eq_global = ecualizacion_histograma_global(imagen)
    eq_local = ecualizacion_histograma_local(imagen)
    media_global_img, varianza_global_img = calcular_media_varianza_global(imagen)
    media_local, varianza_local = calcular_media_varianza_local(imagen)
    
    # Crear una figura de 3x3 para mostrar todas las imágenes, incluyendo media y varianza globales
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
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

    # Mostrar la media local
    axs[1, 0].imshow(media_local, cmap='gray')
    axs[1, 0].set_title("Media Local")
    axs[1, 0].axis('off')

    # Mostrar la varianza local
    axs[1, 1].imshow(varianza_local, cmap='gray')
    axs[1, 1].set_title(f"Varianza Local (Varianza global: {np.var(imagen):.2f})")
    axs[1, 1].axis('off')

    # Mostrar la media global como imagen constante
    axs[1, 2].imshow(media_global_img, cmap='gray')
    axs[1, 2].set_title(f"Media Global: {np.mean(imagen):.2f}")
    axs[1, 2].axis('off')

    # Mostrar la varianza global como imagen constante
    axs[2, 0].imshow(varianza_global_img, cmap='gray')
    axs[2, 0].set_title(f"Varianza Global: {np.var(imagen):.2f}")
    axs[2, 0].axis('off')

    # Espacios vacíos
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')
    
    plt.show()


def ecualizacion_histograma_global(img):
    # Ecualización de histograma global
    return cv.equalizeHist(img)

def ecualizacion_histograma_local(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Ecualización de histograma local (CLAHE)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def calcular_media_varianza_global(img):
    # Calcular la media y varianza global de la imagen
    media = np.mean(img)
    varianza = np.var(img)
    
    # Crear imágenes constantes con los valores de media y varianza para visualización
    imagen_media = np.full_like(img, media, dtype=np.float32)
    imagen_varianza = np.full_like(img, varianza, dtype=np.float32)
    
    return imagen_media, imagen_varianza


def calcular_media_varianza_local(img, kernel_size=5):
    # Calcular media y varianza local usando un filtro de media y desviación estándard
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    img_blurred = cv.filter2D(img, -1, kernel)
    
    img_diff = img - img_blurred
    variances = cv.filter2D(img_diff**2, -1, kernel)
    
    return img_blurred, variances

def menu_opciones_operaciones():
    print("Opciones:")
    print("1. Mostrar imagen original")
    print("2. Ecualización de histograma global")
    print("3. Ecualización de histograma local")
    print("4. Calcular media y varianza global")
    print("5. Calcular media y varianza local")
    print("6. Aplicar todas las operaciones a la vez")
    print("0. Volver al menú de selección de imágenes")
    opciones = input("Elige una o varias opciones separadas por comas (ej. 1,3,5): ")
    opciones = [int(op) for op in opciones.split(",") if op.isdigit()]
    return opciones

# Menu para escojer las imagenes
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
            opciones = menu_opciones_operaciones()
            
            # Verificar si la opción 0 está seleccionada
            if 0 in opciones:
                print("Volviendo al menú de selección de imágenes...")
                break
            
            for opcion in opciones:
                if opcion == 1:
                    mostrar_imagen("Imagen Original", imagen, descripcion_imagen)
                
                elif opcion == 2:
                    resultado = ecualizacion_histograma_global(imagen)
                    mostrar_imagen("Ecualización de Histograma Global", resultado, descripcion_imagen)
                
                elif opcion == 3:
                    resultado = ecualizacion_histograma_local(imagen)
                    mostrar_imagen("Ecualización de Histograma Local", resultado, descripcion_imagen)
                
                elif opcion == 4:
                    media_global, varianza_global = calcular_media_varianza_global(imagen)
                    print(f"Media global: {np.mean(media_global)}, Varianza global: {np.var(varianza_global)}")
                    mostrar_imagen("Media Global", media_global, descripcion_imagen)
                    mostrar_imagen("Varianza Global", varianza_global, descripcion_imagen)
                
                elif opcion == 5:
                    media_local, varianza_local = calcular_media_varianza_local(imagen)
                    mostrar_imagen("Media Local", media_local, descripcion_imagen)
                    mostrar_imagen("Varianza Local", varianza_local, descripcion_imagen)
                
                elif opcion == 6:
                    mostrar_todas_operaciones(imagen, descripcion_imagen)
                
                else:
                    print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()