'''
Utilizando una herramiente de PDI, realice lo siguiebte 
- Implemente los filtros suavisantes y realzantes, promedio, mediana, maximo y minimo, Laplaciano y gradiente. Debera utilizar 
    distintos tamanos de mascaras(3x3, 5x,5) exepto para Laplaciano y gradiente, en el laplaciano mostrar el Laplaciano y 
    el laplaciano mas la img original, osea sumando las imagenes 
- Pruebe los filtro suavizanres y realizes sobre las imagenes descargadas. Debera recomendar cada imagen \
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar imágenes en escala de grises
def cargar_imagen(ruta):
    return cv.imread(ruta, cv.IMREAD_GRAYSCALE)

# Función para mostrar imagen original junto con la procesada
def mostrar_imagen_comparativa(titulo, img_original, img_procesada):
    plt.figure(figsize=(10, 5))
    
    # Mostrar la imagen original
    plt.subplot(1, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(img_original, cmap='gray')
    plt.axis('off')
    
    # Mostrar la imagen procesada
    plt.subplot(1, 2, 2)
    plt.title(titulo)
    plt.imshow(img_procesada, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Función para aplicar filtro promedio
def filtro_promedio(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    return cv.filter2D(img, -1, kernel)

# Función para aplicar filtro de mediana
def filtro_mediana(img, kernel_size):
    return cv.medianBlur(img, kernel_size)

# Función para aplicar filtro máximo
def filtro_maximo(img, kernel_size):
    return cv.dilate(img, np.ones((kernel_size, kernel_size), dtype=np.uint8))

# Función para aplicar filtro mínimo
def filtro_minimo(img, kernel_size):
    return cv.erode(img, np.ones((kernel_size, kernel_size), dtype=np.uint8))

# Función para aplicar Laplaciano
def filtro_laplaciano(img):
    laplaciano = cv.Laplacian(img, cv.CV_64F)
    laplaciano_abs = cv.convertScaleAbs(laplaciano)
    laplaciano_sumado = cv.addWeighted(img, 1, laplaciano_abs, 1, 0)
    return laplaciano_abs, laplaciano_sumado

# Función para calcular gradiente (magnitud del gradiente usando Sobel)
def filtro_gradiente(img):
    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    grad_magnitud = cv.magnitude(grad_x, grad_y)
    return cv.convertScaleAbs(grad_magnitud)

# Menú para elegir las operaciones
def menu_opciones_operaciones(img):
    kernel_size = 3  # Valor predeterminado del kernel
    while True:
        print("\nOpciones:")
        print("1. Cambiar tamaño de kernel (actual: {}x{})".format(kernel_size, kernel_size))
        print("2. Filtro Promedio")
        print("3. Filtro de Mediana")
        print("4. Filtro Máximo")
        print("5. Filtro Mínimo")
        print("6. Filtro Laplaciano")
        print("7. Gradiente (Sobel)")
        print("0. Salir al menú de imágenes")
        
        opcion = int(input("Elige una opción: "))
        
        if opcion == 1:
            kernel_size = int(input("Ingresa el tamaño del kernel (solo valores impares, 3 o 5): "))
            if kernel_size % 2 == 0:
                print("El tamaño del kernel debe ser impar. Se usará el valor predeterminado de 3.")
                kernel_size = 3

        elif opcion == 2:
            resultado = filtro_promedio(img, kernel_size)
            mostrar_imagen_comparativa(f"Filtro Promedio {kernel_size}x{kernel_size}", img, resultado)
        
        elif opcion == 3:
            resultado = filtro_mediana(img, kernel_size)
            mostrar_imagen_comparativa(f"Filtro Mediana {kernel_size}x{kernel_size}", img, resultado)
        
        elif opcion == 4:
            resultado = filtro_maximo(img, kernel_size)
            mostrar_imagen_comparativa(f"Filtro Máximo {kernel_size}x{kernel_size}", img, resultado)
        
        elif opcion == 5:
            resultado = filtro_minimo(img, kernel_size)
            mostrar_imagen_comparativa(f"Filtro Mínimo {kernel_size}x{kernel_size}", img, resultado)
        
        elif opcion == 6:
            laplaciano, laplaciano_sumado = filtro_laplaciano(img)
            mostrar_imagen_comparativa("Filtro Laplaciano", img, laplaciano)
            mostrar_imagen_comparativa("Laplaciano + Imagen Original", img, laplaciano_sumado)
        
        elif opcion == 7:
            resultado = filtro_gradiente(img)
            mostrar_imagen_comparativa("Filtro Gradiente (Magnitud Sobel)", img, resultado)
        
        elif opcion == 0:
            print("Regresando al menú principal...")
            break
        else:
            print("Opcion no valida")
# Menú para seleccionar la imagen
def menu_imagen():
    while True:
        print("\nMenú de imágenes:")
        print("1. Imagen de alto contraste")
        print("2. Imagen de bajo contraste")
        print("3. Imagen de alta iluminación")
        print("4. Imagen de baja iluminación")
        print("0. Salir del programa")
        
        opcion = int(input("Elige una opción: "))
        
        if opcion == 1:
            return cargar_imagen("alto_contraste_img.jpg"), "Imagen de Alto Contraste"
        elif opcion == 2:
            return cargar_imagen("bajo_contraste.jpeg"), "Imagen de Bajo Contraste"
        elif opcion == 3:
            return cargar_imagen("alta_iluminacion.jpg"), "Imagen de Alta Iluminación"
        elif opcion == 4:
            return cargar_imagen("baja_iluminacion.jpg"), "Imagen de Baja Iluminación"
        elif opcion == 0:
            print("Saliendo del programa...")
            return None, None
        else:
            print("Opción no válida. Intente nuevamente.")

# Función principal
def main():
    while True:
        img, descripcion = menu_imagen()
        if img is None:
            break
        print(f"\nSeleccionaste: {descripcion}")
        menu_opciones_operaciones(img)

if __name__ == "__main__":
    main()
