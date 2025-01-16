import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Función para cargar imágenes en escala de grises
def cargar_imagen(ruta):
    return cv.imread(ruta, cv.IMREAD_GRAYSCALE)

# Mostrar imágenes originales y procesadas lado a lado
def mostrar_imagen_comparativa(titulo, img_original, img_procesada):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(img_original, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title(titulo)
    plt.imshow(img_procesada, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Transformada de Fourier
def transformar_fourier(img):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift

# Transformada inversa
def inversa_fourier(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(dft_ishift)
    return np.abs(img_back)

# Crear un filtro ideal de paso bajo
def filtro_ideal_paso_bajo(shape, radio):
    filas, columnas = shape
    centro = (filas // 2, columnas // 2)
    filtro = np.zeros(shape, dtype=np.float32)
    for i in range(filas):
        for j in range(columnas):
            distancia = np.sqrt((i - centro[0])**2 + (j - centro[1])**2)
            if distancia <= radio:
                filtro[i, j] = 1
    return filtro

# Crear un filtro Butterworth de paso bajo
def filtro_butterworth_paso_bajo(shape, radio, orden):
    filas, columnas = shape
    centro = (filas // 2, columnas // 2)
    filtro = np.zeros(shape, dtype=np.float32)
    for i in range(filas):
        for j in range(columnas):
            distancia = np.sqrt((i - centro[0])**2 + (j - centro[1])**2)
            filtro[i, j] = 1 / (1 + (distancia / radio)**(2 * orden))
    return filtro

# Crear un filtro Gaussiano de paso bajo
def filtro_gaussiano_paso_bajo(shape, radio):
    filas, columnas = shape
    centro = (filas // 2, columnas // 2)
    filtro = np.zeros(shape, dtype=np.float32)
    for i in range(filas):
        for j in range(columnas):
            distancia = np.sqrt((i - centro[0])**2 + (j - centro[1])**2)
            filtro[i, j] = np.exp(-(distancia**2) / (2 * (radio**2)))
    return filtro

# Aplicar filtro en el dominio de la frecuencia
def aplicar_filtro(img, filtro):
    dft_shift = transformar_fourier(img)
    dft_filtrado = dft_shift * filtro
    img_filtrada = inversa_fourier(dft_filtrado)
    return img_filtrada

# Menú de filtros
def menu_opciones_operaciones(img):
    while True:
        print("\nOpciones de Filtros:")
        print("1. ILPF (Filtro Ideal Pasa Bajo)")
        print("2. BLPF (Filtro Butterworth Pasa Bajo)")
        print("3. GLPF (Filtro Gaussiano Pasa Bajo)")
        print("4. IHPF (Filtro Ideal Pasa Alto)")
        print("5. BHPF (Filtro Butterworth Pasa Alto)")
        print("6. GHPF (Filtro Gaussiano Pasa Alto)")
        print("0. Salir al menú de imágenes")
        
        opcion = int(input("Elige una opción: "))
        if opcion == 0:
            print("Regresando al menú principal...")
            break

        radio = int(input("Ingresa el radio del filtro: "))
        if opcion in [2, 5]:
            orden = int(input("Ingresa el orden del filtro Butterworth: "))
        
        if opcion == 1:
            filtro = filtro_ideal_paso_bajo(img.shape, radio)
        elif opcion == 2:
            filtro = filtro_butterworth_paso_bajo(img.shape, radio, orden)
        elif opcion == 3:
            filtro = filtro_gaussiano_paso_bajo(img.shape, radio)
        elif opcion == 4:
            filtro = 1 - filtro_ideal_paso_bajo(img.shape, radio)
        elif opcion == 5:
            filtro = 1 - filtro_butterworth_paso_bajo(img.shape, radio, orden)
        elif opcion == 6:
            filtro = 1 - filtro_gaussiano_paso_bajo(img.shape, radio)
        else:
            print("Opción no válida.")
            continue

        img_procesada = aplicar_filtro(img, filtro)
        mostrar_imagen_comparativa(f"Filtro Aplicado (Opción {opcion})", img, img_procesada)

# Menú de selección de imágenes
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
