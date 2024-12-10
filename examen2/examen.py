import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv 



def carga_imagen(ruta):
    return cv.imread(ruta, cv.IMREAD_GRAYSCALE)

def mostrar_imagen(titulo, img):
    cv.imshow(titulo, img) 
    cv.waitKey(0)      
    cv.destroyAllWindows() 
#tecnicas basadas en el procesamiento del histograma

def ecualizacion_local(img, clip_limit = 2.0, tile_grid_size=(8,8)):
    clahe = cv.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_grid_size)
    return clahe.apply(img)

def ecualizacion_global(img):
    return cv.equalizeHist(img)

def calcular_media_varianza_global(img):
    media = np.mean(img)
    varianza  = np.var(img)

    return media, varianza

def calcular_media_varianza_local(img, kernel_size = 5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    img_blurred= cv.filter2D(img, -1, kernel)

    img_diff = img - img_blurred
    variances = cv.filter2D(img_diff**2, -1, kernel)
    return img_blurred, variances

# filtros suavisantes y realzantes

def filtro_promedio(img, kernelsize):
    kernel = np.ones((kernelsize, kernelsize), dtype=np.float32) / (kernelsize **2)
    return cv.filter2D(img, -2, kernel)

def filtro_mediana (img, kernel_size):
    return cv.medianBlur(img, kernel_size)

def filtro_maximo(img, kernel_size):
    return cv.dilate(img, np.ones((kernel_size, kernel_size), dtype=np.unit8))

def filtro_minimo(img, kernel_size):
    return cv.erode(img, np.ones((kernel_size, kernel_size), dtype=np.unit8))

def filtro_laplaciano(img):
    laplaciano = cv.Laplacian(img, cv.CV_64F)
    laplaciano_abs = cv.convertScaleAbs(laplaciano)
    laplaciano_sumado = cv.addWeigthed(img, 1, laplaciano_abs, 1,0)
    return laplaciano_sumado

def filtro_gradiente(img):
    grad_x = cv.Sobel(img, cv.CV_64F, 1,0, ksize=3)
    grad_y = cv.Sobel(img, cv.CV_64F, 0,1, ksize=3)
    grad_magnitud = cv.magnitude(grad_x, grad_y)
    return cv.convertScaleAbs(grad_magnitud)


img = carga_imagen("./alto_contraste_img.jpg")

if img is not None: 
    img_filtro = ecualizacion_global(img)
    mostrar_imagen("Imagen Ecualizada Localmente", img_filtro)
else:
    print("Error: No se pudo cargar la imagen. Verifica la ruta proporcionada.")




