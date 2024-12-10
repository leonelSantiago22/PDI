import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv 

def cargar_imagen(img):
    return cv.imread(img, cv.IMREAD_GRAYSCALE)


def mostrar_imagen(titulo, img):
    cv.imshow(titulo, img)



def ecualizacion_local(img, clip_limit = 2.0, tale_grid_sice = (8,8)):
    clahe = cv.createCLAHE(clipLimit = clip_limit, tileGridSize = tale_grid_sice)
    return clahe.apply(img)


def ecualizacion_global(img):
    return cv.equalizeHist(img)


def media_varianza_local(img, kernel = 5 ):
    kernel = np.ones((kernel, kernel), dtype=np.float32)
    img_blurred = cv.filter2D(img, -1, kernel)    

    img_diff = img - img_blurred
    variances = cv.filter2D(img_diff**2, -1, kernel)

    return img_blurred, variances



imagen = cargar_imagen("alto_contraste_img.jpg")

if imagen is not None:
    image1, image12 = media_varianza_local(imagen)
    mostrar_imagen("media local",image1)
    mostrar_imagen("varianza local", image12)

    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("la imagen no se pudo abrir")