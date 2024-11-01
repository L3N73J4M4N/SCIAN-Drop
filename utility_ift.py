import numpy as np
import cv2 as cv


# Dada una curva (x, y), elimina sus elementos repetidos y los
# ordena. Se realiza este proceso con el fin de que pueda
# definir una curva correctamente y se pueda integrar
def repeter(x, y):
    n = len(y)
    k = -1
    for i in range(0, n):
        if y[i] == np.max(y):
            k = i
    izqy, dery = [], []
    izqx, derx, = [], []
    for i in range(0, n):
        if x[i] < x[k]:
            if y[i] not in izqy:
                izqy.append(y[i])
                izqx.append(x[i])
        else:
            if y[i] not in dery:
                dery.append(y[i])
                derx.append(x[i])
    izqx.sort()
    derx.sort()
    izqy.sort()
    dery.sort(reverse=True)
    yn = izqy + dery
    xn = izqx + derx
    return [np.array(xn), np.array(yn)]


# función que elimina, en caso de haber, coordenadas negativas
# y las ordena correctamente. Dada la implementación
# para obtener ROI en OpenCV, no debería ser necesaria
def positive(c):
    if c == -1:
        return -1
    (xi, yi), (xf, yf) = c
    if xi < 0:
        xi = 0
    if yi < 0:
        yi = 0
    if xf < 0:
        xf = 0
    if yf < 0:
        yf = 0
    if xf < xi and yf < yi:
        (xi, yi, xf, yf) = (xf, yf, xi, yi)
    return (xi, yi), (xf, yf)


# retorna y calcula surf1 - surf2 en altura. Por ejemplo:
# | |  -  | | =
# | |            | |
def diff(surf1, surf2):
    new = []
    for i in surf1:
        if i[1] not in surf2[:, 1]:
            new.append(i)
    return np.array(new)


# retorna la superficie del objeto, dadas las coordenadas
# en donde se ubica. im debe estar binarizada con sus
# bordes obtenidos.
def surf(im, c=-1):
    if c == -1:
        m, n = np.shape(im)
        c = [(0, 0), (m, n)]
    (xi, yi), (xf, yf) = c
    sur = []
    for j in range(yi, yf):
        k = 0
        x1, x2 = 0, 0
        for i in range(xi + 1, xf - 1):
            if im[j, i] != 0 and im[j, i - 1] == 0 and k == 0:
                x1 = i
                k = 1
            if im[j, i] != 0 and im[j, i + 1] == 0 and k == 1:
                x2 = i
        if x1 != 0:
            sur.append([x1, j])
        if x2 != 0:
            sur.append([x2, j])
    return np.array(sur)


# fórmula del número de bond
def bond_omar(sigma):
    return 0.12836 - 0.7577 * sigma + 1.7713 * sigma ** 2 - 0.5426 * sigma ** 3


# bond con aproximación de orden 6. Para obtener esta aproximación
# se resolvió la EDP que define la superficie de una gota colgante
# para distintos números de Bond. Luego, se les calculó sigma
# y, con mínimos cuadrados, se obtuvo Bo(sigma)
def bond(x):
    c = [0.331402284,
         -1.20035969,
         0.998790320,
         0.153195902,
         0.326178807,
         -0.0152367310,
         0.000515911091]
    pol = 0
    for k in range(0, len(c)):
        pol += c[k] * x ** (len(c) - k - 1)
    return pol


# filtro Canny diseñado para esta implementación.
# Consigue binarizar la imagen de manera efectiva,
# sin dejar ruido y marcando los bordes.
def canny_filter(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    can = cv.Canny(img, np.min(img), np.max(img))
    kernel1 = np.array([[0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    dilated = cv.dilate(can, kernel1)
    kernel2 = np.array([[0, 0, 0],
                        [0, 1, 1],
                        [0, 0, 0]], np.uint8)
    dilated = cv.dilate(dilated, kernel2)
    n, output, stats, centroids = cv.connectedComponentsWithStats(dilated, connectivity=8)
    sizes = stats[1:, -1]
    img2 = np.zeros(output.shape, np.uint8)
    for i in range(0, n - 1):
        if sizes[i] >= 1000:
            img2[output == i + 1] = 255
    return img2


# obtiene especificamente la aguja, de la imagen
# img binarizada. Puede no retornar toda la altura de la aguja.
# Por ejemplo, no es totalmente exacta:
# | | -> puede retornar -> | |
# | |
def need(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    can = cv.Canny(gray, np.min(gray), np.max(gray))
    kernel1 = np.array([[0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0]], np.uint8)
    kernel2 = np.array([[0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0]], np.uint8)
    eroded = cv.erode(can, kernel1, iterations=5)
    dilated = cv.dilate(eroded, kernel2, iterations=10)
    n, output, stats, centroids = cv.connectedComponentsWithStats(dilated, connectivity=8)
    sizes = stats[1:, -1]
    img2 = np.zeros(output.shape)
    for i in range(0, n - 1):
        if sizes[i] >= 100:
            img2[output == i + 1] = 255
    return dilated


# dado un valor de calibre de aguja
# retorna su medida en milímetros
def g(value):  # mm en función del calibre
    gauge = {36: 0.004,  # G: pulgada
             35: 0.005,
             34: 0.007,
             33: 0.008,
             32: 0.009,
             31: 0.010,
             30: 0.012,
             29: 0.013,
             28: 0.014,
             27: 0.016,
             26: 0.018,
             25: 0.020,
             24: 0.022,
             23: 0.025,
             22: 0.028,
             21: 0.032,
             20: 0.035,
             19: 0.042,
             18: 0.049,
             17: 0.058,
             16: 0.065,
             15: 0.072,
             14: 0.083,
             13: 0.095,
             12: 0.109,
             11: 0.120,
             10: 0.134,
             9: 0.148,
             8: 0.165,
             7: 0.180,
             6: 0.203,
             5: 0.220,
             4: 0.238,
             3: 0.259,
             2: 0.284,
             1: 0.300}
    return gauge[value] * 25.4  # pulgada -> mm


# forma de hacerlo con transformada de Hough ajustando
# sus parámetros y dejando fijo un radio máximo mx
def get_circle_hough(gray, mx, force_debug):
    try:
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 50, param1=50,
                                  param2=40, minRadius=0, maxRadius=mx)  # Método de Hough
        cir = np.uint16(np.around(circles))[0][0]  # Aproximar los decimales a int
        return cir if not force_debug else force_debug[0]
    except TypeError:
        circles = []
        lens = []
        for i in reversed(range(1, 40)):
            circle = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 50, param1=50,
                                     param2=i, minRadius=0, maxRadius=mx)
            if circle is not None:
                if len(circle[0]) == 1:
                    return np.uint16(np.around(circle))[0][0]
                circ = np.uint16(np.around(circle))[0]
                circles.append(circ)
                lens.append(len(circ))
        return circles[np.argmin(lens)][0]


# obtención de un círculo con transformada de Hough dejando sus
# parámetros iguales, pero agrandando el radio máximo de a poco
# para hallar el menor de los mayores
def get_circle_radio(gray, mx):
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 50, param1=50,
                              param2=40, minRadius=0, maxRadius=mx)  # Método de Hough
    if circles is not None:
        cir = np.uint16(np.around(circles))[0][0]  # Aproximar los decimales a int
    else:
        cir = get_circle_radio(gray, int(round(mx * 1.025)))
    return cir


# obtención del círculo de manera clásica. Vale decir,
# teniendo un apex, el de y la superficie es posible tomar
# el apex y generar un círculo desde este punto de radio R.
# El R irá variando partiendo de un valor alto máximo De // 2
# hasta encontrar la circunferencia que quepa completamente dentro
def get_circle_apex(apex, de, surface):
    r = de // 2  # radio máximo
    y = apex[1] - r  # y desde que comenzará la superficie a comparar
    surf_left = []  # superficie izquierda inferior
    xc_min, xc_max = apex[0] - r, apex[0] + r  # bordes máximo del círculo mayor
    for i in range(0, len(surface)):
        if apex[0] > surface[i, 0] > xc_min and surface[i, 1] > y:
            surf_left.append(surface[i])
    surf_left = np.array(surf_left, int)
    surf_left = surf_left[surf_left[:, 1].argsort()]  # se ordenan los y sin razón alguna jaja (o sí? no me acuerdo)
    while r > 0:
        circle = [apex[0], apex[1] - r, r]
        for j in range(y, apex[1]):
            x = apex[0] - np.sqrt(r ** 2 - (j - (apex[1] - r)) ** 2)  # fórmula de una circunferencia
            if j in surf_left[:, 1]:
                js = np.where(surf_left[:, 1] == j)[0]
                if x < surf_left[js, 0]:  # el punto está afuera de la gota
                    break  # bye
                elif j == apex[1] - 1:  # llegó al final, todos están dentro
                    return circle
        r -= 1  # avanzamos
    return [apex[0], apex[1] - de // 2, de // 2]  # [x0, y0, r] -> [centro, radio]



