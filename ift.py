import numpy as np
import cv2 as cv
from scipy import integrate
from utility_ift import repeter, positive, canny_filter, \
    surf, diff, bond, g, get_circle_apex, get_circle_radio


# clase Calc, permite obtener todo lo necesario para calcular gamma
class SurfaceTension:
    def __init__(self, param, img, binary_value=False):
        drop, med, gauge = param
        self.deltaRho = drop - med  # dif. densidades [Kg/m3]
        self.g = 9.80  # ac. de gravedad [m/s2]
        self.G = g(gauge) / 1000  # [G] -> [m]
        self.Ro = np.inf  # radio Ro
        self.Bo = 0  # numero de Bond
        self.trans = 1  # factor [px] -> [m]
        self.sigma = 0  # sigma = Ds/De
        self.status = False  # estado de "se ejecutó primero drop()"
        self.circle = None  # círculo encontrado
        self.apex = None  # apex encontrado
        self.gamma = 0  # gamma encontrado, IFT
        self.surface = None  # superficie de la gota hallada
        self.center = 0  # centro de la gota
        self.Wo = 0  # número de Worthington
        self.Ne = 0  # número de Neumann
        self.needle_surf = None  # superficie de la aguja
        self.bin = binary_value  # indica si la imagen previamente es binaria
        self.img = img  # imagen
        self.copy = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)  # copia de la imagen, donde se dibuja
        self.copy = cv.cvtColor(self.copy, cv.COLOR_GRAY2BGR)  # copia de la imagen, donde se dibuja
        self.px = None  # radio en [px]
        self.process = None  # imagen procesada
        self.img_without_circle = self.copy.copy()  # imagen sin el círculo

    # needle detecta la aguja y su grosor, retornando la imagen con
    # la aguja marcada y un factor de [px] -> [m]
    def needle(self, coord):
        img = self.img
        canny = canny_filter(img) if not self.bin else cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # filtro Canny para bordes
        self.process = canny
        coord = positive(coord)  # coordenadas ajustadas
        self.needle_surf = surf(canny, coord)
        for s in self.needle_surf:
            cv.circle(self.copy, s, 2, (255, 0, 0), 2)

        # función que calcula el ancho de la aguja en [px]
        # luego, dado que sabemos su ancho en [m]
        # genera una conversión de [px] -> [m]
        def trans(sur):
            sur_y = sur[:, 1]
            y = np.median(sur_y)  # mitad de aguja
            if int(y) in sur_y:
                y = int(y)
            else:
                y = int(y + 1)
            x1, x2, k = 0, 0, 0
            for s in sur:
                if s[1] == y and k == 0:
                    x1 = s[0]
                    k = 1
                if s[1] == y and k == 1:
                    x2 = s[0]
            px = x2 - x1  # ancho en [px]
            return self.G / px, [(x1, y), (x2, y)], px

        trans = trans(self.needle_surf)
        self.trans = trans[0]
        med = trans[1]
        cv.line(self.copy, med[0], med[1], (255, 0, 0), 3)
        if self.status is True:
            self.px = self.Ro
            self.Ro = self.Ro * self.trans
            self.surface = diff(self.surface, self.needle_surf)
            for s in self.surface:
                cv.circle(self.copy, s, 2, (255, 255, 0), 2)
                cv.circle(self.img_without_circle, s, 2, (255, 255, 0), 2)
            cv.line(self.img_without_circle, med[0], med[1], (255, 255, 0), 3)
            for s in self.needle_surf:
                cv.circle(self.img_without_circle, s, 2, (255, 0, 0), 2)
        rad = trans[2] if self.Ro is np.inf else self.Ro
        return self.copy, rad, self.trans

    # drop detecta la gota y el círculo más exacto dentro de ella,
    # retornando la imagen con el círculo marcado y el radio de este.
    def drop(self, coord, hough=False):
        img = self.img
        coord = positive(coord)  # coordenadas ajustadas
        canny = canny_filter(img) if not self.bin else cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # filtro Canny para bordes
        self.process = canny if self.process is None else self.process
        self.surface = surf(canny, coord)  # superficie
        # puntos i y j son los índices que tienen a De
        i, j = np.where(self.surface[:, 1] == self.surface[np.argmax(self.surface[:, 0]), 1])[0]
        de = abs(self.surface[i, 0] - self.surface[j, 0])  # De
        self.center = min(self.surface[i, 0], self.surface[j, 0]) + de // 2  # Centro
        self.apex = (self.center, np.max(self.surface[:, 1]))  # Apex

        # ---------------- antigua implementación ------------------- #
        if hough is True:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if not self.bin else canny  # BGR -> Gray
            mini, maxi = np.min(self.surface[:, 0]), np.max(self.surface[:, 0])
            mx = int(abs(maxi - mini) / 2.0)  # máximo radio
            cir = get_circle_radio(gray, mx)

        # obtención del círculo
        else:
            cir = get_circle_apex(self.apex, de, self.surface)
        self.circle = cir
        if self.trans == 1:
            self.status = True
        self.px = cir[2]
        self.Ro = cir[2] * self.trans  # obtener Ro en [m]
        if self.status is False:
            self.surface = diff(self.surface, self.needle_surf)
            for i in self.surface:
                cv.circle(self.copy, i, 2, (255, 255, 0), 2)
            self.img_without_circle = self.copy.copy()
        cv.circle(self.copy, (cir[0], cir[1]), cir[2], (0, 255, 0), 3)  # dibujar círculo
        cv.circle(self.copy, self.apex, 3, (0, 0, 255), 5)  # dibujar apex
        return self.copy, self.Ro

    # solver obtiene la tensión gamma gracias a los datos ingresados
    # utilizando la fórmula del número de Bond o Eotvos.
    def solver(self):
        self.gamma = 1000 * (self.deltaRho * self.g * self.Ro ** 2) / self.Bo
        return self.gamma

    # status nos sirve para identificar cuál resultado de Ro
    # nos sirve para visualizar, si needle o drop. En resumen
    # indica si primero se uso drop y luego needle.
    def first_drop(self):
        return self.status

    # bond_sigma sirve para obtener el número de bond y sigma.
    # Para esto, se tiene que detectar De y Ds.
    def bond_sigma(self):
        x1 = np.min(self.surface[:, 0])  # punto izq. de De
        x2 = np.max(self.surface[:, 0])  # punto der. de De
        h = 0  # altura de De
        de = abs(x2 - x1)
        self.center = x1 + de / 2
        y = self.apex[1] - de
        x3, x4 = 0, 0
        for s in self.surface:
            if s[0] == x1:
                h = s[1]
            if s[0] == x2 and s[1] > h:
                h = s[1]
            if s[1] == y:
                if x3 == 0:
                    x3 = s[0]
                elif x4 == 0:
                    x4 = s[0]
        ds = abs(x4 - x3)
        cv.line(self.img_without_circle, (x1, h), (x2, h), (0, 255, 255), 3)  # De sin O
        cv.line(self.img_without_circle, (x3, y), (x4, y), (0, 255, 255), 3)  # Ds sin O
        cv.line(self.copy, (x1, h), (x2, h), (0, 255, 255), 3)  # De
        cv.line(self.copy, (x3, y), (x4, y), (0, 255, 255), 3)  # Ds
        self.sigma = ds / de
        self.Bo = bond(self.sigma)
        return self.copy, self.sigma, self.Bo

    # worthington retorna el número de worthington, Wo. Para esto,
    # separa la superficie de la gota en dos mitades, left y right.
    # Con esto, se obtiene el volumen del sólido de revolución
    # generado por cada una de estas superficies. Luego, Wo = Vd / Vmax
    def worthington(self):
        # separa en dos mitades la superficie de la drop
        def r(sur, cent):
            der = []
            izq = []
            for s in sur:
                if s[0] < cent:
                    izq.append([s[1], cent - s[0]])
                else:
                    der.append([s[1], s[0] - cent])
            return np.array(izq), np.array(der)

        r = r(self.surface, self.center)
        (left, right) = r[0], r[1]
        xl, yl = left[:, 0] * self.trans, left[:, 1] * self.trans
        xr, yr = right[:, 0] * self.trans, right[:, 1] * self.trans
        xl, yl = repeter(xl, yl)
        xr, yr = repeter(xr, yr)
        vmax = np.pi * self.G * (self.gamma / 1000) / (self.deltaRho * self.g)
        vl = np.pi * integrate.simpson(yl ** 2, xl)  # pi * integral r(x)^2 dx
        vr = np.pi * integrate.simpson(yr ** 2, xr)  # "
        vd = (vl + vr) / 2
        self.Wo = vd / vmax
        return self.Wo, vd * 10 ** 9

    # neumann retorna el número de Neumann, Ne. Para esto
    # es necesario obtener la altura h.
    def neumann(self):
        mini = np.min(self.surface[:, 1])
        h = (self.apex[1] - mini) * self.trans
        self.Ne = self.deltaRho * self.Ro * self.g * h / (self.gamma / 1000)
        return self.Ne

    # cambiar el radio
    def change_radius(self, rad):
        self.px = rad  # radio en [px] nuevo
        self.Ro = self.px * self.trans  # nuevo radio en [m]
        return self.Ro
