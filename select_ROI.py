import cv2 as cv
import numpy as np


class Draw:
    def __init__(self, image):
        (self.xi, self.yi) = (-1, -1)
        (self.xo, self.yo) = (-1, -1)
        self.img = cv.resize(image, (640, 480))
        self.copy = image

    def convert(self):
        img = cv.cvtColor(self.copy, cv.COLOR_BGR2GRAY)
        (m, n) = np.shape(img)
        a = n / 640
        b = m / 480
        self.xi = round(a * self.xi)
        self.yi = round(b * self.yi)
        self.xo = round(a * self.xo)
        self.yo = round(b * self.yo)
        return [(self.xi, self.yi), (self.xo, self.yo)]

    def select2(self, color='green', title='test'):
        if color == 'green':
            c = (0, 255, 0)
        if color == 'red':
            c = (255, 0, 0)
        if color == 'blue':
            c = (0, 0, 255)

        def draw(action, x, y, flags, *userdata):
            if action is cv.EVENT_LBUTTONDOWN:
                (self.xi, self.yi) = (x, y)
            elif action is cv.EVENT_LBUTTONUP:
                (self.xo, self.yo) = (x, y)
                cv.rectangle(self.img, (self.xi, self.yi), (self.xo, self.yo), c, 2, 8)
                cv.imshow(title, self.img)
        cv.namedWindow(title)
        cv.setMouseCallback(title, draw)
        cv.imshow(title, self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return [(self.xi, self.yi), (self.xo, self.yo)]

    def select(self, title='Select and press enter'):
        roi = cv.selectROI(windowName=title, img=self.img, showCrosshair=False, printNotice=False)
        cv.destroyAllWindows()
        self.xi, self.yi, a, b = roi
        self.xo = self.xi + a
        self.yo = self.yi + b
        return [(self.xi, self.yi), (self.xo, self.yo)]
