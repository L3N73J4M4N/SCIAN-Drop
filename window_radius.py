from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ift import SurfaceTension
from select_ROI import Draw
import numpy as np
import cv2 as cv


class RadiusAdjust:
    def __init__(self, img, rad, apex, color, step, title=-1):
        win = Toplevel()
        title = 'test' if title == -1 else title
        win.title(title)
        x_screen, y_screen = win.winfo_screenwidth(), win.winfo_screenheight()
        x = x_screen // 2 - 390
        y = y_screen // 2 - 265
        win.geometry('780x530+' + str(x) + '+' + str(y))
        self.n = rad
        self.apex = apex
        self.step = step

        # RADIO
        def rad_up():
            self.n += self.step
            copy_2 = self.img.copy()
            center = (self.apex[0], self.apex[1] - self.n)
            cv.circle(copy_2, center, self.n, color, 1)
            img_2 = Image.fromarray(copy_2)
            img_2 = ImageTk.PhotoImage(image=img_2)
            self.image.configure(image=img_2)
            self.image.image = img_2

        def rad_down():
            self.n -= self.step
            copy_2 = self.img.copy()
            center = (self.apex[0], self.apex[1] - self.n)
            cv.circle(copy_2, center, self.n, color, 1)
            img_2 = Image.fromarray(copy_2)
            img_2 = ImageTk.PhotoImage(image=img_2)
            self.image.configure(image=img_2)
            self.image.image = img_2

        def set_rad():
            self.result()
            win.destroy()

        radius = Frame(win)
        radius_up = Button(radius, text='Radius up', command=rad_up)
        radius_down = Button(radius, text='Radius down', command=rad_down)
        set_rad = Button(radius, text='Set radius', command=set_rad)
        radius_up.grid(column=0, row=0)
        radius_down.grid(column=1, row=0)
        set_rad.grid(column=2, row=0)
        radius.pack()

        # IMAGEN
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        copy = self.img.copy()
        center = (self.apex[0], self.apex[1] - self.n)
        cv.circle(copy, center, self.n, color, 1)
        img = Image.fromarray(copy)
        img = ImageTk.PhotoImage(image=img)
        self.image = Label(win, image=img)
        self.image.pack()

    def result(self):
        return self.n

