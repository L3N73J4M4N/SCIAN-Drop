from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ift import SurfaceTension
from select_ROI import Draw
import numpy as np
import cv2 as cv
from icons import icon_drop_karina, icon_warn
import os
import csv


class Interface:
    def __init__(self):
        self.img = None  # imagen
        self.copy = None  # copia de la imagen
        self.param = None  # parámetros
        self.solver = None  # método SurfaceTension para obtener los parámetros
        self.image = None  # label de la imagen
        self.ima = None  # frame de la imagen
        # lista con los resultados
        self.result = [('Ro', 0), ('Bo', 0), ('σ', 0),
                       ('γ', 0), ('Wo', 0), ('Ne', 0),
                       ('V', 0), ('Ro', 0), ('α', 0)]
        # lista con los contadores, cuándo se ocupan las opciones de selección
        self.counter = [False, False, False]
        self.radius = None  # radio ajustado
        win = Tk()
        win.title("SCIAN-Drop (v2024.04.24)")
        x_screen, y_screen = win.winfo_screenwidth(), win.winfo_screenheight()
        x = x_screen // 2 - 390
        y = y_screen // 2 - 285
        win.geometry('780x550+' + str(x) + '+' + str(y))
        win.resizable(None, None)
        icon16, icon32 = icon_drop_karina()
        icon16, icon32 = ImageTk.PhotoImage(icon16), ImageTk.PhotoImage(icon32)
        win.iconphoto(False, icon32, icon16)
        left = Frame(win)
        right = Frame(win, padding=(10, 0))
        left.grid(row=0, column=0)
        right.grid(row=0, column=1)
        self.hough = BooleanVar()  # usar hough o no
        self.coord = ['drop', 'need']  # coordenadas
        self.name = None  # nombre del archivo

        # Parámetros
        # Marco para los datos
        par = LabelFrame(left, width=250, text="Parameters", padding=20)
        par.pack()

        # Densidad de la gota
        # Texto
        drop = Label(par, text="Drop density [Kg/m³]:")
        drop.grid(row=0, column=0)
        # Entrada
        res_drop = Entry(par, width=30)
        res_drop.grid(row=0, column=1)

        # Densidad del medio
        # Texto
        med = Label(par, text="Environment density [Kg/m³]:")
        med.grid(row=1, column=0)
        # Entrada
        res_med = Entry(par, width=30)
        res_med.grid(row=1, column=1)

        # Diámetro de la aguja
        # Texto
        needle = Label(par, text="Needle thickness [G]:")
        needle.grid(row=2, column=0)
        # Entrada
        res_needle = Entry(par, width=30)
        res_needle.grid(row=2, column=1)

        def enter_data():
            d = res_drop.get()
            m = res_med.get()
            n = res_needle.get()
            self.param = [float(d), float(m), float(n)]
            warning.configure(text='--- Entered data ✓ ---')

        enter = Button(par, text='Enter data', command=enter_data)
        enter.grid(row=3, columnspan=2, pady=5)
        warning = Label(left, text='--- Before loading the image, enter the data ---')
        warning.pack()

    # Imagen
        self.ima = LabelFrame(left, height=50, text="Image")
        self.ima.pack()

        def alert():
            win_alert = Toplevel()
            win_alert.geometry('300x50+' + str(x + 240) + '+' + str(y + 225))
            win_alert.resizable(None, None)
            win_alert.title('Warning')
            warn16 = ImageTk.PhotoImage(icon_warn())
            win_alert.iconphoto(False, warn16)
            win_alert.config(width=300, height=50)
            exit = Button(win_alert, text='Ok', command=win_alert.destroy)
            label = Label(win_alert, text='Before loading the image, enter the data')
            label.pack()
            exit.pack()

        def alert_binary():
            win_alert = Toplevel()
            win_alert.geometry('300x50+' + str(x + 240) + '+' + str(y + 225))
            win_alert.resizable(None, None)
            win_alert.title('Warning')
            warn16 = ImageTk.PhotoImage(icon_warn())
            win_alert.iconphoto(False, warn16)
            win_alert.config(width=300, height=50)
            exit = Button(win_alert, text='Ok', command=win_alert.destroy)
            label = Label(win_alert, text='The image is not binary  ')
            label.pack()
            exit.pack()

        def alert_debug():
            win_alert = Toplevel()
            win_alert.geometry('300x50+' + str(x + 240) + '+' + str(y + 225))
            win_alert.resizable(None, None)
            win_alert.title('Warning')
            warn16 = ImageTk.PhotoImage(icon_warn())
            win_alert.iconphoto(False, warn16)
            win_alert.config(width=300, height=50)
            exit = Button(win_alert, text='Ok', command=win_alert.destroy)
            label = Label(win_alert, text='Calculating has not been made')
            label.pack()
            exit.pack()

        def drop():
            d = Draw(self.copy)
            ds = d.select('Select the drop and press enter')
            dc = d.convert()
            self.coord[0] = dc
            if self.solver is None:
                self.solver = SurfaceTension(self.param, self.img, self.binary.get())
            drop_data = self.solver.drop(dc, hough=self.hough.get())
            if self.solver.first_drop() is False:
                bond_data = self.solver.bond_sigma()
                resize = cv.resize(bond_data[0], (385, 285), interpolation=cv.INTER_CUBIC)
                self.result[1] = ('Bo', bond_data[2])
                self.result[2] = ('σ', bond_data[1])
            else:
                resize = cv.resize(drop_data[0], (385, 285), interpolation=cv.INTER_CUBIC)
            img = Image.fromarray(resize)
            img = ImageTk.PhotoImage(image=img)
            self.image.configure(image=img)
            self.image.image = img
            self.result[0] = ('Ro', drop_data[1])
            if self.counter[1] is False:
                warning1.configure(text='--- Entered image ✓ - Entered drop ✓ - Now, select needle ---')
            else:
                warning1.configure(text='--- All ready to calculate ✓ ---')
            self.counter[0] = True

        def need():
            d = Draw(self.copy)
            ds = d.select('Select the needle and press enter')
            dc = d.convert()
            self.coord[1] = dc
            if self.solver is None:
                self.solver = SurfaceTension(self.param, self.img, self.binary.get())
            need_data = self.solver.needle(dc)
            if self.counter[0] is False:
                warning1.configure(text='--- Entered image ✓ - Entered needle ✓ - Now, select drop ---')
            else:
                warning1.configure(text='--- All ready to calculate ✓ ---')
            self.counter[1] = True
            resize = cv.resize(need_data[0], (385, 285), interpolation=cv.INTER_CUBIC)
            if self.solver.first_drop() is True:
                bond_data = self.solver.bond_sigma()
                self.result[0] = ('Ro', need_data[1])
                self.result[1] = ('Bo', bond_data[2])
                self.result[2] = ('σ', bond_data[1])
                resize = cv.resize(bond_data[0], (385, 285), interpolation=cv.INTER_CUBIC)
            img = Image.fromarray(resize)
            img = ImageTk.PhotoImage(image=img)
            self.image.configure(image=img)
            self.image.image = img

        def delete():
            enter_data()
            warning1.configure(text='--- Entered image ✓ - Now, select drop and needle ---')
            self.solver = SurfaceTension(self.param, self.img, self.binary.get())
            resize = cv.resize(self.copy, (385, 285), interpolation=cv.INTER_CUBIC)
            img = Image.fromarray(resize)
            img = ImageTk.PhotoImage(image=img)
            self.image.configure(image=img)
            self.image.image = img
            if self.counter[2]:
                result0.delete(0, 'end')
                result1.delete(0, 'end')
                result2.delete(0, 'end')
                result3.delete(0, 'end')
                result4.delete(0, 'end')
                result5.delete(0, 'end')
                result6.delete(0, 'end')
                result7.delete(0, 'end')
                result8.delete(0, 'end')
            self.counter = [False, False, False]

        def openfile():
            filename = ''
            if self.param is not None:
                filename = filedialog.askopenfilename(initialdir='Desktop',
                                                      title='Select Image',
                                                      filetypes=(('all files', '*.*'),
                                                                 ('bmp files', '*.bmp'),
                                                                 ('jpeg files', '*.jpg'),
                                                                 ('png files', '*.png')))
            else:
                alert()
            if len(filename) > 0:
                # Acondicionamiento de la imagen
                self.img = cv.imread(filename)
                self.copy = self.img.copy()
                self.copy = cv.cvtColor(self.copy, cv.COLOR_BGR2GRAY)
                self.copy = cv.cvtColor(self.copy, cv.COLOR_GRAY2BGR)
                if self.binary.get() is True:
                    if not np.array_equal(self.copy, self.img):
                        return alert_binary()
                if self.solver is not None:
                    self.solver = SurfaceTension(self.param, self.img, self.binary.get())
                resize = cv.resize(self.copy, (385, 285), interpolation=cv.INTER_CUBIC)
                img = Image.fromarray(resize)
                img = ImageTk.PhotoImage(image=img)
                # Elemento donde irá la imagen
                self.image = Label(self.ima, image=img)
                self.image.image = img

                # botones para trabajar la imagen
                button_drop = Button(self.ima, text='Select drop', command=drop)
                button_needle = Button(self.ima, text='Select needle', command=need)
                button_reset = Button(self.ima, text='Delete', command=delete)
                check_hough = Checkbutton(self.ima, text='Hough transform', variable=self.hough)
                name = Label(self.ima, text=os.path.split(filename)[1])
                self.name = os.path.split(filename)[1]
                check_hough.grid(row=0, column=2, columnspan=2)
                button_drop.grid(row=1, column=2)
                button_needle.grid(row=1, column=1)
                button_reset.grid(row=1, column=3)
                name.grid(row=2, columnspan=4)
                self.image.grid(row=3, columnspan=4)
                warning1.configure(text='--- Entered image ✓ - Now, select drop and needle ---')

        self.binary = BooleanVar()
        option = Checkbutton(self.ima, text='Binary image', variable=self.binary)
        option.grid(row=0, column=0, columnspan=2)
        button = Button(self.ima, text='Select an image', command=openfile)
        button.grid(row=1, column=0)
        warning1 = Label(right, text='\n --- Before calculating, enter an image ---')
        warning1.pack()

        # Realizar los cálculos
        def calculus():
            self.counter[2] = True
            sol = self.solver.solver()
            wo = self.solver.worthington()
            ne = self.solver.neumann()
            self.result[3] = ('γ', sol)
            self.result[4] = ('Wo', wo[0])
            self.result[5] = ('Ne', ne)
            self.result[6] = ('V', wo[1])
            self.result[7] = ('px', self.solver.px)
            self.result[8] = ('α', self.solver.trans)
            (_, s0), (_, s1), (_, s2), (_, s3), \
                (_, s8), (_, s5), (_, s6), (_, s7), (_, s4) = self.result  # a no se usa
            s0, s1, s2, s3 = np.round(s0 * 1000, 5), np.round(s1, 5), np.round(s2, 5), np.round(s3, 5)
            s4, s5, s6, s8 = np.round(s4 * 10 ** 6, 5), np.round(s5, 5), np.round(s6, 5), np.round(s8, 5)
            result0.delete(0, 'end')
            result1.delete(0, 'end')
            result2.delete(0, 'end')
            result3.delete(0, 'end')
            result4.delete(0, 'end')
            result5.delete(0, 'end')
            result6.delete(0, 'end')
            result7.delete(0, 'end')
            result8.delete(0, 'end')
            result0.insert(0, str(s0))
            result1.insert(0, str(s1))
            result2.insert(0, str(s2))
            result3.insert(0, str(s3))
            result4.insert(0, str(s4))
            result5.insert(0, str(s5))
            result6.insert(0, str(s6))
            result7.insert(0, str(s7))
            result8.insert(0, str(s8))
            self.counter[2] = True

        calculate = Button(right, text='Calculate', command=calculus)
        calculate.pack()
        result = LabelFrame(right, text='Results')
        result.pack()

        # Borrar all
        def reset():
            warning.configure(text='--- Before loading the image, enter the data ---')
            warning1.configure(text='\n --- Before calculating, enter an image ---')
            self.image = None
            self.solver = None
            self.copy = None
            self.param = None
            self.solver = None
            self.result = [('Ro', 0), ('Bo', 0), ('σ', 0), ('γ', 0),
                           ('Wo', 0), ('Ne', 0), ('V', 0), ('px', 0), ('α', 0)]
            self.counter = [False, False, False]
            res_drop.delete(0, 'end')
            res_needle.delete(0, 'end')
            res_med.delete(0, 'end')
            result0.delete(0, 'end')
            result1.delete(0, 'end')
            result2.delete(0, 'end')
            result3.delete(0, 'end')
            result4.delete(0, 'end')
            result5.delete(0, 'end')
            result6.delete(0, 'end')
            result7.delete(0, 'end')
            result8.delete(0, 'end')
            self.ima.destroy()
            self.ima = LabelFrame(left, width=250, text="Image")
            self.ima.pack()
            option = Checkbutton(self.ima, text='Binary image', variable=self.binary)
            option.grid(row=0, column=0, columnspan=2)
            button = Button(self.ima, text='Select an image', command=openfile)
            button.grid(row=1, column=0)

        res = Frame(result)
        res.pack()
        (r0, _), (r1, _), (r2, _), (r3, _), (r4, _), \
            (r5, _), (r6, _), (r7, _), (r8, _) = self.result
        # primer resultado
        text0 = Label(res, text=r0 + ' [mm]:', padding=5)
        result0 = Entry(res, width=10)
        # segundo resultado
        text1 = Label(res, text=r1 + ':', padding=5)
        result1 = Entry(res, width=10)
        # tercero resultado
        text2 = Label(res, text=r2 + ':', padding=5)
        result2 = Entry(res, width=10)
        # cuarto resultado
        text3 = Label(res, text=r3 + ' [mN/m]:', padding=5)
        result3 = Entry(res, width=10)
        # quinto resultado
        text4 = Label(res, text=r8 + ' ×10⁶:', padding=5)
        result4 = Entry(res, width=10)
        # sexto resultado
        text5 = Label(res, text=r5 + ':', padding=5)
        result5 = Entry(res, width=10)
        # séptimo resultado
        text6 = Label(res, text=r6 + ' [mm³]:', padding=5)
        result6 = Entry(res, width=10)
        # octavo resultado
        text7 = Label(res, text=r7 + ' [px]:', padding=5)
        result7 = Entry(res, width=10)
        # noveno resultado
        text8 = Label(res, text=r4 + ':', padding=5)
        result8 = Entry(res, width=10)

        # pack en forma de grid, para cada uno
        text0.grid(row=0, column=0)
        result0.grid(row=0, column=1)
        text1.grid(row=0, column=2)
        result1.grid(row=0, column=3)
        text2.grid(row=0, column=4)
        result2.grid(row=0, column=5)
        text3.grid(row=1, column=0)
        result3.grid(row=1, column=1)
        text4.grid(row=1, column=2)
        result4.grid(row=1, column=3)
        text5.grid(row=1, column=4)
        result5.grid(row=1, column=5)
        text6.grid(row=2, column=0)
        result6.grid(row=2, column=1)
        text7.grid(row=2, column=2)
        result7.grid(row=2, column=3)
        text8.grid(row=2, column=4)
        result8.grid(row=2, column=5)

        # Debug
        def adjust():
            if self.counter[2] is False:
                alert_debug()
                return None
            c_drop, c_need = self.coord
            (x0, y0), (xf, yf) = c_drop
            apex = (self.solver.apex[0] - x0, self.solver.apex[1] - y0)
            self.radius = self.solver.px
            binary_image = self.solver.process
            binary_image = binary_image.astype('uint8')
            binary_image = binary_image[y0: yf, x0: yf]
            self.image_adjust = cv.cvtColor(binary_image, cv.COLOR_GRAY2BGR)
            win_radius = Toplevel()
            icon16 = ImageTk.PhotoImage(icon_drop_karina()[0])
            win_radius.iconphoto(False, icon16)
            win_radius.title('Adjust radius')
            self.apex_adjust = apex

            # FUNCIONES
            def rad_up():
                try:
                    h = int(step.get())
                except ValueError:
                    h = 1
                self.radius += h
                rad.config(text='R =' + str(self.radius))
                copy_2 = self.image_adjust.copy()
                center_circ = (apex[0], apex[1] - self.radius)
                cv.circle(copy_2, center_circ, self.radius, [255, 0, 0], 1)
                img_2 = Image.fromarray(copy_2)
                img_2 = ImageTk.PhotoImage(image=img_2)
                self.image_adjust_label.configure(image=img_2)
                self.image_adjust_label.image = img_2

            def rad_down():
                try:
                    h = int(step.get())
                except ValueError:
                    h = 1
                self.radius -= h
                rad.config(text='R =' + str(self.radius))
                copy_2 = self.image_adjust.copy()
                center_circ = (apex[0], apex[1] - self.radius)
                cv.circle(copy_2, center_circ, self.radius, [255, 0, 0], 1)
                img_2 = Image.fromarray(copy_2)
                img_2 = ImageTk.PhotoImage(image=img_2)
                self.image_adjust_label.configure(image=img_2)
                self.image_adjust_label.image = img_2

            def set_rad():
                image_new = self.solver.img_without_circle.copy()
                cv.circle(image_new, (self.solver.apex[0], self.solver.apex[1] - self.radius),
                          self.radius, (0, 255, 0), 3)  # dibujar círculo
                cv.circle(image_new, self.solver.apex, 3, (0, 0, 255), 5)  # dibujar apex
                image_new = cv.resize(image_new, (385, 285), interpolation=cv.INTER_CUBIC)
                image_new = Image.fromarray(image_new)
                image_new = ImageTk.PhotoImage(image=image_new)
                self.image.configure(image=image_new)
                self.image.image = image_new
                rad = self.solver.change_radius(self.radius)
                self.result[0] = ('Ro', rad)
                calculus()
                win_radius.destroy()

            # BOTONES
            radius = Frame(win_radius)
            step = Entry(radius, width=10)
            step_name = Label(radius, text='Radius step =')
            rad = Label(radius, text='R =' + str(self.radius))
            radius_up = Button(radius, text='Radius up', command=rad_up)
            radius_down = Button(radius, text='Radius down', command=rad_down)
            set_rad = Button(radius, text='Set radius', command=set_rad)
            step_name.grid(column=0, row=0)
            step.grid(column=1, row=0)
            radius_up.grid(column=3, row=0)
            radius_down.grid(column=2, row=0)
            set_rad.grid(column=4, row=0)
            rad.grid(column=5, row=0)
            radius.pack()

            # IMAGEN
            copy = self.image_adjust.copy()
            center = (apex[0], apex[1] - self.radius)
            cv.circle(copy, center, self.radius, [255, 0, 0], 1)
            img = Image.fromarray(copy)
            img = ImageTk.PhotoImage(image=img)
            self.image_adjust_label = Label(win_radius, image=img)
            self.image_adjust_label.image = img
            self.image_adjust_label.pack()

        def exp_surf():
            if self.counter[2] is False:
                alert_debug()
                return None
            path = filedialog.asksaveasfilename(initialdir="Desktop",
                                                initialfile=self.name.split('.')[0]
                                                + '_contours',
                                                defaultextension='.csv',
                                                title="Save as",
                                                filetypes=(("csv files", "*.csv"),
                                                           ("all files", "*.*")))
            if len(path) > 0:
                if path[-4::] != '.csv':
                    path += '.csv'
                with open(path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['droplet contour'])
                    writer.writerows(self.solver.surface)
                    writer.writerow('')
                    writer.writerow(['needle contour'])
                    writer.writerows(self.solver.needle_surf)

        def exp_res():
            if self.counter[2] is False:
                alert_debug()
                return None
            path = filedialog.asksaveasfilename(initialdir="Desktop",
                                                initialfile=self.name.split('.')[0]
                                                + '_results',
                                                defaultextension='.csv',
                                                title="Save as",
                                                filetypes=(("csv files", "*.csv"),
                                                           ("all files", "*.*")))
            if len(path) > 0:
                if path[-4::] != '.csv':
                    path += '.csv'
                with open(path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['results'])
                    param = ['Ro[m]', 'Bo', 'sigma',
                             'gamma[mN/m]', 'Wo', 'Ne',
                             'V[mm3]', 'Ro[px]', 'alpha']
                    for i in range(0, len(self.result)):
                        _, value = self.result[i]
                        writer.writerow([param[i], value])

        debug = LabelFrame(right, text='Debugging')
        export = Button(debug, text='Export results', command=exp_res)
        export_surf = Button(debug, text='Export contours', command=exp_surf)
        adjust_rad = Button(debug, text='Adjust radius', command=adjust)
        debug.pack(pady=5)
        export.pack(padx=141)
        export_surf.pack()
        adjust_rad.pack()

        # Reset
        button_reset = Button(right, text='New calculation', command=reset)
        button_reset.pack(pady=5)

        # Inicio del programa
        win.mainloop()


Interface()
