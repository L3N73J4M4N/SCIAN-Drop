from select_ROI import Draw
from ift import SurfaceTension
import cv2 as cv
from utility_ift import *
import numpy as np
import matplotlib.pyplot as plt

num = 2
string = r"C:\Users\matia\OneDrive - Universidad de Chile\Escritorio\imagenes curso\Medio" + str(num) + ".bmp"
img = cv.imread(string)
coord_drop = [(501, 477), (1099, 1030)]
param = [1.93, 1.0066, 30]

i_conj = range(0, 460, 10)
gamma_conj = np.zeros(len(i_conj))
alpha_conj = np.zeros(len(i_conj))
j = 0
for i in i_conj:
    coord_needle = [(600, i), (946, 460)]
    ift = SurfaceTension(param, img)
    needle, rad, alpha = ift.needle(coord_needle)
    drop, radius = ift.drop(coord_drop)
    bond, sigma, bond_number = ift.bond_sigma()
    gamma = ift.solver()
    alpha_conj[j] = alpha
    gamma_conj[j] = gamma
    if j == 0 or j == len(i_conj) // 2 or j == len(i_conj) - 1:
        plt.imshow(bond)
        plt.title(r'$\alpha = $' + str(alpha)[0: 9] + 'e-06, ' + 'γ = ' + str(gamma)[0: 9])
        plt.show()
    j += 1

mean1 = np.mean(alpha_conj)
std1 = np.std(alpha_conj)
mean2 = np.mean(gamma_conj)
std2 = np.std(gamma_conj)
print('mean alpha = ', mean1)
print('std alpha = ', std1)
print('mean gamma = ', mean2)
print('std gamma = ', std2)


# plt.imshow(bond)
# plt.title(r'$\alpha = $' + str(alpha))
# plt.show()
# print('α = ', alpha)
# print('Ro:', rad, '[px]', '-->', radius * 1000, '[mm]')
# print('γ = ', gamma)


plt.plot(i_conj, 5 * 10 ** 3 * alpha_conj, '.', label=r'$\alpha$')
plt.plot(i_conj, gamma_conj, '.', color='g',  label=r'$\gamma$')
# plt.title('medio ' + str(num) + r': $\alpha = $' + str(np.round(mean1, 12)) + r' $\pm$ ' + str(np.round(std1, 12)))
# plt.show()
plt.plot(i_conj, np.ones(len(i_conj)) * mean2, '--', label=r'$\bar{\gamma}$')
plt.legend()
plt.title(r'Tendencia de $5 \cdot \alpha \times 10^{4}$ y $\gamma$')

plt.show()
