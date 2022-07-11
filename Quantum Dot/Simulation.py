from matplotlib import ticker
import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def disk_capacitance(R, eps_r):
    return 8*eps_r*eps_0*R

hbar = 1.05457182e-34
k_b = 1.380649e-23 #m^2 kg/s^2/K
v_f = 1e6
dos_slope = 2/pi/hbar**2/v_f**2

m_e = 9.1093837e-31
R = 5e-8
q_e = 1.602e-19
eps_0 = 8.854e-12
eps_SiC = 9.66 # https://www.microwaves101.com/encyclopedias/silicon-

mu_left = 5e-3 * 1.6e-19 # 5meV
mu_right = 0
V_sd = 5e-3 # source-drain voltage of 5 mV

# 4 times self disk capacitance as an approximation
total_capacitance = 4 * (disk_capacitance(R, eps_SiC))

# setting up prediction of energy level from N
A = 4*(R)**2
c= 2*hbar*v_f*sqrt(pi/A)
E_N = lambda x, c=c: c*np.sqrt(x)

def mu_dot(N):
    return E_N(N) - 1/2 * q_e**2/total_capacitance

plt.plot(mu_dot(np.linspace(1,100)))
plt.plot(mu_dot(np.linspace(1,100))+ 1/2 * q_e**2/total_capacitance)
plt.show()
print(total_capacitance)

