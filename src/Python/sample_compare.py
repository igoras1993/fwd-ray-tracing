# -*- coding: utf-8 -*-

import numpy as np
from fwdraytracing.geo import *
from fwdraytracing.utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import dblquad

def podcalk(H,z0,theta):
    if 0 <= theta < np.pi:
        o = np.exp(-(2*H - z0)/np.abs(np.sin(theta)))*np.abs(np.cos(theta))
    elif np.pi <= theta < 2*np.pi:
        o = np.exp(-(z0)/np.abs(np.sin(theta)))*np.abs(np.cos(theta))
    return o

def LY(H):
    f = lambda z0,th: podcalk(H, z0,th)
    
    return dblquad(f, a = 0, b = 2*np.pi, gfun = lambda th: 0, hfun = lambda th: H)[0]/(4*H)
    

data_sphere = np.loadtxt("results.data")
data_torus = np.loadtxt("results_torus.data")

rads = data_sphere[:,0]
rads_tor = data_torus[:,1]
heights = np.sqrt(2*np.pi/3)*rads

Ly3D = np.array([LY(H) for H in heights]) 
Lysphere = data_sphere[:,1]
Lytorus = data_torus[:,2]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(rads, Ly3D, color='b')
ax.plot(rads, Lysphere, color='g')
ax.plot(rads, Lytorus, color='r')

ax.set_xlabel("R-sphere")
ax.set_ylabel("LY(R)")

blue_patch = mpatches.Patch(color='blue', label='Cube LY')
green_patch = mpatches.Patch(color='green', label='Half-Sphere LY')
red_patch = mpatches.Patch(color='red', label='Half-Torus LY')
plt.legend(handles = [blue_patch, green_patch, red_patch])

plt.show()

