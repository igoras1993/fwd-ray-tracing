# -*- coding: utf-8 -*-

import numpy as np
from fwdraytracing.geo import *
from fwdraytracing.utils import *


crystal = Scintillator('u v',
                       '[sin(u)*cos(v),sin(u)*sin(v),cos(u)]', 
                        ((0,np.pi/2 +0.01),(0,2*np.pi)), 
                        precision_cnt=41, 
                        alpha = 1, R=1)

dirs = np.array([[0,0,1]])
dirshor = np.array([np.cos(np.linspace(0, 2*np.pi,9, endpoint=False)), np.sin(np.linspace(0, 2*np.pi,9,endpoint=False)), np.zeros(9)]).T

fig, ax = sampleTrace(crystal.surface, pos = np.array([0,0.5,0.5]), dirs = 4, maxRecursion = 99)

fig2, ax2 = sampleTrace(crystal.surface, pos = np.array([0.6,0.6,0.01]), dirs = dirs, maxRecursion = 99)
fig3, ax3 = sampleTrace(crystal.surface, pos = np.array([0.7,0.7,0.01]), dirs = dirs, maxRecursion = 99)

fig4, ax4 = sampleTrace(crystal.surface, pos = np.array([0,0,0.5]), dirs = dirshor, maxRecursion = 99)

#crystal.screenMap(direct_prec=100000, maxref=99, pointSource=np.array([[0.1,0.1,0.2]]), pixels = 100)