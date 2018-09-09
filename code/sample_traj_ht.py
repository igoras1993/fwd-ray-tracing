# -*- coding: utf-8 -*-

import numpy as np
from fwdraytracing.geo import *
from fwdraytracing.utils import *

R = 2.5
r = 1
crystal = Scintillator('u v',
                       '[(2.5 + cos(u))*cos(v),(2.5 + cos(u))*sin(v),sin(u)]', 
                        ((-0.01,np.pi+0.01),(0,2*np.pi)), 
                        precision_cnt=41, 
                        alpha = 1, R=1)

dirs = np.array([[-0.35,0,0.9]])
dirshor = np.array([np.cos(np.linspace(0, 2*np.pi,9, endpoint=False)), np.sin(np.linspace(0, 2*np.pi,9,endpoint=False)), np.zeros(9)]).T

fig1, ax1 = sampleTrace(crystal.surface, pos = np.array([0,3.49,0.01]), dirs = dirs, maxRecursion = 99)

fig2, ax2 = sampleTrace(crystal.surface, pos = np.array([0,2.5,0.5]), dirs = dirshor, maxRecursion = 99)
