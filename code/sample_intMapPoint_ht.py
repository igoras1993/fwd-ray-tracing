# -*- coding: utf-8 -*-


import numpy as np
from fwdraytracing.geo import *
from fwdraytracing.utils import *


crystal = Scintillator('u v',
                       '[(2.5 + cos(u))*cos(v),(2.5 + cos(u))*sin(v),sin(u)]', 
                        ((-0.01,np.pi+0.01),(0,2*np.pi)),
                        precision_cnt=41, 
                        alpha = 1, R=1)
                        
map1, ax1 = crystal.screenMap(direct_prec=300000, maxref=50, pixels=200, pointSource=np.array([[0,2.2,.5]]))