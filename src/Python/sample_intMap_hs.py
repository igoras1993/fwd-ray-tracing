# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
from fwdraytracing.geo import *
from fwdraytracing.utils import *


crystal = Scintillator('u v',
                       '[sin(u)*cos(v),sin(u)*sin(v),cos(u)]', 
                        ((0,np.pi/2 +0.01),(0,2*np.pi)), 
                        precision_cnt=41, 
                        alpha = 1, R=0.95)
                        
map1, ax1 = crystal.screenMap(spatial_prec=20, direct_prec=300, maxref=50, pixels=200)