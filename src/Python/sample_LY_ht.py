# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from fwdraytracing.geo import *
from fwdraytracing.utils import *
from time import clock

Rt = 2.5
rads = np.linspace(0,3.5, 40)[1:]
rads_tor = np.sqrt((rads**3)*2.0/(3*np.pi*Rt))
base = '[(2.5 + %f*cos(u))*cos(v),(2.5 +%f*cos(u))*sin(v),%f*sin(u)]'
argstr = []
for radii in rads_tor:
    argstr.append( base % (radii, radii, radii))

ress = np.zeros(len(rads_tor))
times = np.zeros(len(rads_tor))

for i in xrange(len(ress)):
    arg = argstr[i]
    t0 = clock()
    crystal = Scintillator('u v', arg,((-0.01,np.pi+0.01),(0,2*np.pi)))
    if rads_tor[i] < 0.30:
        LY = crystal.lightYield(spatial_prec = 40, direct_prec = 400)
    else:
        LY = crystal.lightYield(spatial_prec = 20, direct_prec = 400)
    t1 = clock()
    ress[i] = LY
    times[i] = t1-t0
    print (i, times[i])

    


with open('results_torus.data','w') as f:
    for i in xrange(len(ress)):
        f.write(str(rads[i]) + '\t')
        f.write(str(rads_tor[i]) + '\t')
        f.write(str(ress[i]) + '\t')
        f.write(str(times[i]) + '\n')

data = np.loadtxt('results_torus.data')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data[:,1], data[:,2])
ax.set_xlabel("R-torus")
ax.set_ylabel("LY(R)")