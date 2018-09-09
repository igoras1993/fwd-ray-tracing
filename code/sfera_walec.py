from fwdraytracing.geo import *
from fwdraytracing.utils import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sferaD = DifferentialManifold('u v','[sin(u)*cos(v),sin(u)*sin(v),cos(u)]',((0,np.pi),(np.pi,2*np.pi)))
sferaM = SurfaceModel(30)
sferaM.addSheet(sferaD)
sferaM.bakeMesh()

walecD = DifferentialManifold('u v','[cos(u),sin(u),v]',((0,np.pi),(-1,1)))
walecM = SurfaceModel(30)
walecM.addSheet(walecD)
walecM.bakeMesh()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

for tri in walecM.mesh:
    ax.plot(tri[:,0],tri[:,1],tri[:,2], color = 'g', alpha = 0.9)
for tri in sferaM.mesh:
    ax.plot(tri[:,0],tri[:,1],tri[:,2], color = 'b', alpha = 0.9)

plt.show()


print sferaM.mesh.shape
