# -*- coding: utf-8 -*-
import numpy as np
from geo import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def unique_rows(arr):
    ind = np.lexsort(arr.T)
    ind = ind[np.concatenate(([True],np.any(np.logical_not(np.isclose(arr[ind[1:]],arr[ind[:-1]])),axis=1)))]
    return arr[ind].copy()

def seedDirections(Npoints):
    """
    Generates a set of random directions that are uniformly distributed on a
    unit sphere. For detail math see: http://mathworld.wolfram.com/SpherePointPicking.html
    
    seedDirections(Npoints) -> arr
     Npoints - number of points to seed, should be in for of C^2,
              otherwise number of seeded points will be n = floot(sqrt(N))^2
     arr - numpy array of shape (n,3)
    """
    n = int(np.sqrt(Npoints))
    phiv = 2*np.pi*np.random.rand(n,n)
    thetav = np.arccos(2*np.random.rand(n,n) - 1)
    
    x_table = (np.sin(thetav)*np.cos(phiv)).flatten()
    y_table = (np.sin(thetav)*np.sin(phiv)).flatten()
    z_table = np.cos(thetav).flatten()
    
    return np.array([x_table, y_table, z_table]).T
    
def meshDirections(Npoints):
    """
    Generates a set of directions that are uniformly distributed on a
    unit sphere. For detail math see: http://mathworld.wolfram.com/SpherePointPicking.html
    
    seedDirections(Npoints) -> arr
     Npoints - number of points to seed, should be in for of C^2,
              otherwise number of seeded points will be n = floot(sqrt(N))^2
     arr - numpy array of shape (n,3)
    """
    n = int(np.sqrt(Npoints))
    phi = 2*np.pi*np.linspace(0,1,n)
    theta = np.arccos(2*np.linspace(0.001,1-0.001,n) - 1)
    phiv, thetav = np.meshgrid(phi,theta)
    x_table = (np.sin(thetav)*np.cos(phiv)).flatten()
    y_table = (np.sin(thetav)*np.sin(phiv)).flatten()
    z_table = np.cos(thetav).flatten()
    
    return unique_rows(np.array([x_table, y_table, z_table]).T)
    
def jitter(arr, size = 0.01):
    """
    """
    return arr + (np.random.rand(*(arr.shape)) - 0.5)*size

def meshVolume(model, N):
    """
    """
    
    n = N
    zmax = np.max(model.mesh[:,:,2])
    zmesh = np.linspace(0,zmax,n+1, endpoint = False)[1:]
    #DEBUG:zmesh = np.array([zmax/2])
    DEBUGCNT = 0

    x_max = np.max(model.mesh[:,:,0])
    x_min = np.min(model.mesh[:,:,0])
    y_max = np.max(model.mesh[:,:,1])
    y_min = np.min(model.mesh[:,:,1])
    
    xseed = np.linspace(x_min, x_max, n)
    yseed = np.linspace(y_min, y_max, n)
    
    xyseed = jitter(np.transpose([np.tile(xseed, len(yseed)), np.repeat(yseed, len(xseed))]))
    
    globalMesh = np.array([[0,0,0]])
    
    for zi in zmesh:
        lines = []
        for i in xrange(len(model.mesh)):
            u0a = (zi - model.mesh[i,2,2])/(model.mesh[i,1,2] - model.mesh[i,2,2])
            v0b = (zi - model.mesh[i,2,2])/(model.mesh[i,0,2] - model.mesh[i,2,2])
            uc =  (model.mesh[i,0,2] - zi)/(model.mesh[i,0,2] - model.mesh[i,1,2])
            vc =  (zi - model.mesh[i,1,2])/(model.mesh[i,0,2] - model.mesh[i,1,2])
            
            testA = 0 <= u0a <= 1
            testB = 0 <= v0b <= 1
            testC = (0 <= uc <= 1) and (0 <= vc <= 1)
            
            #swt = np.array([testA,testB,testC]).dot([0,2,4])
            
            if (testA and testB): #crossing u and v axis
                x1 = model.mesh[i,2,0] + (model.mesh[i,1,0]-model.mesh[i,2,0])*u0a
                y1 = model.mesh[i,2,1] + (model.mesh[i,1,1]-model.mesh[i,2,1])*u0a
                x2 = model.mesh[i,2,0] + (model.mesh[i,0,0]-model.mesh[i,2,0])*v0b
                y2 = model.mesh[i,2,1] + (model.mesh[i,0,1]-model.mesh[i,2,1])*v0b
                p = np.array([[x1, y1, zi],
                               [x2, y2, zi]])
                lines.append(p)
                
            elif (testA and testC): #crossing u and line
                x1 = model.mesh[i,2,0] + (model.mesh[i,1,0]-model.mesh[i,2,0])*u0a
                y1 = model.mesh[i,2,1] + (model.mesh[i,1,1]-model.mesh[i,2,1])*u0a
                x2 = model.mesh[i,2,0] + (model.mesh[i,1,0]-model.mesh[i,2,0])*uc + (model.mesh[i,0,0]-model.mesh[i,2,0])*vc
                y2 = model.mesh[i,2,1] + (model.mesh[i,1,1]-model.mesh[i,2,1])*uc + (model.mesh[i,0,1]-model.mesh[i,2,1])*vc
                p = np.array([[x1, y1, zi],
                               [x2, y2, zi]])
                lines.append(p)
                
            elif (testB and testC): #crossing v and line
                x1 = model.mesh[i,2,0] + (model.mesh[i,1,0]-model.mesh[i,2,0])*uc + (model.mesh[i,0,0]-model.mesh[i,2,0])*vc
                y1 = model.mesh[i,2,1] + (model.mesh[i,1,1]-model.mesh[i,2,1])*uc + (model.mesh[i,0,1]-model.mesh[i,2,1])*vc
                x2 = model.mesh[i,2,0] + (model.mesh[i,0,0]-model.mesh[i,2,0])*v0b
                y2 = model.mesh[i,2,1] + (model.mesh[i,0,1]-model.mesh[i,2,1])*v0b
                p = np.array([[x1, y1, zi],
                               [x2, y2, zi]])
                lines.append(p)
                
            else:
                pass
            
        
        #########################################################
        ######## lines now contains polygons
        npLines = np.array(lines)
        
        insideIdx = []
        
        for j in xrange(len(xyseed)):
            point = xyseed[j,:]
            crossCnt = 0
            for i in xrange(len(npLines)):
                tLine = (point[1] - npLines[i,1,1])/(npLines[i,0,1]-npLines[i,1,1])
                
                tp = npLines[i,1,0] + (npLines[i,0,0] - npLines[i,1,0])*tLine + point[0]
                
                if ((0 <= tp < np.inf) and ((0 < tLine <= 1) or np.isclose(tLine,0) or np.isclose(tLine,1))): #half-line (tp>0) and line section crossed (0<<1)
                    crossCnt += 1

            if (crossCnt % 2) == 1: #point is inside polygon
                insideIdx.append(j)
        
        insidePts = xyseed[insideIdx]
        ziPts = np.concatenate((insidePts,np.repeat(np.array([[zi]]),len(insidePts), axis = 0)), axis = 1)
        
        globalMesh = np.concatenate((globalMesh, ziPts), axis = 0)
        
        #print DEBUGCNT
        DEBUGCNT += 1
    #end for zi
    globalMesh = globalMesh[1:,:]
    
    return globalMesh
    
def zeroPoly(model, zero = 0):
    zi = zero
    lines = []
    for i in xrange(len(model.mesh)):
        u0a = (zi - model.mesh[i,2,2])/(model.mesh[i,1,2] - model.mesh[i,2,2])
        v0b = (zi - model.mesh[i,2,2])/(model.mesh[i,0,2] - model.mesh[i,2,2])
        uc =  (model.mesh[i,0,2] - zi)/(model.mesh[i,0,2] - model.mesh[i,1,2])
        vc =  (zi - model.mesh[i,1,2])/(model.mesh[i,0,2] - model.mesh[i,1,2])
        
        testA = 0 <= u0a <= 1
        testB = 0 <= v0b <= 1
        testC = (0 <= uc <= 1) and (0 <= vc <= 1)
        
        #swt = np.array([testA,testB,testC]).dot([0,2,4])
        
        if (testA and testB): #crossing u and v axis
            x1 = model.mesh[i,2,0] + (model.mesh[i,1,0]-model.mesh[i,2,0])*u0a
            y1 = model.mesh[i,2,1] + (model.mesh[i,1,1]-model.mesh[i,2,1])*u0a
            x2 = model.mesh[i,2,0] + (model.mesh[i,0,0]-model.mesh[i,2,0])*v0b
            y2 = model.mesh[i,2,1] + (model.mesh[i,0,1]-model.mesh[i,2,1])*v0b
            p = np.array([[x1, y1, zi],
                           [x2, y2, zi]])
            lines.append(p)
            
        elif (testA and testC): #crossing u and line
            x1 = model.mesh[i,2,0] + (model.mesh[i,1,0]-model.mesh[i,2,0])*u0a
            y1 = model.mesh[i,2,1] + (model.mesh[i,1,1]-model.mesh[i,2,1])*u0a
            x2 = model.mesh[i,2,0] + (model.mesh[i,1,0]-model.mesh[i,2,0])*uc + (model.mesh[i,0,0]-model.mesh[i,2,0])*vc
            y2 = model.mesh[i,2,1] + (model.mesh[i,1,1]-model.mesh[i,2,1])*uc + (model.mesh[i,0,1]-model.mesh[i,2,1])*vc
            p = np.array([[x1, y1, zi],
                           [x2, y2, zi]])
            lines.append(p)
            
        elif (testB and testC): #crossing v and line
            x1 = model.mesh[i,2,0] + (model.mesh[i,1,0]-model.mesh[i,2,0])*uc + (model.mesh[i,0,0]-model.mesh[i,2,0])*vc
            y1 = model.mesh[i,2,1] + (model.mesh[i,1,1]-model.mesh[i,2,1])*uc + (model.mesh[i,0,1]-model.mesh[i,2,1])*vc
            x2 = model.mesh[i,2,0] + (model.mesh[i,0,0]-model.mesh[i,2,0])*v0b
            y2 = model.mesh[i,2,1] + (model.mesh[i,0,1]-model.mesh[i,2,1])*v0b
            p = np.array([[x1, y1, zi],
                           [x2, y2, zi]])
            lines.append(p)
            
        else:
            pass
        
    npLines = np.array(lines)

    return npLines
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1) #[0,0,0]
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    
class Scintillator(object):
    def __init__(self, param_def=None, shape_def=None, ranges=None, precision_cnt = 40, alpha = 1, R=1):
        if not ((param_def is None) or (shape_def is None) or (ranges is None) or (precision_cnt is None)):
            self.bound = DifferentialManifold(param_def, shape_def, ranges)
            self.surface = SurfaceModel(precision_cnt)
            self.surface.addSheet(self.bound)
            self.surface.bakeMesh()

        self.alpha = alpha
        self.R = R
    
        self.globalMesh = None
        self._gMeshSpatPrec = 0
        self._gMeshDirPrec = 0
        
        #self.horzMap = []
        self.beta = None

    def loadRaw(self, raw):
        S = SurfaceModel(int(np.sqrt(len(raw))))
        S.fromRawMesh(raw)
        self.surface = S

        
    def lightYield(self, spatial_prec=15, direct_prec=100, maxref=20):
        if self.globalMesh is None:
            self.globalMesh = meshVolume(self.surface, spatial_prec)
            self._gMeshSpatPrec = spatial_prec
            self._gMeshDirPrec = direct_prec
        elif ((self._gMeshSpatPrec != spatial_prec) or (self._gMeshDirPrec != direct_prec)):
            self.globalMesh = meshVolume(self.surface, spatial_prec)
            self._gMeshSpatPrec = spatial_prec
            self._gMeshDirPrec = direct_prec

        phasePointsCnt = len(self.globalMesh) * len(seedDirections(self._gMeshDirPrec))
        lengths = np.zeros(phasePointsCnt)
        reflections = np.zeros(phasePointsCnt)
        
        #ranges = 
            
        
        lastEntry = 0;
        j = 0
        for initPos in self.globalMesh:
            initDirs = seedDirections(self._gMeshDirPrec)
            j += 1
            for initDir in initDirs:
                beam = Ray(initPos, initDir)
                beam.placeEnviorment(self.surface)
                beam.trace(maxref)
                if beam.reflectCnt != maxref:
                    #self.horzMap.append(beam.path[beam.reflectCnt+1])
                    lengths[lastEntry] = beam.pathLength()
                    reflections[lastEntry] = beam.reflectCnt
                    lastEntry += 1
        
        self.beta = np.mean(np.exp((-1)*self.alpha*lengths[:lastEntry+1])*(self.R**reflections[:lastEntry+1]))
        
        return self.beta

        
    def screenMap(self, spatial_prec=15, direct_prec=100, maxref=20, pixels = 20, pointSource = None, dirSeed = True, name = 'Screen intensity map'):
        if self.globalMesh is None:
            self.globalMesh = meshVolume(self.surface, spatial_prec)
            self._gMeshSpatPrec = spatial_prec
            self._gMeshDirPrec = direct_prec
        elif ((self._gMeshSpatPrec != spatial_prec) or (self._gMeshDirPrec != direct_prec)):
            self.globalMesh = meshVolume(self.surface, spatial_prec)
            self._gMeshSpatPrec = spatial_prec
            self._gMeshDirPrec = direct_prec
        
        if (pointSource is None):
            mesh = self.globalMesh
        else:
            mesh = pointSource        
        
        phasePointsCnt = len(mesh) * len(seedDirections(self._gMeshDirPrec))
        xy_hist = np.zeros((phasePointsCnt,3))
        weights = np.zeros(phasePointsCnt)
        
        zeroBound = zeroPoly(self.surface)
        rmin = np.min(zeroBound[:,:,0:2])
        rmax = np.max(zeroBound[:,:,0:2])
        ranges = [[rmin, rmax],
                  [rmin, rmax]]
        
        lastEntry = 0;
        
        for initPos in mesh:
            initDirs = seedDirections(self._gMeshDirPrec) if dirSeed else meshDirections(self._gMeshDirPrec)
            
            for initDir in initDirs:
                beam = Ray(initPos, initDir)
                beam.placeEnviorment(self.surface)
                beam.trace(maxref)
                if beam.reflectCnt != maxref:
                    xy_hist[lastEntry,0] = beam.path[beam.reflectCnt+1, 0]
                    xy_hist[lastEntry,1] = beam.path[beam.reflectCnt+1, 1]
                    xy_hist[lastEntry,2] = beam.path[beam.reflectCnt+1, 2]
                    weights[lastEntry] = np.exp((-1)*self.alpha*beam.pathLength())*(self.R**(beam.reflectCnt))
                    lastEntry += 1
        self.H, self.xedges, self.yedges = np.histogram2d(xy_hist[:(lastEntry+1),0], xy_hist[:(lastEntry+1),1], weights = weights[:(lastEntry+1)], 
                                                        bins = pixels, range = ranges)
        #self.H = self.H.T
        
        fig = plt.figure()
        fig.canvas.set_window_title(name)
        ax = fig.add_subplot(111, title = name)
        ax.set_aspect('equal')
        plt.imshow(self.H, interpolation = 'bilinear', origin = 'low', 
                   extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
        plt.colorbar()
        return fig, ax
        
def plotSurf(model, axes = False, z0 = True, color = 'g', alpha = 1):
    
    fig = plt.figure()
    fig.canvas.set_window_title('Surface mesh model')
    ax = fig.add_subplot(111, projection = '3d')
    
#    m = model.mesh.copy()
#    zlut = model.mesh[:,:,2] > 0.5
#    m[:,:,2] = zlut 
#    m[:,:,1] = zlut
#    m[:,:,0] = zlut
    for tri in model.mesh:
        lut = tri[:,2] >= 0
        ax.plot(tri[lut,0], tri[lut,1], tri[lut,2], color = color, alpha = alpha)
        if lut[0] and lut[-1]:
            ax.plot(tri[[0,-1],0], tri[[0,-1],1], tri[[0,-1],2], color = color, alpha = alpha)
    axisEqual3D(ax)        
    zeroBound = zeroPoly(model)
    if z0:
        for line in zeroBound:
            ax.plot(line[:,0], line[:,1], line[:,2], color = 'm')
    
    x_ranges = np.array([np.max(zeroBound[:,:,0])*1.2, np.min(zeroBound[:,:,0])*1.2])
    y_ranges = np.array([np.max(zeroBound[:,:,1])*1.2, np.min(zeroBound[:,:,1])*1.2])
    X,Y, = np.meshgrid(x_ranges, y_ranges)
    ax.plot_surface(X,Y,0, color = 'c', alpha = 0.2)
    
    if not axes:
        ax.set_axis_off()
    
    plt.show()
    return fig, ax
    
def sampleTrace(model, pos, dirs = None, ax = None, fig = None, maxRecursion = 15):
    
    if (ax is None) or (fig is None):
        figp, axp = plotSurf(model, axes = False, alpha = 0.3)
    else:
        figp = fig
        axp = ax
    axisEqual3D(axp)
    figp.canvas.set_window_title('Trajectories')
    
    if dirs is None:
        init_dirs = seedDirections(9)
    elif type(dirs) is int:
        init_dirs = seedDirections(dirs)
    else:
        init_dirs = dirs
    
    for direction in init_dirs:
        beam = Ray(pos, direction)
        beam.placeEnviorment(model)
        beam.trace(maxRecursion)
        axp.plot(beam.path[:beam.reflectCnt+2,0], beam.path[:beam.reflectCnt+2,1], beam.path[:beam.reflectCnt+2,2], color = 'b', alpha = 0.6)
        axp.scatter(beam.path[1:beam.reflectCnt+2,0], beam.path[1:beam.reflectCnt+2,1], beam.path[1:beam.reflectCnt+2,2], color = 'b', alpha = 0.6)
    
    plt.show()
    
    return figp, axp
            
            