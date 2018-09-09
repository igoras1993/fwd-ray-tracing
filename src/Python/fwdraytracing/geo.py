# -*- coding: utf-8 -*-

import sympy as sym
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import ctrace as ct


def primitives(prim, params):
    if prim == 'cuboid':
        A = params[0] #x
        L = params[1] #y
        H = params[2] #z
        
        p =   np.array([[0.5*A, -.5*L, 0],
                        [0.5*A, 0.5*L, 0],
                        [-.5*A, 0.5*L, 0],
                        [-.5*A, -.5*L, 0],
                        [0.5*A, -.5*L, H],
                        [0.5*A, 0.5*L, H],
                        [-.5*A, 0.5*L, H],
                        [-.5*A, -.5*L, H]])
        tris = np.array([[p[0], p[3], p[4]], [p[7], p[4], p[3]],
                         [p[0], p[1], p[4]], [p[5], p[4], p[1]],
                         [p[1], p[2], p[5]], [p[6], p[5], p[2]],
                         [p[7], p[6], p[2]], [p[3], p[2], p[7]],
                         [p[5], p[4], p[6]], [p[4], p[7], p[6]]])
        return tris.copy()



class Ray(object):
    """
    
    """
    def __init__(self, init_pos, init_dir, regCnt = 100):
        self.init_pos = init_pos
        self.init_dir = init_dir
        self.pos = init_pos.copy()
        self.dir = init_dir.copy()
        self.path = np.zeros((regCnt,3))
        self.reflectCnt = 0
        
        self.model = None
        self._lastRefIdx = None
        self._triLen = None
        self._regCnt = regCnt
        
        
    def placeEnviorment(self, model):
        self.model = model
        self._lastRefIdx = len(model.mesh)
        self._triLen = len(self.model.mesh)
        
    def reflect(self):
        """
        """
        cres = ct.reflect(self.pos, self.dir, self.model.mesh, self.model.normal, self._lastRefIdx)
        if not (cres is None):
            self.pos = cres[0]
            self.dir = cres[1]
            self._lastRefIdx = cres[2]
            return cres[0], cres[1]
            
    def trace(self, maxRecursion = 100):
        """
        """
        if maxRecursion >= self._regCnt:
            raise ValueError("maxRecursion can't be greater than regCnt (%d)" % self._regCnt)

        self.path[0,:] = self.pos.copy()
        self.reflectCnt = 0
        for i in xrange(maxRecursion):
            self.reflect()
            self.path[i+1,:] = self.pos.copy()
            
            if np.isclose(self.pos[2], 0):
                break
            self.reflectCnt += 1
        
        return self.reflectCnt
        
    def pathLength(self):
        """
        """
        divs = self.path[:(self.reflectCnt+1),:] - self.path[1:(self.reflectCnt+2),:]
        return np.sqrt(np.square(divs).sum(1)).sum()


class SurfaceModel(object):
    """
    Class to store model of surface for further Ray Tracing purposes
    members:
    self.precision
    self.sheet
    self.mesh
    self.normal
    """
    def __init__(self, precision):
        """

        """
        #self.sheets = []
        #self.bounds = []
        self.precision = precision
        
    def addSheet(self, manifold):
        """
        
        """
        self.sheet = manifold

        
                
    def bakeMesh(self):
        """
        
        """
        #lets set the entries
        uTable = np.linspace(self.sheet.u_range[0],self.sheet.u_range[1],self.precision)
        vTable = np.linspace(self.sheet.v_range[0],self.sheet.v_range[1],self.precision)
        uLen = len(uTable)
        vLen = len(vTable)
        pts = np.zeros((uLen*vLen,3))
        uvLUT = np.zeros((uLen*vLen,2))
        
        #next, lets calculate a mesh points
        for i in xrange(uLen):
            for j in xrange(vLen):
                uvLUT[vLen*i + j, 0] = uTable[i]
                uvLUT[vLen*i + j, 1] = vTable[j]
                pts[vLen*i + j,:] = self.sheet.nR(uTable[i], vTable[j])[0]
        #identify duplicates and extract uniques:
        
        ##idx = unique_rows(pts)
        #idx now contains indices of unique rows
        #overwrite pts and LUT:
        ##pts = pts[idx,:]
        ##uvLUT = uvLUT[idx,:]
        #now perform Delaunay triangulation on flat parameter space:
        uvDelTri = Delaunay(uvLUT)
        uvTri = uvLUT[uvDelTri.simplices]
        triLen = len(uvTri)
        surfTri = np.zeros((triLen,3,3))
        normalGrid = np.zeros((triLen,3))
        for i in xrange(triLen):
            surfTri[i,0,:] = self.sheet.nR(*uvTri[i,0,:])[0]
            surfTri[i,1,:] = self.sheet.nR(*uvTri[i,1,:])[0]
            surfTri[i,2,:] = self.sheet.nR(*uvTri[i,2,:])[0]
            normalGrid[i,:] = np.cross((surfTri[i,2,:] - surfTri[i,0,:]),
                                (surfTri[i,2,:] - surfTri[i,1,:]))
            normalGrid[i,:] = normalGrid[i,:]/np.linalg.norm(normalGrid[i,:])
        self.mesh = surfTri
        self.normal = normalGrid
        
    def fromRawMesh(self, raw):
        surfTri = raw.copy()
        triLen = len(surfTri)
        normalGrid = np.zeros((triLen, 3))
        for i in xrange(triLen):
            normalGrid[i,:] = np.cross((surfTri[i,2,:] - surfTri[i,0,:]),
                                (surfTri[i,2,:] - surfTri[i,1,:]))
            normalGrid[i,:] = normalGrid[i,:]/np.linalg.norm(normalGrid[i,:])
        self.mesh = surfTri
        self.normal = normalGrid


#==============================================================================
#==============================================================================
class DifferentialManifold(object):
    """
    General purpose class to store a curved surfaces.

    ===========================================================================
    
    attributes:
    ========
    self.u - symbol for storing first variable
    self.v - symbol for storing second variable
    self.parameterization - sympy Matrix storing a guiding vector in its 
        only row. Matrix elements are regular sympy expressions 
        (/w self.u, self.v as a symbols)
    self.u_range - two element tuple storing range: (u_min, u_max)
    self.v_range - two element tuple storing range: (v_min, v_max)
    self.f_tangent_bundle - stores result of self.tangent_bundle() 
    self.f_normal_bundle - stores result of self.normal_bundle() 
    self.f_riemann_metric - stores result of self.riemann_metric()
    self.f_hesse_matrix - stores result of self.hesse_matrix()

    ===========================================================================
    
    ========
    NEXT METHODS ARE ATRIBUTES WHICH ARE JUST LAMBDA FUNCTIONS HANDLES
    TO USE FOR FAST NUMERICAL PERFORMANCE
    ========    
    self.nR(u, v) -> r
     u,v - arguments (numerical) for guiding vector
     r - numpy 2D matrix (row) of coordinates of guiding vector. 
         e.g. np.array([[x,y,z]])
    ========
    self.nDuR(u, v) -> DuR
    u,v - arguments (numerical) for guiding vector
     DuR - numpy 2D matrix (row) of coordinates of first p. der. (u) of guiding 
         vec. e.g. np.array([[Dux,Duy,Duz]])
    ========        
    self.nDvR(u, v) -> DvR
     u,v - arguments (numerical) for guiding vector
     DvR - numpy 2D matrix (row) of coordinates of first p. der. (v) of guiding
         vec. e.g. np.array([[Dvx,Dvy,Dvz]])
    ========
    self.nNR(u, v) -> N
     u,v - arguments (numerical) for guiding vector
     N - numpy 2D matrix (row) of coordinates of ortogonal vector to the plane 
         in point r(u,v) e.g. np.array([[Nx,Ny,Nz]])
    ========
    self.nDuuR(u, v) -> DuuR
     u,v - arguments (numerical) for guiding vector
     DuuR - numpy 2D matrix (row) of coordinates of second p. der. (u,u) of 
         guiding vec. e.g. np.array([[Duux,Duuy,Duuz]])
    ========
    self.nDuvR(u, v) -> DuvR
     u,v - arguments (numerical) for guiding vector
     DuvR - numpy 2D matrix (row) of coordinates of second p. der. (u,v) of 
         guiding vec. e.g. np.array([[Duvx,Duvy,Duvz]]). 
     Note: DuvR = DvuR, so there is no need of defining func. for DvuR
    ========
    self.nDvvR(u, v) -> DvvR
     u,v - arguments (numerical) for guiding vector
     DuuR - numpy 2D matrix (row) of coordinates of second p. der. (v,v) of 
         guiding vec. e.g. np.array([[Dvvx,Dvvy,Dvvz]])
    ===========================================================================
    """
    def __init__(self, args_str, param_str, ranges_tpl):
        """
        __init__(self, args_str, param_str, ranges_tpl)
         args_str - string of two named variables separated by space e.g. 'u v'
         param_str - string of guiding vector with free variables given in args_str,
            e.g. for unit sphere: '[sin(u)*cos(v), sin(u)*sin(v), cos(v)]'
         ranges_tpl - tuple of tuples, storing ranges for free variables, 
            e.g. ((-3.14,3.14),(0,6.28))
        """
        self.u, self.v = sym.symbols(args_str)
        self.parameterization = sym.Matrix(sym.sympify(param_str)).T
        self.u_range = ranges_tpl[0]
        self.v_range = ranges_tpl[1]
        
        self.f_tangent_bundle = self.tangent_bundle()
        self.f_normal_bundle = self.normal_bundle()
        self.f_riemann_metric = self.riemann_metric()
        self.f_hesse_matrix = self.hesse_matrix()
        
        self.nR = sym.lambdify((self.u, self.v), self.parameterization[0,:])
        self.nDuR = sym.lambdify((self.u, self.v), self.f_tangent_bundle[0,:])
        self.nDvR = sym.lambdify((self.u, self.v), self.f_tangent_bundle[1,:])
        self.nNR = sym.lambdify((self.u, self.v), self.f_normal_bundle[0,:])
        self.nDuuR = sym.lambdify((self.u, self.v), self.f_hesse_matrix[0,0])
        self.nDvvR = sym.lambdify((self.u, self.v), self.f_hesse_matrix[1,1])
        self.nDuvR = sym.lambdify((self.u, self.v), self.f_hesse_matrix[0,1])
        
    def bring(self, u, v):
        """
        bring(self, u, v) -> point
         u,v - numbers in range defined in .u_range, .v_range
         point - numpy array of coordinates coresponding to given u,v        
        """
        if not ((self.u_range[0] <= u <= self.u_range[1]) 
                and
                (self.v_range[0] <= v <= self.v_range[1])):
            raise ValueError(
            'Arguments out of range: u:{0}; v:{1}'.format(self.u_range,self.v_range))
            
        evParameterization = self.parameterization.evalf(subs = {self.u : u, self.v : v})
        return np.array(evParameterization.tolist()).astype(np.float64)
    
    def tangent_bundle(self):
        """
        tangent_bundle(self) -> t_vecs
         t_vecs - sympy Matrix with rows as a tangent vectors to surface (partial 
             derivatives of a guiding vector along u and v). Matrix elements are 
             just regular sympy expressions (w/ u,v as a symbols)
        """
        DuR = sym.diff(self.parameterization, self.u)
        DvR = sym.diff(self.parameterization, self.v)
        return sym.Matrix([DuR,DvR])
        
    def normal_bundle(self):
        """
        normal_bundle(self) -> n_vec
         n_vec - sympy Matrix with only one row as a perpendicular vector to the 
             given surface. Note that vector is not normalized, it has a lenght 
             equal to the surface curvature (DuR x DvR). Matrix elements are 
             just regular sympy expressions (w/ u,v as a symbols)
        """
        DuR = sym.diff(self.parameterization, self.u)
        DvR = sym.diff(self.parameterization, self.v)
        normal = DuR.cross(DvR)
        return sym.simplify(normal)
        
    def hesse_matrix(self):
        """
        hesse_matrix(self) -> H
         H - hesse matrix (second derivative matrix) of a guiding vector. 
             This is actually a sympy 2D matrix which entries are as well 
             a sympy 2D matrices. E.g. the second partial derivative of R(u,v)
             after u and v will be H[0,1].
        """
        DuR = sym.diff(self.parameterization, self.u)
        DvR = sym.diff(self.parameterization, self.v)
        DuuR = sym.simplify(sym.diff(DuR, self.u))
        DvvR = sym.simplify(sym.diff(DvR, self.v))
        DuvR = sym.simplify(sym.diff(DuR, self.v))
        H = sym.Matrix([[DuuR,DuvR],[DuvR,DvvR]])
        return H
    def riemann_metric(self):
        """
        riemann_metric(self) -> gij
         gij - sympy Matrix equal to the riemann metric tensor of given surface. 
             Matrix elements are just regular sympy expressions (w/ u,v as a symbols)
        """
        DuR = sym.diff(self.parameterization, self.u)
        DvR = sym.diff(self.parameterization, self.v)
        guu = DuR.dot(DuR)
        gvv = DvR.dot(DvR)
        guv = DuR.dot(DvR)
        g = sym.Matrix([
            [guu, guv],
            [guv, gvv]]
            )
        return sym.simplify(g)

    def __qDist(self, x, q):
        """
        __qDist(self, x, q) -> dist
        to use for fast numerical performance
         x - array-like of arguments e.g. [u,v]
         q - numpy array repr. 3D point (x,y,z)
         dist - euclidean distance between r(u,v) and q
        """
        return np.sqrt((self.nR(*x)[0]-q).dot((self.nR(*x)[0]-q)))
        
    def __qGrad(self, x, q):
        """
        __qGrad(self, x, q) -> grad
        to use for fast numerical performance
         x - array-like of arguments e.g. [u,v]
         q - numpy array repr. 3D point (x,y,z)
         grad - numpy array representing gradient of a |r(u,v) - q|
        """
        return np.array(
                        (
                        ((self.nR(*x)[0]-q).dot(self.nDuR(*x)[0]))
                        /(np.sqrt((self.nR(*x)[0]-q).dot((self.nR(*x)[0]-q)))),
                        ((self.nR(*x)[0]-q).dot(self.nDvR(*x)[0]))
                        /(np.sqrt((self.nR(*x)[0]-q).dot((self.nR(*x)[0]-q))))
                        ))
    def __qHesse(self, x, q):
        """
        __qHesse(self, x, q) -> hessian
        to use for fast numerical performance
         x - array-like of arguments e.g. [u,v]
         q - numpy array repr. 3D point (x,y,z)
         hessian - numpy 2d array representing hesse matrix of a d =|r(u,v) - q|,
             ((Duud,Duvd),(Dvud,Dvvd))
        """
        qr = self.nR(*x)[0] - q
        dqr = np.sqrt(qr.dot(qr))
        DuR = self.nDuR(*x)[0]
        DvR = self.nDvR(*x)[0]
        
        Duudqr = (((np.power(dqr,2)*(DuR.dot(DuR)+qr.dot(self.nDuuR(*x)[0])))
                -(qr.dot(DuR))*(qr.dot(DuR)))
                / np.power(dqr,3))
                
        Duvdqr = (((np.power(dqr,2)*(DuR.dot(DvR)+qr.dot(self.nDuvR(*x)[0])))
                -(qr.dot(DuR))*(qr.dot(DvR)))
                / np.power(dqr,3))
                
        Dvvdqr = (((np.power(dqr,2)*(DvR.dot(DvR)+qr.dot(self.nDvvR(*x)[0])))
                -(qr.dot(DvR))*(qr.dot(DvR)))
                / np.power(dqr,3))
        return np.array(((Duudqr,Duvdqr),(Duvdqr,Dvvdqr)))

    def surface_point(self, q, start = 'def', eps = 0.00005):
        """
        surface_point(self, q, start = 'def', eps = 0.00005) -> x,system
        This method computes the nearest point x from q on the surfrace, and 
        local orthonolmal coordiate system 'system' at this point
         q - a numpy 1D array repr. point in 3D from which distance is calculated
         start - starting point parameters (u,v) as array-like object. If not
             given it will be a random (u,v) in its ranges.
         eps - precision goal. Default eps = 0.00005
         x - numpy array of calculated parameters [u,v]
         system - numpy 2D array. 1st row is a first tangent vector at the 
             calculated point, 2nd row is the second tangent vector, 3rd row 
             is normal vector at the calculated point. The system is orthonormal
        """
        # computing nearest point
        x = np.zeros(2)
        if start == 'def':
            x[0] = self.u_range[0] + np.random.rand(1)*(self.u_range[1]-self.u_range[0])
            x[1] = self.v_range[0] + np.random.rand(1)*(self.v_range[1]-self.v_range[0])
        else:
            x = np.copy(start)
        fun = lambda arg: self.__qDist(arg,q)
        grad = lambda arg: self.__qGrad(arg,q)
        hesse = lambda arg: self.__qHesse(arg,q)
        res = minimize(fun, x, method='Newton-CG', jac=grad, hess=hesse, options = {'disp':True, 'xtol':eps})
        
        # local othonolmal coordinate system:
        ort = self.nNR(*(res.x))[0]
        n = ort/np.sqrt(ort.dot(ort))
        # according to Erich Hartmann:
        if (n[0] > 0.5 or n[1] > 0.5):
            t1 = np.array((n[1],-n[0],0))
        else:
            t1 = np.array((-n[2],0,n[0]))
        t1 = t1/np.sqrt(t1.dot(t1))
        t2 = np.cross(n,t1)
        
        return res.x,np.array((t1,t2,n))
        

