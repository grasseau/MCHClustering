#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 15, 2020 4:32:19 PM$"

"""
Main differences with v1
- Reorganize the implementation structure to deal with 2 or more grids
"""
import sys, traceback
import inspect
import numpy as np
import math

import matplotlib.pyplot as plt
import plotUtil as plu

DataType = np.float32
InternalType = np.float64

TwoPi = 2 * np.pi
SqrtTwoPi = np.sqrt( TwoPi )

def printGM( str, w, mu, var, firstItems=3, printVar=False):
  K = w.size
  if firstItems == -1:
    firstItems = K
  print(str)
  K0 = min( K, firstItems)
  if printVar:
    print("   k    w     mux   muy   varx   vary" )
  else:
    print("   k    w     mux   muy" )
  for k in range(K0):
    if printVar:
      print("  {:2d} {:4.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3f}".format( k, w[k], mu[k,0], mu[k,1], var[k,0], var[k,1] ) )
    else:
      print("  {:2d} {:4.3f} {:5.3f} {:5.3f}".format( k, w[k], mu[k,0], mu[k,1] ) )
  if K0 < K:
    print("  ...")
    for k in range(K-1,K):
      if printVar:
        print("  {:2d} {:4.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3f}".format( k, w[k], mu[k, 0], mu[k,1], var[k,0], var[k,1] ) )
      else:
        print("  {:2d} {:4.3f} {:5.3f} {:5.3f}".format( k, w[k], mu[k,0], mu[k,1] ) )
  return

def processEMClusterV4( x0, y0, dx0, dy0, ch0, 
                        x1, y1, dx1, dy1, ch1, 
                        cID ):
    discretizedPDF = False
    em = EM2D( discretizedPDF )
    # ???
    grids = []
    grid0 = Grid2D( 3, 3)
    grids.append( grid0 )
    grid1 = Grid2D( 3, 3)
    grids.append( grid1 )

    print("########### Cluster cID=", cID)
    
    if ( x0.shape[0] == 0): return ([],[],[])

    xm, xsig = plu.getBarycenter( x0, x1, ch0, ch1)
    ym, ysig = plu.getBarycenter( y0, y1, ch0, ch1)
    print( "Barycenter [x, y, sigx, sigy]", xm, ym, xsig, ysig )

    # Normalization
    ch0 = ch0 / (4*dx0*dy0)
    ch1 = ch1 / (4*dx1*dy1)
    s = np.sum( ch0)
    ch0 = ch0 /s
    s = np.sum( ch1)
    ch1 = ch1 /s
    #
    xy = []; dxy = []; z = []
    xy.append( np.array( [x0, y0] ) )
    xy.append( np.array( [x1, y1]) )
    dxy.append( np.array( [dx0, dy0]) )
    dxy.append( np.array( [dx1, dy1] ) )
    z.append( np.array( ch0 ) )
    z.append( np.array( ch1) )
    for g in range(2):
      grids[g].setGridValues( xy[g], dxy[g], z[g] )
    wi = np.array( [1.0], InternalType )
    mui = np.array( [[xm,ym]], InternalType )
    vari = np.array( [[xsig*xsig, ysig*ysig]], InternalType )
    (wf, muf, varf) = em.weightedEMLoopOnGrids( grids,  wi, mui, vari, dataCompletion=False, plotEvery=0)
    print( "sum charge each cathodes", np.sum(ch0), np.sum(ch1) )
    # ??? print "Cathodes charge input", clusters.cFeatures["ChargeCath0"][i], clusters.cFeatures["ChargeCath1"][i]
    # print "Clustering/Fitting X, Y ", clusters.cFeatures["X"][i], clusters.cFeatures["Y"][i]
    print( "Barycenter [x, y, sigx, sigy]", xm, ym, xsig, ysig )
    return wf, muf, varf, xm, xsig, ym, ysig

class Grid2D:
    def __init__(self, nX, nY, w0=[], mu0=[], var0=[], startX=0.0, startY=0.0, endX=1.0, endY=1.0 ):
        self.nX = nX; self.nY = nY
        if (nX == 1 ):
            deltaX = endX - startX
        else:
            deltaX = (endX - startX) / (nX-1)
        if (nY == 1 ):
            deltaY = endY - startY
        else:          
            deltaY = (endY - startY) / (nY-1)
        # V3 Inv: self.deltaX = deltaX; self.deltaY = deltaY
        self.startX = startX; self.startY = startY
        self.endX = endX; self.endY = endY
        # Position xy at the center of the rectangle/pad
        self.xy = np.mgrid[ startX+deltaX*0.5: endX: deltaX, startY+deltaY*0.5: endY: deltaY].reshape(2,-1)

        # V3 Inv self.dxy = np.array( [deltaX, deltaY], dtype=InternalType)
        self.dxy = np.full( (self.xy.shape[1], self.xy.shape[0]), [0.5*deltaX, 0.5*deltaY] ).T
        if ( self.xy.shape[1] != nX*nY ):
            print( "Grid2D.__init__: problem xu.shape[1],  nX, nY", self.xy.shape[1], nX, nY )
        self.z = []
        if ( len(mu0) != 0 ):
          self.z = EM2D.generateMixedGaussians2D( self.xy,  self.dxy, w0, mu0, var0,  normalize=False)
        self.zSum = np.sum( self.z )
        return
    
    def setGridValues(self, xy, dxy, z):
        self.xy = xy
        self.dxy = dxy
        self.z = z
        self.nX = None;
        self.nY = None;
        self.startX = None; self.startY = None
        self.endX = None; self.endY = None

def getMin( x, xmin ):
    """
    x : np.array
    xmin or xmax : scalar
    """
    min_ = np.min(x);
    return max( xmin, min_ )

def getMax( x, xmax ):
    """
    x : np.array
    xmin or xmax : scalar
    """
    max_ = np.max(x);
    return max( xmax, max_ )

def getMinMax( x, xmin, xmax ):
    """
    x : np.array
    xmin or xmax : scalar
    """
    min_ = np.min(x);
    min_ = min( xmin, min_ )
    max_ = np.max(x);
    max_ = max( xmax, max_ )
    return min_, max_

class GMModel:
    @staticmethod
    def set1popId( verbose=False ):
        """
        Set (w, mu, sig) = ( w0, mu0, sig0 )
        Use to check
        """
        w0  = np.array( [1.0], dtype=DataType )
        mu0= np.array( [[0.0, 0.0]], dtype=InternalType )
        sig0 = np.array( [[0.1, 0.1]], dtype=InternalType )
        (w, mu, sig) = ( w0, mu0, sig0 )
        if verbose:
          print( "GM data")
          print( "  ", w0,)
          print( "  ", mu0)
          print( "  ", sig0)
          print( "initial GM")
          print( "  ", w,)
          print( "  ", mu)
          print( "  ", sig)
        #
        return w0, mu0, sig0, w, mu, sig

    @staticmethod
    def set1pop( self, verbose=False ):
        w0  = np.array( [1.0], dtype=DataType )
        mu0= np.array( [[0.2, 0.5]], dtype=InternalType )
        sig0 = np.array( [[0.2, 0.1]], dtype=InternalType )
        (w, mu, sig) = ( np.array( [1.0]), np.array( [[0.2, 0.2]] ), np.array( [ [0.1, 0.2] ]) )
        if verbose:
          print( "GM data")
          print( "  ", w0,)
          print( "  ", mu0 )
          print( "  ", sig0 )
          print( "initial GM" )
          print( "  ", w, )
          print( "  ", mu )
          print( "  ", sig )
        #
        return w0, mu0, sig0, w, mu, sig

    @staticmethod
    def set2popId( verbose=False ):
        """
        Set (w, mu, sig) = ( w0, mu0, sig0 )
        Use to check
        """
        w0   = np.array( [0.6, 0.4], dtype=DataType )
        mu0 = np.array( [[0.3,0.3],[0.7, 0.7]], dtype=InternalType )
        sig0 = np.array( [ [0.1, 0.2], [0.2, 0.1] ], dtype=InternalType )
        (w, mu, sig) = ( w0, mu0, sig0 )
        if verbose:
          print( "GM data")
          print( "  ", w0,)
          print( "  ", mu0)
          print( "  ", sig0)
          print( "initial GM")
          print( "  ", w,)
          print( "  ", mu)
          print( "  ", sig)
        #
        return w0, mu0, sig0, w, mu, sig

    @staticmethod
    def set4popId( verbose=False ):
        """
        Set (w, mu, sig) = ( w0, mu0, sig0 )
        Use to check
        """
        w0   = np.array( [0.2, 0.4, 0.3, 0.1], dtype=DataType )
        mu0 = np.array( [[0.2,0.2], [0.2,0.8],[0.8,0.2],[0.8, 0.8]], dtype=InternalType )
        sig0 = np.array( [ [0.1, 0.2], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2] ], dtype=InternalType )
        (w, mu, sig) = ( w0, mu0, sig0 )
        if verbose:
          print( "GM data")
          print( "  ", w0,)
          print( "  ", mu0)
          print( "  ", sig0)
          print( "initial GM")
          print( "  ", w,)
          print( "  ", mu)
          print( "  ", sig)
        #
        return w0, mu0, sig0, w, mu, sig
    
    @staticmethod
    def set4pop( verbose=False ):
        """
        Set (w, mu, sig) = ( w0, mu0, sig0 )
        Use to check
        """
        w0   = np.array( [0.2, 0.4, 0.3, 0.1], dtype=DataType )
        mu0 = np.array( [[0.2,0.2], [0.2,0.8],[0.8,0.2],[0.8, 0.8]], dtype=InternalType )
        sig0 = np.array( [ [0.1, 0.2], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2] ], dtype=InternalType )
        # sig0 = np.array( [ [0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1] ], dtype=InternalType )
        w   = np.array( [0.25, 0.25, 0.25, 0.25], dtype=DataType )
        mu = np.array( [[0.25,0.25], [0.15,0.85],[0.75,0.15],[0.75, 0.85]], dtype=InternalType )
        sig = np.array( [ [0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2] ], dtype=InternalType )
        if verbose:
          print( "GM data")
          print( "  ", w0,)
          print( "  ", mu0)
          print( "  ", sig0)
          print( "initial GM")
          print( "  ", w,)
          print( "  ", mu)
          print( "  ", sig)
        #
        return w0, mu0, sig0, w, mu, sig



def normalizeDistrib( y0 ):
    s = np.sum( y0 )
    s = 1./s
    return y0 * s

def computeGaussian1D( x,  mu, var):
    # print ( "???", __name__, x[-1], x.shape )
    # print "x", x
    sig = np.sqrt(var)
    u = (x - mu) / sig
    # print "u", u
    u = - 0.5 * u * u
    cst = 1.0 / ( sig * SqrtTwoPi)
    y = cst * np.exp( u )
    return y

def computeGaussian2D( xy,  mu, var):
    # WARNING: Only for diagonal covariance matrice var
    # xy : array[2,n]  of x, y coordinates
    # mu : array[2] of x, y coordinates of the center
    # sig2 : array[2] of x & y variance

    sig = np.sqrt(var)
    u = (xy[0] - mu[0]) / sig[0]
    u = - 0.5 * u * u
    v =  (xy[1] - mu[1]) / sig[1]
    v = - 0.5 * v * v
    cst = 1.0 / ( sig[0] * sig[1] * TwoPi)
    y = cst * np.exp( u + v)
    return y

def vErf(x):
    y = np.zeros( x.shape, dtype=InternalType )
    for i in range( x.shape[0]):
        y[i] = math.erf( x[i] )
    return y

def pderivateOfDiscretizedGaussian2D( derivate, xy, dxy, mu, var,  ):
    # TODO: to optimize
    N = xy.shape[1]
    sig = np.sqrt( var )
    
    # V2 xinf   = xy[0,:] - 0.5 * dxy[0]
    xinf   = xy[0,:] - dxy[0,:]
    xsup = xy[0,:] + dxy[0,:]
    yinf   = xy[1,:] - dxy[1,:]
    ysup = xy[1,:] + dxy[1,:]
    """
    ISupSup = 0.25 * ( 1.0 + vErf( (xsup - mu[0]) * cstx ) ) * ( 1.0 + vErf( (ysup - mu[1]) * csty ) )
    IInfInf    = 0.25 * ( 1.0 + vErf( (xinf  - mu[0]) * cstx ) ) * ( 1.0 + vErf( (yinf - mu[1]) * csty ) )
    IInfSup   = 0.25 * ( 1.0 + vErf( (xinf  - mu[0]) * cstx ) ) * ( 1.0 + vErf( (ysup - mu[1]) * csty ) )
    ISupInf   = 0.25 * ( 1.0 + vErf( (xsup  - mu[0]) * cstx ) ) * ( 1.0 + vErf( (yinf - mu[1]) * csty ) )
    """
    if derivate == "mux":
      if var[1] < 1.e-15:
        IYInf = np.where( (yinf - mu[1]) < 0.0, 0.0, 2.0)
        IYSup = np.where( (ysup - mu[1]) < 0.0, 0.0, 2.0)
        IY = IYSup - IYInf
      else:
        csty = 1.0 / (np.sqrt(2.0) * sig[1])
        IYInf = 1.0 + vErf( (yinf - mu[1]) * csty )
        IYSup = 1.0 + vErf( (ysup - mu[1]) * csty )
        IY = IYSup - IYInf
      if var[0] < 1.e-15:
        IXInf = np.where( (xinf == mu[0]), 1.0, 0.0)
        IXSup = np.where( (xsup == mu[0]), 1.0, 0.0)        
        IX = IXInf - IXSup
      else:
        IXInf = computeGaussian1D(xinf, mu[0], var[0] )
        IXSup = computeGaussian1D(xsup, mu[0], var[0] )
        IX = IXInf - IXSup
      I = IX * IY
    if derivate == "muy":
      if var[0] < 1.e-15:
        IXInf = np.where( (xinf - mu[0]) < 0.0, 0.0, 2.0)
        IXSup = np.where( (xsup - mu[0]) < 0.0, 0.0, 2.0)
        IX = IXSup -IXInf
      else:
        cstx = 1.0 / (np.sqrt(2.0) * sig[0])
        IXInf = 1.0 + vErf( (xinf - mu[0]) * cstx )
        IXSup = 1.0 + vErf( (xsup - mu[0]) * cstx )
        IX = IXSup - IXInf
      if var[1] < 1.e-15:
        #
        IYInf = np.where( (yinf == mu[1]), 1.0, 0.0)
        IYSup = np.where( (ysup == mu[1]), 1.0, 0.0)
        IY = IYInf - IYSup
      else:
        IYInf = computeGaussian1D(yinf, mu[1], var[1] )
        IYSup = computeGaussian1D(ysup, mu[1], var[1] )      
        IY = IYInf - IYSup 
      I = IX * IY

    if (I.shape[0] == 0):
        print( "xy.shape", xy.shape )
        traceback.print(_stack());
        exit()
    return I

def computeDiscretizedGaussian2D( xy, dxy, mu, var ):
    # TODO: to optimize
    N = xy.shape[1]
    sig = np.zeros( (2) )
    # V2 xinf   = xy[0,:] - 0.5 * dxy[0]
    xinf   = xy[0,:] - dxy[0,:]
    xsup = xy[0,:] + dxy[0,:]
    yinf   = xy[1,:] - dxy[1,:]
    ysup = xy[1,:] + dxy[1,:]
    """
    ISupSup = 0.25 * ( 1.0 + vErf( (xsup - mu[0]) * cstx ) ) * ( 1.0 + vErf( (ysup - mu[1]) * csty ) )
    IInfInf    = 0.25 * ( 1.0 + vErf( (xinf  - mu[0]) * cstx ) ) * ( 1.0 + vErf( (yinf - mu[1]) * csty ) )
    IInfSup   = 0.25 * ( 1.0 + vErf( (xinf  - mu[0]) * cstx ) ) * ( 1.0 + vErf( (ysup - mu[1]) * csty ) )
    ISupInf   = 0.25 * ( 1.0 + vErf( (xsup  - mu[0]) * cstx ) ) * ( 1.0 + vErf( (yinf - mu[1]) * csty ) )
    """
    if var[0] < 1.e-15:
      # IXInf = np.ones((N)) * 2.0 
      # IXSup = np.ones((N)) * 2.0 
      IXInf = np.where( (xinf - mu[0]) < 0.0, 0.0, 2.0)
      IXSup = np.where( (xsup - mu[0]) < 0.0, 0.0, 2.0)
    else: 
      sig[0] = np.sqrt( var[0] )
      cstx = 1.0 / (np.sqrt(2.0) * sig[0])
      IXInf = 1.0 + vErf( (xinf - mu[0]) * cstx )
      IXSup = 1.0 + vErf( (xsup - mu[0]) * cstx )
    if var[1] < 1.e-15:
      # IYInf = np.ones((N)) * 2.0
      # IYSup = np.ones((N)) * 2.0 
      IYInf = np.where( (yinf - mu[1]) < 0.0, 0.0, 2.0)
      IYSup = np.where( (ysup - mu[1]) < 0.0, 0.0, 2.0)
    else:
      sig[1] = np.sqrt( var[1] )
      csty = 1.0 / (np.sqrt(2.0) * sig[1])
      IYInf = 1.0 + vErf( (yinf - mu[1]) * csty )
      IYSup = 1.0 + vErf( (ysup - mu[1]) * csty )
    #  
    # print "Debug: IXInf, IXSup, IYInf, IYSup =", IXInf, IXSup, IYInf, IYSup
    ISupSup = 0.25 * IXSup * IYSup
    IInfInf = 0.25 * IXInf * IYInf
    IInfSup = 0.25 * IXInf * IYSup
    ISupInf = 0.25 * IXSup * IYInf
    # print "Debug: IInfInf, ISupInf, IInfSup, ISupSup = ", IInfInf, ISupInf, IInfSup, ISupSup
    I = ISupSup - IInfSup - ISupInf + IInfInf
    """ Debug
    print "xy", xy
    print "dxy", dxy
    print "xinf", xinf
    print "xsup", xsup
    print "xy.shape, dxy.shape", xy.shape, dxy.shape
    print "I ", I
    """
    if (I.shape[0] == 0):
        print( "xy.shape", xy.shape )
        traceback.print(_stack());
        exit()
    return I

class EMResults:
  def __init__(self):
    self.w = []
    self.mu = []
    self.var = []
    
class EM2D:

    @staticmethod
    def generateMixedGaussians2D( xy, dxy, w, mu, var, discretized=True, normalize=False ):
        """
        xy : array[2,n] x,y values
        # V2: dxy: array[2]
        dxy: array[2, n]
        w: array[K]  magnitude between [0,1]
        mu : array[K, 2] average, center of the Gaussian
        var : array[K, 2] variance  (sigma^2)
        Options:
          discretized :compute the mixture on discretized gaussian
        """
        y = np.zeros(xy.shape[1], dtype=DataType)
        K = w.shape[0]
        for k in range(K):
            if discretized:
                y += w[k] * computeDiscretizedGaussian2D( xy, dxy, mu[k], var[k] )
            else:
                print( "generateMixedGaussians2D: must be discretized" )
                exit()
                y  += w[k] * computeGaussian2D( xy, mu[k], var[k])
            idx = np.where( y == 0)
            """ Debug
            print( "y", y )
            if idx[0].shape[0] != 0:
                print "MG y = 0", len(idx))
                print( "idx", idx
                print "xxyy", xy [:,idx]
                xxyy = xy [:,idx].reshape( 2, -1 )
                print "  xxyy", xxyy
                print "  ??? compDG2D", computeDiscretizedGaussian2D( xxyy, dxy, mu[k], var[k] )
            """
            # Debug
            # print "a, mu, var, at k", a[k], mu[k], var[k]
            # print "xy", xy
            # print "y", y
        if normalize:
          y = normalizeDistrib( y )
        #
        return y

    def __init__(self, discretized):
        self.discretized = discretized

    def generateMixedGaussians2DOnGrids( self, grids,  w, mu, var, normalize=False ):
        """
        grids : list[nGrids]
        w: array[K]  magnitude between [0,1]
        mu : array[K, 2] average, center of the Gaussian
        var : array[K, 2] variance  (sigma^2)
        Options:
          discretized :compute the mixture on discretized gaussian
        return:
          z : list[nGrids].array[n] values
        """
        NGrids = len(grids)
        z = []
        # Test if there are at least one grid
        if ( NGrids !=0 ):
            for g in range(NGrids):
              ztmp = generateMixedGaussians2D( grids[g].xy, grids[g].dxy, w, mu, var, normalize)
              z.append(ztmp)
        else:
            print( "Error in generateMixedGaussians2DOnGrids:", " not in Grid format")
            exit()
        #
        return z

    def completeData( self, grid, z0, w, mu, var, nSig=3, Check= False):
        """
        xy : array[2,n] x,y values
        w: array[K]  magnitude between [0,1]
        mu : array[K, 2] average, center of the Gaussian
        var : array[K, 2] variance  (sigma^2)
        """
        if Check:
          print( "completeData:")
        #
        # Compute left, right top, bottom data completion to have p x sig
        #
        xy0 = grid.xy
        dxy = grid.dxy
        # Numbers of row and colum to extend ( matrix[2,2] )
        nExtend = np.zeros( (2,2), dtype=np.int32 )
        Nx = grid.nX
        Ny = grid.nY
        # loop on x, y
        for i in range(2):
          sig = np.sqrt( var[:,i] )
          #
          muMin  = mu[:,i] - nSig*sig
          muMax = mu[:,i] + nSig*sig
          if Check:
              print( " mumin,max, axis=", i, muMin, muMax )
          xMin = np.min( muMin )
          xMax = np.max( muMax )
          dxBefore = (xy0[i,0]-xMin) / dxy[i]
          dxAfter = (xMax - xy0[i,-1]) / dxy[i]
          nBefore = int( np.round (dxBefore ) )
          nAfter = int( np.round (dxAfter) )
          if nBefore < 0:
            nBefore = 0
          if nAfter < 0:
            nAfter = 0
          nExtend[i,0] = nBefore
          nExtend[i,1] = nAfter
        #
        # Build the final array xy, z (Data completed)
        #
        NX = Nx + nExtend[0,0] + nExtend[0,1]
        NY = Ny + nExtend[1,0] + nExtend[1,1]
        dx =  dxy[0]
        dy =  dxy[1]
        print( "Nx, Ny", Nx, Ny )
        print( "dx dy", dx, dy)
        print( "Origin:", grid.startX, grid.startY)
        print( "z0.shape", z0.shape)
        xInf = grid.startX - nExtend[0,0] * dx
        xSup = xInf  + (NX-1) * dx
        yInf =  grid.startY - nExtend[1,0] * dy
        ySup = yInf  + (NY-1) * dy
        if Check:
          print( "X Extend left, right", nExtend[0,:],)
          print( "Y Extend bottom, top", nExtend[1,:],)
          print( "NX, NY", NX, NY,)
          print( "xInf, xSup, yInf, ySup", xInf, xSup, yInf, ySup,)
        #
        #  New Grid xyg[2, NX, NY] and z[ NX, NY]
        #
        # 0.5*dx[dy] is added to avoid rounding pbs.
        xyg = np.mgrid[ xInf: xSup+0.5*dx: dx,   yInf: ySup+0.5*dy: dy]
        # Reshape xyg[ 2, NX, NY] -> xyg[ 2, NX * NY]
        xy = xyg.reshape( (2,-1) )
        z = np.zeros( (NX, NY), dtype=InternalType )
        # Reshape z0[ Nx * Ny ] -> z0[ Nx, Ny ]
        z0 =z0.reshape( (Nx, Ny) )
        #
        #  Central copy
        #
        # Copy z0 in new array z
        if Check:
          print( "  Central copy z0 -> z[", nExtend[0,0], ":",  NX-nExtend[0,1], ",", nExtend[1,0], ":",  NY-nExtend[1,1], "]")
        z[ nExtend[0,0]: NX-nExtend[0,1], nExtend[1,0]:NY-nExtend[1,1] ]  = z0[:,:]
        #
        #  EAST
        #
        if nExtend[0,0] != 0:
          if Check : print( " EAST (left)")
          # Reshape  : xygrid[ 2, range_X, range_Y] -> xy[ 2, range_X * range_Y]
          ztmp = self.generateMixedGaussians2D( xyg[ :, 0: nExtend[0,0], :].reshape(2,-1), dxy, w, mu, var)
          if (Check):
            print( "  z before copy,  z[0:", nExtend[0,0]+1, ", : ] =", z[ 0:nExtend[0,0]+1, :])
          # Reshape ztmp[ range_X * range_Y] -> ztmp[ range_X, range_Y]
          print( "xyg", xyg.shape)
          print( "ztmp", ztmp.shape)
          print( "z", z.shape)
          z[0: nExtend[0,0], :] = ztmp.reshape( (nExtend[0,0], -1) )
          if (Check):
            print( "  z after", z[ 0: nExtend[0,0]+1, :])
        #
        #  WEST
        #
        if nExtend[0,1] != 0:
          # Reshape  : xygrid[ 2, range_X, range_Y] -> xy[ 2, range_X * range_Y]
          ztmp = self.generateMixedGaussians2D( xyg[ :, - nExtend[0,1]:, :].reshape(2,-1), dxy, w, mu, var)
          if (Check):
            print( " WEST (right)")
            print( "  z before copy,  z[", -nExtend[0,1]-1, ":-1, : ] =", z[ -nExtend[0,1]-1:, :])
          # Reshape ztmp[ range_X * range_Y] -> ztmp[ range_X, range_Y]
          z[-nExtend[0,1]:, :] = ztmp.reshape( (nExtend[0,1], -1) )
          if (Check):
            print( "  z after", z[ -nExtend[0,1]-1 :, :])
        #
        #  SOUTH
        #
        if nExtend[1,0] != 0:
          # Reshape  : xygrid[ 2, range_X, range_Y] -> xy[ 2, range_X * range_Y]
          ztmp = self.generateMixedGaussians2D( xyg[ :, :, 0: nExtend[1,0] ].reshape(2,-1), dxy, w, mu, var)
          if (Check):
            print( "  SOUTH (bottom)")
            print( "  z before copy,  z[:, 0:", nExtend[1,0]+1, "] =", z[ :, 0:nExtend[1,0]+1])
          # Reshape ztmp[ range_X * range_Y] -> ztmp[ range_X, range_Y]
          z[:, 0: nExtend[1,0]] = ztmp.reshape(-1, nExtend[1,0]  )
          if (Check):
            print( "  z after", z[ :, 0: nExtend[1,0]+1] )
        #
        #  NORTH
        #
        if nExtend[1,1] != 0:
          # Reshape  : xygrid[ 2, range_X, range_Y] -> xy[ 2, range_X * range_Y]
          ztmp = self.generateMixedGaussians2D( xyg[ :, :, - nExtend[1,1]: ].reshape(2,-1), dxy, w, mu, var)
          if (Check):
            print( "  NORTH (top)" )
            print( "  z before copy,  z[ :,", -nExtend[1,1]-1, ":-1] =", z[ : -nExtend[1,1]-1: -1] )
          # Reshape ztmp[ range_X * range_Y] -> ztmp[ range_X, range_Y]
          z[:, -nExtend[1,1]: ] = ztmp.reshape( (-1, nExtend[1,1]) )
          if (Check):
            print( "  z after", z[ :, -nExtend[1,1]-1 : ])
        #
        # Reshape z[ range_X, range_Y] -> ztmp[ range_X * range_Y]
        return xy,z.reshape( NX*NY )

    def computeWeightedLikelihoods( self, xy, dxy, z, w, mu, var):
        N = xy.shape[1]
        K = mu.shape[0]
        W = np.sum( w)
        # Already normalized ???
        # li : log-likelihood of x_i
        li = np.zeros( (N), dtype =InternalType)
        li[:] = self.generateMixedGaussians2D( xy[:,:],  dxy, w[:], mu[:,:], var[:,:] )
        # Debug
        # print( "li.shape", li.shape, z.shape
        # print 'li', li
        #
        # Weighted
        # Avoiding log(0) calculation
        # For small proba replace by proba=1.0
        idx = np.where( li == 0)
        # Debug
        if len(idx[0]) != 0:
          print( "len idx",len(idx[0]))
          # print "idx, li[idx], xy[0, idx], xy[1,idx]",idx, li[idx], xy[0, idx], xy[1,idx]
          # xxyy = xy[:, idx].reshape(2, -1)
          # print xy.shape, xxyy.shape
          # print generateMixedGaussians2D( xxyy,  dxy, w[:], mu[:,:], var[:,:] )
        li[idx] = 1.0
        li[:] = np.log( li[:] ) * z[:]
        return li

    def EStep( self, xy, dxy, w, mu, var, check=False):
          """
          Compute new eta(i,k), fraction/proba that point i
          belongs to kth gaussian
          """

          K = w.shape[0]
          N = xy.shape[1]
          eta = np.zeros( ( N, K), dtype=InternalType )
          kSum = np.zeros( N, dtype=InternalType )
          for k in range(K):
             # ??? computeGaussian2D or computeDicretizedGaussian2D
             eta[ :, k] = w[k] * computeDiscretizedGaussian2D( xy[:,:], dxy, mu[k,:], var[k,:] ) 
             kSum += eta[:,k]
          # Normalization
          for i in range(N) :
              if kSum[i] !=0 :
                eta[ i, :] = eta[i, :] / kSum[i]
              else:
                print( "WARNING sum eta(i,:) is null, i=",i , "xy=", xy[:,i], "dxy=",dxy[:,i])
                eta[i,:] = 0

          if check:
              kSum = np.sum( eta, axis=1 )
              print( "EStep: shape, sum_k (eta) min & max ", kSum.shape, np.min( kSum), np.max( kSum) )
              ssum = np.sum( eta, axis= 0 )
              print( "EStep: shape, sum_i (eta) min & max ", ssum.shape, np.min( ssum), np.max( ssum) )
              print( "EStep: sum (eta) ", np.sum( eta))

          return eta

    def weightedMStep( self, xy, z, eta, updateVar=None):
        """
        xy: [2, N]
        z : [N]
        eta : [K, N]
        """
        N = xy.shape[1]
        K = eta.shape[1]
        
        w = np.zeros( (K), dtype=InternalType)
        mu = np.zeros( (K,2), dtype=InternalType)
        var = np.zeros( (K,2), dtype=InternalType)

        u =  eta.T * z
        u = u.T
        Wk = np.sum( u, axis=0)
        # ??? print "Wk.shape", Wk.shape
        # ??? print "Wk", Wk
        #
        # Mu x
        u = xy[0,:]*z
        mu[:, 0] = np.matmul( u, eta)
        # mu[:, 0]  = mu[:, 0]  / Wk
        mu[:, 0]  = np.where( Wk == 0.0, 0, mu[:, 0] / Wk)
        # Mu y
        u = xy[1,:]*z
        mu[:, 1] = np.matmul( u, eta)
        # mu[:, 1] = mu[:, 1] / Wk
        mu[:, 1] = np.where( Wk == 0.0, 0, mu[:, 1] / Wk)
        if (updateVar is None):
          # Calculate sigma2
          for k in range(K):
            X2 = xy[0,:] - mu[k,0]
            X2 = X2 * X2
            X2 = X2 * z
            Y2 = xy[1,:] - mu[k,1]
            Y2 = Y2 * Y2
            Y2 = Y2 * z
            var[k,0] =  np.matmul( X2, eta[:,k] )
            var[k,1] =  np.matmul( Y2, eta[:,k] )
            # var[k] /= Wk[k]
            if Wk[k] == 0.0 :
              var[k,:] = 0.0
            else:
              var[k] /= Wk[k]
        else:
            var = updateVar
        # Calculate w
        #  Non weighted case :
        #  w = Wk / N
        w = Wk / np.sum(z)
        #
        return w, mu, var

    def weightedEMLoopOnGrids( self, grids0, w0, mu0, var0, LConvergence = 0.000001, nIterMin=20, dataCompletion=True, cstVar=False, verbose=0, plotEvery=0):
        """
        xy0 : list[ array[2, nx*ny] ], list of grids
        z0 : list[ array[ nx*ny] ], list of values at grid coordinates
        """
        nGrids = len( grids0 )
        # nGrids = 1
        w = np.copy( w0 )
        mu = np.copy( mu0 )
        var = np.copy( var0 )
        eta = []
        # Plot
        plot = True if (plotEvery != 0) else False
        colors = plu.getColorMap()
        #
        # For graphics usage
        #
        maxdX0 = 0.0; maxdY0 = 0.0
        if grids0[0].xy[0].size != 0 :
          maxX0  = grids0[0].xy[0][0]; minX0  = maxX0
          maxY0  = grids0[0].xy[1][0]; minY0  = maxY0
          maxZ0 = grids0[0].z[0]; minZ0 = maxZ0
        else:
          maxX0  = grids0[1].xy[0][0]; minX0  = maxX0
          maxY0  = grids0[1].xy[1][0]; minY0  = maxY0
          maxZ0 = grids0[1].z[0]; minZ0 = maxZ0
        for g in range(nGrids):
          if (grids0[g].dxy[0].size != 0):
            maxdX0 = max( np.max( grids0[g].dxy[0, :]) , maxdX0 )
            maxdY0 = max( np.max( grids0[g].dxy[1, :]), maxdY0 )
            maxX0  = max( np.max( grids0[g].xy[0]), maxX0 )
            minX0  = min( np.min( grids0[g].xy[0]), minX0 )
            maxY0  = max( np.max( grids0[g].xy[1]), maxY0 )
            minY0  = min( np.min( grids0[g].xy[1]), minY0 )
            minZ0  = min( np.min( grids0[g].z ), minZ0 )
            maxZ0 = max( np.max( grids0[g].z ), maxZ0 )
          #
        #
        # Initial Likelihood
        L= 0
        for g in range(nGrids):
          if grids0[g].xy.size != 0:
            li = self.computeWeightedLikelihoods( grids0[g].xy, grids0[g].dxy, grids0[g].z, w, mu, var )
            L += np.sum(li)
          #
        if verbose >= 1 : print( "EMLoop, initial cond.:", w, mu, var, "Log-L=", L)
        #
        prevL = L; L = 1.0 + prevL;
        it =0
        while ( (( np.abs( (L - prevL)/L )  > LConvergence)  or  (it < nIterMin)) and ( it < 2000 ) ):
            prevL = L
            # Used for plotting
            minX = minX0; maxX = maxX0
            minY = minY0; maxY = maxY0
            minZ = minZ0; maxZ = maxZ0
            #
            # Data completion
            xyg = []; zg = []; dxyg = [] ;
            if dataCompletion:
              print( "dataCompletion: Not implemented")
              exit()
              # V2 RegularGrid : Inv
              """
              # Op tim: the grid is cst, the data in the 4 edges can be filled separately
              xy = np.array( [], dtype=InternalType)
              dxy = np.array( [], dtype=InternalType)
              z = np.array( [], dtype=InternalType)
              for g in range(nGrids):
                  xyt, zt = self.completeData( grids0[g], grids0[g].z,  w, mu, var, Check=False )
                  dxyt =  np.full(  (xyt.shape[1], xyt.shape[0]) , grids0[g].dxy )
                  dxyt  = dxyt.T
                  minX, maxX = getMinMax( xyt[0,:] , minX, maxX );
                  minY, maxY = getMinMax( xyt[1,:] , minY, maxY );
                  minZ, maxZ  = getMinMax( zt, minZ, maxZ )
                  xyg.append( xyt ); zg.append( zt ), dxyg.append( dxyt )
                  if g==0:
                    xy = np.copy(xyt)
                    dxy = np.copy(dxyt)
                    z = np.copy(zt)
                  else:
                    xy = np.concatenate( (xy, xyt), axis=1)
                    dxy = np.concatenate( (dxy, dxyt), axis=1)
                    z = np.concatenate( (z, zt), axis=0
              """
            else:
              # Concatenate grids data into the variable xy, dxy, z
              xy   = np.copy( grids0[0].xy)
              dxy = np.copy( grids0[0].dxy)
              z     = np.copy( grids0[0].z)
              xyg.append(   grids0[0].xy )
              dxyg.append( grids0[0].dxy )
              zg.append(     grids0[0].z )
              for g in range(1, nGrids):
                xy   = np.concatenate( (xy, grids0[g].xy), axis=1)
                dxy = np.concatenate( (dxy, grids0[g].dxy), axis=1)
                z     = np.concatenate( (z, grids0[g].z), axis=0)
                xyg.append( grids0[g].xy )
                dxyg.append( grids0[g].dxy )
                zg.append( grids0[g].z )
            #
            #  Plot init.
            #
            if ( plot and ( (it % plotEvery) == 0) ):
              fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(13, 7) )

            # EM Steps
            eta = self.EStep( xy, dxy, w, mu, var, check= False)
            if cstVar :
              (w, mu, var) = self.weightedMStep( xy, z, eta, updateVar=var )
            else:
              (w, mu, var) = self.weightedMStep( xy, z, eta )
                
            # To avoid numerical pb
            var = np.where( var < 1.e-15, 0.0, var)
            #
            # Likelihood
            L= 0
            for g in range(nGrids):
              if grids0[g].dxy.size != 0:
                li = self.computeWeightedLikelihoods( xyg[g], grids0[g].dxy, zg[g], w, mu, var )
                L += np.sum(li)
              #
            """ Debug
            idx = np.where( var < 1.e-15 )
            if idx[0].shape[0] != 0:
                print "Negative values of var: ", var[idx]
                print "Negative values of var (index): ", idx[0]
                exit()
            """
            if verbose >= 1: print( "EM step ", it, "L=", L, "dL", L - prevL)
            if verbose >= 2:
              print( "------ w =")
              print(  w )
              print( "------ mu =")
              print( mu )
              print( "------ sig =" )
              print( np.sqrt( var ) )    #  % (i, L, w, mu, np.sqrt( var ))
            if ( plot and ( (it % plotEvery) == 0) ):
              #
              # Set plot limits
              #
              newZg = []
              for g in range(nGrids):
                zz = self.generateMixedGaussians2D( xyg[g], grids0[g].dxy, w, mu, var,  normalize = False )
                # Min, max
                minZ, maxZ  = getMinMax( zz, minZ, maxZ ) 
                newZg.append( zz )
              #
              # Set x,y plot limits
              maxdX_ = maxdX0
              maxdY_ = maxdY0
              for i in range(3):
                for j in range(3):
                  ax[i,j].set_xlim( minX-maxdX_, maxX+maxdX_)
                  ax[i,j].set_ylim( minY-maxdY_, maxY+maxdY_)
              # Set Lut scale
              plu.setLUTScale( 0, max( maxZ0, maxZ) )
              #
              # Graph of Initial data
              #
              
              plu.drawPads( ax[0,0], grids0[0].xy[0,:], grids0[0].xy[1,:], grids0[0].dxy[0], grids0[0].dxy[1], grids0[0].z[:],  title="Plane 1", doLimits=False)
              plu.drawPads( ax[0,1], grids0[1].xy[0,:], grids0[1].xy[1,:], grids0[1].dxy[0], grids0[1].dxy[1], grids0[1].z[:],  title="Plane 2", doLimits=False)
              plu.drawPads( ax[0,2], grids0[0].xy[0,:], grids0[0].xy[1,:], grids0[0].dxy[0], grids0[0].dxy[1], grids0[0].z[:],  title="", alpha=1.0, doLimits=False)
              plu.drawPads( ax[0,2], grids0[1].xy[0,:], grids0[1].xy[1,:], grids0[1].dxy[0], grids0[1].dxy[1], grids0[1].z[:],  title="Both planes", alpha=0.5, doLimits=False)
              #
              #  Graph of Completed data
              #
              plu.drawPads( ax[1,0], xyg[0][0,:], xyg[0][1,:], grids0[0].dxy[0], grids0[0].dxy[1], zg[0][:],  title="", doLimits=False)
              plu.drawPads( ax[1,1], xyg[1][0,:], xyg[1][1,:], grids0[1].dxy[0], grids0[1].dxy[1], zg[1][:],  title="", doLimits=False)
              plu.drawPads( ax[1,2], xyg[0][0,:], xyg[0][1,:], grids0[0].dxy[0], grids0[0].dxy[1], zg[0][:],  title="", alpha=1.0, doLimits=False)
              plu.drawPads( ax[1,2], xyg[1][0,:], xyg[1][1,:], grids0[1].dxy[0], grids0[1].dxy[1], zg[1][:],  title="", alpha=0.5, doLimits=False)
              #
              # New Mixture
              #
              plu.drawPads( ax[2,0], xyg[0][0,:], xyg[0][1,:], grids0[0].dxy[0], grids0[0].dxy[1], newZg[0][:],  title="", doLimits=False)
              plu.drawPads( ax[2,1], xyg[1][0,:], xyg[1][1,:], grids0[1].dxy[0], grids0[1].dxy[1], newZg[1][:],  title="", doLimits=False)
              plu.drawPads( ax[2,2], xyg[0][0,:], xyg[0][1,:], grids0[0].dxy[0], grids0[0].dxy[1], newZg[0][:],  title="", alpha=1.0, doLimits=False)
              plu.drawPads( ax[2,2], xyg[1][0,:], xyg[1][1,:], grids0[1].dxy[0], grids0[1].dxy[1], newZg[1][:],  title="", alpha=0.5, doLimits=False)
              print(xyg[0][0,:])
              print( grids0[0].dxy[0] )
              print( zg[0][:] )
              xcg, xsig = plu.getBarycenter( xyg[0][0,:], grids0[0].dxy[0], xyg[1][0,:], grids0[1].dxy[0], zg[0][:], zg[1][:] )
              ycg, ysig = plu.getBarycenter( xyg[0][1,:], grids0[0].dxy[1], xyg[1][1,:], grids0[1].dxy[1], zg[0][:], zg[1][:] )
              plu.drawModelComponents(ax[2,2], w0, mu0, var0, pattern="cross" )
              plu.drawModelComponents(ax[2,2], np.array([1.0]), np.array([ [ xcg, ycg ] ] ),  np.array([ [ xsig*0.5, ysig*0.5 ] ] ), pattern="rect",color="blue")              
              plu.drawModelComponents(ax[2,2], w, mu, var, pattern="diam", color="red" )
              #
              ax[0,0].set_ylabel("Input Data" )
              ax[1,0].set_ylabel("Data Completion" )
              ax[2,0].set_ylabel("New Mixture" )
              plu.displayLUT( ax[2, 3], colors )
              plt.show()

            it+=1
        #
        # To find composantes connexes
        """"
        K = eta.shape[1]
        if K > 1 :
          N = eta.shape[0]
          nbrInGrp = 0
          grp = np.zeros( N )
          grpFlag = False
          for i in range( N ):
            idx = np.where( (eta[i,:] + 10.0e-6) > 1.0 )
            # print( "??? idx", idx )
            if idx[0].size == 1:
              grp[i] = idx[0][0]
              nbrInGrp += 1
          if (nbrInGrp == N):
              grpFlag = True
          print( "grpFlag, nbrInGrp, N, nbrOfGroups, K", grpFlag, nbrInGrp, "/", N, np.unique( grp ).size, "/", K )    
          input("Next") 
        """
        return (w, mu, var)

if __name__ == "__main__":

    N = 10 + 1
    a0  = np.array( [0.75, 0.25], dtype=DataType )
    # shape[k, x ]
    mu0= np.array( [[0.25, 0.75], [0.70, 0.30] ], dtype=InternalType )
    sig0 = np.array( [[0.1, 0.1], [0.2, 0.2]], dtype=InternalType )
    var0 = sig0 * sig0

    print (var0)
    dxy = 1./N
    xy = np.mgrid[0.0: 1.0+dxy: dxy, 0.0: 1.0+dxy: dxy]
    # xy = np.reshape( xy, ( xy.shape[0], xy.shape[1]*xy.shape[2] ) )
    # Verify dimensions ???
    z0 = generateMixedGaussians2D( xy, a0, mu0, var0)

def simpleProcessEMCluster( x, y, dx, dy, cathode, charge, wi, mui, vari, cstVar=False, discretizedPDF=True ):
    # discretizedPDF = False
    
    em = EM2D( discretizedPDF )
    # ???
    grids = []
    grid0 = Grid2D( 3, 3)
    grids.append( grid0 )
    grid1 = Grid2D( 3, 3)
    grids.append( grid1 )
    
    if ( x.shape[0] == 0): return ([],[],[])
    #
    # Check if the undelying grid is regular
    #
    cathIdx0 = np.where( cathode ==0 )
    x0 = x[cathIdx0]
    y0 = y[cathIdx0]
    dx0 =dx[cathIdx0]
    dy0 =dy[cathIdx0]
    cathIdx1 = np.where( cathode == 1 )
    x1 = x[cathIdx1]
    y1 = y[cathIdx1]
    dx1 =dx[cathIdx1]
    dy1 =dy[cathIdx1]
    # x0, y0, dx0, dy0, x1, y1, dx1, dy1 = computeOverlapingPads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 )
    """
    c0 = cl.charge[cathIdx0]
    c0 = c0 / (4*dx0*dy0)
    l0 = x0.shape[0]
    """
    """
    if l0 !=0:
      checkRegularGrid( x0, dx0 )
      checkRegularGrid( y0, dy0 )
    """
    """
    cathIdx = np.where( cl.cathode == 1 )
    x1 = x[cathIdx]
    y1 = y[cathIdx]
    dx1 =dx[cathIdx]
    dy1 =dy[cathIdx]
    c1 = cl.charge[cathIdx]
    c1 = c1 / (4*dx1*dy1)
    l1 = x1.shape[0]
    """
    """
    if l1 !=0:
      checkRegularGrid( x1, dx[cathIdx] )
      checkRegularGrid( y1, dy[cathIdx] )
    """
    """
    xm, xsig = plu.getBarycenter( x0, dx0, x1, dx1, c0, c1)
    ym, ysig = plu.getBarycenter( y0, dy0, y1, dy1, c0, c1)
    print( "Barycenter [x, y, sigx, sigy]", xm, ym, xsig, ysig )
    """
    # Normalization
    # ds = dx0 * dy0 * 4
    # c0 = c0 / ds
    s = np.sum( charge[cathIdx0])
    c0 = charge[cathIdx0] / s
    # ds = dx1 * dy1 * 4
    # c1 = c1 / ds
    s = np.sum( charge[cathIdx1])
    c1 = charge[cathIdx1] / s
    xy = []; dxy = []; z = []
    xy.append( np.array( [x0, y0] ) )
    xy.append( np.array( [x1, y1]) )
    dxy.append( np.array( [dx0, dy0]) )
    dxy.append( np.array( [dx1, dy1] ) )
    z.append( np.array( c0 ) )
    z.append( np.array( c1) )
    for g in range(2):
      grids[g].setGridValues( xy[g], dxy[g], z[g] )
    """
    wi = np.array( [1.0], EM.InternalType )
    mui = np.array( [[xm,ym]], EM.InternalType )
    vari = np.array( [[xsig*xsig, ysig*ysig]], EM.InternalType )
    """
    # (wf, muf, varf) = em.weightedEMLoopOnGrids( grids,  wi, mui, vari, dataCompletion=False, plotEvery=0)
    (wf, muf, varf) = em.weightedEMLoopOnGrids( grids,  wi, mui, vari, dataCompletion=False, cstVar=cstVar, plotEvery=0)
    print( "sum charge each cathodes", np.sum(c0), np.sum(c1) )
    return wf, muf, varf