# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import numpy as np
import C.PyCWrapper as PCWrap

def buildPads( nx, ny, xMin, xMax, yMin, yMax ):
    dx = (xMax - xMin)/nx
    dy = (yMax - yMin)/ny
    x1d = np.arange( xMin, xMax - 0.1 * dx, dx) + 0.5 * dx
    y1d = np.arange( yMin, yMax - 0.1 * dy, dy) + 0.5 * dy
    x, y = np.meshgrid( x1d, y1d)
    x = x.ravel( )
    y = y.ravel( )
    #
    dx = np.ones(x.shape) * 0.5 * dx
    dy = np.ones(y.shape) * 0.5 * dy
    #
    return x, y, dx, dy

def buildPadsAsXYdXY( nx, ny, xMin, xMax, yMin, yMax ):
    
    dx = (xMax - xMin)/nx
    dy = (yMax - yMin)/ny
    x1d = np.arange( xMin, xMax - 0.1 * dx, dx) + 0.5 * dx
    y1d = np.arange( yMin, yMax - 0.1 * dy, dy) + 0.5 * dy
    x, y = np.meshgrid( x1d, y1d)
    N = x.size
    xyDxy = np.zeros((4*N))
    xyDxy[0:N] = x.ravel( )
    xyDxy[N:2*N] = y.ravel( )
    #
    xyDxy[2*N:3*N] = np.ones(x.shape) * 0.5 * dx
    xyDxy[3*N:4*N] = np.ones(y.shape) * 0.5 * dy
    #
    return xyDxy

def asXYdXY( x, y, dx, dy):
    N = x.size
    xyDxy = np.zeros((4*N))
    xyDxy[0*N:1*N] = x
    xyDxy[1*N:2*N] = y
    xyDxy[2*N:3*N] = dx
    xyDxy[3*N:4*N] = dy
    return xyDxy

def asXYdXdY( xyDxy):
    N = int( xyDxy.size / 4 )
    x = xyDxy[0*N:1*N]
    y = xyDxy[1*N:2*N]
    dx = xyDxy[2*N:3*N]
    dy = xyDxy[3*N:4*N]
    return (x, y, dx, dy)

def mergePads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 ):
  x = np.hstack( [x0, x1] )
  y = np.hstack( [y0, y1] )    
  dx = np.hstack( [dx0, dx1] )
  dy = np.hstack( [dy0, dy1] )   
  return (x, y, dx, dy)

def removePads( x, y, dx, dy, cath, z, flags ):
  # not operator
  keep = np.invert(flags)
  return ( x[keep], y[keep], dx[keep], dy[keep], cath[keep], z[keep] )

def padToXYInfSup( x, y, dx, dy):
  N = x.size
  xyInfSup = np.zeros( 4 * N)
  xyInfSup[0*N:1*N] = x - dx
  xyInfSup[1*N:2*N] = y - dy 
  xyInfSup[2*N:3*N] = x + dx
  xyInfSup[3*N:4*N] = y + dy
  return xyInfSup

def padToXYdXY( x, y, dx, dy):
  N = x.size
  xyDxy = np.zeros( 4 * N)
  xyDxy[0*N:1*N] = x
  xyDxy[1*N:2*N] = y 
  xyDxy[2*N:3*N] = dx
  xyDxy[3*N:4*N] = dy
  return xyDxy
    
def asTheta( w, muX, muY, varX=None, varY=None):
  K = w.size
  theta = np.zeros((5*K))
  if varX is not None:
    theta[0*K:1*K] = varX
  if varY is not None:
    theta[1*K:2*K] = varY
  theta[2*K:3*K] = muX
  theta[3*K:4*K] = muY
  theta[4*K:5*K] = w
  return theta


def compute2DPadIntegrals( xInf, xSup, yInf, ySup, chId ):
    N = xInf.size
    xyInfSup = np.zeros((4*N))
    #
    xyInfSup[0*N:1*N] = xInf[:]
    xyInfSup[1*N:2*N] = yInf[:]
    xyInfSup[2*N:3*N] = xSup[:]
    xyInfSup[3*N:4*N] = ySup[:]
    z = PCWrap.compute2DPadIntegrals( xyInfSup, chId )
    return z