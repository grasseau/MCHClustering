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

def thetaAsWMuVar( theta):
  K = int(theta.size / 5 )
  w = np.zeros((K))
  muX = np.zeros((K))
  muY = np.zeros((K))
  varX = np.zeros((K))
  varY = np.zeros((K))
  #
  varX = theta[0*K:1*K]
  varY = theta[1*K:2*K]
  muX = theta[2*K:3*K]
  muY = theta[3*K:4*K]
  w = theta[4*K:5*K]
  return (w, muX, muY, varX, varY)

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

def buildPreCluster( w, muX, muY, xyDxDy0, xyDxDy1, chId, minRatioCh, maxRatioCh):
  #
  K = w.size
  # Theta
  varX = 0.1 * np.ones( K )
  varY = 0.1 * np.ones( K )
  theta = asTheta( w, muX, muY, varX, varY)
  # Mathieson
  x0, y0, dx0, dy0 = xyDxDy0
  xyInfSup0 = padToXYInfSup( x0, y0, dx0, dy0)
  z0 = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup0, theta, chId )
  x1, y1, dx1, dy1 = xyDxDy1
  xyInfSup1 = padToXYInfSup( x1, y1, dx1, dy1)
  z1 = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup1, theta, chId )
  # Cathodes
  cath0 = np.zeros( x0.size, dtype=np.int16 )
  cath1 = np.ones( x1.size, dtype=np.int16 )

  # Low Charge cutoff
  zMinBefore = min( np.min(z0), np.min(z1) )
  zMaxBefore = max( np.max(z0), np.max(z1) )
  lowCutoff  = minRatioCh * zMaxBefore
  (x0r, y0r, dx0r, dy0r, cath0r, z0r) = removePads( x0, y0, dx0, dy0, cath0, z0, ( z0 < lowCutoff ) )
  (x1r, y1r, dx1r, dy1r, cath1r, z1r) = removePads( x1, y1, dx1, dy1, cath1, z1, ( z1 < lowCutoff ) )
  N0 = x0r.size
  N1 = x1r.size
  (x, y, dx, dy) = mergePads( x0r, y0r, dx0r, dy0r, x1r, y1r, dx1r, dy1r)
  xyDxy = padToXYdXY( x, y, dx, dy)
  z = np.hstack( [z0r, z1r] )
  cath = np.hstack( [ np.zeros( (N0), dtype=np.int16), 
                      np.ones( (N0), dtype=np.int16) ] )
  # High Charge cutoff - saturation
  highCutoff = maxRatioCh * zMaxBefore
  idx = np.where( z > highCutoff)
  z[ idx ] = highCutoff
  saturated = np.zeros( (N0+N1), dtype=np.int16)
  saturated[idx] = 1
  # Print info
  print( "buildPreCluster")
  print( "  min/max z before cutting-off", zMinBefore, zMaxBefore)
  print( "  min/max z cutoff", lowCutoff, highCutoff )
  print( "  # saturated pads", np.sum( saturated ))
  # 
  return ( xyDxy, cath, saturated, z )