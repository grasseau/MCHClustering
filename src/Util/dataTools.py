# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import numpy as np
import numpy.random as npr
import C.PyCWrapper as PCWrap
import pickle

class SimulCluster:
  fileName = "sEvent.obj"
  
  def read( ):
    file = open( SimulCluster.fileName, "rb" )
    obj = pickle.load( file )
    file.close()
    return obj

  def write( self):
   file = open( SimulCluster.fileName, "wb" )
   pickle.dump( self, file )
   file.close()
   
  def __init__(self, nX, nY, xGrid = [-2.0, 2.0], yGrid = [-2.0, 2.0]):
    self.gridNx = nX
    self.gridNy = nY
    self.gridXLimits = xGrid
    self.gridYLimits = yGrid
  
  def buildCluster(self, chId, K, minPadCh, maxPadCh, zMax  ):
    # Random:
    # - theta
    # - cluster amplitude
    #
    self.nbrOfSeeds = K
    self.padMinCh = minPadCh
    self.padMaxCh = maxPadCh   # saturated pads
    self.chId = chId
    
    # Grid
    x0, y0, dx0, dy0 = buildPads( self.gridNx, self.gridNy, 
        self.gridXLimits[0], self.gridXLimits[1], self.gridYLimits[0], self.gridYLimits[1] )
    x1, y1, dx1, dy1 = buildPads( self.gridNy, self.gridNx, 
        self.gridXLimits[0], self.gridXLimits[1], self.gridYLimits[0], self.gridYLimits[1] )
    cath0 = np.zeros( x0.size, dtype=np.int32 )
    cath1 = np.ones ( x1.size, dtype=np.int32 )

    # Seeds/Hits
    xSize = self.gridXLimits[1] - self.gridXLimits[0]
    ySize = self.gridYLimits[1] - self.gridYLimits[0]
    # Limit the domain of drawns seed
    xSeedDomain = xSize * 3 / 4
    ySeedDomain = ySize * 3 / 4
    # Theta
    ok = False
    while not ok:
      w = npr.random( self.nbrOfSeeds )
      wLast = 1.0 - np.sum( w[0:self.nbrOfSeeds-1]) 
      w[self.nbrOfSeeds-1] = wLast
      ok = (wLast >= 0) and (wLast <= 1)
      if not ok : print("bad draw w", w)
    xShift = self.gridXLimits[0] + xSize * 1 / 8
    yShift = self.gridYLimits[0] + ySize * 1 / 8    
    muX = npr.random( self.nbrOfSeeds ) * xSeedDomain + xShift
    muY = npr.random( self.nbrOfSeeds ) * ySeedDomain + yShift
    print(" xShift, yShift, xSize, ySize", xShift, yShift, xSize, ySize)
    print("npr.random( self.nbrOfSeeds )", npr.random( self.nbrOfSeeds ))
    print("w", w)
    print("muX", muX)
    print("muY", muY)
    cstVar = 0.1 * 0.1
    varX = np.ones( self.nbrOfSeeds ) * cstVar
    varY = np.ones( self.nbrOfSeeds ) * cstVar
    theta = asTheta( w, muX, muY, varX, varY)
    xyInfSup0 = padToXYInfSup( x0, y0, dx0, dy0)
    z0 = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup0, theta, chId )
    xyInfSup1 = padToXYInfSup( x1, y1, dx1, dy1)
    z1 = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup1, theta, chId )
    # ratio = npr.random( 1 )[0] * zMax / max( np.max( z0 ), np.max( z1 ) )
    ratio = 1.0 * zMax / max( np.max( z0 ), np.max( z1 ) )
    z0 = z0 * ratio
    z1 = z1 * ratio
    
    print("sum z0 MMixture", np.sum(z0))
    print("sum z1 MMixture", np.sum(z1))
    
     
    (x0r, y0r, dx0r, dy0r, cath0r, z0r) = removePads( x0, y0, dx0, dy0, cath0, z0, ( z0 < self.padMinCh) )
    (x1r, y1r, dx1r, dy1r, cath1r, z1r) = removePads( x1, y1, dx1, dy1, cath1, z1, ( z1 < self.padMinCh) )
    
    # Saturated
    saturated0 = z0r > self.padMaxCh
    saturated1 = z1r > self.padMaxCh
    z0r = np.where( saturated0, self.padMaxCh, z0r)
    z1r = np.where( saturated1, self.padMaxCh, z1r)

    print("sum z0 filtered MMixture", np.sum(z0r))
    print("sum z1 filtered MMixture", np.sum(z1r))
    self.padCath0 = (x0r, y0r, dx0r, dy0r, cath0r.astype( np.int16 ), saturated0.astype( np.int16 ), z0r)
    self.padCath1 = (x1r, y1r, dx1r, dy1r, cath1r.astype( np.int16 ), saturated1.astype( np.int16 ), z1r)
    self.theta = theta
    
    return ( self.padCath0, self.padCath1, self.theta )

  def getMergedPads(self):
    (x0r, y0r, dx0r, dy0r, cath0r, saturated0, z0r) = self.padCath0
    (x1r, y1r, dx1r, dy1r, cath1r, saturated1, z1r) = self.padCath1
    N0 = x0r.size
    N1 = x1r.size
    (x, y, dx, dy) = mergePads( x0r, y0r, dx0r, dy0r, x1r, y1r, dx1r, dy1r)
    xyDxy = padToXYdXY( x, y, dx, dy)
    z = np.hstack( [z0r, z1r] )
    cath = np.hstack( [cath0r, cath1r] )
    saturated = np.hstack( [saturated0, saturated1] )
    return ( xyDxy, cath, saturated, z )

   


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

def xyDxyToInfSup( xyDxy ):
  N = xyDxy.size // 4
  x = xyDxy[0*N:1*N]
  y = xyDxy[1*N:2*N]
  dx = xyDxy[2*N:3*N]
  dy = xyDxy[3*N:4*N]  
  xyInfSup = np.zeros( 4 * N)
  xyInfSup[0*N:1*N] = x - dx
  xyInfSup[1*N:2*N] = y - dy 
  xyInfSup[2*N:3*N] = x + dx
  xyInfSup[3*N:4*N] = y + dy
  return xyInfSup

def ungroupXYInfSup( xyInfSup ):
  N = xyInfSup.size // 4
  xInf = xyInfSup[0*N:1*N]
  yInf = xyInfSup[1*N:2*N]
  xSup = xyInfSup[2*N:3*N]
  ySup = xyInfSup[3*N:4*N]
  return (xInf, yInf, xSup, ySup )

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

def thetaListAsTheta( theta):
  (w, muX, muY, varX, varY) = theta
  K = len(w)
  thetaR = np.zeros((K*5))
  w = np.array(w)
  muX = np.array(muX)
  muY = np.array(muY)
  varX = np.array(varX)
  varY = np.array(varY)
  #
  oneOverSumW = 1.0 / np.sum( w )
  w = oneOverSumW * w
  #
  print("??? K=", K, muX)
  thetaR[0*K:1*K] = varX[:]
  thetaR[1*K:2*K] = varY
  thetaR[2*K:3*K] = muX
  thetaR[3*K:4*K] = muY
  thetaR[4*K:5*K] = w
  return thetaR

def printTheta(str, theta):
  K = theta.size // 5 
  muX = theta[2*K:3*K]
  muY = theta[3*K:4*K]
  w = theta[4*K:5*K] 
  print(str)
  print("  w", w)
  print("  muX", muX)
  print("  muY", muY)
  return

def maxXYDistance( theta0, thetaMask, theta1):
  K = int( theta0.size / 5 )
  mu0X = theta0[2*K:3*K]
  mu0Y = theta0[3*K:4*K]
  w0 = theta0[4*K:5*K]  
  mu1X = theta1[2*K:3*K]
  mu1Y = theta1[3*K:4*K]
  w1 = theta1[4*K:5*K]
  xMax = np.max( np.abs(mu0X - mu1X)*thetaMask )
  yMax = np.max( np.abs(mu0Y - mu1Y*thetaMask) )
  return (xMax, yMax)


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
  print("??? z1", z1)
  # Cathodes
  cath0 = np.zeros( x0.size, dtype=np.int16 )
  cath1 = np.ones( x1.size, dtype=np.int16 )

  # Low Charge cutoff
  # zMinBefore = min( np.min(z0), np.min(z1) )
  zMinBefore = np.min( np.hstack( [z0, z1] ) )
  # zMaxBefore = max( np.max(z0), np.max(z1) )
  zMaxBefore = np.max( np.hstack( [z0, z1] ) )
  lowCutoff  = minRatioCh * zMaxBefore
  (x0r, y0r, dx0r, dy0r, cath0r, z0r) = removePads( x0, y0, dx0, dy0, cath0, z0, ( z0 < lowCutoff ) )
  (x1r, y1r, dx1r, dy1r, cath1r, z1r) = removePads( x1, y1, dx1, dy1, cath1, z1, ( z1 < lowCutoff ) )
  N0 = x0r.size
  N1 = x1r.size
  (x, y, dx, dy) = mergePads( x0r, y0r, dx0r, dy0r, x1r, y1r, dx1r, dy1r)
  xyDxy = padToXYdXY( x, y, dx, dy)
  z = np.hstack( [z0r, z1r] )
  cath = np.hstack( [ np.zeros( (N0), dtype=np.int16), 
                      np.ones( (N1), dtype=np.int16) ] )
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