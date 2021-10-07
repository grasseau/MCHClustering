#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt
import math

import C.PyCWrapper as PCWrap
import Util.dataTools as tUtil
import Util.plot as uPlt

sigX = 0.1734
sigY = 0.1781
sqrt2 = math.sqrt( 2.0 )

def getPad( x, y, dx, dy, mux, muy ):
  eps = 10.0e-3
  xInf = x - dx 
  xSup = x + dx 
  yInf = y - dy 
  ySup = y + dy
  maskX = np.bitwise_and( (xInf - eps) < mux, mux < (xSup +eps))
  maskY = np.bitwise_and( (yInf - eps) < muy, muy < (ySup +eps))
  mask = np.bitwise_and( maskX, maskY )
  idx = np.where( mask) [0]
  if (idx.size !=1):
     print("getPad. several pads or 0", idx.size )
  return idx[0]

def spanMu( x, y, dx, dy, z, mux, muy, chId, cutOff ):
  muIdx = getPad( x, y, dx, dy, mux, muy )
  zMuMax = z[muIdx] 
  xInfM = x - dx - mux
  xSupM = x + dx - muy
  yInfM = y - dy - muy
  ySupM = y + dy - muy
  f = np.max (tUtil.compute2DPadIntegrals( xInfM, xSupM, yInfM, ySupM, chId ))
  zz = zMuMax/ f * tUtil.compute2DPadIntegrals( xInfM, xSupM, yInfM, ySupM, chId )
  idx = ( zz > cutOff)
  z = z[idx]
  x = x[idx]
  y = y[idx]
  dx = dx[idx]
  dy = dy[idx]
  xMin = np.min( x - dx)
  xMax = np.min( x + dx)
  yMin = np.min( y - dy)
  yMax = np.min( y + dy)
  xWidth = 0.5*(xMin + xMax)
  yWidth = 0.5*(yMin + yMax)
  return xWidth, yWidth

def dI( a ):
  dX = (1.0 - math.erf( (a*sigX ) / (sqrt2*sigX ) )) 
  # dY = (1.0 - math.erf( (a*sigY ) / (sqrt2*sigY ) )) / 2.0 
  # return np.sqrt( dX*dY )
  return dX 

def findSigmaFactor( x, y, dx, dy, z, muX, muY, chId, cutoff ):
    
  xInf = x - dx - muX
  xSup = x + dx - muX
  yInf = y - dy - muY
  ySup = y + dy - muY
  
  # zMax = np.max( z )
  idx = getPad( x, y, dx, dy, muX, muY )
  zMax = z[idx]
  zTh = tUtil.compute2DPadIntegrals( xInf, xSup, yInf, ySup, chId )
  zMaxTh = np.max( zTh )
  # zSum = np.sum( z )
  zSumTh = np.sum( zTh )
  norm = zMax / zMaxTh 
  IResidu = 1.0 - zSumTh
  
  aMin = 0.0 
  aMax = 100.0
  u = 1.0 + IResidu
  a = (aMax + aMin)*0.5
  while ( np.abs( u -  IResidu)/IResidu > 0.1 ):
    a = (aMax + aMin)*0.5
    u = dI(a) 
    print("a, u", a, u)
    if ( u <  IResidu ):
      aMax = a
    else:
      aMin = a
  return a

def findSigmafactorV0( I, cutoff ):
  aMin = 0.0 
  aMax = 100.0
  u = 1.0 + cutoff
  a = (aMax + aMin)*0.5
  while ( np.abs( u - cutoff )/cutoff > 0.1 ):
    a = (aMax + aMin)*0.5
    u = dI(a) * I
    print("a, u", a, u)
    if ( u < cutoff ):
      aMax = a
    else:
      aMin = a
  return a

if __name__ == "__main__":
    
    pcWrap = PCWrap.setupPyCWrapper()
    pcWrap.initMathieson()
    
    K = 1
    
    w   = np.array( [1.0] )
    muX = np.array( [0.5] )
    muY = np.array( [0.5] )
    varX = np.array( [0.1 * 0.1] )
    varY = np.array( [0.2 * 0.2] )
    
    
    Nx = 20
    Ny = 20
    chId = 1
    
    x, y, dx, dy = tUtil.buildPads( Nx, Ny, -0.2, 1.2, -0.2, 1.2 )
    minCh = 5.0
    maxCh = 500.0
    K = 1
    """"
    simul = tUtil.SimulCluster( Nx, Ny )
    ( padCath0, padCath1, thetai) = simul.buildCluster( chId, K, minCh, maxCh, 1.1*maxCh)
    x, y, dx, dy, cath, saturated, z = simul.padCath0 
    thetai = simul.theta
    # ( xyDxy, cath, saturated, z ) = simul.getMergedPads()
    ( w, muX, muY, varX, varY ) = tUtil.thetaAsWMuVar( thetai )
    """
    #
    # Merge pads
    N = x.size
    # (x, y, dx, dy) = tUtil.mergePads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 )
    # cath = np.zeros( x.size, dtype=np.int32 )
    #
    # Compute charge on the pads
    #
    # xyInfM
    xInfM = x - dx - muX[0]
    xSupM = x + dx - muX[0]
    yInfM = y - dy - muY[0]
    ySupM = y + dy - muY[0]
    
    # Compute Mathieson
    ICth = 1000
    z = ICth * tUtil.compute2DPadIntegrals( xInfM, xSupM, yInfM, ySupM, chId )
    print("sum z", np.sum(z))    
    print("max z", np.max(z))    
    
    # Suppress pad < 5
    idx = np.where( z>=5 )[0]
    z = z[idx]
    x = x[idx]
    y = y[idx]
    dx = dx[idx]
    dy = dy[idx]
    print(z)
   
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    uPlt.setLUTScale( 0, np.max(z) )
    uPlt.drawPads( fig, ax[0,0], x, y, dx, dy, z,  title="Mathieson (%d,%d)" % (0,0))
    idx0 = np.where( z < 10.0)[0]
    # uPlt.drawPoints( ax[0,0], x[idx], y[idx], color="white" )
    uPlt.drawPoints( ax[0,0], muX, muY)
    du = sigX * 2
    dv = sigY * 2
    diamondX = np.array( [muX+du, muX, muX-du, muX, muX+du] )
    diamondY = np.array( [muY, muY+dv, muY, muY-dv, muY] )
    spanX, spanY = spanMu( x, y, dx, dy, z, muX, muY, chId, 5.0 )
    print("spanX, spanY", spanX, spanY)
    # uPlt.drawPoints( ax[0,0], diamondX, diamondY)  
    
    ax[0,0].plot( diamondX, diamondY, color="white")
    """
    X = x[0:Nx]
    Y = y[::Nx]
    Z = z.reshape( (Nx,Ny))
    print("X", X)
    print("y", Y)
    print("Z", Z)
    """
    # uPlt.setLUTScale( 0, np.max(zi) )
    # uPlt.drawPads( fig, ax[0,1], x, y, dx, dy, zi,  title="Init. Gaussian (%d,%d)" % (0,0))
    # uPlt.setLUTScale( 0, np.max(zf) )
    # uPlt.drawPads( fig, ax[1,0], x, y, dx, dy, zf,  title="Final Gaussian (%d,%d)" % (0,0))
    uPlt.setLUTScale( 0, np.max(z) )
    uPlt.drawPads( fig, ax[1,1], x, y, dx, dy, z,  title="Levels (%d,%d)" % (0,0))
    # u = plt.contour( X, Y, Z )


    print( math.erf( (0 ) / (sqrt2* sigX )) )
    print( math.erf( (2.0*sigX ) / (sqrt2*sigX ) ))
    # a = findSigmafactor( 1.0, 0.005)
    # a = findSigmafactor( 1.0, 0.12)
    # a = findSigmafactor( 1.0, 1.0 - (0.12 + 0.2))
    a = findSigmaFactor( x, y, dx, dy, z, muX, muY, chId, 5 )
    print("a res, a*sigX, a*sigY", a, a*sigX, a*sigY )
    # dx = sigX * 2
    # dy = sigY * 2
    du = sigX * a
    dv = sigY * a
    plt.plot( [0.5+du, 0.5, 0.5-du, 0.5, 0.5+du], [0.5, 0.5+dv, 0.5, 0.5-dv, 0.5], color="white")
    argMaxIdx = np.argmax( z )
    zMax = z[argMaxIdx] 
    xMax = x[argMaxIdx] 
    yMax = y[argMaxIdx] 
    dxMax = dx[argMaxIdx] 
    dyMax = dy[argMaxIdx] 
    print( "x,y,z max", xMax, yMax, dxMax, dyMax, zMax)
    print("sum Z", np.sum( z))
    IC = np.sum(z)
    f = np.max (tUtil.compute2DPadIntegrals( xInfM[idx], xSupM[idx], yInfM[idx], ySupM[idx], chId ))
    
    zz = zMax/ f * tUtil.compute2DPadIntegrals( xInfM[idx], xSupM[idx], yInfM[idx], ySupM[idx], chId )
    print("sum zz rebuilt", np.sum(zz))
    print("max zz rebult", np.max(zz))
    # print(u.__dict__)
    # print(u.levels)
    # uPlt.drawPads( fig, ax[0,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], z[cath==1],  title="Mathieson (%d,%d)" % (0,0))
    # uPlt.drawPads( fig, ax[1,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], zGauss[cath==0],  title="Gaussian (%d,%d)" % (0,0))
    # uPlt.drawPads( fig, ax[1,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], zGauss[cath==1],  title="Gaussian (%d,%d)" % (0,0))    
    plt.show()
