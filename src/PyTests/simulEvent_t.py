#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

import C.PyCWrapper as PCWrap
import Util.plot as uPlt
import Util.dataTools as tUtil
import Util.geometry as uGeom

def spacialFilterTheta( theta, dx, dy ):
  K = theta.size // 5
  if K == 0 : return
  ( w, muX, muY, varX, varY) = tUtil.thetaAsWMuVar( theta )
  newW = []
  newMuX = []
  newMuY = []
  newVarX = []
  newVarY = []
  done = np.zeros( K )
  for k0 in range(0,K):
    if not done[k0] :
      muX0 = muX[k0]
      muY0 = muY[k0]
      # for k range(k0+1,K):
      k = k0+1
      inX = np.abs( muX0 - muX[k:])  < dx
      inY = np.abs( muY0 - muY[k:])  < dy
      inXY = inX * inY
      if np.sum( inXY > 0):
        newMuX.append( muX0 )
        newMuY.append( muY0 )
        newVarX.append( varX[k0] )
        newVarY.append( varY[k0] )
        idx = np.where( inXY == 1)[0] + k
        done[idx] = 1
        # newIdx.append( idx ))
        newW.append( np.sum(w[idx]) + w[k0])
  print( newW )
  print( newMuX)
  
  w = np.array( newW )
  muX = np.array( newMuX )
  muY = np.array( newMuY )
  varX = np.array( newVarX )
  varY = np.array( newVarY )
  theta = tUtil.asTheta( w, muX, muY, varX, varY )
  return theta
  
def seedGenerator(x, y, dx, dy, z, chId):
  N = x.size
  
  sumZ = np.sum( z)
  w = z / sumZ
  muX = x
  muY = y
  varX = dx
  varY = dy
  """ 
  sumZ = np.sum( z[::4])
  w = z[::4] / sumZ
  muX = x[::4]
  muY = y[::4]
  varX = dx[::4] 
  varY = dy[::4]  
  """
  theta = tUtil.asTheta( w, muX, muY, varX, varY)
  return theta

def compute2DMathiesonMixturePadIntegrals(x, y, dx, dy, theta, chId ):
  xyInfSup = tUtil.padToXYInfSup( x, y, dx, dy )
  z = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, chId )
  return z

if __name__ == "__main__":
    
    pcWrap = PCWrap.setupPyCWrapper()
    pcWrap.initMathieson()
    
 
    chId = 2
    
    # - Theta  (-1.5 < muX/Y < 1.5)
    # - Max Charge
    # - Min Charge
    K = 3
    Nx = 20
    Ny = 2
    minCh = 5.0
    maxCh = 600.0
    readObj = True
    if ( not readObj ):
      simul = tUtil.SimulCluster( Nx, Ny )
      # Build the pads
      #
      ( padCath0, padCath1, thetai) = simul.buildCluster( chId, K, minCh, maxCh, 1.1 * maxCh)

    else:
      simul = tUtil.SimulCluster.read()
    #
    x0, y0, dx0, dy0, cath0, saturated0, z0 = simul.padCath0
    x1, y1, dx1, dy1, cath1, saturated1, z1 = simul.padCath1
    thetai = simul.theta
    ( xyDxy, cath, saturated, z ) = simul.getMergedPads()  
    simul.write()
    #
    nSaturated = np.sum(saturated)
    print("# nSaturated", nSaturated)
    nFigRow = 2; nFigCol = 4
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(15, 7))
    for iRow in range( nFigRow):
      for iCol in range( nFigCol):
        ax[iRow,iCol].set_xlim( simul.gridXLimits[0], simul.gridXLimits[1])
        ax[iRow,iCol].set_ylim( simul.gridYLimits[0], simul.gridYLimits[1])

    uPlt.setLUTScale( 0, max( np.max(z0), np.max(z1) )  )
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  title="Mathieson  cath-0", doLimits=False )
    uPlt.drawPads( fig, ax[1,0], x1, y1, dx1, dy1, z1,  title="Mathieson  cath-1", doLimits=False )
    ax[0,0].plot( x0[saturated0==1], y0[saturated0==1], "o", color='black', markersize=3 )
    ax[1,0].plot( x1[saturated1==1], y1[saturated1==1], "o", color='black', markersize=3 )
    uPlt.drawPads( fig, ax[0,1], x0, y0, dx0, dy0, z0, doLimits=False, displayLUT=False, alpha=0.5)
    uPlt.drawPads( fig, ax[0,1], x1, y1, dx1, dy1, z1,  title="Mathieson  both cath", doLimits=False, alpha=0.5)    
    # uPlt.drawPads( fig, ax[1,0], x0r, y0r, dx0r, dy0r, z0r,  title="Mathieson  cath-0 - Removed pads" )
    # uPlt.drawPads( fig, ax[1,1], x1r, y1r, dx1r, dy1r, z1r,  title="Mathieson  cath-1 - Removed pads" )
    #
    uPlt.drawModelComponents( ax[0,1], thetai, color='black', pattern="+" ) 
    uPlt.drawModelComponents( ax[0,1], thetai, color='black', pattern="show w" ) 
    
    fig.suptitle( "Filtering pads to a Mathieson Mixture")
    
  
    # Projection on one plane
    (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.projectChargeOnOnePlane(
                        x0, dx0, y0, dy0, x1, dx1, y1, dy1, z0, z1)
    chProj = 0.5*(chA + chB)
    uPlt.setLUTScale( 0, 1.2 * np.max( chProj) )
    uPlt.drawPads( fig, ax[1,1], xProj, yProj, dxProj, dyProj, chProj,  title="Projection",  doLimits=False)
    # uPlt.drawPads( fig, ax[0,3], xProj, yProj, dxProj, dyProj, chA,  title="Projection",  doLimits=False)
    # uPlt.drawPads( fig, ax[1,3], xProj, yProj, dxProj, dyProj, chB,  title="Projection",  doLimits=False)
    
    # Connected-Components
    #
    nbrGroups, padGrp = PCWrap.getConnectedComponentsOfProjPads()
    #
    print( "# of groups, pad group ", nbrGroups, padGrp, np.max(padGrp) )
    # uPlt.setLUTScale( 0, np.max(padGrp)  )
    # uPlt.drawPads( fig, ax[0,2], xProj, yProj, dxProj, dyProj, padGrp, alpha=1.0, title="Pad group", doLimits=False ) 
    # Process
    nbrHits = PCWrap.clusterProcess( xyDxy, cath, saturated, z, chId )
    (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
    # EM
    thetaEMFinal = PCWrap.collectThetaEMFinal()
    uPlt.drawModelComponents( ax[1,1], thetaEMFinal, color="gray", pattern='o')
    uPlt.drawModelComponents( ax[1,1], thetaResult, color='black', pattern="show w" )     
    ( w, muX, muY, _, _) = tUtil.thetaAsWMuVar( thetai)
    # print("theta0 w", w)
    # print("       muX", muX)
    # print("       muY", muY)
    ( w, muX, muY, _, _) = tUtil.thetaAsWMuVar( thetaResult)
    # print("thetaResult w", w)
    # print("            muX", muX)
    # print("            muY", muY)
    # Residual
    residual = PCWrap.collectResidual( )
    maxResidual = np.max( residual )
    minResidual = np.min( residual )  
    uPlt.setLUTScale( minResidual, maxResidual  )
    print("mi/mx residual", minResidual, maxResidual)
    print("residual.size, xProj.size", residual.size, xProj.size)
    if residual.size == xProj.size :
      uPlt.drawPads( fig, ax[1,2], xProj, yProj, dxProj, dyProj, residual, alpha=1.0, title="Residual", doLimits=False ) 
    # Residual Mathieson
    nGroup = np.max( padGrp )
    zMat = np.zeros( chProj.size )
    for g in range(nGroup):
        idx = np.where( padGrp == (g+1) )[0]
        sumProj = np.sum( chProj[idx])
        thetaIdx = np.where( thetaToGrp == (g+1) )
        (w, muX, muY, varX, varY) = tUtil.thetaAsWMuVar( thetaResult )
        wg = w[thetaIdx]
        muXg = muX[thetaIdx]
        muYg = muY[thetaIdx]
        varXg = varX[thetaIdx]
        varYg = varY[thetaIdx]
        thetaGrp = tUtil.asTheta(  wg, muXg, muYg, varXg, varYg )
        
        zg = sumProj * compute2DMathiesonMixturePadIntegrals( xProj[idx], yProj[idx], dxProj[idx], dyProj[idx], thetaGrp, chId)
        zMat[idx] = zg
    residual = chProj - zMat
    maxResidual = np.max( residual )
    minResidual = np.min( residual )
    uPlt.setLUTScale( minResidual, maxResidual  )
    uPlt.drawPads( fig, ax[1,3], xProj, yProj, dxProj, dyProj, residual, alpha=1.0, title="Residual", doLimits=False )
    # Laplacian
    laplacian = PCWrap.collectLaplacian( )
    uPlt.setLUTScale( 0.0, np.max(laplacian))
    uPlt.drawPads( fig, ax[0,2], xProj, yProj, dxProj, dyProj, laplacian, doLimits=False, alpha=1.0 )
    # uPlt.drawModelComponents( ax[0,2], thetaInit, color="red", pattern='o')
    ax[0,2].set_title("Laplacian & theta init.")
    # Min/Max proj
    minProj, maxProj = PCWrap.collectProjectedMinMax( xProj.size )
    # zzz = maxProj - minProj
    zzz = minProj
    uPlt.setLUTScale( 0.0, np.max( zzz) )
    uPlt.drawPads( fig, ax[0,3], xProj, yProj, dxProj, dyProj, zzz, doLimits=False, alpha=1.0 )
    # uPlt.drawModelComponents( ax[0,2], thetaInit, color="red", pattern='o')
    # Local Max
    xyDxy0 = tUtil.asXYdXY(x0, y0, dx0, dy0)
    xyDxy1 = tUtil.asXYdXY(x1, y1, dx1, dy1)
    (localXMax, localYMax) = uGeom.findLocalMax( xyDxy0, xyDxy1, z0, z1 )
    print("localMax", localXMax, localYMax)
    input("next ?")
    uPlt.drawPoints(  ax[0,3], localXMax, localYMax, color='black', pattern="o" )
    ax[0,3].set_title("Min Proj & theta init.")
    # 
    # Generator
    #
    if 0:
      verbose = 0
      # mode : cstVar (1bit)
      mode = 1 
      LConv = 1.0e-6
      N = xProj.size
      x = np.hstack( [x0, x1])
      y = np.hstack( [y0, y1])
      dx = np.hstack( [dx0, dx1])
      dy = np.hstack( [dy0, dy1])
      z = np.hstack( [z0, z1])
      
      thetaT = seedGenerator( x, y, dx, dy, z, chId)
      # xyDxy = tUtil.asXYdXY ( xProj, yProj, dxProj, dyProj)
      # def weightedEMLoop( xyDxy, saturated, zObs, thetai, thetaMask, mode, LConvergence, verbose):
      saturated = np.zeros( N, dtype=np.int16)
      thetaMask = np.ones( N, dtype=np.int16 )
      theta, logL = PCWrap.weightedEMLoop( xyDxy, saturated, z, thetaT, thetaMask, mode, LConv, verbose ) 
      uPlt.drawModelComponents( ax[1,3], theta, color="gray", pattern='rect')
  
      # Filter
      thetaFilter = spacialFilterTheta( theta, dxProj[0], dyProj[0] )
      thetaMask = np.ones( thetaFilter.size // 5, dtype=np.int16 )
      # theta, logL = PCWrap.weightedEMLoop( xyDxy, saturated, chProj, thetaFilter, thetaMask, mode, LConv, verbose )     
      print("filtered theta", theta)
      uPlt.drawModelComponents( ax[1,3], theta, color="black", pattern='x')
      ( w, muX, muY, _, _) = tUtil.thetaAsWMuVar( thetai)
      print("theta0 w", w)
      print("       muX", muX)
      print("       muY", muY)
      ( w, muX, muY, _, _) = tUtil.thetaAsWMuVar( theta)
      print("thetaResult w", w)
      print("            muX", muX)
      print("            muY", muY)    
    plt.show()
    
    # free memory in Pad-Processing
    PCWrap.freeMemoryPadProcessing()

     