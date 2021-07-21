#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import sys
import numpy as np
import matplotlib.pyplot as plt

# Cluster Processing
import C.PyCWrapper as PCWrap
import Util.plot as uPlt
import Util.dataTools as tUtil
import Util.geometry as geom

# Reading MC, Reco, ... Data
import Util.IORun2 as IO

def processPreCluster( pc, display=False ):
  displayBefore = False
  (id, pads, hits ) = pc
  ( bc, orbit, iROF, DEId, nbrOfPads) = id
  chId = DEId // 100
  ( xi, yi, dxi, dyi, chi, saturated, cathi, adc ) = pads
  (nHits, xr, yr, errX, errY, uid, startPadIdx, nPadIdx) = hits
  
  print("###")
  print("### New Pre Cluster bc=", bc,", orbit=", orbit, ", iROF=", iROF)
  print("###")
  # Print cluster info
  print("# DEIds", DEId)
  print("# Nbr of pads:", xi.size)
  print("# Saturated pads", np.sum( saturated))
  # print("# Calibrated pads", np.sum( preClusters.padCalibrated[ev][pc]))
  # xyDxy
  xyDxy = tUtil.padToXYdXY( xi, yi, dxi, dyi)
  x0  = xi[cathi==0]
  y0  = yi[cathi==0]
  dx0 = dxi[cathi==0]
  dy0 = dyi[cathi==0]
  z0  = chi[cathi==0]
  x1  = xi[cathi==1]
  y1  = yi[cathi==1]
  dx1 = dxi[cathi==1]
  dy1 = dyi[cathi==1]
  z1  = chi[cathi==1]
    
  if 0 and displayBefore:
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )

    xMin=np.min(xi-dxi)
    xMax=np.max(xi+dxi)
    yMin=np.min(yi-dyi)
    yMax=np.max(yi+dyi)
    zMax = np.max( chi )
    uPlt.setLUTScale( 0.0, zMax ) 
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=1.0)
    uPlt.drawPads( fig, ax[0,1], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=1.0, )
    uPlt.drawPads( fig, ax[1,0], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[1,0], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    # Reco 
    ax[1,0].plot( xr, yr, "+", color='white', markersize=3 )
    # 
    ax[0,0].set_xlim( xMin, xMax )
    ax[0,0].set_ylim( yMin, yMax )
    ax[0,1].set_xlim( xMin, xMax )
    ax[0,1].set_ylim( yMin, yMax )
    ax[1,0].set_xlim( xMin, xMax )
    ax[1,0].set_ylim( yMin, yMax )
    #
    t = r'bCrossing=%d orbit=%d iROF=%d, DEId=%d' % (bc, orbit, iROF, DEId )
    fig.suptitle(t)
    plt.show()
    
  #
  #  Do Clustering
  #
  try:
    nbrHits = PCWrap.clusterProcess( xyDxy, cathi, saturated, chi, chId )
    print("nbrHits", nbrHits)
    (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
    print("Cluster Processing Results:")
    print("  theta   :", thetaResult)
    print("  thetaGrp:", thetaToGrp)
    # Returns the fit status
    (xyDxyResult, chResult, padToGrp) = PCWrap.collectPadsAndCharges()
    # print("xyDxyResult ... ", xyDxyResult)
    # print("charge ... ", chResult)
    # print("padToGrp", padToGrp)
    # Return the projection
    (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.copyProjectedPads()
    zProj = (chA + chB)*0.5

  except:
    print("except ???")
  plt.show()
  # New find local Max
  xyDxy0 = tUtil.asXYdXY( x0, y0, dx0, dy0 )
  xyDxy1 = tUtil.asXYdXY( x1, y1, dx1, dy1 )
  xl, yl = geom.findLocalMax( xyDxy0, xyDxy1, z0, z1 )
  
  # display = True
  diffNbrOfSeeds = (xr.size != xl.size)
  if chId > 5:
      display = True
  else : display = False
  
  display = True
  display = True if nbrHits > 10 else False
  display = diffNbrOfSeeds and chId > 5
  if display:
    nFigRow = 2; nFigCol = 3
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(15, 7))
    x0  = xi[cathi==0]
    y0  = yi[cathi==0]
    dx0 = dxi[cathi==0]
    dy0 = dyi[cathi==0]
    sat0  = saturated[cathi==0]
    z0  = chi[cathi==0]
    x1  = xi[cathi==1]
    y1  = yi[cathi==1]
    dx1 = dxi[cathi==1]
    dy1 = dyi[cathi==1]
    z1  = chi[cathi==1]
    sat1  = saturated[cathi==1]
    xMin=np.min(xi-dxi)
    xMax=np.max(xi+dxi)
    yMin=np.min(yi-dyi)
    yMax=np.max(yi+dyi)
    zMax = np.max( chi )
    uPlt.setLUTScale( 0.0, zMax ) 

    for iRow in range( nFigRow):
      for iCol in range( nFigCol):
        ax[iRow,iCol].set_xlim( xMin, xMax)
        ax[iRow,iCol].set_ylim( yMin, yMax)
    uPlt.drawPads( fig, ax[0,1], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[0,1], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  doLimits=False )
    uPlt.drawPads( fig, ax[1,0], x1, y1, dx1, dy1, z1,  doLimits=False )  
    # Saturated
    ax[0,0].plot( x0[sat0==1], y0[sat0==1], "o", color='blue', markersize=3 )
    ax[1,0].plot( x1[sat1==1], y1[sat1==1], "o", color='blue', markersize=3 )
    
    # Reco 
    ax[0,1].plot( xr, yr, "+", color='white', markersize=4 )
    ax[0,0].plot( xr, yr, "+", color='white', markersize=4 )
    ax[1,0].plot( xr, yr, "+", color='white', markersize=4 )
    #
    (xr, yr, dxr, dyr) = tUtil.asXYdXdY( xyDxyResult)
    # uPlt.drawPads( fig, ax[1,0], xProj, yProj, dxProj, dyProj, zProj, doLimits=False, alpha=1.0 )
    uPlt.drawPads( fig, ax[1,1], xProj, yProj, dxProj, dyProj, zProj, doLimits=False, alpha=1.0 )
    uPlt.drawModelComponents( ax[0,1], thetaResult, color="black", pattern='x')
    uPlt.drawModelComponents( ax[1,1], thetaResult, color="black", pattern='x')
    # uPlt.drawPoints( ax[1,1], xr, yr, color='black', pattern="o" )
    
    # New find Local max
    uPlt.drawPoints( ax[0,0], xl, yl, color='black', pattern='+')
    uPlt.drawPoints( ax[1,0], xl, yl, color='black', pattern='+')
    
    ax[1,0].set_xlim( xMin, xMax )
    ax[1,0].set_ylim( yMin, yMax )
    ax[1,1].set_xlim( xMin, xMax )
    ax[1,1].set_ylim( yMin, yMax )
    
    #
    t = r'bCrossing=%d orbit=%d iROF=%d, DEId=%d sat=%d' % (bc, orbit, iROF, DEId, np.sum( saturated ))
    fig.suptitle(t)
    plt.show()

  #
  # free memory in Pad-Processing
  PCWrap.freeMemoryPadProcessing()

  #
  return

def processEvent( preClusters,  ev):
  nbrOfPreClusters = len( preClusters.padId[ev] )
  for pc in range(0, nbrOfPreClusters):
    processPreCluster( preClusters, ev, pc, display=False)


if __name__ == "__main__":
    
  pcWrap = PCWrap.setupPyCWrapper()
  pcWrap.initMathieson()
  
  # Read MC data
  reco = IO.Run2PreCluster(fileName="../Run2Data/recoRun2-100.dat")
  """
  for pc in reco:
    (id, pads, hits ) = pc
    ( bc, orbit, iROF, DEId, nbrOfPads) = id
    if (iROF == 85): 
      processPreCluster ( pc )
      sys.exit()
  """
  for pc in reco:
    processPreCluster ( pc, display=True )
  """
  pc = reco.readPreCluster( 0, 93, 464)
  processPreCluster ( pc )
  """
  
  # reco.read(verbose=True)
  # nPreClusters = len(reco.padX )
  # for ipc in range(0, nPreClusters ):
    # processEvent( reco, ipc )    

    