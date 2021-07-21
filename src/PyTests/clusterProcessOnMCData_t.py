#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

# Cluster Processing
import C.PyCWrapper as PCWrap
import Util.plot as uPlt
import Util.dataTools as tUtil

# Reading MC, Reco, ... Data
import Util.IOv5 as IO

def processPreCluster( preClusters, ev, pc, display=False ):
  displayBefore = False
  print("###")
  print("### New Pre Cluster", pc,"/", ev)
  print("###")
  xi = preClusters.padX[ev][pc]
  dxi = preClusters.padDX[ev][pc]
  yi = preClusters.padY[ev][pc]
  dyi = preClusters.padDY[ev][pc]
  cathi = preClusters.padCath[ev][pc].astype( np.int16)
  chi = preClusters.padCharge[ev][pc]
  chIds = preClusters.padChId[ev][pc]
  chIds = np.unique( chIds )
  DEIds = preClusters.padDEId[ev][pc]
  DEIds = np.unique( DEIds )
  if (DEIds.size != 1):
    input("Bab number of DEIds")
  if (chIds.size != 1):
    input("Bab number of DEIds")
  chId = np.unique( chIds )[0]

 
  # Print cluster info
  print("# DEIds", DEIds)
  print("# Nbr of pads:", xi.size)
  print("# Saturated pads", np.sum( preClusters.padSaturated[ev][pc]))
  print("# Calibrated pads", np.sum( preClusters.padCalibrated[ev][pc]))
  if ( xi.size !=  np.sum( preClusters.padCalibrated[ev][pc])) :
      input("pb with calibrated pads")
  saturated = preClusters.padSaturated[ev][pc].astype( np.int16)
  # xyDxy
  xyDxy = tUtil.padToXYdXY( xi, yi, dxi, dyi)
  
  if displayBefore:
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )
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
    ax[0,0].set_xlim( xMin, xMax )
    ax[0,0].set_ylim( yMin, yMax )
    ax[0,1].set_xlim( xMin, xMax )
    ax[0,1].set_ylim( yMin, yMax )
    ax[1,0].set_xlim( xMin, xMax )
    ax[1,0].set_ylim( yMin, yMax )
    plt.show()
  #
  #  Do Clustering
  #
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
  
  if display:
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )
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
    xMin=np.min(xi-dxi)
    xMax=np.max(xi+dxi)
    yMin=np.min(yi-dyi)
    yMax=np.max(yi+dyi)
    zMax = np.max( chi )
    uPlt.setLUTScale( 0.0, zMax ) 
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[0,0], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    ax[0,0].set_xlim( xMin, xMax )
    ax[0,0].set_ylim( yMin, yMax )
    #
    (xr, yr, dxr, dyr) = tUtil.asXYdXdY( xyDxyResult)
    uPlt.drawPads( fig, ax[1,0], xProj, yProj, dxProj, dyProj, chA, doLimits=False, alpha=1.0 )
    uPlt.drawPads( fig, ax[1,1], xProj, yProj, dxProj, dyProj, chB, doLimits=False, alpha=1.0 )
    ax[1,0].set_xlim( xMin, xMax )
    ax[1,0].set_ylim( yMin, yMax )
    ax[1,1].set_xlim( xMin, xMax )
    ax[1,1].set_ylim( yMin, yMax )
    
    #
    fig.suptitle( "Test" )
    plt.show()

  #
  # free memory in Pad-Processing
  PCWrap.freeMemoryPadProcessing()

  #
  return

def processEvent( preClusters,  ev):
  nbrOfPreClusters = len( preClusters.padId[ev] )
  for pc in range(0, nbrOfPreClusters):
    processPreCluster( preClusters, ev, pc, display=True)


if __name__ == "__main__":
    
  pcWrap = PCWrap.setupPyCWrapper()
  pcWrap.initMathieson()
  
  # Read MC data
  mcData = IO.MCData(fileName="../MCData/MCDataDump.dat")
  mcData.read()
  # Read PreClusters
  recoData = IO.PreCluster(fileName="../MCData/RecoDataDump.dat")
  recoData.read()
  print( "recoData.padChIdMinMax=", recoData.padChIdMinMax )
  print( "recoData.rClusterChIdMinMax=", recoData.rClusterChIdMinMax )
  nEvents = len( recoData.padX )
  for ev in range(0, nEvents):
    processEvent( recoData, ev )    

    