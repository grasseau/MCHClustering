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
import Util.geometry as geom
import Util.dataTools as dUtil
# Reading MC, Reco, ... Data
import Util.IOv5 as IO
# Analyses Tool Kit
import analyseToolKit as aTK

def inspectPreCluster( preClusters, ev, pc, mcObj, display=True ):
  displayBefore = False
  print("[python] ###")
  print("[python] ### New Pre Cluster", pc,"/", ev)
  print("[python] ###")
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
  nPads = xi.size
  
  # Print cluster info
  sumSaturated = np.sum( preClusters.padSaturated[ev][pc])
  print("[python] # DEIds", DEIds)
  print("[python] # Nbr of pads:", nPads)
  print("[python] # Saturated pads", np.sum( preClusters.padSaturated[ev][pc]))
  print("[python] # Calibrated pads", np.sum( preClusters.padCalibrated[ev][pc]))
  if ( xi.size !=  np.sum( preClusters.padCalibrated[ev][pc])) :
      input("pb with calibrated pads")
  saturated = preClusters.padSaturated[ev][pc].astype( np.int16)
  # xyDxy
  xyDxy = dUtil.padToXYdXY( xi, yi, dxi, dyi)

  if displayBefore:
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(17, 7) )
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
  print("[python] nbrHits", nbrHits)
  (thetaf, thetaToGrp) = PCWrap.collectTheta( nbrHits)
  print("[python] Cluster Processing Results:")
  print("[python]   theta   :", thetaf)
  print("[python]   thetaGrp:", thetaToGrp)
  # Returns the fit status
  (xyDxyResult, chResult, padToGrp) = PCWrap.collectPadsAndCharges()
  # print("xyDxyResult ... ", xyDxyResult)
  # print("charge ... ", chResult)
  # print("padToGrp", padToGrp)
  # Return the projection
  
  (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.copyProjectedPads()
  chProj = (chA + chB)*0.5
  res = aTK.matchingThetaWithMC(thetaf, preClusters, ev, pc, mcObj)
  (pc_, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) = res
  # Ratio
  if (TP - FP + FN) != 0 : 
    ratioTP = TP / (TP - FP + FN)
  else: 
    ratioTP = 0
  #  Reco Infos
  res = aTK.matchingRecoWithMC(preClusters, ev, pc, mcObj)
  ( _, _, _, recoTP, recoFP,recoFN, recoDMin, recoDxMin, recoDyMin, _, _) = res
  recoRatioTP = (recoTP - TP) 
  recoRatioFP = (recoFP - FP) 
  recoRatioFN = (recoFN - FN) 
  
  if (match > 0.33):
      print( "[python] EM   ev=", ev, "pc=", pc," TP, FP, FN", TP, FP, FN)
      print( "[python] Reco ev=", ev, "pc=", pc," TP, FP, FN", recoTP, recoFP, recoFN)

  if (dxMin.size != 0):
      maxDxMin = np.max( dxMin )
      maxDyMin = np.max( dyMin )
      maxRecoDxMin = np.max( recoDxMin ) if recoDxMin.size != 0 else 1.0
      maxRecoDyMin = np.max( recoDyMin ) if recoDyMin.size != 0 else 1.0
  else:
      maxDxMin = 1.0
      maxDyMin = 1.0
      maxRecoDxMin = 1.0
      maxRecoDyMin = 1.0      
  #
  # Ratio seeds / MC hit
  thetaInit = PCWrap.collectThetaInit()
  nSeeds = thetaInit.size // 5
  frame = uPlt.getPadBox( xi, yi, dxi, dyi )
  (xdummy, ydummy) = uPlt.getMCHitsInFrame( frame, mcObj, ev, DEIds )
  if xdummy.size > 0 :
    seedRatio = nSeeds / xdummy.size
  else:
    seedRatio = 0
  #
  (w, muX, muY, varX, varY) = dUtil.thetaAsWMuVar( thetaf )
  wMin = np.min( w )
  print("[python] w", w )
  print("[python] wMin=", wMin )
  
  # Residual
  residual = PCWrap.collectResidual( )
  maxResidual = np.max( residual )
  minResidual = np.min( residual )
  sumAbsResidual =  np.sum( np.abs( residual) ) / chProj.size
  maxAbsResidual = max( maxResidual, abs( minResidual) )
  maxProj = np.max( chProj )
  ratioResidual = maxAbsResidual / maxProj
  print("[python] sumResidual", np.sum( residual) )
  print("[python] minResidual",minResidual)
  print("[python] maxResidual",maxResidual)
  print("[python] maxProj", maxProj)
  print("[python] ratioResidual", ratioResidual)
  print("[python] sumAbsResidual", sumAbsResidual)
  print("[python] sumSaturated", sumSaturated )

  #
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
  matchFlag = (match > 0.33)
  notSaturated = ( sumSaturated == 0)
  TPFlag = (( recoRatioTP > 0) or ( recoRatioFP < 0) or ( recoRatioFN < 0) )
  residualFlag = (ratioResidual > 0.70)
  severalGrp = ( max(padToGrp) > 1)
  # MergedGroup or CathGroups 
  padToCathGrp = PCWrap.collectPadToCathGroup( nPads )
  padCathGrpMax = np.max( padToCathGrp )
  thetaMaxGrp = np.max( thetaToGrp )
  padGrpMax = np.max( padToGrp )
  nRecoSeeds = preClusters.rClusterX[ev][pc].size
  xyDxy0 = dUtil.asXYdXY(  x0, y0, dx0, dy0 )
  xyDxy1 = dUtil.asXYdXY(  x1, y1, dx1, dy1 )
  xl, yl = geom.findLocalMax( xyDxy0, xyDxy1, z0, z1 )
  nNewSeeds = xl.size
  
  drawPlot = display and (match > 0.33) 
  drawPlot = display and (match > 0.33)  and ( np.sum( saturated) > 0 )
  drawPlot = display and (match > 0.33)  and ( ratioTP < 0.8 )
  drawPlot = display and (match > 0.33)  and ( ( seedRatio < 0.8 ) )
  drawPlot = display and (match > 0.33) and ( wMin < 0.005 )
  drawPlot = display
  
  drawPlot = display and (match > 0.33) 
  drawPlot = display and (match > 0.33) and (maxDxMin > 0.15 or maxDyMin > 0.1) 
  drawPlot = display and (match > 0.33) and ( (maxDxMin < (maxRecoDxMin )) or (maxDyMin < (maxRecoDyMin )) )
  drawPlot = display and (match > 0.33) and ( (maxDxMin > (maxRecoDxMin + 0.05)) or (maxDyMin > (maxRecoDyMin + 0.03)) )
  drawPlot = display and (match > 0.33) and TPFlag
  drawPlot = display  and notSaturated and (residualFlag or TPFlag)
  drawPlot = display and severalGrp 
  drawPlot = False
  drawPlot = display and residualFlag
  drawPlot = display and (match > 0.33)  and ( ( seedRatio < 0.8 ) or ( seedRatio > 1.2 ) )
  drawPlot = display and (nRecoSeeds != nNewSeeds)
  #
  """
  if (padCathGrpMax != thetaMaxGrp):
      print("padCathGrpMax differs from thetaMaxGrp; padCathGrpMax", padCathGrpMax, thetaMaxGrp, padGrpMax)
      print("padCathGrp", padToCathGrp)
      input("next")
      drawPlot = True
  """    
  if drawPlot:
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(17, 13) )

    """ ????
    xMin=np.min(xi-dxi)
    xMax=np.max(xi+dxi)
    yMin=np.min(yi-dyi)
    yMax=np.max(yi+dyi)
    """
    (xMin, xMax, yMin, yMax ) = frame
    # Pads & cathodes
    zMax = np.max( chi )
    uPlt.setLUTScale( 0.0, zMax ) 
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[0,0], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    ax[0,0].set_xlim( xMin, xMax )
    ax[0,0].set_ylim( yMin, yMax )
    uPlt.drawPoints( ax[0,0],  preClusters.rClusterX[ev][pc], preClusters.rClusterY[ev][pc], color='green', pattern='o')
    uPlt.drawMCHitsInFrame( ax[0,0], frame, mcObj, ev, DEIds )
    # ???? uPlt.drawModelComponents(ax[0,0], theta, color="black", pattern='o')
    ax[0,0].set_title("MC with Cathodes")
    #
    (xr, yr, dxr, dyr) = dUtil.asXYdXdY( xyDxyResult)
    # uPlt.drawPads( fig, ax[1,0], xr, yr, dxr, dyr, chResult, doLimits=True, alpha=1.0 )
    
    # Projection
    uPlt.setLUTScale( 0.0, np.max( chProj) * 1.2 )
    uPlt.drawPads( fig, ax[0,1], xProj, yProj, dxProj, dyProj, chProj, doLimits=False, alpha=1.0 )
    uPlt.drawModelComponents( ax[0,1], thetaf, color="red", pattern='o')
    uPlt.drawMCHitsInFrame( ax[0,1], frame, mcObj, ev, DEIds )
    ax[0, 1].set_title("Projection & theta final")
    #
    # Laplacian
    laplacian = PCWrap.collectLaplacian( )
    uPlt.setLUTScale( 0.0, 1.2)
    uPlt.drawPads( fig, ax[0,2], xProj, yProj, dxProj, dyProj, laplacian, doLimits=False, alpha=1.0 )
    uPlt.drawModelComponents( ax[0,2], thetaInit, color="red", pattern='o')
    uPlt.drawMCHitsInFrame( ax[0,2], frame, mcObj, ev, DEIds )
    ax[0,2].set_title("Laplacian & theta init.")

    #
    # EM Final
    thetaEMFinal = PCWrap.collectThetaEMFinal()
    uPlt.setLUTScale( 0.0, 1.2)
    uPlt.drawPads( fig, ax[0,3], xProj, yProj, dxProj, dyProj, laplacian, doLimits=False, alpha=1.0 )
    uPlt.drawModelComponents( ax[0,3], thetaEMFinal, color="red", pattern='o')
    uPlt.drawMCHitsInFrame( ax[0,3], frame, mcObj, ev, DEIds )
    ax[0,3].set_title("Laplacian & theta EM Final.")
    #
    # Residual
    uPlt.setLUTScale( np.min( residual ), np.max( residual ) * 1.2 )
    uPlt.drawPads( fig, ax[1,0], xProj, yProj, dxProj, dyProj, residual, doLimits=False, alpha=1.0 )
    uPlt.drawModelComponents( ax[1,0], thetaEMFinal, color="red", pattern='o')
    uPlt.drawMCHitsInFrame( ax[1,0], frame, mcObj, ev, DEIds )
    ax[1,0].set_title("Residual & theta EM final.")

    
    # Groups
    #
    # Projected Grp
    """
    grpMax = np.max( padToGrp )
    uPlt.setLUTScale( 0.0, grpMax )
    uPlt.drawPads( fig, ax[1,1], xr, yr, dxr, dyr, padToGrp, doLimits=False, alpha=0.5 )
    # uPlt.drawPads( fig, ax[1,1], xProj, yProj, dxProj, dyProj, padToGrp, doLimits=False, alpha=1.0 )
    """
    #
    # Merged Groups
    uPlt.setLUTScale( 0.0, padCathGrpMax ) 
    uPlt.drawPads( fig, ax[1,1], xi, yi, dxi, dyi, padToCathGrp,  doLimits=False, alpha=0.5 )
    ax[1,1].set_title("Group of pads")

    
    # MC versus EM
    uPlt.drawModelComponents( ax[1,2], thetaEMFinal, color="red", pattern='')
    uPlt.drawModelComponents( ax[1,2], thetaf, color="red", pattern='o')
    uPlt.drawMCHitsInFrame( ax[1,2], frame, mcObj, ev, DEIds )
    ax[1,2].set_title("MC versus EM")

    # EM versus Reco
    uPlt.drawPoints( ax[1,3],  preClusters.rClusterX[ev][pc], preClusters.rClusterY[ev][pc], color='green', pattern='o')
    uPlt.drawModelComponents( ax[1,3], thetaf, color="red", pattern='+')
    ax[1,3].set_title("EM versus Reco")
    
    # The two cathodes
    zMax = np.max( chi )
    uPlt.setLUTScale( 0.0, zMax ) 
    uPlt.drawPads( fig, ax[0,4], x0, y0, dx0, dy0, z0,  doLimits=False)
    uPlt.drawPads( fig, ax[1,4], x1, y1, dx1, dy1, z1,  doLimits=False )

    uPlt.drawPoints( ax[0,4], xl, yl, color='black', pattern='x')
    uPlt.drawPoints( ax[1,4], xl, yl, color='black', pattern='x')
    ax[0,4].set_title("Cathode 0")
    ax[1,4].set_title("Cathode 1")
    
    for i in range(ax.shape[0]):
      for j in range(ax.shape[1]):
        ax[i,j].set_xlim( xMin, xMax )
        ax[i,j].set_ylim( yMin, yMax )
    
    
    
    #
    # t = r'$\sigma=%.3f$' % xStd
    t = r'Event=%d preCluster=%d DEId=%d Saturated=%d' % (ev, pc, DEIds[0], not notSaturated)
    fig.suptitle(t)
    plt.show()

  #
  # free memory in Pad-Processing
  PCWrap.freeMemoryPadProcessing()

  #
  return thetaf

def inspectEvent( preClusters,  ev, mcObj):
  nbrOfPreClusters = len( preClusters.padId[ev] )
  evTP=0; evFP=0; evFN=0
  evNbrOfHits=0
  # Matching preClusters description
  assign = []
  allDMin = []
  # for pc in range(286, 287):
  # for pc in range(18, 19):
  # for pc in range(4, 5):
  for pc in range(0, nbrOfPreClusters):
    thetaf = inspectPreCluster( preClusters, ev, pc, mcObj, display=True)
    res = aTK.matchingThetaWithMC(thetaf, preClusters, ev, pc, mcObj)
    (pc_, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) = res
    if (match > 0.33):
        assign.append( res )
        evTP += TP; evFP += FP; evFN += FN
        allDMin.append( dMin )
        evNbrOfHits += nbrOfHits
  #
  return (evNbrOfHits, evTP, evFP, evFN, allDMin, assign)

if __name__ == "__main__":
    
  pcWrap = PCWrap.setupPyCWrapper()
  pcWrap.initMathieson()
  
  # Read MC data
  mcData = IO.MCData(fileName="../MCData/MCDataDump.dat")
  mcData.read()
  # Read PreClusters
  recoData = IO.PreCluster(fileName="../MCData/RecoDataDump.dat")
  recoData.read()
  print( "[python] recoData.padChIdMinMax=", recoData.padChIdMinMax )
  print( "[python] recoData.rClusterChIdMinMax=", recoData.rClusterChIdMinMax )
  # recoMeasure
  recoMeasure = IO.readPickle("../MCData/recoMeasure.obj")
  # EM measure
  nEvents = len( recoData.padX )
  emMeasure = []
  # for ev in range(2, 3):
  # for ev in [14, 39]:
  # inspectPreCluster( recoData, 0, 57, mcData, display=True)

  for ev in range(0, nEvents):
    emMeasure.append( inspectEvent( recoData, ev, mcData ) )    

    