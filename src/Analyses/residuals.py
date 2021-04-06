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
import PyTests.utilitiesForTests as tUtil
# Reading MC, Reco, ... Data
import Util.IOv5 as IO
# Analyses Tool Kit
import analyseToolKit as aTK

#
# TP, FN, FP, ... analysis
#
def matchingWithMC( theta, preClusters, ev, pc, mcObj, rejectPadFactor=0.5):
  ( w, muX, muY, _, _) = tUtil.thetaAsWMuVar( theta )
  if ( w.size != 0 ):
    """ Not necessary
    idx = np.where( wFinal != 0.0)
    wFinal = wFinal[idx]
    muFinal = muFinal[idx]
    varFinal = varFinal[idx]
    """
    print(" wFinal.size", w.size)
    print(" # reco Hits", preClusters.rClusterX[ev][pc].size)
    # input("next")

    #    
    # ???
    """
    emObj.w[ev].append(wFinal)
    emObj.mu[ev].append(muFinal)
    emObj.var[ev].append(varFinal)
    """
    #
    # Assign Cluster hits to MC hits
    spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
    rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
    rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
    spanBox = geom.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                           preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
    match, nbrOfHits, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved = \
                 aTK.matchMCTrackHits(muX, muY, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
  return (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) 
      
def processPreCluster( preClusters, ev, pc, mcObj, display=False ):
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
  return thetaResult

def processEvent( preClusters,  ev, mcObj):
  nbrOfPreClusters = len( preClusters.padId[ev] )
  evTP=0; evFP=0; evFN=0
  evNbrOfHits=0
  # Matching preClusters description
  assign = []
  allDMin = []
  for pc in range(0, nbrOfPreClusters):
    thetaf = processPreCluster( preClusters, ev, pc, mcObj, display=False)
    res = matchingWithMC(thetaf, preClusters, ev, pc, mcObj)
    (pc_, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) = res
    if (match > 0.33):
        assign.append( res )
        evTP += TP; evFP += FP; evFN += FN
        allDMin.append( dMin )
        evNbrOfHits += nbrOfHits
        print( "ev=", ev, "pc=", pc," TP, FP, FN", TP, FP, FN)
  #
  return (evNbrOfHits, evTP, evFP, evFN, allDMin, assign)

def statOnDistances( measure, mcObj, recoObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP = [0]*11
  allFP = [0]*11
  allFN = [0]*11
  allDMin = [ [] for _ in range(11) ]
  allDxMin = [ [] for _ in range(11) ]
  allDyMin = [ [] for _ in range(11) ]
  allPC = []*11
  allEv = []*11 
  totalnValues = 0
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    for m in assign :
      (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) = m
      # print("PreCluster pc, DEId", pc, recoObj.rClusterDEId[ev][pc], recoObj.rClusterX[ev][pc])
      # print("mcHitsInvolved", mcHitsInvolved)
      # for mch in mcHitsInvolved:
      #    tid, idx = mch
      #    print("MC part, DEId", mcObj.trackParticleId[ev][tid], mcObj.trackDEId[ev][tid], mcObj.trackX[ev][tid])
      # input('next')
      chIds = recoObj.padChId[ev][pc] 
      diffChIds = np.unique(chIds)
      if diffChIds.size != 1 : 
        print("several Chambers")
        print(chIds)
        input("Warning on Chamber")
        continue
      chId = diffChIds[0]
      allTP[chId] += TP 
      allFP[chId] += FP 
      allFN[chId] += FN
      allDxMin[chId].append(dxMin)
      allDyMin[chId].append(dyMin)
      allDMin[chId].append( np.concatenate( evDMin ).ravel() )
  # ev loop


  U = np.empty( shape=(0) )
  V = np.empty( shape=(0) )
  # X
  fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(13, 7) ) 
  for ch in range(1,11,2):
    print( "ch=", ch, "All matched hits TP, FP, FN", allTP[ch], allFP[ch], allFN[ch] )
    print( "ch=", ch+1, "All matched hits TP, FP, FN", allTP[ch+1], allFP[ch+1], allFN[ch+1] )
    allDMin[ch] = np.concatenate( allDMin[ch] ).ravel()
    dxMin = np.concatenate(allDxMin[ch]).ravel()
    dxMin = np.hstack( [dxMin, np.concatenate(allDxMin[ch+1]).ravel()])
    dyMin = np.concatenate(allDyMin[ch]).ravel()
    dyMin = np.hstack( [dyMin, np.concatenate(allDyMin[ch+1]).ravel()])
    # dyMin = np.concatenate(allDyMin[ch]).ravel()
    nValues = dxMin[ch].size
    totalnValues += nValues 
    U = np.hstack( [ U, dxMin ])
    V = np.hstack( [ V, dyMin ])
    # aver = float( np.sum( allDMin[ch] ) ) / nValues
    # print("Distances", aver, np.max( allDMin[ch] ) )
    jp = (ch - 1)  // 2
    ax[0,jp].grid(True)
    ax[1,jp].grid(True)

    n, bins, patches  = ax[0, jp].hist(dxMin, bins=100, range= (-0.5, 0.5))
    n, bins, patches  = ax[1, jp].hist(dyMin, bins=100, range= (-0.1, 0.1))
    # X Std dev
    xMean = np.mean( dxMin)
    xStd = np.std( dxMin )
    t = r'$\sigma=%.3f$' % xStd
    uPlt.setText( ax[0,jp], (0.6, 0.9), t, ha='left', fontsize=10)
    t = r'St %1d' % (jp+1)
    uPlt.setText( ax[0,jp], (0.1, 0.9), t, ha='left', fontsize=11)
    # Y Std dev
    yMean = np.mean( dyMin)
    yStd = np.std( dyMin )
    t = r'$\sigma=%.3f$' % yStd
    uPlt.setText( ax[1,jp], (0.6, 0.9), t, ha='left', fontsize=10)
    print( "mean / std", xMean, xStd)
    t = r'St %1d' % (jp+1)
    uPlt.setText( ax[1,jp], (0.1, 0.9), t, ha='left', fontsize=11)
    
  ax[0,0].set_ylabel( "Hist. of X residues" )
  ax[1,0].set_ylabel( "Hist. of Y residues" )
  label = "Residues (cm)"
  ax[1,0].set_xlabel( label )
  ax[1,1].set_xlabel( label )
  ax[1,2].set_xlabel( label )
  ax[1,3].set_xlabel( label )
  ax[1,4].set_xlabel( label )

  plt.suptitle('Histogram of the residues between Reco & MC clusters')
  plt.show()
  # All chambers      plt.show()

  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13, 7) ) 
  n, bins, patches  = ax[0,0].hist(U, bins=100, range=(-0.5, 0.5))
  n, bins, patches  = ax[0,1].hist(V, bins=100, range=(-0.1, 0.1))
  xMean = np.mean( U )
  xStd = np.std( U )
  t = r'$\sigma=%.3f$' % xStd
  uPlt.setText( ax[0,0], (0.8, 0.9), t, ha='left', fontsize=10)
  xMean = np.mean( V )
  xStd = np.std( V )
  t = r'$\sigma=%.3f$' % xStd
  uPlt.setText( ax[0,1], (0.8, 0.9), t, ha='left', fontsize=10)
  ax[0,0].set_title( "Histogram of X residues" )
  ax[0,1].set_title( "Histogram of Y residues" )
  ax[0,0].set_xlabel( "X residues (cm)" )
  ax[0,1].set_xlabel( "Y residues (cm)" )
  plt.show()
  return

if __name__ == "__main__":
    
  pcWrap = PCWrap.setupPyCWrapper()
  pcWrap.initMathieson()
  
  # Read MC data
  mcData = IO.MCData(fileName="../Data/MCDataDump.dat")
  mcData.read()
  # Read PreClusters
  recoData = IO.PreCluster(fileName="../Data/RecoDataDump.dat")
  recoData.read()
  print( "recoData.padChIdMinMax=", recoData.padChIdMinMax )
  print( "recoData.rClusterChIdMinMax=", recoData.rClusterChIdMinMax )
  # recoMeasure
  recoMeasure = IO.readPickle("../Data/recoMeasure.obj")
  # EM measure
  nEvents = len( recoData.padX )
  emMeasure = []
  for ev in range(0, nEvents):
    emMeasure.append( processEvent( recoData, ev, mcData ) )    

  statOnDistances( recoMeasure, mcData, recoData, range(nEvents) )
  statOnDistances( emMeasure, mcData, recoData, range(nEvents) )
    