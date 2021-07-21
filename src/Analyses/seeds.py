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

diffISeedsPerChamberWithMC = [ [] for _ in range(11) ]
diffISeedsPerChamberWithReco = [ [] for _ in range(11) ]
diffFSeedsPerChamberWithMC = [ [] for _ in range(11) ]
diffFSeedsPerChamberWithReco = [ [] for _ in range(11) ]

def plotHistoSeeds():
  fig, ax = plt.subplots(nrows=4, ncols=10, figsize=(13, 7) ) 
  for ch in range(1,11):
    mcI = np.hstack( diffISeedsPerChamberWithMC[ch] )
    recoI = np.hstack( diffISeedsPerChamberWithReco[ch] ) 
    mcF = np.hstack( diffFSeedsPerChamberWithMC[ch] )
    recoF = np.hstack( diffFSeedsPerChamberWithReco[ch] ) 
    jp =  (ch - 1) % 5
    ip = (ch - 1)  // 5
    
    ax[ip,jp].grid(True)
    ax[ip,jp].grid(True)
    print("???", ip, jp)

    n, bins, patches  = ax[0, ch-1].hist(mcI, bins=20, color="blue", log=True ) # range= (-0.5, 0.5))
    n, bins, patches  = ax[1, ch-1].hist(recoI, bins=20, color="red", log=True ) # , range= (-0.1, 0.1))
    n, bins, patches  = ax[2, ch-1].hist(mcF, bins=20, color="lightblue", log=True ) # range= (-0.5, 0.5))
    n, bins, patches  = ax[3, ch-1].hist(recoF, bins=20, color="purple", log=True ) # , range= (-0.1, 0.1))
    if ch == 1 :
      ax[0, 0].set_ylabel("EM-MC seeds Init")
      ax[1, 0].set_ylabel("EM-Reco seeds Init")
      ax[2, 0].set_ylabel("EM-MC seeds Final")
      ax[3, 0].set_ylabel("EM-Reco seeds Final")
  #
  plt.show()
    
def preClusterSeeds( preClusters, ev, pc, mcObj):
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
  xyDxy = dUtil.padToXYdXY( xi, yi, dxi, dyi)
  
  #
  #  Do Clustering
  #
  nbrHits = PCWrap.clusterProcess( xyDxy, cathi, saturated, chi, chId )
  print("nbrHits", nbrHits)
  (thetaf, thetaToGrp) = PCWrap.collectTheta( nbrHits)
  print("Cluster Processing Results:")
  print("  theta   :", thetaf)
  print("  thetaGrp:", thetaToGrp)
  # Returns the fit status
  # (xyDxyResult, chResult, padToGrp) = PCWrap.collectPadsAndCharges()
  # Return the projection
  # (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.copyProjectedPads()
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
      print( "EM   ev=", ev, "pc=", pc," TP, FP, FN", TP, FP, FN)
      print( "Reco ev=", ev, "pc=", pc," TP, FP, FN", recoTP, recoFP, recoFN)
      spanBox = geom.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                         preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
      xHits, yHits   = aTK.getMatchingMCTrackHits( DEIds, spanBox, mcObj, ev ) 
      nMCSeeds   = xHits.size
      print("nMCSeeds ", nMCSeeds)
      nRecoSeeds = preClusters.rClusterX[ev][pc].size
      # Ratio seeds / MC hit
      thetaInit = PCWrap.collectThetaInit()
      nSeeds = thetaInit.size // 5
      if ( nSeeds - nMCSeeds ) < -20:
        input( "next")
      diffISeedsPerChamberWithMC [ chId ].append( nSeeds - nMCSeeds )  
      diffISeedsPerChamberWithReco [ chId ].append( nSeeds -  nRecoSeeds)  
      diffFSeedsPerChamberWithMC[ chId ].append( thetaf.size // 5- nMCSeeds )
      diffFSeedsPerChamberWithReco[ chId ].append( thetaf.size // 5 - nRecoSeeds )
      

  #


  #
  # free memory in Pad-Processing
  PCWrap.freeMemoryPadProcessing()

  #
  return 

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
    preClusterSeeds( preClusters, ev, pc, mcObj )
    """
    res = aTK.matchingThetaWithMC(thetaf, preClusters, ev, pc, mcObj)
    (pc_, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) = res
    if (match > 0.33):
        assign.append( res )
        evTP += TP; evFP += FP; evFN += FN
        allDMin.append( dMin )
        evNbrOfHits += nbrOfHits
    """
  #
  return ( [])

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
  # for ev in range(2, 3):
  # for ev in range(20, 21):
  # for ev in range(0, 10):
  for ev in range(0, nEvents):
    inspectEvent( recoData, ev, mcData )
  #
  plotHistoSeeds()
    