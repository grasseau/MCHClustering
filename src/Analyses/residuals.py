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
  
  return (allTP, allFP, allFN )

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
  # for ev in range(37, 38):
    emMeasure.append( aTK.processEvent( recoData, ev, mcData ) )    

  recoConfMatrix = statOnDistances( recoMeasure, mcData, recoData, range(nEvents) )
  
  emConfMatrix = statOnDistances( emMeasure, mcData, recoData, range(nEvents) )
  
  (recoTP, recoFP, recoFN) = recoConfMatrix
  (emTP, emFP, emFN) = emConfMatrix
  for ch in range(1,11,1):
    print('reco ', recoTP[ch], recoFP[ch], recoFN[ch])
    print('em   ', emTP[ch], emFP[ch], emFN[ch])
    