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
import Util.dataTools as tUtil
# Reading MC, Reco, ... Data
import Util.IOv5 as IO
# Analyses Tool Kit
import analyseToolKit as aTK

def statOnClusterSize( measure, mcObj, recoObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP = [0]*11
  allFP = [0]*11
  allFN = [0]*11
  allDMin = [ [] for _ in range(11) ]
  allSizes = [ [] for _ in range(11) ]
  allPC = []*11
  allEv = []*11 
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    for k, m in enumerate( assign ) :
      ( pc, match, nbrOfHits, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved) = m
      # ??? To remove (pc, match, nbrOfHits, TP, FP,FN, dxMin, dyMin, mcHitsInvolved ) = m
      # print( "??? ", recoObj.rClusterX[ev][pc].size, len(mcHitsInvolved) )
      """
      if len(mcHitsInvolved) != 1:
         print( "??? len pb", mcHitsInvolved )
         exit()
      """
      # ( trackIdx, hitIdx) =  mcHitsInvolved[0]
      chIds = recoObj.padChId[ev][pc] 
      diffChIds = np.unique(chIds)
      if diffChIds.size != 1 : 
        print("several Chambers")
        print(m)
        print(chIds)
        input("Warning on Chamber")
        continue
      chId = chIds[0]
      # print( "??? # reco/mc", recoObj.rClusterX[ev][pc].size, len(mcHitsInvolved) )
      # print("??? dMin", evDMin[k])
      # print(evDMin)
      clSize = len(mcHitsInvolved)
      if clSize  > 60:
        print (evDMin[k])
        print( "??? # reco/fullEM/mc", recoObj.rClusterX[ev][pc].size, len(mcHitsInvolved) )
        input("next")  
      allDMin[chId].append( evDMin[k]  )
      allSizes[chId].append( np.ones(evDMin[k].shape)*clSize)
      # input("next")
  nFigRow = 2
  nFigCol = 5
  fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(10, 7) )
  for ch in range(1,11):
    print( "ch=", ch, "All matched hits TP, FP, FN", allTP[ch], allFP[ch], allFN[ch] )
    y = np.concatenate( allDMin[ch] ).ravel() 
    x = np.concatenate( allSizes[ch] ).ravel() 
    nValues = x.size
    fX = (ch - 1) // nFigCol
    fY = (ch - 1) % nFigCol
    ax[fX, fY].plot( x, y,".")
    if (nValues != 0):
      aver = float( np.sum( y ) ) / nValues
      print("Distances", aver )
    xMin = np.min(x)
    xMax = np.max(x)
    
    ax[fX, fY].plot( [ xMin, xMax], [aver, aver], color="r" )
    xSet = np.unique( x )
    for k in xSet:
      yk = y[ np.where( x == k)]
      averk = float( np.sum( yk ) ) / yk.size
      ax[fX, fY].plot( [ k ], [averk], "o", color="r" )
    
    ax[fX, fY].plot( [xMin, xMax], [0.07, 0.07], "--", color="green" )
    print("xSet ", xSet)
    if fX == 1 :
      ax[fX, fY].set_xlabel('# mcHits')
    if fY == 0 :
      ax[fX, fY].set_ylabel('distances (cm)')
    ax[fX, fY].set_ylim( 0.0, 0.4 )
  # fig.title('Hit accuracy versus cluster size (# of MC Hits)')
  plt.show()
  return allSizes, allDMin

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

  statOnClusterSize( recoMeasure, mcData, recoData, range(nEvents) )
  statOnClusterSize( emMeasure, mcData, recoData, range(nEvents) )
    