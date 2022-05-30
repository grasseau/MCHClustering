#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

# Cluster Processing
# import C.PyCWrapper as PCWrap
import O2_Clustering.PyCWrapper as PCWrap
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
  supDxMin = [ [] for _ in range(11) ]
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
      nSup = np.sum( ( np.abs(dxMin) > 0.1) )
      supDxMin[chId].append( (ev, nSup, dxMin.size))
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
  
  fig, ax = plt.subplots(nrows=2, ncols=10, figsize=(13, 7) ) 
  for ch in range(1,11):
    evList = []; nSupList = []; nTotList = [];
    nSupWeight=0;nTotWeight=0; 
    for i, val in enumerate(supDxMin[ch]):
      ev, nSup, nTot = val
      evList.append(ev)
      nSupList.append(nSup)
      nTotList.append(nTot)
      nSupWeight += nSup
      nTotWeight += nTot
    print("ch=", ch, "weigh of dxMin > 0.1 :", nSupWeight/nTotWeight)
    evList = np.array(evList)
    nSupList = np.array( nSupList)
    nTotList = np.array( nTotList)
    
    ax[0, ch-1].plot( evList, nSupList, "+")
    
  plt.show()    
  
  return (allTP, allFP, allFN )

def compareDistances( set1, set2, mcObj, recoObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP1 = [0]*11
  allFP1 = [0]*11
  allFN1 = [0]*11
  allTP2 = [0]*11
  allFP2 = [0]*11
  allFN2 = [0]*11
  allDMin = [ [] for _ in range(11) ]
  allDxMin1 = [ [] for _ in range(11) ]
  allDyMin1 = [ [] for _ in range(11) ]
  allDxMin2 = [ [] for _ in range(11) ]
  allDyMin2 = [ [] for _ in range(11) ]
  allPC = []*11
  allEv = []*11 
  totalnValues = 0
  PC1, match1, DXMin1, DYMin1, ChIds1, TP1, FP1, FN1 = set1
  PC2, match2, DXMin2, DYMin2, ChIds2, TP2, FP2, FN2 = set2
  
  for ev in evRange:
    PCa = np.array( PC1[ev] )
    PCb = np.array( PC2[ev] )
    maxPC = max( np.max( PCa ), np.max( PCb ) )
    for pc in range( maxPC):
      jdx1 = np.where( PCa == pc)[0]
      jdx2 = np.where( PCb == pc)[0]
      print( "??? pc, jdx1", pc, jdx1.size, jdx1, "<", PCa.size)
      idx1 = jdx1[0] if (jdx1.size == 1) else -1
      idx2 = jdx2[0] if (jdx2.size == 1) else -1
      ok1 = (jdx1.size > 0) and (match1[ev][idx1] > 0.33)
      ok2 = (jdx2.size > 0) and (match2[ev][idx2] > 0.33)
      
      if not ok1 and ok2:
        print( "Find an EM and not a Reco")
        print( "  Reco (set 1): Ev/pc", ev,"/", pc, jdx1.size, match1[ev][idx1])
        print( "  EM   (set 2): Ev/pc", ev,"/", pc, jdx2.size, match2[ev][idx2])
        input("next")
      if not ok2 and ok1:
        print( "Find a RECO and not an EM")
        print( "  Reco (set 1): Ev/pc", ev,"/", pc, jdx1.size, match1[ev][idx1])
        print( "  EM   (set 2): Ev/pc", ev,"/", pc, jdx2.size, match2[ev][idx2])
        input("next")
      if ok1 and ok2:
        print("???", )
        chid1 = ChIds1[ev][idx1]
        chid2 = ChIds2[ev][idx2]
        if (chid1 != chid2):
          print("pc ...", pc, PC1[ev][idx1], PC1[ev][idx1])
          print("pb chids", chid1, chid2)
          input( "next")
        chId = chid1
        allTP1[chId] += TP1[ev][idx1] 
        allFP1[chId] += FP1[ev][idx1] 
        allFN1[chId] += FN1[ev][idx1]
        allTP2[chId] += TP2[ev][idx2] 
        allFP2[chId] += FP2[ev][idx2] 
        allFN2[chId] += FN2[ev][idx2]
        allDxMin1[chId].append( DXMin1[ev][idx1] )
        allDyMin1[chId].append( DYMin1[ev][idx1] )
        allDxMin2[chId].append( DXMin2[ev][idx2] )
        allDyMin2[chId].append( DYMin2[ev][idx2] )
  # ev loop

  U = np.empty( shape=(0) )
  V = np.empty( shape=(0) )
  # X
  fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(13, 7) ) 
  for ch in range(1,11,2):
    # print( "ch=", ch, "All matched hits TP, FP, FN", allTP[ch], allFP[ch], allFN[ch] )
    # print( "ch=", ch+1, "All matched hits TP, FP, FN", allTP[ch+1], allFP[ch+1], allFN[ch+1] )
    # allDMin[ch] = np.concatenate( allDMin[ch] ).ravel()
    dxMin1 = np.concatenate(allDxMin1[ch]).ravel()
    dxMin1 = np.hstack( [dxMin1, np.concatenate(allDxMin1[ch+1]).ravel()])
    dyMin1 = np.concatenate(allDyMin1[ch]).ravel()
    dyMin1 = np.hstack( [dyMin1, np.concatenate(allDyMin1[ch+1]).ravel()])
    dxMin2 = np.concatenate(allDxMin2[ch]).ravel()
    dxMin2 = np.hstack( [dxMin2, np.concatenate(allDxMin2[ch+1]).ravel()])
    dyMin2 = np.concatenate(allDyMin2[ch]).ravel()
    dyMin2 = np.hstack( [dyMin2, np.concatenate(allDyMin2[ch+1]).ravel()])
    # dyMin = np.concatenate(allDyMin[ch]).ravel()
    
    # nValues = dxMin[ch].size
    #totalnValues += nValues 
    U = np.hstack( [ U, dxMin1 ])
    V = np.hstack( [ V, dyMin1 ])
    # aver = float( np.sum( allDMin[ch] ) ) / nValues
    # print("Distances", aver, np.max( allDMin[ch] ) )
    jp = (ch - 1)  // 2
    ax[0,jp].grid(True)
    ax[1,jp].grid(True)

    n, bins, patches  = ax[0, jp].hist(dxMin1, bins=100, range= (-0.5, 0.5), alpha=0.5, color="red")
    n, bins, patches  = ax[1, jp].hist(dxMin2, bins=100, range= (-0.5, 0.5), alpha=0.5, color="blue")
    # n, bins, patches  = ax[1, jp].hist(dyMin1, bins=100, range= (-0.1, 0.1))
    # n, bins, patches  = ax[1, jp].hist(dyMin2, bins=100, range= (-0.1, 0.1))
    # X Std dev
    xMean = np.mean( dxMin1)
    xStd = np.std( dxMin1 )
    t = r'$\sigma=%.3f$' % xStd
    uPlt.setText( ax[0,jp], (0.6, 0.9), t, ha='left', fontsize=10)
    t = r'St %1d' % (jp+1)
    uPlt.setText( ax[0,jp], (0.1, 0.9), t, ha='left', fontsize=11)
    
    xMean = np.mean( dxMin2)
    xStd = np.std( dxMin2 )
    t = r'$\sigma=%.3f$' % xStd
    uPlt.setText( ax[1,jp], (0.6, 0.9), t, ha='left', fontsize=10)
    t = r'St %1d' % (jp+1)
    uPlt.setText( ax[1,jp], (0.1, 0.9), t, ha='left', fontsize=11)    
    """
    # Y Std dev
    yMean = np.mean( dyMin)
    yStd = np.std( dyMin )
    t = r'$\sigma=%.3f$' % yStd
    uPlt.setText( ax[1,jp], (0.6, 0.9), t, ha='left', fontsize=10)
    print( "mean / std", xMean, xStd)
    t = r'St %1d' % (jp+1)
    uPlt.setText( ax[1,jp], (0.1, 0.9), t, ha='left', fontsize=11)
    """
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
  """
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
  """
  return


def classifyOnEvAndPC( measure, mcObj, recoObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP = []
  allFP = []
  allFN = []
  allDMin = []
  allDXMin = []
  allDYMin = []
  allPC = []
  allEv = [] 
  allMatch = [] 
  allChId = [] 
  totalnValues = 0
  for ev in evRange:
    evNbrOfHits, _, _, _, evDMin, assign = measure[ev]
    evPC = []
    evMatch = [] 
    evDXMin = [] 
    evDYMin = [] 
    evChId = []
    evTP = []
    evFP = []
    evFN = []
    for m in assign :
      (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) = m
      # print("PreCluster pc, DEId", pc, recoObj.rClusterDEId[ev][pc], recoObj.rClusterX[ev][pc])
      # print("mcHitsInvolved", mcHitsInvolved)
      # for mch in mcHitsInvolved:
      #    tid, idx = mch
      #    print("MC part, DEId", mcObj.trackParticleId[ev][tid], mcObj.trackDEId[ev][tid], mcObj.trackX[ev][tid])
      # input('next')
      evPC.append( pc ) 
      evMatch.append( match ) 
      evDXMin.append( dxMin ) 
      evDYMin.append( dyMin ) 
      chIds = recoObj.padChId[ev][pc] 
      diffChIds = np.unique(chIds)
      if diffChIds.size != 1 : 
        print("several Chambers")
        print(chIds)
        input("Warning on Chamber")
        continue
      chId = diffChIds[0]
      evChId.append( chId ) 
      evTP.append ( TP ) 
      evFP.append ( FP ) 
      evFN.append ( FN ) 
      
      
    allPC.append( evPC ) 
    allMatch.append( evMatch ) 
    allDXMin.append( evDXMin ) 
    allDYMin.append( evDYMin ) 
    allChId.append( evChId ) 
    allTP.append ( evTP ) 
    allFP.append ( evFP ) 
    allFN.append ( evFN ) 
  # ev loop
  
  return (allPC, allMatch, allDXMin, allDYMin, allChId, allTP, allFP, allFN )

if __name__ == "__main__":
    
  pcWrap = PCWrap.setupPyCWrapper()
  pcWrap.o2_mch_initMathieson()
  
  # Read MC data
  mcData = IO.MCData(fileName="../MCData/MCDataDump.dat")
  mcData.read()
  # Read PreClusters
  recoData = IO.PreCluster(fileName="../MCData/RecoDataDump.dat")
  recoData.read()
  print( "recoData.padChIdMinMax=", recoData.padChIdMinMax )
  print( "recoData.rClusterChIdMinMax=", recoData.rClusterChIdMinMax )
  # recoMeasure
  recoMeasure = IO.readPickle("../MCData/recoMeasure.obj")
  # EM measure
  nEvents = len( recoData.padX )
  emMeasure = []
  #for ev in range(0, 5):
  for ev in range(0, nEvents):
    emMeasure.append( aTK.processEvent( recoData, ev, mcData ) )    
  
  # recoConfMatrix = statOnDistances( recoMeasure, mcData, recoData, range(0, 6) )
 
  recoConfMatrix = statOnDistances( recoMeasure, mcData, recoData, range(0,nEvents) )
  
  # emConfMatrix = statOnDistances( emMeasure, mcData, recoData, range(0, 6) )
  emConfMatrix = statOnDistances( emMeasure, mcData, recoData, range(0,nEvents) )
  
  (recoTP, recoFP, recoFN) = recoConfMatrix
  (emTP, emFP, emFN) = emConfMatrix
  allRecoTP = 0; allRecoFP = 0; allRecoFN = 0;
  allEMTP = 0; allEMFP = 0; allEMFN = 0;
  for ch in range(1,11,1):
    print('reco chId=', ch, "TP/FP/FN", recoTP[ch], recoFP[ch], recoFN[ch])
    print('em   chId=', ch, "TP/FP/FN", emTP[ch], emFP[ch], emFN[ch])
    allRecoTP += recoTP[ch]
    allRecoFP += recoFP[ch]
    allRecoFN += recoFN[ch]
    allEMTP += emTP[ch]
    allEMFP += emFP[ch]
    allEMFN += emFN[ch] 
  print('reco all', ch, "TP/FP/FN", allRecoTP, allRecoFP, allRecoFN)
  print('em   all', ch, "TP/FP/FN", allEMTP, allEMFP, allEMFN)
  print("Reco TP/FP/FN")
  print(recoTP)
  print(recoFP)
  print(recoFN)
  print("EM TP/FP/FN")
  print(emTP)
  print(emFP)
  print(emFN)
  
  """
  recoSet = classifyOnEvAndPC( recoMeasure, mcData, recoData, range(0,5) )
  EMSet  = classifyOnEvAndPC( emMeasure, mcData, recoData, range(0,5) ) 
  compareDistances( recoSet, EMSet, mcData, recoData, range(0,5))

  recoSet = classifyOnEvAndPC( recoMeasure, mcData, recoData, range(nEvents) )
  EMSet   = classifyOnEvAndPC( emMeasure, mcData, recoData, range(nEvents) ) 
  compareDistances( recoSet, EMSet, mcData, recoData, range(nEvents))
  """
