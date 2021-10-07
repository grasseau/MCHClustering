#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

# Cluster Processing
# import C.PyCWrapper as PCWrap
#import LaplacianProj.PyCWrapper as PCWrap
import O2_Clustering.PyCWrapper as PCWrap

import Util.plot as uPlt
import Util.dataTools as tUtil
import Util.geometry as geom

# Reading MC, Reco, ... Data
import Util.IORun2 as IO
import Util.IOTracks as IOTracks

RecoTracks = []

sqrt2 = math.sqrt( 2.0 )

def getPad( x, y, dx, dy, z, mux, muy ):
  eps = 10.0e-3
  xInf = x - dx 
  xSup = x + dx 
  yInf = y - dy 
  ySup = y + dy
  maskX = np.bitwise_and( (xInf - eps) < mux, mux < (xSup +eps))
  maskY = np.bitwise_and( (yInf - eps) < muy, muy < (ySup +eps))
  mask = np.bitwise_and( maskX, maskY )
  idx = np.where( mask) [0]
  k = 0
  if (idx.size != 0):
     print("getPad. several pads or 0", idx.size )
     k = np.argmax( z[idx])
  else: 
    return -1
  return idx[k]

def dI( a, chId ):
  if chId < 3:
    sigX = tUtil.cstSigXCh1ToCh2
    sigY = tUtil.cstSigYCh1ToCh2
  else:
    sigX = tUtil.cstSigXCh3ToCh10
    sigY = tUtil.cstSigYCh3ToCh10     
  #
  dX = (1.0 - math.erf( (a*sigX ) / (sqrt2*sigX ) )) 
  # dY = (1.0 - math.erf( (a*sigY ) / (sqrt2*sigY ) )) / 2.0 
  # return np.sqrt( dX*dY )
  return dX 

def findSigmaFactor( x, y, dx, dy, z, theta, chId, cutoff ):

  (w, MuX, MuY, varX, varY) = tUtil.thetaAsWMuVar(theta)
  
  K = w.size
  sigFactor =np.zeros( K)
  for k in range(K):
    # print("k=", k, "w[k]=", w[k])
    muX = MuX[k]
    muY = MuY[k]
    xInf = x - dx - muX
    xSup = x + dx - muX
    yInf = y - dy - muY
    ySup = y + dy - muY
    # zMax = np.max( z )
    idx = getPad( x, y, dx, dy, z, muX, muY )
    if idx == -1:
      sigFactor[k] = 1.0
      continue
    zMax = z[idx]
    zTh = tUtil.compute2DPadIntegrals( xInf, xSup, yInf, ySup, chId )
    zMaxTh = np.max( zTh )
    # zSum = np.sum( z )
    # norm = zMax / zMaxTh 
    zSumTh = np.sum( zTh )
    IResidu = 2.0 - zSumTh
    # print("zMaxTh, zSumTh", zMaxTh, zSumTh)
    
    aMin = 0.0 
    aMax = 100.0
    u = 100 * IResidu
    a = (aMax + aMin)*0.5
    # print(IResidu)
    # print("np.abs( u -  IResidu)/IResidu", np.abs( u -  IResidu)/IResidu)
    while ( np.abs( u -  IResidu)/IResidu > 0.1 ):
      a = (aMax + aMin)*0.5
      u = 2 * dI(a, chId) 
      #print("a, u", a, u,"/",IResidu)
      if ( u <  IResidu ):
        aMax = a
      else:
        aMin = a
    #
    sigFactor[k] = a

  return sigFactor

def getHitsInTracks( ev, DEId ):
    tracks = RecoTracks.tracks[ev]
    x = []
    y = []
    for track in tracks:
      ( trackIdx, chi2, nHits, DEIds, UIDs, X, Y, Z, errX, errY) = track
      for h in range(DEIds.size):
        if (DEIds[h] == DEId):
          x.append( X[h])
          y.append( Y[h])
    return (len(x), x, y )

def processPreCluster( pcReco, track, d, gemHits, display=False ):
  ( id, pads, hits ) = pcReco
  ( bc, orbit, iROF, DEId, nbrOfPads ) = id
  chId = DEId // 100
  ( xi, yi, dxi, dyi, chi, saturated, cathi, adc ) = pads
  (nHits, xr, yr, errX, errY, uid, startPadIdx, nPadIdx) = hits
  
  # Track info
  ( trackIdx, trackChi2, nTrackHits, DEIds, UIDs, trackX, trackY, trackZ, errX, errY) = track
  
  print("###")
  print("### New Pre Cluster bc=", bc,", orbit=", orbit, ", iROF=", iROF)
  print("###")
  # Print cluster info
  print("# DEIds", DEId)
  print("# Nbr of pads:", xi.size)
  print("# Nbr of pads per cathodes:", xi[cathi==0].size, xi[cathi==1].size)
  if ( xi.size > 800):
    idx = np.where( chi > 2.0)
    chi =chi[idx]
    xi =xi[idx]
    yi =yi[idx]
    dxi =dxi[idx]
    dyi =dyi[idx]
    saturated =saturated[idx]
    cathi = cathi[idx]
    adc = adc[idx]
    
  if xi[cathi==0].size == 0:
    sumCh0 = 0; minCh0 = 0;  maxCh0 = 0;
  else:
    sumCh0 = np.sum( chi[cathi==0]) ; 
    minCh0 = np.min( chi[cathi==0]);  
    maxCh0 = np.max( chi[cathi==0]);
  if xi[cathi==1].size == 0:
    sumCh1 = 0; minCh1 = 0;  maxCh1 = 0;
  else:
    sumCh1 = np.sum( chi[cathi==1]) ; 
    minCh1 = np.min( chi[cathi==1]);  
    maxCh1 = np.max( chi[cathi==1]);
  print("# Total charge on cathodes", sumCh0, sumCh1)
  print("# Min charge on cathodes", minCh0, minCh1 )
  print("# Max charge on cathodes", maxCh0, maxCh1 )
  print("# Saturated pads", np.sum( saturated))
  # print("# Calibrated pads", np.sum( preClusters.padCalibrated[ev][pc]))
  # xyDxy
  xyDxy = tUtil.padToXYdXY( xi, yi, dxi, dyi)
  """
  idx = np.where( chi > 2.0 )[0]
  nPads = np.sum(chi > 2.0  )
  xi = xi[idx]
  dxi = dxi[idx]
  yi = yi[idx]
  dyi = dyi[idx]
  chi = chi[idx]
  cathi = cathi[idx]
  saturated = saturated[idx]
  adc = adc[idx]
  """
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
  """
  if displayBefore:
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
  """
  #
  #  Do Clustering
  #
  try:
    nbrHits = 0
    
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
  # xl, yl = geom.findLocalMax( xyDxy0, xyDxy1, z0, z1 )
  
  # 
  recoTracks = getHitsInTracks( orbit , DEId)
  # display = True
  diffNbrOfSeeds = (xr.size != nbrHits)

 
  selected = diffNbrOfSeeds and chId > 8
  selected = True
  
  if display and selected:
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
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[0,0], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    uPlt.drawPads( fig, ax[0,1], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[0,1], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    uPlt.drawPads( fig, ax[1,0], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[1,0], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    uPlt.drawPads( fig, ax[1,1], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[1,1], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    uPlt.drawPads( fig, ax[0,2], x0, y0, dx0, dy0, z0,  doLimits=False )
    uPlt.drawPads( fig, ax[1,2], x1, y1, dx1, dy1, z1,  doLimits=False )  
    ax[0,2].set_title("Cath-0 with sat. pads")
    ax[1,2].set_title("Cath-1 with sat. pads")
    # Saturated
    ax[0,2].plot( x0[sat0==1], y0[sat0==1], "o", color='blue', markersize=3 )
    ax[1,2].plot( x1[sat1==1], y1[sat1==1], "o", color='blue', markersize=3 )
    
    # Reco 
    ax[0,0].plot( xr, yr, "+", color='black', markersize=4 )
    ax[0,1].plot( xr, yr, "+", color='black', markersize=4 )
    ax[0,0].set_title("Reco")
    ax[0,1].set_title("Reco with track-hit")

    # GEM resul   
    (xr, yr, dxr, dyr) = tUtil.asXYdXdY( xyDxyResult)
    uPlt.setLUTScale( 0.0, np.max(zProj) )
    # uPlt.drawPads( fig, ax[1,1], xProj, yProj, dxProj, dyProj, zProj, doLimits=False, alpha=1.0 )
    sigFactors = findSigmaFactor( xi, yi, dxi, dyi, chi, thetaResult, chId, 0.0 )
    thetaTmp = tUtil.setThetaVar(thetaResult, sigFactors, chId)
    uPlt.drawModelComponents( ax[1,1], thetaTmp, color="black", pattern='cross')
    uPlt.drawModelComponents( ax[1,0], thetaTmp, color="black", pattern='x')
    # uPlt.drawPoints( ax[1,1], xr, yr, color='black', pattern="o" )
    
    # New find Local max
    # uPlt.drawPoints( ax[0,0], xl, yl, color='black', pattern='+')
    # uPlt.drawPoints( ax[1,0], xl, yl, color='black', pattern='+')

    #
    # Tracks
    #
    # Reco hits
    print( "Track Id", trackIdx, "Track d=", d, " DEId=", DEIds[d], ",", trackX[d], "", trackY[d], "", trackZ[d])
    # ax[0,1].plot( trackX[d], trackY[d], "o", color='blue', markersize=3 )
    uPlt.drawPoints( ax[0,1], trackX[d], trackY[d], color='black', pattern='circle')
    # GEM Hits
    gemX, gemY = gemHits
    # ax[1,1].plot( gemX, gemY, marker='o', color='black', markersize=3 )
    print( "???", gemX)
    if gemX.size != 0:
      uPlt.drawPoints( ax[1,1], gemX, gemY, color='black', pattern='circle')
    ax[1,0].set_title("GEM")
    ax[1,1].set_title("GEM with track-hit")
    
    ax[1,0].set_xlim( xMin, xMax )
    ax[1,0].set_ylim( yMin, yMax )
    ax[1,1].set_xlim( xMin, xMax )
    ax[1,1].set_ylim( yMin, yMax )
    
    #
    t = r'orbit=%d track=%d cluster=%d, DEId=%d sat=%d' % (orbit, trackIdx, iROF, DEId, np.sum( saturated ))
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
    processPreCluster( preClusters, ev, pc)

def findGEMClustersInDEId( GEMTracks, ev, trackDEId, box):
  xHits = []
  yHits = []
  tracks = GEMTracks.tracks[ev]
  for track in tracks:
    ( trackIdx, chi2, nHits, DEIds, UIDs, X, Y, Z, errX, errY) = track
    for tHit in range(DEIds.size):
      DEId = DEIds[tHit]
      print(trackDEId, DEId, X[tHit], box)
      if trackDEId == DEId and geom.isInBox( X[tHit], Y[tHit], box ):
         xHits.append( X[tHit] )   
         yHits.append( Y[tHit] )   

  return (np.array(xHits), np.array(yHits) )

def findPreClusterInDEId( allPC, ev, trackDEId, trackHitX, trackHitY):
  pcList = []
  boxes = []
  for pc in allPC:
    # print("pc" , pc)
    (id, pads, hits ) = pc
    ( bc, orbit, iROF, DEId, nbrOfPads) = id
    if ( orbit == ev) and (DEId == trackDEId):
      ( xi, yi, dxi, dyi, chi, saturated, cathi, adc ) = pads
      xMin = np.min(xi-dxi)
      xMax = np.max(xi+dxi)
      yMin = np.min(yi-dyi)
      yMax = np.max(yi+dyi)
      box = (xMin, xMax, yMin, yMax )
      if geom.isInBox( trackHitX, trackHitY, box ):
        print( "findPreClusterInDEId: found ev=", ev, ", DEId=", trackDEId, "=",DEId)
        input("next")
        pcList.append(pc)
        boxes.append( box )
    elif ( orbit > ev):
      break
  print("findPreClusterInDEId len:", len(pcList))

  return (pcList, boxes )

if __name__ == "__main__":
    
  pcWrap = PCWrap.setupPyCWrapper()
  pcWrap.initMathieson()
  
  # Read PreClusters
  reco = IO.Run2PreCluster(fileName="../Run2Data/recoRun2-100.dat")
  """
  for pc in reco:
    (id, pads, hits ) = pc
    ( bc, orbit, iROF, DEId, nbrOfPads) = id
    if (iROF == 85): 
      processPreCluster ( pc )
      sys.exit()
  """
  RecoTracks = IOTracks.Tracks("/home/grasseau/TracksReco-1.dat")
  RecoTracks.read()

  # GEMTracks = IOTracks.Tracks("/home/grasseau/TracksGEM.dat")
  GEMTracks = IOTracks.Tracks("/home/grasseau/TracksGEM-gsl.dat")
  GEMTracks.read()
  
  # Read PC
  allPC = []
  for pc in reco:
    allPC.append( pc )
  
  nEvents = len( RecoTracks.tracks )
  for ev in range(0, nEvents):
    tracks = RecoTracks.tracks[ev]
    for track in tracks:
      ( trackIdx, chi2, nHits, DEIds, UIDs, X, Y, Z, errX, errY) = track
      print( "New track nHits=", nHits, "(=", DEIds.size, "=", X.size, "=", Y.size, ")")
      input("next")
      for tHit in range(DEIds.size):
        DEId = DEIds[tHit]
        (PCs, boxes) = findPreClusterInDEId ( allPC, ev, DEId, X[tHit], Y[tHit])
        for (i, pc) in enumerate(PCs):
          print("ev=", ev, ", track Id =", trackIdx, 'DEId=', DEId, " trackHitId=", tHit, "" )
          gemHits = findGEMClustersInDEId( GEMTracks, ev, DEId, boxes[i] )
          processPreCluster ( pc, track, tHit, gemHits, display=True  )
        
  """
  if 0:
    for pc in reco:
      # processPreCluster ( pc, display=True, displayBefore=False )
      processPreCluster ( pc, display=False, displayBefore=False )
  else:
    # pc = reco.readPreCluster( 0, 1, 757)
    # pc = reco.readPreCluster( 0, 1, 627)
    # pc = reco.readPreCluster( 0, 2, 1137)
    # pc = reco.readPreCluster( 0, 7, 319) # Pb of group not took into account
    # pc = reco.readPreCluster( 0, 3, 295)
    # pc = reco.readPreCluster( 0, 87, 853)
    # pc = reco.readPreCluster( 0, 0, 468)
    # pc = reco.readPreCluster( 0, 0, 78)
    # pc = reco.readPreCluster( 0, 0, 133)
    # pc = reco.readPreCluster( 0, 0, 177)
    # pc = reco.readPreCluster( 0, 8, 1105) # Not enough space in neighbours list 13 > MaxNeighbors=12
    #pc = reco.readPreCluster( 0, 8, 1207) # Not enough space in neighbours list 13 > MaxNeighbors=12
    # pc = reco.readPreCluster( 0, 0, 79)
    # pc = reco.readPreCluster( 0, 0, 667)
    # pc = reco.readPreCluster( 0, 0, 737)
    # pc = reco.readPreCluster( 0, 0, 737) # GEM meilleur !
    # pc = reco.readPreCluster( 0, 5, 521) # Pb of groups merging
    # pc = reco.readPreCluster( 0, 11, 1332) # Isolated pad and grp
    # pc = reco.readPreCluster( 0, 46, 1007)
    pc = reco.readPreCluster( 0, 0, 376)
    #processPreCluster ( pc, display=True, displayBefore=True)
    # processPreCluster ( pc, display=True, displayBefore=True)
    processPreCluster ( pc, display=True, displayBefore=True)
  """

    