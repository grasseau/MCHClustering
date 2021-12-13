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
#import LaplacianProj.PyCWrapper as PCWrap
import O2_Clustering.PyCWrapper as PCWrap
# import C.PyCWrapper as PCWrap

import Util.plot as uPlt
import Util.dataTools as tUtil
import Util.geometry as geom

# Reading MC, Reco, ... Data
import Util.IORun2 as IO
import Util.IOTracks as IOTracks
import Analyses.analyseToolKit as aTK

RecoTracks = []

sqrt2 = math.sqrt( 2.0 )

def  clusterProcessWithPET( xyDxy, cathi, saturated, chi, chId, proj=None):
  xi, yi, dxi, dyi = tUtil.asXYdXdY( xyDxy )
  x0  = xi[cathi==0]
  y0  = yi[cathi==0]
  dx0 = dxi[cathi==0]
  dy0 = dyi[cathi==0]
  q0  = chi[cathi==0]
  x1  = xi[cathi==1]
  y1  = yi[cathi==1]
  dx1 = dxi[cathi==1]
  dy1 = dyi[cathi==1]
  q1  = chi[cathi==1]
  xyDxy0 = tUtil.asXYdXY( x0, y0, dx0, dy0 )  
  xyDxy1 = tUtil.asXYdXY( x1, y1, dx1, dy1 )  
  (theta, pixInit, pixTheta0, pixTheta1) = aTK.findLocalMaxWithPET(xyDxy0, xyDxy1, q0, q1, chId, proj=proj )
  nbrHits = theta.size // 5
  return nbrHits, theta, pixTheta0, pixTheta1

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

def processPreCluster( pc, display=False, displayBefore=False ):

  # Current Reco
  (id, pads, hits ) = pc
  ( bc, orbit, iROF, DEId, nbrOfPads) = id
  chId = DEId // 100
  ( xi, yi, dxi, dyi, chi, saturated, cathi, adc ) = pads
  (nHits, xr, yr, errX, errY, uid, startPadIdx, nPadIdx) = hits
  
  print("[python] ###")
  print("[python] ### New Pre Cluster bc=", bc,", orbit=", orbit, ", iROF=", iROF)
  print("[python] ###")
  # Print cluster info
  print("[python] # DEIds", DEId)
  print("[python] # Nbr of pads:", xi.size)
  print("[python] # Nbr of pads per cathodes:", xi[cathi==0].size, xi[cathi==1].size)
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
  print("[python] # Total charge on cathodes", sumCh0, sumCh1)
  print("[python] # Min charge on cathodes", minCh0, minCh1 )
  print("[python] # Max charge on cathodes", maxCh0, maxCh1 )
  print("[python] # Saturated pads", np.sum( saturated))
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
    
  #
  #  Do Clustering
  #
  try:
    nbrHits = PCWrap.clusterProcess( xyDxy, cathi, saturated, chi, chId )
    print("[python] nbrHits", nbrHits)
    (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
    nbrOfGroups = np.max(thetaToGrp)
    print("[python] Cluster Processing Results:")
    print("[python]   theta   :", thetaResult)
    print("[python]   thetaGrp:", thetaToGrp)
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
  # nPETSeeds, thetaPET, pixTheta0, pixTheta1  = clusterProcessWithPET( xyDxy, cathi, saturated, chi, chId,  proj = (xProj, dxProj, yProj, dyProj, zProj) )
  # tUtil.printTheta("clusterProcessWithPET", thetaPET)  
  # New find local Max
  
  xyDxy0 = tUtil.asXYdXY( x0, y0, dx0, dy0 )
  xyDxy1 = tUtil.asXYdXY( x1, y1, dx1, dy1 )
  # xl, yl = geom.findLocalMax( xyDxy0, xyDxy1, z0, z1 )
  
  # 
  recoTracks = getHitsInTracks( orbit , DEId)
  # display = True
  diffNbrOfSeeds = (xr.size != nbrHits)
  if chId > 5:
      selected = True
  else : selected = False
  
  selected = True
  selected = True if nbrHits > 10 else False
  selected = diffNbrOfSeeds and chId > 8
  selected = diffNbrOfSeeds 
  selected = (nbrOfGroups > 1)
  
  if display and selected:
    nFigRow = 2; nFigCol = 4
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

    #
    # Reco
    #
    # uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  doLimits=False )
    # uPlt.drawPads( fig, ax[0,1], x1, y1, dx1, dy1, z1,  doLimits=False ) 
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[0,0], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    # Saturated
    # ax[0,0].plot( x0[sat0==1], y0[sat0==1], "o", color='white', markersize=3 )
    #ax[0,1].plot( x1[sat1==1], y1[sat1==1], "o", color='white', markersize=3 )
    
    #ax[0,0].plot( xr, yr, "+", color='black', markersize=4 )
    #ax[0,1].plot( xr, yr, "+", color='black', markersize=4 )
    ax[0,0].plot( xr, yr, "+", color='black', markersize=4 )
    # ax[0,0].set_title("Cath0 & Current Algo")
    # ax[0,1].set_title("Cath1 & Current Algo")
    ax[0,0].set_title("Both Cath & Current Algo")
    
        #
    # Merged Groups
    padToCathGrp = PCWrap.collectPadToCathGroup( xi.size )
    padCathGrpMax = 0
    if padToCathGrp.size != 0:
      padCathGrpMax = np.max( padToCathGrp )
    uPlt.setLUTScale( 0.0, padCathGrpMax ) 
    uPlt.drawPads( fig, ax[0,1], xi, yi, dxi, dyi, padToCathGrp,  doLimits=False, alpha=0.5 )
    ax[0,1].set_title("Group of pads")
      
    #
    # ??? (xr, yr, dxr, dyr) = tUtil.asXYdXdY( xyDxyResult)
    
    #
    # Projection
    #
    """
    uPlt.setLUTScale( 0.0, np.max(zProj) )
    uPlt.drawPads( fig, ax[0,2], xProj, yProj, dxProj, dyProj, zProj, doLimits=False, alpha=1.0 )
    uPlt.drawModelComponents( ax[0,2], thetaResult, color="black", pattern='x')
    ax[0,2].set_title("Projection & PET Algo")
    """
    
    """
    sigFactors = findSigmaFactor( xi, yi, dxi, dyi, chi, thetaResult, chId, 0.0 )
    thetaTmp = tUtil.setThetaVar(thetaResult, sigFactors, chId)
    """
    #uPlt.drawModelComponents( ax[0,1], thetaResult, color="black", pattern='x')
    # uPlt.drawPoints( ax[1,1], xr, yr, color='black', pattern="o" )
    
    #
    # Pixels
    #    
    # EM Final
    thetaEMFinal = PCWrap.collectThetaEMFinal()
    #
    for p in range(7, -1, -1):
      (nPix, xyDxyPix, qPix) = PCWrap.collectPixels(p)
      if nPix != 0: pEnd = p; break
    
    print( "pEnd ???", pEnd, )
    (nPix, xyDxyPix, qPix) = PCWrap.collectPixels(pEnd-1)
    if (nPix >0 ):
      (xPix, yPix, dxPix, dyPix) = tUtil.asXYdXdY( xyDxyPix)
      # qPix, xPix, yPix, dxPix, dyPix = tUtil.thetaAsWMuVar( pixTheta1 )
      uPlt.setLUTScale( 0.0, np.max(qPix))
      uPlt.drawPads( fig, ax[1,1], xPix, yPix, dxPix, dyPix, qPix, doLimits=False, alpha=1.0 )
      #uPlt.drawModelComponents( ax[1,1], thetaTmp, color="black", pattern='x')
      uPlt.drawModelComponents( ax[1,1], thetaResult, color="black", pattern='x', markersize=4)
      uPlt.drawModelComponents( ax[1,1], thetaEMFinal, color="lightgrey", pattern='x')
    ax[1,1].set_title("Pixels & PET Algo (1)")
    #
    (nPix, xyDxyPix, qPix) = PCWrap.collectPixels(pEnd)
    if (nPix >0 ):
      (xPix, yPix, dxPix, dyPix) = tUtil.asXYdXdY( xyDxyPix)
      # qPix, xPix, yPix, dxPix, dyPix = tUtil.thetaAsWMuVar( pixTheta1 )
      uPlt.setLUTScale( 0.0, np.max(qPix))  
      uPlt.drawPads( fig, ax[1,2], xPix, yPix, dxPix, dyPix, qPix, doLimits=False, alpha=1.0 )
      #uPlt.drawModelComponents( ax[1,1], thetaTmp, color="black", pattern='x')
      uPlt.drawModelComponents( ax[1,2], thetaResult, color="black", pattern='x', markersize=4)
      uPlt.drawModelComponents( ax[1,2], thetaEMFinal, color="lightgrey", pattern='x')
      uPlt.drawModelComponents( ax[1,2], thetaResult, color="black", pattern='x')
    ax[1,2].set_title("Pixels & PET Algo (2)")    
    #
    (nPix, xyDxyPix, qPix) = PCWrap.collectPixels(0)
    if (nPix >0 ):
      (xPix, yPix, dxPix, dyPix) = tUtil.asXYdXdY( xyDxyPix)
      # qPix, xPix, yPix, dxPix, dyPix = tUtil.thetaAsWMuVar( pixTheta1 )
      uPlt.setLUTScale( np.min(qPix), np.max(qPix)) 
      uPlt.drawPads( fig, ax[0,2], xPix, yPix, dxPix, dyPix, qPix, doLimits=False, alpha=1.0 )
      #uPlt.drawModelComponents( ax[1,1], thetaTmp, color="black", pattern='x')
      uPlt.drawModelComponents( ax[0,2], thetaResult, color="black", pattern='x', markersize=4)
      uPlt.drawModelComponents( ax[0,2], thetaEMFinal, color="lightgrey", pattern='x')
      uPlt.drawModelComponents( ax[0,2], thetaResult, color="black", pattern='x')
    ax[0,2].set_title("dPixels & PET Algo")    
    #ax[0,2].set_xlim( np.min(xPix), np.max(xPix))
    #ax[0,2].set_ylim( np.min(yPix), np.max(yPix))
    
    # New find Local max
    # uPlt.drawPoints( ax[0,0], xl, yl, color='black', pattern='+')
    # uPlt.drawPoints( ax[1,0], xl, yl, color='black', pattern='+')
    
 
    #
    # PET Algo
    #
    uPlt.setLUTScale( 0.0, zMax ) 
    uPlt.drawPads( fig, ax[1,0], x0, y0, dx0, dy0, z0,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[1,0], x1, y1, dx1, dy1, z1,  doLimits=False, alpha=0.5, )
    ax[1,0].plot( xr, yr, "o", color='red', markersize=4 )

    uPlt.drawModelComponents( ax[1,0], thetaResult, color="black", pattern='x')
    ax[1,0].set_title("PET Algo")

    #   
    # The max Laplacian of the 2 cathodes
    #
    xl, yl = geom.findLocalMax( xyDxy0, xyDxy1, z0, z1 )
    zMax = np.max( chi )
    uPlt.setLUTScale( 0.0, zMax ) 
    uPlt.drawPads( fig, ax[0,3], x0, y0, dx0, dy0, z0,  doLimits=False)
    uPlt.drawPads( fig, ax[1,3], x1, y1, dx1, dy1, z1,  doLimits=False )
    # Saturated
    xSat = x0[ sat0==1 ]
    ySat = y0[ sat0==1 ]
    uPlt.drawPoints( ax[0,3], xSat, ySat, color='white', pattern='o', markerSize=4)
    xSat = x1[ sat1==1 ]
    ySat = y1[ sat1==1 ]
    uPlt.drawPoints( ax[1,3], xSat, ySat, color='white', pattern='o', markerSize=4)
    #
    uPlt.drawPoints( ax[0,3], xl, yl, color='black', pattern='o')
    uPlt.drawPoints( ax[1,3], xl, yl, color='black', pattern='o')

    ax[0,3].set_title("Cathode 0 & max Laplacian")
    ax[1,3].set_title("Cathode 1 & max Laplacian")
    
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
    processPreCluster( preClusters, ev, pc)


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
  RecoTracks = IOTracks.Tracks("/home/grasseau/TracksReco.dat")
  RecoTracks.read()
  
  if 1:
    for pc in reco:
      # processPreCluster ( pc, display=True, displayBefore=False )
      processPreCluster ( pc, display=True, displayBefore=False )
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
    # processPreCluster ( pc, display=True, displayBefore=True)
    
    # Grp pb
    grpPb = [1, 6, 44, 133, 134, 173, 177, 198]
    # Pix pb
    pixPb = [21, 44, 71, 72, 91, 94, 96, 101, 133, 134, 144, 146, 173]
    # Others
    othersPix = [1, 8, 68]
    # PCList = grpPb + pixPb
    # PCList = pixPb 
    PCList = [ 238 ]
    
    #
    for pc in reco:
      (id, pads, hits ) = pc
      (bc, orbit, irof, _, _) = id
      print(id)
      if (orbit == 0) and (irof in PCList):
        processPreCluster ( pc, display=True, displayBefore=False )
      elif (orbit==7) and (irof==319):
        processPreCluster ( pc, display=True, displayBefore=False )
  # reco.read(verbose=True)
  # nPreClusters = len(reco.padX )
  # for ipc in range(0, nPreClusters ):
    # processEvent( reco, ipc )    

    