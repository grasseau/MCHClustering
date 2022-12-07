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

def processPreCluster( pc, display=False, displayBefore=False,  ):

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
    
  # Min/Max/sum pad charge 
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
  twoCath = True
  if (x0.size ==0) or (x1.size ==0):
    twoCath = False
  
  if xr.size >= 1 :
    wr = np.ones( xr.size ) * 1.0/xr.size
    thetaRecoi = tUtil.asTheta( wr, xr, yr)
    thetaRecof = PCWrap.fitMathieson( xyDxy, chi, cathi, saturated, chId, thetaRecoi)
    tUtil.printTheta("[python] Reco thetaRecof", thetaRecof) 

  #
  #  Do Clustering
  #
  nbrOfGroups = 0
  try:
    nbrHits = PCWrap.clusterProcess( xyDxy, cathi, saturated, chi, chId )
    print("[python] nbrHits", nbrHits)
    (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
    nbrOfGroups = np.max(thetaToGrp)

    print("[python] Cluster Processing Results:")
    print("[python]   theta   :", thetaResult)
    print("[python]   thetaGrp:", thetaToGrp)
    (w, muX, muY, varX, varY) = tUtil.thetaAsWMuVar( thetaResult )
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
  # tUtil.printUtiltTheta("clusterProcessWithPET", thetaPET)  
  # New find local Max
  
  xyDxy0 = tUtil.asXYdXY( x0, y0, dx0, dy0 )
  xyDxy1 = tUtil.asXYdXY( x1, y1, dx1, dy1 )
  # xl, yl = geom.findLocalMax( xyDxy0, xyDxy1, z0, z1 )
  
  # 
  # recoTracks = getHitsInTracks( orbit , DEId)
  # display = True
  diffNbrOfSeeds = (xr.size != nbrHits)
  
   # Compute the max of the Reco and EM seeds/hits
  maxDxMinREM, maxDyMinREM  = (0.0, 0.0 )
  if nbrHits != 0 :
    maxDxMinREM, maxDyMinREM = aTK.minDxDy( muX, muY, xr, yr)
  
  selected = True if nbrHits > 10 else False
  selected = diffNbrOfSeeds and chId > 8
  selected = diffNbrOfSeeds 
  # select good ones
  selected = (nbrOfGroups > 1)
  selected = not twoCath
  selected = (xr.size > nbrHits)
  selected = (diffNbrOfSeeds) or (maxDxMinREM > 0.07) or (maxDyMinREM > 0.07)
  selected = True
  selected = (chId > 4) and (nbrOfPads > 50)
  
  #f
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
    pEnd = 0
    for p in range(7, -1, -1):
      (nPix, xyDxyPix, qPix) = PCWrap.collectPixels(p)
      if nPix != 0: pEnd = p; break
    
    (nPix, xyDxyPix, qPix) = PCWrap.collectPixels(pEnd-1)
    if (nPix >0 ):
      (xPix, yPix, dxPix, dyPix) = tUtil.asXYdXdY( xyDxyPix)
      # qPix, xPix, yPix, dxPix, dyPix = tUtil.thetaAsWMuVar( pixTheta1 )
      # print("????????????????? maxQPixel", pEnd-1, " ", np.max(qPix))
      # print( "????????????????? QPixel", nPix, " ", qPix)
      # print("xPix.size", xPix.size)
      
      uPlt.setLUTScale( 0.0, np.max(qPix))
      uPlt.drawPads( fig, ax[1,1], xPix, yPix, dxPix, dyPix, qPix, doLimits=False, alpha=1.0 )
      #uPlt.drawModelComponents( ax[1,1], thetaTmp, color="black", pattern='x')
      uPlt.drawModelComponents( ax[1,1], thetaResult, color="black", pattern='x', markersize=4)
      uPlt.drawModelComponents( ax[1,1], thetaEMFinal, color="lightgrey", pattern='x')
    ax[1,1].set_title( 'Pixels & PET Algo {:1d}'.format(pEnd-1) )
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
    ax[1,2].set_title( 'Pixels & PET Algo {:1d}'.format(pEnd) )
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
    ax[0,2].set_title( 'Pixels & PET Algo {:1d}'.format(0) )
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
    uPlt.drawPads( fig, ax[0,3], x0, y0, dx0, dy0, z0,  doLimits=False, showEdges=True)
    uPlt.drawPads( fig, ax[1,3], x1, y1, dx1, dy1, z1,  doLimits=False, showEdges=True )
    # Saturated
    xSat = x0[ sat0==1 ]
    ySat = y0[ sat0==1 ]
    uPlt.drawPoints( ax[0,3], xSat, ySat, color='white', pattern='o', markersize=4)
    xSat = x1[ sat1==1 ]
    ySat = y1[ sat1==1 ]
    uPlt.drawPoints( ax[1,3], xSat, ySat, color='white', pattern='o', markersize=4)
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
  # PCWrap.freeMemoryPadProcessing()

  #
  return

def processEvent( preClusters,  ev):
  nbrOfPreClusters = len( preClusters.padId[ev] )
  for pc in range(0, nbrOfPreClusters):
    processPreCluster( preClusters, ev, pc)

def analyseSymetry(reco, nbrPadCutoff):
  lowNbrPads = []
  clusterInfo = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
  nLowQCluster = 0
  #for pc in reco:
  for k in range(100000):
    pc = reco.__next__()
    # processPreCluster ( pc, display=True, displayBefore=False )    
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
    # Min/Max/sum pad charge
    cath0Size = xi[cathi==0].size 
    cath1Size = xi[cathi==1].size 
    nbrCath = (cath0Size > 0) + (cath1Size > 0) 
    nbrPads = xi.size
    if cath0Size == 0:
      sumCh0 = 0; minCh0 = 0;  maxCh0 = 0;
    else:
      sumCh0 = np.sum( chi[cathi==0]) ; 
      minCh0 = np.min( chi[cathi==0]);  
      maxCh0 = np.max( chi[cathi==0]);
    if cath1Size == 0:
      sumCh1 = 0; minCh1 = 0;  maxCh1 = 0;
    else:
      sumCh1 = np.sum( chi[cathi==1]) ; 
      minCh1 = np.min( chi[cathi==1]);  
      maxCh1 = np.max( chi[cathi==1]);
    print("[python] # Total charge on cathodes", sumCh0, sumCh1)
    print("[python] # Min charge on cathodes", minCh0, minCh1 )
    print("[python] # Max charge on cathodes", maxCh0, maxCh1 )
    print("[python] # Saturated pads", np.sum( saturated))
    minQPads = np.min( chi)
    
    sumQPads = (sumCh0 + sumCh1 ) / nbrCath
    
    if nbrPads < nbrPadCutoff:
      lowNbrPads.append( pads )
      clusterInfo[0].append( sumCh0 )
      clusterInfo[1].append( minCh0 )
      clusterInfo[2].append( maxCh0 )
      clusterInfo[3].append( sumCh1 )
      clusterInfo[4].append( minCh1 )
      clusterInfo[5].append( maxCh1 )
      clusterInfo[6].append( minQPads )
      clusterInfo[7].append( max(xi[cathi==0].size,  xi[cathi==1].size) ) 
      clusterInfo[8].append( sumQPads ) 
      clusterInfo[9].append( nbrPads ) 
      clusterInfo[10].append( nbrCath ) 
      clusterInfo[11].append( np.all( saturated ) ) 
      clusterInfo[12].append( xi[cathi==0].size ) 
      clusterInfo[13].append( xi[cathi==1].size ) 
      
  #
  # nClusters = np.zeros( len(low) )
  minQPads = np.array( clusterInfo[6] )
  clusterSizes = np.array( clusterInfo[7] )
  sumQPads = np.array( clusterInfo[8] )
  nbrCathodes = np.array( clusterInfo[10] )
  nbrPads = np.array( clusterInfo[9] )
  """
  for i, cl in enumerate(lowNbrPads):
    ( xi, yi, dxi, dyi, chi, saturated, cathi, adc ) = cl
    nClusters[i] = xi.size
    if xi.size == 0 :
      print( "xi", xi)
      print( "yi", yi)
      print( "chi", chi)
      print( "cathi", cathi)
      input( "next ?")
  """   
  fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(18, 10) )
  
  # ax[0,0].scatter( nbrPads, sumQPads, marker="+")  
  # ax[0,0].set(xlabel='Nbr of Pads', ylabel="Total cluster charge")
  # Sum of charge
  singleCathIdx = np.where( np.array( clusterInfo[10] ) == 1 ) [0]
  ax[0,0].scatter( nbrPads, np.array( clusterInfo[0] ), marker="+", color="orange")  
  ax[0,0].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath0")
  ax[1,0].scatter( nbrPads, np.array( clusterInfo[3] ), marker="+", color="blue")  
  ax[1,0].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath1")
  # Single
  ax[0,3].scatter( nbrPads[singleCathIdx], np.array( clusterInfo[0] )[singleCathIdx], marker="+", color="orange")  
  ax[1,3].scatter( nbrPads[singleCathIdx], np.array( clusterInfo[3] )[singleCathIdx], marker="+", color="blue")  
  ax[0,3].set(xlabel='Nbr of Pads', ylabel="Total cluster charge with single cath0")
  ax[1,3].set(xlabel='Nbr of Pads', ylabel="Total cluster charge with single cath1")
  # Min Charge
  ax[0,1].scatter( nbrPads, np.array( clusterInfo[1] ), marker="+", color="orange")  
  ax[1,1].scatter( nbrPads, np.array( clusterInfo[4] ), marker="+", color="blue")  
  ax[0,1].set(xlabel='NbrOfPads', ylabel="Min charge cath 0/1")
  ax[1,1].set(xlabel='NbrOfPads', ylabel="Min charge cath 0/1")
  # Single
  ax[0,4].scatter( nbrPads[singleCathIdx], np.array( clusterInfo[1] )[singleCathIdx], marker="+", color="orange")  
  ax[1,4].scatter( nbrPads[singleCathIdx], np.array( clusterInfo[4] )[singleCathIdx], marker="+", color="blue")  
  ax[0,4].set(xlabel='Nbr of Pads', ylabel="Min charge with single cath0")
  ax[1,4].set(xlabel='Nbr of Pads', ylabel="Min charge with single cath1")
  
  # Max Charge
  ax[0,2].scatter( nbrPads, np.array( clusterInfo[2] ), marker="+", color="orange")  
  ax[1,2].scatter( nbrPads, np.array( clusterInfo[5] ), marker="+", color="blue")  
  ax[0,2].set(xlabel='NbrOfPads', ylabel="Max charge cath 0/1")
  ax[1,2].set(xlabel='NbrOfPads', ylabel="Max charge cath 0/1")
  # Single
  ax[0,5].scatter( nbrPads[singleCathIdx], np.array( clusterInfo[2] )[singleCathIdx], marker="x", color="orange")  
  ax[1,5].scatter( nbrPads[singleCathIdx], np.array( clusterInfo[5] )[singleCathIdx], marker="x", color="blue")  
  ax[0,5].set(xlabel='Nbr of Pads', ylabel="Max charge with single cath0")
  ax[1,5].set(xlabel='Nbr of Pads', ylabel="Max charge with single cath1")
  
  
  # Single
  doubleCathIdx = np.where( np.array( clusterInfo[10] ) == 2 ) [0]
  ax[2,0].scatter( nbrPads, np.array( clusterInfo[12] ) - np.array( clusterInfo[13] ), marker="x", color="purple")  
  ax[2,0].set(xlabel='Nbr of Pads', ylabel="difference nCath0 -nCath1")
  ax[2,1].scatter( nbrPads, np.array( clusterInfo[0] ) - np.array( clusterInfo[3] ), marker="x", color="purple")  
  ax[2,1].set(xlabel='Nbr of Pads', ylabel="difference Sum charge0 - Sum charge1 ")
  ax[2,2].scatter( nbrPads[doubleCathIdx], np.array( clusterInfo[0] )[doubleCathIdx] - np.array( clusterInfo[3] )[doubleCathIdx], marker="x", color="purple")  
  ax[2,2].set(xlabel='Nbr of Pads', ylabel="difference Sum charge0 - Sum charge1 ")
  
  doubleCathIdx = np.where( np.array( clusterInfo[10] ) == 2 ) [0]
  # DOuble cath
  ax[2,3].scatter( nbrPads[doubleCathIdx], np.array( clusterInfo[0] )[doubleCathIdx], marker="x", color="orange")  
  ax[2,3].scatter( nbrPads[doubleCathIdx], np.array( clusterInfo[3] )[doubleCathIdx], marker="+", color="blue")  
  ax[2,3].set(xlabel='Nbr of Pads', ylabel="Total cluster charge with both cath")
  #
  ax[2,4].scatter( nbrPads[doubleCathIdx], np.array( clusterInfo[1] )[doubleCathIdx], marker="x", color="orange")  
  ax[2,4].scatter( nbrPads[doubleCathIdx], np.array( clusterInfo[4] )[doubleCathIdx], marker="+", color="blue")  
  ax[2,4].set(xlabel='Nbr of Pads', ylabel="Min charge with both cath")  
  #
  ax[2,5].scatter( nbrPads[doubleCathIdx], np.array( clusterInfo[2] )[doubleCathIdx], marker="x", color="orange")  
  ax[2,5].scatter( nbrPads[doubleCathIdx], np.array( clusterInfo[5] )[doubleCathIdx], marker="+", color="blue")  
  ax[2,5].set(xlabel='Nbr of Pads', ylabel="Max charge with both cath")    
  """
  ax[0,1].scatter( minQPads, clusterSizes, marker="+")  
  ax[1,0].scatter( minQPads, sumQPads, marker="+")  
  ax[1,0].set(xlabel='Min Pad charge in the cluster', ylabel="Sum of Charge")
  ax[1,1].scatter( sumQPads, clusterSizes, marker="+")  
  ax[1,1].set(xlabel='Sum of Charge', ylabel="Number of Pads")
  #
  idx = np.where( clusterSizes == 1)
  ax[0,2].scatter( minQPads[idx], sumQPads[idx], marker="+")  
  #
  """
  plt.show()
  
def analyseLowPads(reco, nbrPadCutoff):
  lowNbrPads = []
  clusterInfo = [ [], [], [], [], [], [], [], [], [], [], [], []]
  nLowQCluster = 0
  #for pc in reco:
  for k in range(100000):
    pc = reco.__next__()
    # processPreCluster ( pc, display=True, displayBefore=False )    
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
    # Min/Max/sum pad charge
    cath0Size = xi[cathi==0].size 
    cath1Size = xi[cathi==1].size 
    nbrCath = (cath0Size > 0) + (cath1Size > 0) 
    nbrPads = xi.size
    if cath0Size == 0:
      sumCh0 = 0; minCh0 = 0;  maxCh0 = 0;
    else:
      sumCh0 = np.sum( chi[cathi==0]) ; 
      minCh0 = np.min( chi[cathi==0]);  
      maxCh0 = np.max( chi[cathi==0]);
    if cath1Size == 0:
      sumCh1 = 0; minCh1 = 0;  maxCh1 = 0;
    else:
      sumCh1 = np.sum( chi[cathi==1]) ; 
      minCh1 = np.min( chi[cathi==1]);  
      maxCh1 = np.max( chi[cathi==1]);
    print("[python] # Total charge on cathodes", sumCh0, sumCh1)
    print("[python] # Min charge on cathodes", minCh0, minCh1 )
    print("[python] # Max charge on cathodes", maxCh0, maxCh1 )
    print("[python] # Saturated pads", np.sum( saturated))
    minQPads = np.min( chi)
    
    sumQPads = (sumCh0 + sumCh1 ) / nbrCath
    
    if nbrPads < nbrPadCutoff:
      lowNbrPads.append( pads )
      clusterInfo[0].append( sumCh0 )
      clusterInfo[1].append( minCh0 )
      clusterInfo[2].append( maxCh0 )
      clusterInfo[3].append( sumCh1 )
      clusterInfo[4].append( minCh1 )
      clusterInfo[5].append( maxCh1 )
      clusterInfo[6].append( minQPads )
      clusterInfo[7].append( max(xi[cathi==0].size,  xi[cathi==1].size) ) 
      clusterInfo[8].append( sumQPads ) 
      clusterInfo[9].append( nbrPads ) 
      clusterInfo[10].append( nbrCath ) 
      clusterInfo[11].append( np.all( saturated ) ) 
      
  #
  # nClusters = np.zeros( len(low) )
  minQPads = np.array( clusterInfo[6] )
  clusterSizes = np.array( clusterInfo[7] )
  sumQPads = np.array( clusterInfo[8] )
  nbrCathodes = np.array( clusterInfo[10] )
  nbrPads = np.array( clusterInfo[9] )
  """
  for i, cl in enumerate(lowNbrPads):
    ( xi, yi, dxi, dyi, chi, saturated, cathi, adc ) = cl
    nClusters[i] = xi.size
    if xi.size == 0 :
      print( "xi", xi)
      print( "yi", yi)
      print( "chi", chi)
      print( "cathi", cathi)
      input( "next ?")
  """   
  fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(13, 7) )
  ax[0,0].scatter( nbrPads, sumQPads, marker="+")  
  ax[0,0].set(xlabel='Nbr of Pads', ylabel="Total cluster charge")
  ax[0,1].scatter( nbrPads, np.array( clusterInfo[0] ), marker="+")  
  ax[0,1].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath0")
  ax[0,2].scatter( nbrPads, np.array( clusterInfo[3] ), marker="+")  
  ax[0,2].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath1")
  ax[1,0].scatter( np.array( clusterInfo[11] ), np.array( clusterInfo[2] ), marker="+", color="orange")  
  ax[1,0].scatter( np.array( clusterInfo[11] ), np.array( clusterInfo[5] ), marker="+", color="blue")  
  ax[1,0].set(xlabel='Saturate', ylabel="Min charge cath 0/1")
  # Low charge
  ax[0,3].scatter( nbrPads, np.array( clusterInfo[1] ), marker="+", color="orange")  
  ax[1,3].scatter( nbrPads, np.array( clusterInfo[4] ), marker="+", color="blue")    
  
  """
  ax[0,1].scatter( minQPads, clusterSizes, marker="+")  
  ax[1,0].scatter( minQPads, sumQPads, marker="+")  
  ax[1,0].set(xlabel='Min Pad charge in the cluster', ylabel="Sum of Charge")
  ax[1,1].scatter( sumQPads, clusterSizes, marker="+")  
  ax[1,1].set(xlabel='Sum of Charge', ylabel="Number of Pads")
  #
  idx = np.where( clusterSizes == 1)
  ax[0,2].scatter( minQPads[idx], sumQPads[idx], marker="+")  
  #
  """
  plt.show()
  
def analyseLowCharges(reco, padChargeCutoff):
  lowQCluster = []
  clusterInfo = [ [], [], [], [], [], [], [], [], [], [], [], []]
  nLowQCluster = 0
  #for pc in reco:
  for k in range(100000):
    pc = reco.__next__()
    # processPreCluster ( pc, display=True, displayBefore=False )    
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
    # Min/Max/sum pad charge
    cath0Size = xi[cathi==0].size 
    cath1Size = xi[cathi==1].size 
    nbrCath = (cath0Size > 0) + (cath1Size > 0) 
    if cath0Size == 0:
      sumCh0 = 0; minCh0 = 0;  maxCh0 = 0;
    else:
      sumCh0 = np.sum( chi[cathi==0]) ; 
      minCh0 = np.min( chi[cathi==0]);  
      maxCh0 = np.max( chi[cathi==0]);
    if cath1Size == 0:
      sumCh1 = 0; minCh1 = 0;  maxCh1 = 0;
    else:
      sumCh1 = np.sum( chi[cathi==1]) ; 
      minCh1 = np.min( chi[cathi==1]);  
      maxCh1 = np.max( chi[cathi==1]);
    print("[python] # Total charge on cathodes", sumCh0, sumCh1)
    print("[python] # Min charge on cathodes", minCh0, minCh1 )
    print("[python] # Max charge on cathodes", maxCh0, maxCh1 )
    print("[python] # Saturated pads", np.sum( saturated))
    minQPads = np.min( chi)
    
    sumQPads = (sumCh0 + sumCh1 ) / nbrCath
    
    if minQPads < padChargeCutoff:
      lowQCluster.append( pads )
      clusterInfo[0].append( sumCh0 )
      clusterInfo[1].append( minCh0 )
      clusterInfo[2].append( maxCh0 )
      clusterInfo[3].append( sumCh1 )
      clusterInfo[4].append( minCh1 )
      clusterInfo[5].append( maxCh1 )
      clusterInfo[6].append( minQPads )
      clusterInfo[7].append( max(xi[cathi==0].size,  xi[cathi==1].size) ) 
      clusterInfo[8].append( sumQPads ) 
      nLowQCluster += 1
      if xi.size == 0 :
        print( "xi", xi)
        print( "yi", yi)
        print( "chi", chi)
        print( "cathi", cathi)
        input( "next ?")
  #
  nClusters = np.zeros( len(lowQCluster) )
  minQPads = np.array( clusterInfo[6] )
  clusterSizes = np.array( clusterInfo[7] )
  sumQPads = np.array( clusterInfo[8] )
  print( "len(lowQCluster),  nLowQCluster, minQClusters.size ", len(lowQCluster),  nLowQCluster, minQPads.size)
  for i, cl in enumerate(lowQCluster):
    ( xi, yi, dxi, dyi, chi, saturated, cathi, adc ) = cl
    nClusters[i] = xi.size
    if xi.size == 0 :
      print( "xi", xi)
      print( "yi", yi)
      print( "chi", chi)
      print( "cathi", cathi)
      input( "next ?")
        
  fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 9) )
  ax[0,0].scatter( minQPads, nClusters, marker="+")  
  ax[0,0].set(xlabel='Min Pad charge in the cluster', ylabel="Number of Pads")
  ax[0,1].scatter( minQPads, clusterSizes, marker="+")  
  ax[1,0].scatter( minQPads, sumQPads, marker="+")  
  ax[1,0].set(xlabel='Min Pad charge in the cluster', ylabel="Sum of Charge")
  ax[1,1].scatter( sumQPads, clusterSizes, marker="+")  
  ax[1,1].set(xlabel='Sum of Charge', ylabel="Number of Pads")
  #
  idx = np.where( clusterSizes == 1)
  ax[0,2].scatter( minQPads[idx], sumQPads[idx], marker="+")  
  #
  plt.show()
  
def analyzeChargeCluster( reco ):
  # Result container
  cluster = {"Pads":[], "nPads_0":[], "nPads_1":[],"totalCharge_0":[], "totalCharge_1":[], "surfProj":[], 
       "maxCharge":[], "padSurface":[], "padSurface_0":[],"padSurface_1":[],"DEId":[], "xSeeds":[], "ySeeds":[]}
  """
  cluster["nPads"].append( xi.size ) 
  cluster["totalCharge"].append( np.sum( chi ) ) 
  cluster["maxCharge"].append( np.max( chi ) )
  """
  #for pc in reco:
  for k in range(100000):
    pc = reco.__next__()
    # processPreCluster ( pc, display=True, displayBefore=False )    
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
    # Min/Max/sum pad charge 
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
    nbrOfGroups = 0
    nbrHits = 0
    ## try:
    xyDxy = tUtil.padToXYdXY( xi, yi, dxi, dyi)
    nbrHits = PCWrap.clusterProcess( xyDxy, cathi, saturated, chi, chId )
    print("[python] nbrHits", nbrHits)
    if nbrHits != 0 :
      (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
      nbrOfGroups = np.max(thetaToGrp)
    else :
      thetaResult = np.zeros( 0 )
      thetaToGrp = np.zeros( 0 )
      nbrOfGroups = 0
    print("[python] Cluster Processing Results:")
    print("[python]   theta   :", thetaResult)
    print("[python]   thetaGrp:", thetaToGrp)
    (w, muX, muY, varX, varY) = tUtil.thetaAsWMuVar( thetaResult )
      # Returns the fit status
      # (xyDxyResult, chResult, padToGrp) = PCWrap.collectPadsAndCharges()
    # except :
    #  print("Unexpected error:", sys.exc_info()[0])
    #  print("!!! Exception in clustering/fitting !!!")
    
    # Projection
    (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.copyProjectedPads() 
    surfProj = np.dot( dxProj*2, dyProj*2)
    # Selection criteria 
    diffNbrOfSeeds = (xr.size - nbrHits)
    selected = False
    if xr.size == 1 and nbrHits == 1 :
      maxDxMinSeeds = 1.; maxDyMaxSeeds = 1.
      maxDxMinSeeds, maxDyMinSeeds = aTK.minDxDy( muX, muY, xr, yr)
      selected = (maxDxMinSeeds < 0.07) and (maxDyMinSeeds < 0.07)
    if (selected):
      clusterInfo[0].append( sumCh0 )
      clusterInfo[1].append( minCh0 )
      clusterInfo[2].append( maxCh0 )
      clusterInfo[3].append( sumCh1 )
      clusterInfo[4].append( minCh1 )
      clusterInfo[5].append( maxCh1 )
      clusterInfo[6].append( minQPads )
      clusterInfo[7].append( max(xi[cathi==0].size,  xi[cathi==1].size) ) 
      clusterInfo[8].append( sumQPads ) 
      
      cluster["Pads"].append( pads )

      cluster["DEId"].append( DEId ) 
      cluster["nPads_0"].append( xi[cathi==0].size ) 
      cluster["nPads_1"].append( xi[cathi==1].size ) 
      cluster["totalCharge_0"].append( np.sum( chi[cathi==0] ) ) 
      cluster["totalCharge_1"].append( np.sum( chi[cathi==1] ) ) 
      cluster["maxCharge"].append( np.max( chi ) ) 
      cluster["surfProj"].append( surfProj )
      cluster["xSeeds"].append( xr[0] )
      cluster["ySeeds"].append( yr[0] )
      
      dx0 = dxi[cathi == 0]
      dy0 = dyi[cathi == 0]
      surf0 = np.dot( dx0*2, dy0*2)
      dx1 = dxi[cathi == 1]
      dy1 = dyi[cathi == 1]
      surf1 = np.dot( dx1*2, dy1*2)
      cluster["padSurface_0"].append( surf0 ) 
      cluster["padSurface_1"].append( surf1 ) 
      cluster["padSurface"].append( surf0 + surf1 ) 
      
    
  cluster["nPads_0"] = np.array(cluster["nPads_0"])
  cluster["nPads_1"] = np.array(cluster["nPads_1"])
  cluster["DEId"] = np.array(cluster["DEId"])
  cluster["maxCharge"] = np.array(cluster["maxCharge"])
  cluster["totalCharge_0"] = np.array(cluster["totalCharge_0"])
  cluster["totalCharge_1"] = np.array(cluster["totalCharge_1"])
  cluster["padSurface"] = np.array(cluster["padSurface"])
  cluster["padSurface_0"] = np.array(cluster["padSurface_0"])
  cluster["padSurface_1"] = np.array(cluster["padSurface_1"])
  cluster["surfProj"] = np.array(cluster["surfProj"])
  cluster["xSeeds"] = np.array(cluster["xSeeds"])
  cluster["ySeeds"] = np.array(cluster["ySeeds"])
  # cluster[""] = np.array(cluster[""])
  
  DEIds = np.unique( cluster["DEId"] )
  print("DEIds", DEIds)
  for de in DEIds:
    idx = np.where( de == cluster["DEId"] )[0]
    print("DEid", de,  "idx", idx)
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(13, 7) )
    ax[0,0].scatter(cluster["nPads_0"][idx], cluster["totalCharge_0"][idx], marker="+")
    ax[0,1].scatter(cluster["nPads_1"][idx], cluster["totalCharge_1"][idx], marker="+")
    ax[1,0].scatter(cluster["padSurface_0"][idx], cluster["totalCharge_0"][idx], marker="+")
    ax[1,1].scatter(cluster["padSurface_1"][idx], cluster["totalCharge_1"][idx], marker="+")
    ax[0,0].set(ylabel='Total charge')
    #
    ax[0,0].set(xlabel='nbr of pads cath0')
    ax[0,1].set(xlabel='nbr of pads cath1')
    ax[1,0].set(xlabel='Surface cath0')
    ax[1,1].set(xlabel='Surface cath1')
    xx = np.zeros(0)
    yy = np.zeros(0)
    xdx = np.zeros(0)
    ydy = np.zeros(0)
    q = np.zeros(0)
    cathp = np.zeros(0)
    for i in idx:
      ( xd, yd, dxd, dyd, chd, saturatedd, cathd, adcd ) = cluster["Pads"][i]
      xx = np.append( xx, xd )
      yy = np.append( yy, yd )
      xdx = np.append(xdx, dxd )
      ydy = np.append(ydy, dyd )
      q = np.append( q, chd )
      cathp = np.append( cathp, cathd )
    # uPlt.drawPads( fig, ax[0,2], xx, yy, xdx, ydy, q,  doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.setLUTScale( 0.0, np.max(q) ) 
    idx0 = np.where( cathp == 0)[0]
    idx1 = np.where( cathp == 1)[0]
    ax[0,2].set_xlim( np.min(xx-xdx), np.max(xx+xdx))
    ax[0,2].set_ylim( np.min(yy-ydy), np.max(yy+ydy))
    uPlt.drawPads( fig, ax[0,2], xx[idx0], yy[idx0], xdx[idx0], ydy[idx0], q[idx0], doLimits=False, alpha=0.5, displayLUT=False)
    uPlt.drawPads( fig, ax[0,2], xx[idx1], yy[idx1], xdx[idx1], ydy[idx1], q[idx1], doLimits=False, alpha=0.5)
    # Seeds
    print("Seeds x", cluster["xSeeds"][idx])
    print("Seeds y", cluster["ySeeds"][idx])
    ax[0,2].scatter(cluster["xSeeds"][idx], cluster["ySeeds"][idx], marker="o", color="black")
    print("xx.size", xx.size)
    print("yy.size", yy.size)
    ax[1,2].scatter(cluster["surfProj"][idx], cluster["totalCharge_0"][idx], marker="+")
    ax[1,2].scatter(cluster["surfProj"][idx], cluster["totalCharge_1"][idx], marker="x")
    ax[1,2].scatter(cluster["surfProj"][idx], cluster["maxCharge"][idx], marker="o", color="blue")

    plt.show()
  return

def analyzeSingleClusterWith2Cath( reco ):
  # Result container
  clusterInfo = [ [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
  """
  cluster["nPads"].append( xi.size ) 
  cluster["totalCharge"].append( np.sum( chi ) ) 
  cluster["maxCharge"].append( np.max( chi ) )
  """
  #for pc in reco:
  for k in range(100000):
    pc = reco.__next__()
    # processPreCluster ( pc, display=True, displayBefore=False )    
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
    # Min/Max/sum pad charge 
    cath0Size = xi[cathi==0].size 
    cath1Size = xi[cathi==1].size 
    nbrCath = (cath0Size > 0) + (cath1Size > 0)
    nbrPads = cath0Size +cath1Size
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
    nbrOfGroups = 0
    nbrHits = 0
    ## try:
    if cath0Size !=0 and cath1Size != 0:
      xyDxy = tUtil.padToXYdXY( xi, yi, dxi, dyi)
      nbrHits = PCWrap.clusterProcess( xyDxy, cathi, saturated, chi, chId )
      print("[python] nbrHits", nbrHits)
      if nbrHits != 0 :
        (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
        nbrOfGroups = np.max(thetaToGrp)
      else :
        thetaResult = np.zeros( 0 )
        thetaToGrp = np.zeros( 0 )
        nbrOfGroups = 0
      print("[python] Cluster Processing Results:")
      print("[python]   theta   :", thetaResult)
      print("[python]   thetaGrp:", thetaToGrp)
      (w, muX, muY, varX, varY) = tUtil.thetaAsWMuVar( thetaResult )
        # Returns the fit status
        # (xyDxyResult, chResult, padToGrp) = PCWrap.collectPadsAndCharges()
      # except :
      #  print("Unexpected error:", sys.exc_info()[0])
      #  print("!!! Exception in clustering/fitting !!!")
      
      # Projection
      (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.copyProjectedPads() 
      surfProj = np.dot( dxProj*2, dyProj*2)
      # Selection criteria 
      diffNbrOfSeeds = (xr.size - nbrHits)
      selected = False
      if xr.size == 1 and nbrHits == 1 :
        maxDxMinSeeds = 1.; maxDyMaxSeeds = 1.
        maxDxMinSeeds, maxDyMinSeeds = aTK.minDxDy( muX, muY, xr, yr)
        selected = (maxDxMinSeeds < 0.07) and (maxDyMinSeeds < 0.07)
      minQPads = min( minCh0, minCh1)
      sumQPads = sumCh0 + sumCh1
      selected = True
      if (selected):
        clusterInfo[0].append( sumCh0 )
        clusterInfo[1].append( minCh0 )
        clusterInfo[2].append( maxCh0 )
        clusterInfo[3].append( sumCh1 )
        clusterInfo[4].append( minCh1 )
        clusterInfo[5].append( maxCh1 )
        clusterInfo[6].append( minQPads )
        clusterInfo[7].append( max(xi[cathi==0].size,  xi[cathi==1].size) ) 
        clusterInfo[8].append( sumQPads )
        clusterInfo[9].append( nbrPads ) 
        clusterInfo[10].append( nbrCath ) 
        clusterInfo[11].append( np.all( saturated ) ) 
        clusterInfo[12].append( xi[cathi==0].size ) 
        clusterInfo[13].append( xi[cathi==1].size ) 
        clusterInfo[14].append( xr.size )
        clusterInfo[15].append( nbrHits )
        
  nbrPads = np.array( clusterInfo[9] )
  fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(18, 10) )
  
  # Reco
  #
  idx = np.where ( np.array(clusterInfo[14]) != 0)[0]
  ax[0,0].scatter( nbrPads[idx], np.array( clusterInfo[0])[idx], marker="+", color="orange")  
  ax[0,0].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath0")
  ax[0,1].scatter( nbrPads[idx], np.array( clusterInfo[3] )[idx], marker="+", color="blue")  
  ax[0,1].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath1")
  # Min Charge
  ax[1,0].scatter( nbrPads[idx], np.array( clusterInfo[1] )[idx], marker="+", color="orange")  
  ax[1,1].scatter( nbrPads[idx], np.array( clusterInfo[4] )[idx], marker="+", color="blue")  
  ax[1,0].set(xlabel='NbrOfPads', ylabel="Min charge cath 0/1")
  ax[1,1].set(xlabel='NbrOfPads', ylabel="Min charge cath 0/1")
  # Max Charge
  ax[2,0].scatter( nbrPads[idx], np.array( clusterInfo[2] )[idx], marker="+", color="orange")  
  ax[2,1].scatter( nbrPads[idx], np.array( clusterInfo[5] )[idx], marker="+", color="blue")  
  ax[2,0].set(xlabel='NbrOfPads', ylabel="Max charge cath 0/1")
  ax[2,1].set(xlabel='NbrOfPads', ylabel="Max charge cath 0/1")  
  #
  # GEM
  #
  idx = np.where ( np.array(clusterInfo[15]) != 0)[0]
  ax[0,2].scatter( nbrPads[idx], np.array( clusterInfo[0])[idx], marker="+", color="orange")  
  ax[0,2].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath0")
  ax[0,3].scatter( nbrPads[idx], np.array( clusterInfo[3] )[idx], marker="+", color="blue")  
  ax[0,3].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath1")
  # Min Charge
  ax[1,2].scatter( nbrPads[idx], np.array( clusterInfo[1] )[idx], marker="+", color="orange")  
  ax[1,3].scatter( nbrPads[idx], np.array( clusterInfo[4] )[idx], marker="+", color="blue")  
  ax[1,2].set(xlabel='NbrOfPads', ylabel="Min charge cath 0/1")
  ax[1,3].set(xlabel='NbrOfPads', ylabel="Min charge cath 0/1")
  # Max Charge
  ax[2,2].scatter( nbrPads[idx], np.array( clusterInfo[2] )[idx], marker="+", color="orange")  
  ax[2,3].scatter( nbrPads[idx], np.array( clusterInfo[5] )[idx], marker="+", color="blue")  
  ax[2,2].set(xlabel='NbrOfPads', ylabel="Max charge cath 0/1")
  ax[2,3].set(xlabel='NbrOfPads', ylabel="Max charge cath 0/1")  
  #
  # Both equal
  #
  idx = np.where ( np.logical_and( (np.array(clusterInfo[15]) == np.array(clusterInfo[14])), (np.array(clusterInfo[15]) == 1) ))[0]
  ax[0,4].scatter( nbrPads[idx], np.array( clusterInfo[0])[idx], marker="+", color="orange")  
  ax[0,4].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath0")
  ax[0,5].scatter( nbrPads[idx], np.array( clusterInfo[3] )[idx], marker="+", color="blue")  
  ax[0,5].set(xlabel='Nbr of Pads', ylabel="Total cluster charge cath1")
  # Min Charge
  ax[1,4].scatter( nbrPads[idx], np.array( clusterInfo[1] )[idx], marker="+", color="orange")  
  ax[1,5].scatter( nbrPads[idx], np.array( clusterInfo[4] )[idx], marker="+", color="blue")  
  ax[1,4].set(xlabel='NbrOfPads', ylabel="Min charge cath 0/1")
  ax[1,5].set(xlabel='NbrOfPads', ylabel="Min charge cath 0/1")
  # Max Charge
  ax[2,4].scatter( nbrPads[idx], np.array( clusterInfo[2] )[idx], marker="+", color="orange")  
  ax[2,5].scatter( nbrPads[idx], np.array( clusterInfo[5] )[idx], marker="+", color="blue")  
  ax[2,4].set(xlabel='NbrOfPads', ylabel="Max charge cath 0/1")
  ax[2,5].set(xlabel='NbrOfPads', ylabel="Max charge cath 0/1")  
  plt.show()
  return

if __name__ == "__main__":
    
  pcWrap = PCWrap.setupPyCWrapper()
  pcWrap.o2_mch_initMathieson()
  
  # Read MC data
  # reco = IO.Run2PreCluster(fileName="../Run2Data/recoRun2-100.dat")
  reco = IO.Run2PreCluster(fileName="../Run3Data/orig-pp-july-22-r3.dat")
  analyzeSingleClusterWith2Cath(reco)
  # analyseSymetry(reco, 70)
  # analyseLowPads(reco, 20)
  # analyseLowCharges(reco, 100.0 )
  # analyzeChargeCluster( reco )
  
  if 0:    
    # All
    for pc in reco:
      # processPreCluster ( pc, display=True, displayBefore=False )
      processPreCluster ( pc, display=True, displayBefore=False )
  elif 0:
    # Big preClusters
    #
    evts = [ (5,74), (9,939), (18,655), (38,1152), (39,1149), (46,1007), (65,546), (80,755), (87,833) ]
    evts = [ (9,939), (18,655), (38,1152), (39,1149)]
    
    for ev in evts:
      pc = reco.readPreCluster( 0, ev[0], ev[1] )
      processPreCluster ( pc, display=False, displayBefore=False )
      
  elif 0:
    for i in range(100):
      pc = reco.randomReadPreCluster()
      processPreCluster ( pc, display=True, displayBefore=False )
