# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

import C.PyCWrapper as PCWrap
import Util.geometry as geom
import Util.dataTools as dUtil
import Util.plot as uPlt

def getMatchingMCTrackHits( spanDEIds, boxOfRecoPads, mcObj, ev, verbose = False ):
  """
  x1 = []; y1 = []
  nPreClusters = x0.shape[0]
  if (nPreClusters == 0 ): 
    return ( np.empty( shape=(0)), np.empty( shape=(0) ) )
  """
  deIds = spanDEIds
  if verbose :
    print("matchMCTrackHits x0.shape=", x0.shape, "spanDEIds=", spanDEIds)
  # Pads half-size
  nbrOfTracks = len( mcObj.trackDEId[ev] )
  # print("??? nbrOfTracks", nbrOfTracks)
  # print("??? deIds", deIds)
  x = np.empty(shape=[0])
  y = np.empty(shape=[0])
  # Keep the location of McHits involved
  mcHitsInvolved = []
  totalMCHits = 0
  xl = []; yl = []
  for t in range( nbrOfTracks):
    # Select track hits in DEIds
    for k, d in enumerate(deIds):
      # Must belong to the same DE and have a trackCharge in the DEId 
      # AND must be in the box delimited by the reco pads
      flag0 = (mcObj.trackDEId[ev][t] == d) 
      flag1 = (mcObj.trackCharge[ev][t] > 0)
      flag2 = geom.isInBox ( mcObj.trackX[ev][t], mcObj.trackY[ev][t], boxOfRecoPads )
      flags = np.bitwise_and( flag0, flag1)
      flags = np.bitwise_and( flags, flag2) 
      # flag1 = (mcObj.trackDEId[ev][t] == d) 
      # print("??? flag0 DEids", flag0)
      # print("??? flag1 charge", flag1)
      # print("??? flag2 box", flag2)
      # print("??? flags", flags)
      
      idx = np.where( flags )[0]
      nbrOfHits = idx.shape[0]
    #  print("??? nbrOfHits", nbrOfHits)
      totalMCHits += nbrOfHits
      if ( nbrOfHits ) :
        xl.append( mcObj.trackX[ev][t][idx] )
        yl.append( mcObj.trackY[ev][t][idx] )
      if k > 0:
        input("??? Take care")
    # print("??? xl", xl)
    
    if len(xl) > 0 :
      x = np.hstack(xl)
      y = np.hstack(yl)
  # print("??? x", x)
  # input("next")
    
  return x, y


def matchMCTrackHits(x0, y0, spanDEIds, boxOfRecoPads, rejectedDx, rejectedDy, mcObj, ev, verbose = False ):
  x1 = []; y1 = []
  nPreClusters = x0.shape[0]
  if (nPreClusters == 0 ): 
    return ( 0, 0, 0, 0, 0, 
            np.array([],dtype=np.float), np.array([],dtype=np.float), np.array([],dtype=np.float), 
            np.empty( shape=(0, 0)), [] )
  deIds = spanDEIds
  if verbose :
    print("matchMCTrackHits x0.shape=", x0.shape, "spanDEIds=", spanDEIds)
  # Pads half-size
  dr0 = rejectedDx*rejectedDx + rejectedDy*rejectedDy
  nbrOfTracks = len( mcObj.trackDEId[ev] )
  dMatrix = np.empty(shape=[nPreClusters, 0])
  dMin = np.empty(shape=[0])
  x = np.empty(shape=[0])
  y = np.empty(shape=[0])
  # Keep the location of McHits involved
  mcHitsInvolved = []
  totalMCHits = 0
  for t in range( nbrOfTracks):
    xl = []; yl = []
    # Select track hits in DEIds
    for k, d in enumerate(deIds):
      # Must belong to the same DE and have a trackCharge in the DEId 
      # AND must be in the box delimited by the reco pads
      flag0 = (mcObj.trackDEId[ev][t] == d) 
      flag1 = (mcObj.trackCharge[ev][t] > 0)
      flag2 = geom.isInBox ( mcObj.trackX[ev][t], mcObj.trackY[ev][t], boxOfRecoPads )
      flags = np.bitwise_and( flag0, flag1)
      flags = np.bitwise_and( flags, flag2) 
      # flag1 = (mcObj.trackDEId[ev][t] == d) 
      idx = np.where( flags )[0]
      nbrOfHits = idx.shape[0]
      totalMCHits += nbrOfHits
      xl.append( mcObj.trackX[ev][t][idx] )
      yl.append( mcObj.trackY[ev][t][idx] )
      if k > 0:
        input("??? Take care")
    x = np.vstack(xl)
    y = np.vstack(yl)
    if ( x.shape[1] != 0 ):
      # Build distance matrix
      # [nbrOfHits][nbrX0]
      #
      # To debug
      x1.append(x)
      y1.append(y)
      
      dx = np.tile( x0, (nbrOfHits,1) ).T - x
      dy = np.tile( y0, (nbrOfHits,1) ).T - y
      # dxMin = dx[ np.argmin(np.abs(dx)) ]
      # dyMin = dx[ np.argmin(np.abs(dy)) ]
      dist = np.multiply( dx, dx) + np.multiply( dy, dy)
      # Not used
      selected = (dist < dr0 )
      """
      dist = np.where(  dist < dr0, dist, 0 )
      addHit = np.sum(selected)
      """
      # Select all tracks
      dMatrix = np.hstack( [dMatrix, dist])
      mcHitsInvolved.append( (t, idx) )

    # if x.shape[1] != 0
  # loop on track (t)
  #
  if verbose:
    print("matchMCTrackHits  dMatrix(distances) :")
    print(dMatrix)

  #
  # TP selection
  if dMatrix.size != 0 :
    # For debugging
    x1 = np.concatenate(x1).ravel()
    y1 = np.concatenate(y1).ravel()
    # dMin = np.min(dMatrix, axis=1) # Min per line
    # dMatrix = np.where( dMatrix == 0, 10000, dMatrix)
    dMinIdx = np.argmin( dMatrix, axis=0 )
    # Build TF Matrix
    tfMatrix = np.zeros( dMatrix.shape, dtype=np.int )
    for i in range(dMatrix.shape[1]):
      # Process on assigned hits
      if dMatrix[ dMinIdx[i], i ] < dr0 :
        # Search other value on the same row
        jIdx = np.where( tfMatrix[dMinIdx[i],:] == 1)
        # Check empy set
        if jIdx[0].shape[0] == 0:
          # No other occurence in the tfMatrix line
          # So can set it to 1
          tfMatrix[ dMinIdx[i], i ] = 1
        else:
          # One and only one occurence is possible
          j = jIdx[0][0]
          # Search the distance matrix minimun
          if dMatrix[ dMinIdx[i], j ] < dMatrix[ dMinIdx[i], i ]:
             # Other solution is less than the current index
             # Set to 0 the current element 
             tfMatrix[ dMinIdx[i], i ] = 0
          else:
             # the current element is the best solution (minimum)  
             tfMatrix[ dMinIdx[i], i ] = 1
             # Clean the other solution
             tfMatrix[ dMinIdx[i], j ] = 0
          #    
      else:
        tfMatrix[ dMinIdx[i], i ] = 0
    if verbose:
      print("matchMCTrackHits  tfMatrix :")
      print(tfMatrix)

    #
    # TP/FP
    #
    tpRow = np.sum( tfMatrix, axis=1)
    FP = np.sum( tpRow == 0)
    # Debug
    """
    iii = np.where( tpRow == 0 )
    for kk in list( iii[0] ):
      print( x0[ kk ], y0[ kk] )
    print("FP", FP, tpRow.shape)
    """
    TP = np.sum( tpRow > 0 )
    # Remove the TP and count otheroccurence as FP
    tpRow = tpRow - 1
    # print("tpRow", tpRow)
    FP += np.sum( np.where( tpRow > 0, tpRow, 0) )
    # Debug
    """
    iii = np.where( tpRow > 0 )
    for kk in list( iii[0] ):
      print( x0[ kk ], y0[ kk] )
    print("FP", FP)
    """
    #
    # FN and 
    #
    tpCol = np.sum( tfMatrix, axis=0)
    # print("tpCol", tpCol)
    FN = np.sum( tpCol == 0)
    # Debug
    """
    print("FN", FN)
    iii = np.where( tpCol == 0 )
    print(iii)
    print("??? x1",x1)
    for kk in list( iii[0] ):
      print( x1[ kk ], y1[ kk] )    
    """
    # The min has been performed by column
    # FP += np.sum( np.where( tpCol > 0, tpCol, 0) )
    if verbose : 
      print("[python] matchMCTrackHits # Reco Hits, # MC Hits", tfMatrix.shape )  
      print("[python] matchMCTrackHits TP, FP, FN", TP, FP, FN)
    # 
    # Minimal dsitances for assigned hits given by tfMatrix
    dMin = np.sqrt( dMatrix[ np.where(tfMatrix == 1) ].ravel() )
    """
    print("??? dMatrix", dMatrix )
    print("??? dr0", dr0 )
    print("??? tfMatrix", tfMatrix )
    print("???", np.where(tfMatrix == 1) )
    print("??? em x", x0 )
    print("??? mc x", x1 )
    print("??? em y", y0 )
    print("??? mc y", y1 )
    """
    idx0, idx1 = np.where(tfMatrix == 1)
    if idx0.size != idx1.size :
      input("[python] Pb : one and only one TP per reco hit point")
    dxMin = x0[ idx0 ] - x1[ idx1 ]
    dyMin = y0[ idx0 ] - y1[ idx1 ]
    #
    # Verification
    if tfMatrix.size != 0:
      sumLines = np.sum(tfMatrix, axis=0)
      sumColumns = np.sum(tfMatrix, axis=1)
      if ( (np.where( sumLines  > 1)[0].size != 0) or (np.where( sumColumns > 1)[0].size != 0)  ):
        print("[python] tfMatrix", tfMatrix)
        print("[python] sumLines", sumLines)
        print("[python] sumColumns", sumColumns)
        print("[python] len(mcHitsInvolved)", len(mcHitsInvolved))
        input("[python] Pb in tfMatrix")
  else:
    TP=0; FP=0; FN=0; dMin = np.array([],dtype=np.float)
    dxMin = np.array([],dtype=np.float); dyMin = np.array([],dtype=np.float)
    tfMatrix = np.empty( shape=(0, 0) )
  match = float(TP) / nPreClusters  
  # input("One Cluster")
 
  return match, nPreClusters, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved 

#
# TP, FN, FP, ... analysis
#
def matchingRecoWithMC( preClusters, ev, pc, mcObj, rejectPadFactor=0.5):
  print("[python] # reco Hits", preClusters.rClusterX[ev][pc].size)
  # input("next")
  muX = preClusters.rClusterX[ev][pc]
  muY = preClusters.rClusterY[ev][pc]
  #
  # Assign Cluster hits to MC hits
  spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
  rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
  rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
  spanBox = geom.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                         preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
  match, nbrOfHits, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved = \
               matchMCTrackHits(muX, muY, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
  return (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) 
      
#
# TP, FN, FP, ... analysis
#
def matchingThetaWithMC( theta, preClusters, ev, pc, mcObj, rejectPadFactor=0.5):
  ( w, muX, muY, _, _) = dUtil.thetaAsWMuVar( theta )
  match = 0.0; nbrOfHits = 0; TP=0; FP=0; FN=0; dMin=0; 
  dxMin=np.empty([0]); dyMin=np.empty([0])
  tfMatrix=np.empty([0]); mcHitsInvolved=np.empty([0])
  if ( w.size != 0 ):
    print("[python] wFinal.size", w.size)
    print("[python] # reco Hits", preClusters.rClusterX[ev][pc].size)
    # input("next")

    #
    # Assign Cluster hits to MC hits
    spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
    rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
    rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
    spanBox = geom.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                           preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
    match, nbrOfHits, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved = \
                 matchMCTrackHits(muX, muY, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
  return (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) 
      
def processPreCluster( preClusters, ev, pc, mcObj, display=False ):
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

  # Print cluster info
  print("[python] # DEIds", DEIds)
  print("[python] # Nbr of pads:", xi.size)
  print("[python] # Saturated pads", np.sum( preClusters.padSaturated[ev][pc]))
  print("[python] # Calibrated pads", np.sum( preClusters.padCalibrated[ev][pc]))
  if ( xi.size !=  np.sum( preClusters.padCalibrated[ev][pc])) :
      input("[python] pb with calibrated pads")
  saturated = preClusters.padSaturated[ev][pc].astype( np.int16)
  # xyDxy
  xyDxy = dUtil.padToXYdXY( xi, yi, dxi, dyi)
  
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
  print("[python] nbrHits", nbrHits)
  (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
  print("[python] Cluster Processing Results:")
  print("[python]  theta   :", thetaResult)
  print("[python]  thetaGrp:", thetaToGrp)
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
    (xr, yr, dxr, dyr) = dUtil.asXYdXdY( xyDxyResult)
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

def processEvent( preClusters,  ev, mcObj, matchCutOff = 0.33):
  nbrOfPreClusters = len( preClusters.padId[ev] )
  evTP=0; evFP=0; evFN=0
  evNbrOfHits=0
  # Matching preClusters description
  assign = []
  allDMin = []
  for pc in range(0, nbrOfPreClusters):
  # for pc in range(0, 2):
    thetaf = processPreCluster( preClusters, ev, pc, mcObj, display=False)
    res = matchingThetaWithMC(thetaf, preClusters, ev, pc, mcObj)
    (pc_, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) = res
    if (match > matchCutOff):
        assign.append( res )
        evTP += TP; evFP += FP; evFN += FN
        allDMin.append( dMin )
        evNbrOfHits += nbrOfHits
        print( "[python] ev=", ev, "pc=", pc," TP, FP, FN", TP, FP, FN)
  #
  return (evNbrOfHits, evTP, evFP, evFN, allDMin, assign)

def intersectionArea( xrInf,  xrSup,  yrInf,  yrSup, xInf,  xSup,  yInf,  ySup, z ):
  epsilon = 10.0e-5
  # area = np.zeros( x.size )
  area = 0
  indexex = []
  for j in range(xInf.shape[0]):
    xmin = max( xrInf, xInf[j] )
    xmax = min( xrSup, xSup[j] )
    xInter = ( xmin <= (xmax - epsilon) )
    ymin = max( yrInf, yInf[j] )
    ymax = min( yrSup, ySup[j] )
    yInter = ( ymin <= (ymax - epsilon))
    # intersection
    if xInter and yInter:
      area += (xmax-xmin) * (ymax-ymin) / ((xSup[j] - xInf[j]) * (ySup[j] - yInf[j]) ) * z[j]
      # indexes.append(i)
        
  return area

def buildPixels(  x0, dx0, y0, dy0, x1, dx1, y1, dy1, z0, z1, reso=0.07):
  xMin = max( np.min( x0-dx0),  np.min( x1-dx1) )
  xMax = min( np.max( x0+dx0),  np.max( x1+dx1) )
  yMin = max( np.min( y0-dy0),  np.min( y1-dy1) )
  yMax = min( np.max( y0+dy0),  np.max( y1+dy1) )
  
  xMin = xMin - reso
  xMax = xMax + reso
  yMin = yMin - reso
  yMax = yMax + reso
  nXPixels = int( np.ceil( (xMax - xMin) / reso ))
  nYPixels = int( np.ceil( (yMax - yMin) / reso ))
  x0Inf = x0 - dx0
  x0Sup = x0 + dx0
  y0Inf = y0 - dy0
  y0Sup = y0 + dy0
  x1Inf = x1 - dx1
  x1Sup = x1 + dx1
  y1Inf = y1 - dy1
  y1Sup = y1 + dy1
  xPix = np.zeros( (nXPixels, nYPixels))
  dxPix = np.zeros( (nXPixels, nYPixels))
  yPix = np.zeros( (nXPixels, nYPixels))
  dyPix = np.zeros( (nXPixels, nYPixels))
  qPix = np.zeros( (nXPixels, nYPixels))
  for i in range(nXPixels):
    xPixInf = i * reso + xMin
    xPixSup = xPixInf + reso 
    for j in range(nYPixels):
      yPixInf = j * reso + yMin
      yPixSup = yPixInf + reso 
      areaCharge0 = intersectionArea(xPixInf, xPixSup, yPixInf, yPixSup, x0Inf, x0Sup, y0Inf, y0Sup, z0)
      areaCharge1 = intersectionArea(xPixInf, xPixSup, yPixInf, yPixSup, x1Inf, x1Sup, y1Inf, y1Sup, z1)
      qPix[i, j] = min( areaCharge0, areaCharge1 )
      xPix[i,j] = xPixInf + reso*0.5
      dxPix[i,j] = reso*0.5
      yPix[i,j] = yPixInf + reso*0.5
      dyPix[i,j] = reso*0.5
      #qPix[i, j] = 1.0
    #
  #
  xPix = xPix.reshape( nXPixels * nYPixels )
  yPix = yPix.reshape( nXPixels * nYPixels )
  dxPix = dxPix.reshape( nXPixels * nYPixels )
  dyPix = dyPix.reshape( nXPixels * nYPixels )  
  qPix = qPix.reshape( nXPixels * nYPixels )
  # Filter
  uMin = min( np.min(z0), np.min(z1))
  areaRatio =  (reso*reso) / (dx0[0]*dx1[0])
  print( "build Pixels: min charge", uMin, "area ratio", areaRatio, "qPixMin", uMin* areaRatio)
  # input("next")
  
  # qCut = 0.02
  qCut = uMin* areaRatio
  idx = np.where( qPix > qCut )[0]
  xPix = xPix[idx]
  dxPix = dxPix[idx]
  yPix = yPix[idx]
  dyPix = dyPix[idx]
  qPix = qPix[idx]
  print("buildPixel: nbr of pixels=", qPix.size)
  return (xPix, dxPix, yPix, dyPix, qPix)

def iterateEMPoisson( Cij, Ci, Cj, maskCij, qPix, qPad, qPredPad):
  nPads = qPad.size
  nPix = qPix.size
  newQPix = np.zeros( nPix )

  for i in range(nPads):
    qPredPad[i] = np.dot( Cij[ i, 0:nPix], qPix[0:nPix]*maskCij[i,:] )  #  Sum_j cij.qj
    # Avoiding to divide by zero
    if (qPredPad[i] < 1.0e-6):
      if (qPad[i] < 1.0e-6):
         qPredPad[i] = 1.0
      else:
        qPredPad[i] = 0.1 * qPad[i]
      #print("qPredPad[", i,"]set to 1"

  if np.any( qPredPad == 0.0 ):
    print("Warning it=",  "qPredPad = 0:", np.where( qPredPad == 0 )[0], "qPad[qPredPad = 0]", qPad[ np.where( qPredPad == 0 )[0] ] , "sum qPix", np.sum(qPix) ) 
 
  # Update qPix (ie w)
  for j in range(nPix):
    # Normalization term
    # Cj[j] = np.sum( Cij[:,j   
    r = np.sum( Cij[:,j]*qPad[0:nPads] / qPredPad[0:nPads] )
    if ( Cj[j] > 0 ):
      newQPix[j] = r * qPix[j] / Cj[j]
    else :
      newQPix[j] = 0
  #
  return newQPix, qPredPad


def pixelFilter( qPix, maskCij, qCutMode):
  #
  # Filter
  #
  # idx = np.where (qPix > 1.e-02 )[0]
  # idx = np.where (qPix > 5 )[0]
  qPixMax = np.max( qPix )
  qPixCut = 20
  qPixCut = 0.0
  qPixCut = 2.0e-1
  threshold = np.min( qPix )
  threshold = 0.2
  if qCutMode == -1:
    # Percent of the min
    qPixCut = 1.02 * np.min( qPix )
  elif qCutMode == 0:
    qPixCut = 0
  else :
    qPixCut = max( 0.01 * np.max( qPix ),  threshold)
    
  # qPixCut = 0.0

  # qPixCut = max( 0.01 * np.sum( qPix ), 0.2 )
  # idx = np.where (qPix > qPixCut )[0]
  idx = np.where (qPix <= qPixCut )[0]
  # mux = mux[idx]
  # muy = muy[idx]
  # update MaskCij
  for j in idx:
    maskCij[:, j] = 0
  return qPixCut, maskCij
        
def EMPoissonSQR( xyInfSup, qPad, theta, chId, minPadResidu, nIt, qCutMode=1):
 
  # global SaveCij
  Cij = PCWrap.computeCij( xyInfSup, theta, chId )
  
  (q, mux, muy, varx, vary) = dUtil.thetaAsWMuVar( theta)
  qPix = np.copy( q )
  nPix = qPix.size
  nPads = xyInfSup.size // 4
  nPixMax = nPads
  maskCij = np.ones( Cij.shape )
  Cj = np.zeros( nPix )
  for j in range(nPix):
    Cj[j] = np.sum( Cij[:,j])
  Ci = np.zeros( nPads )
  for i in range (nPads):
    Ci[i] = np.sum( Cij[i,:])
  qPredPad = np.zeros( nPads )
    
  converge = False
  it = 0
  residu = 0
  padResidu = 0
  while( not converge ):
    qPixCut, maskCij = pixelFilter( qPix, maskCij, qCutMode)
    # Update Ci, Cj
    for j in range(nPix):
      Cj[j] = np.sum( Cij[:,j] * maskCij[:,j] )  
    for i in range (nPads):
      Ci[i] = np.sum( Cij[i,:] * maskCij[i,:] )
    #
    prevQPix = np.copy( qPix )      
    qPix1, qPredPad = iterateEMPoisson( Cij, Ci, Cj, maskCij, qPix, qPad, qPredPad)
    qPix2, qPredPad = iterateEMPoisson( Cij, Ci, Cj, maskCij, qPix1, qPad, qPredPad)
    r = qPix1 - qPix
    v = (qPix2 - qPix1) - r
    # print("??? it=", it, " np.linalg.norm( r ),  np.linalg.norm( v )", np.linalg.norm( r ),  np.linalg.norm( v ))
    rNorm = np.linalg.norm( r )
    vNorm = np.linalg.norm( v )
    if ( rNorm < 1.0e-12 ) or (vNorm < 1.0e-12 ):
      rNorm = 0
      convergence = True 
      """
    elif (vNorm < 1.0e-12 ): 
        print("Warning: it=", it, " alpha no defined ", rNorm, vNorm )
        qPix, qPredPad = iterateEMPoisson( Cij, Ci, Cj, maskCij, qPix2, qPad, qPredPad)
        continue
      """
    else :
      alpha = - rNorm / vNorm 
      # 
      qPix = qPix - 2.0*alpha*r + alpha*alpha*v
      qPix, qPredPad = iterateEMPoisson( Cij, Ci, Cj, maskCij, qPix, qPad, qPredPad)
    
    residu = np.abs( prevQPix - qPix )
    if (np.sum( residu ) / qPix.size) < 1.0e-12:
      converge = True
    padResidu = np.abs( qPad - qPredPad )
    if 1 :
      # print("it=", it, ", pixResidu", np.sum(residu), ", padResidu", np.sum(padResidu), ", max(padResidu)", np.max(padResidu), ", ratio residu/charge", np.sum(padResidu)/np.sum(qPad))
      print("it=", it, ", <pixResidu>", np.sum(residu)/qPix.size, ", <padResidu>", np.sum(padResidu)/qPad.size, ", max(padResidu)", np.max(padResidu), ", ratio pad residu/charge", np.sum(padResidu)/np.sum(qPad))
      # print("       sum padResidu>=", np.sum(padResidu), ", sum chargepad", np.sum(qPad), ", ratio residu/charge", np.sum(padResidu)/np.sum(qPad))
      # print("       EMPoisson qPixCut=", qPixCut)
    it += 1
    # converge = (it>nIt)
    converge = converge or (np.sum(padResidu)/qPad.size < minPadResidu) or (it>nIt)
    #
  print("it=", it, ", <pixResidu>", np.sum(residu)/qPix.size, ", <padResidu>", np.sum(padResidu)/qPad.size, ", max(padResidu)", np.max(padResidu))
  print("       sum padResidu =", np.sum(padResidu), ", sum chargepad", np.sum(qPad), ", ratio residu/charge", np.sum(padResidu)/np.sum(qPad))
  print("       EMPoisson qPixCut=", qPixCut)
  print( "Total Charge", np.sum(qPad))
  print( "Total Predictited Charge", np.sum(qPredPad))
  print( "Total Pixel Charge", np.sum(qPix))
  #
  print("tn, np.sum(Cij)", np.sum(Cij))
  # ????

  idx = np.where ( qPix > qPixCut )[0]
  qPix = qPix[idx]
  mux = mux[idx]
  muy = muy[idx]
  varx = varx[idx]
  vary = vary[idx]  
  
  theta = dUtil.asTheta(qPix, mux, muy, varx, vary )
  # Save Cij
  """
  SaveCij = np.zeros( (nPads, idx.size) )
  for newJ,j in enumerate(idx):
    SaveCij[:, newJ] = Cij[:,j]
  """
  #
  return theta, padResidu

def EMPoisson( xyInfSup, qPad, theta, chId, nIt, qCutMode=0):
 
  # global SaveCij
  Cij = PCWrap.computeCij( xyInfSup, theta, chId )
  
  (q, mux, muy, varx, vary) = dUtil.thetaAsWMuVar( theta)
  qPix = np.copy( q )
  nPix = qPix.size
  nPads = xyInfSup.size // 4
  nPixMax = nPads
  maskCij = np.ones( Cij.shape )
  Cj = np.zeros( nPix )
  for j in range(nPix):
    Cj[j] = np.sum( Cij[:,j])
  Ci = np.zeros( nPads )
  for i in range (nPads):
    Ci[i] = np.sum( Cij[i,:])
  qPredPad = np.zeros( nPads )
    
  """
  Cij = np.copy( C0ij )
  C0j = np.zeros( nPix )
  for j in range(nPix):
    C0j[j] = np.sum( Cij[:,j])
  C0i = np.zeros( nPads )
  for i in range (nPads):
    C0i[i] = np.sum( Cij[i,:])
  # Estimate Pad Charge (f_i)
  Ci = np.copy(C0i)
  Cj = np.copy(C0j)
  """
  """
  print("t0, np.sum(Cij)", np.sum(Cij))
  print("t0, Ci - pads", Ci.size)
  print(Ci)
  print("t0, Cj - pixels", Cj.size),
  print(Cj)
  input("next")
  """
  converge = False
  it = 0
  while( not converge ):
    # if (it >= 20):
    #  nPixMax = (nPads + 1) // 3
    if ( it % 1) == 0:
      #
      # Filter
      #
      # idx = np.where (qPix > 1.e-02 )[0]
      # idx = np.where (qPix > 5 )[0]
      qPixMax = np.max( qPix )
      qPixCut = 20
      qPixCut = 0.0
      qPixCut = 2.0e-1
      threshold = np.min( qPix )
      threshold = 0.2
      if qCutMode == -1:
        # Percent of the min
        qPixCut = 1.02 * np.min( qPix )
      else :
        qPixCut = max( 0.01 * np.max( qPix ),  threshold)
        
      qPixCut = 0.0

      # qPixCut = max( 0.01 * np.sum( qPix ), 0.2 )
      # idx = np.where (qPix > qPixCut )[0]
      otherIdx = np.where (qPix <= qPixCut )[0]
      # mux = mux[idx]
      # muy = muy[idx]
      # update MaskCij
      for j in otherIdx:
        maskCij[:, j] = 0
      for j in range(nPix):
        Cj[j] = np.sum( Cij[:,j] * maskCij[:,j] )  
      for i in range (nPads):
        Ci[i] = np.sum( Cij[i,:] * maskCij[i,:] )
      # qPix.size must be  less than nPixMax
      
      # Filter with Cij ~ 1 or 2
      maxIdx = np.argmax( Cij[:,j] * maskCij[:,j] )
      
      nPix = qPix.size
      prevQPix = np.copy( qPix )
    for i in range(nPads):
      qPredPad[i] = np.dot( Cij[ i, 0:nPix], qPix[0:nPix]*maskCij[i,:] )  #  Sum_j cij.qj
      # Avoiding to divide by zero
      if (qPredPad[i] < 1.0e-6) and (qPad[i] < 1.0e-6):
        qPredPad[i] = 1.0
        #print("qPredPad[", i,"]set to 1")
        
    if np.any( qPredPad == 0.0 ):
      print("Warning it=", it, "qPredPad = 0:", np.where( qPredPad == 0 )[0], "qPad[qPredPad = 0]", qPad[ np.where( qPredPad == 0 )[0] ] , "sum qPix", np.sum(qPix) ) 
    """
    print("qPredPad ", qPredPad)
    print("qPix ", qPix)
    input("next")
    """
    # Update qPix (ie w)
    for j in range(nPix):
      # Normalization term
      # Cj[j] = np.sum( Cij[:,j])
      # 
      
      r = np.sum( Cij[:,j]*qPad[0:nPads] / qPredPad[0:nPads] )
      if ( Cj[j] > 0 ):
        qPix[j] = r * qPix[j] / Cj[j]
      else :
        qPix[j] = 0
    #
    residu = np.abs( prevQPix - qPix )
    padResidu = np.abs( qPad - qPredPad )
    if it == 0:
      print("it=", it, ", <pixResidu>", np.sum(residu)/qPix.size, ", <padResidu>", np.sum(padResidu)/qPad.size, ", max(padResidu)", np.max(padResidu))
      print("       sum padResidu>=", np.sum(padResidu), ", sum chargepad", np.sum(qPad), ", ratio residu/charge", np.sum(padResidu)/np.sum(qPad))
      print("       EMPoisson qPixCut=", qPixCut)
    it += 1
    converge = (np.sum(padResidu)/np.sum(qPad) < 0.05) or (it>nIt)
    converge = (it>nIt)
    #
  print("it=", it, ", <pixResidu>", np.sum(residu)/qPix.size, ", <padResidu>", np.sum(padResidu)/qPad.size, ", max(padResidu)", np.max(padResidu))
  print("       sum padResidu =", np.sum(padResidu), ", sum chargepad", np.sum(qPad), ", ratio residu/charge", np.sum(padResidu)/np.sum(qPad))
  print("       EMPoisson qPixCut=", qPixCut)
  print( "Total Charge", np.sum(qPad))
  print( "Total Predictited Charge", np.sum(qPredPad))
  print( "Total Pixel Charge", np.sum(qPix))
  #
  print("tn, np.sum(Cij)", np.sum(Cij))
  # ????
  # qPixCut = 0.0
  qPixCut = min( qPixCut, 0.2  )
  idx = np.where ( qPix > qPixCut )[0]
  qPix = qPix[idx]
  mux = mux[idx]
  muy = muy[idx]
  varx = varx[idx]
  vary = vary[idx]  
  theta = dUtil.asTheta(qPix, mux, muy, varx, vary )
  # Save Cij
  """
  SaveCij = np.zeros( (nPads, idx.size) )
  for newJ,j in enumerate(idx):
    SaveCij[:, newJ] = Cij[:,j]
  """
  #
  return theta
  
def addSimpleEdgesPads( x, y, dx, dy, q ):
    if x.size == 0: return x, y, dx, dy, q
    # Direction
    dirX = np.min(dx) < np.min(dy)
    if ( dirX ):
      u = x
      v = y
      du = dx
      dv = dy
    else:
      u = y
      v = x
      du = dy
      dv = dx      
    minIdx = np.argmin(u)
    maxIdx = np.argmax(u)
    u = list( u )
    v = list( v )
    du = list( du )
    dv = list( dv )
    q = list( q )
    u.append( u[minIdx] - 2*du[minIdx])
    du.append( du[minIdx] ) 
    v.append( v[minIdx] ) 
    dv.append( dv[minIdx] )
    q.append( 0.001)
    #
    u.append( u[maxIdx] + 2*du[maxIdx])
    du.append( du[maxIdx] ) 
    v.append( v[maxIdx] ) 
    dv.append( dv[maxIdx] ) 
    q.append( 0.001)
    if dirX:
      x = np.asarray( u )
      y = np.asarray( v )
      dx = np.asarray( du )
      dy = np.asarray( dv )
    else:
      x = np.asarray( v )
      y = np.asarray( u )
      dx = np.asarray( dv )
      dy = np.asarray( du )        
    #
    q = np.asarray( q )
    #
    return x, y, dx, dy, q

def computeChiSq( xyInfSup, q, chId, theta ):
  pixTheta, padResidu = EMPoissonSQR( xyInfSup, q, theta, chId, 1.0, 60,  qCutMode=0) 
  qBar = np.where( q < 1.0e-9, 1, q)
  chisq = np.dot( padResidu/qBar, padResidu/qBar)
  N = xyInfSup.size // 4
  K = theta.size // 5
  dof = N - (3*K -1) 
  print("chisq, dof", chisq, dof)
  return chisq / dof    

def findLocalMaxWithPET( xyDxy0, xyDxy1, q0, q1, chId, proj=None, display=False ):
    x0, y0, dx0, dy0 = dUtil.asXYdXdY( xyDxy0 )
    x1, y1, dx1, dy1 = dUtil.asXYdXdY( xyDxy1 )
    # Add Pads
    x0, y0, dx0, dy0, q0 = geom.addBoundaryPads( x0, y0, dx0, dy0, q0 )
    x1, y1, dx1, dy1, q1 = geom.addBoundaryPads( x1, y1, dx1, dy1, q1 )
        
    # Reso 
    if (x0.size != 0  ):
      xMin  = np.min( x0 - dx0 )
      xMax  = np.max( x0 + dx0 )
      yMin  = np.min( y0 - dy0 )
      yMax  = np.max( y0 + dy0 )
      dxMin = np.min( dx0 )
      dyMin = np.min( dy0 )
    else:
      xMin  = x1[0]
      xMax  = x1[0]
      yMin  = y1[0]
      yMax  = y1[0]
      dxMin = dx1[0]
      dyMin = dy1[0]        
    if ( x1.size != 0  ):
      xMin  = min( xMin, np.min( x1 - dx1 ) )
      xMax  = max( xMax, np.max( x1 + dx1 ) )
      yMin  = min( yMin, np.min( y1 - dy1 ) )
      yMax  = max( yMax, np.max( y1 + dy1 ) )
      dxMin = min( dxMin, np.min( dx1 ) )
      dyMin = min( dyMin, np.min( dy1 ) )
      
    reso = 0.5 * min(dxMin, dyMin )
    print( "findLocalMaxWithPET: reso=", reso)
    #
    # Sub sample proj
    # if 0:
    if proj is not None:
      (xProj, dxProj, yProj, dyProj, qProj) = proj
      n = xProj.size
      xPix = np.zeros(4*n)
      yPix = np.zeros(4*n)
      dxPix = np.zeros(4*n)
      dyPix = np.zeros(4*n)
      qPix = np.zeros(4*n)
      
      # NW
      xPix[0:n] = xProj - 0.5 *dxProj
      yPix[0:n] = yProj + 0.5 *dyProj
      dxPix[0:n] = 0.5 * dxProj
      dyPix[0:n] = 0.5 * dyProj
      qPix[0:n] = 0.25 * qProj

      # NE
      xPix[n:2*n] = xProj + 0.5 *dxProj
      yPix[n:2*n] = yProj + 0.5 *dyProj
      dxPix[n:2*n] = 0.5 * dxProj
      dyPix[n:2*n] = 0.5 * dyProj
      qPix[n:2*n] = 0.25 * qProj
      
      # SW
      xPix[2*n:3*n] = xProj - 0.5 *dxProj
      yPix[2*n:3*n] = yProj - 0.5 *dyProj
      dxPix[2*n:3*n] = 0.5 * dxProj
      dyPix[2*n:3*n] = 0.5 * dyProj
      qPix[2*n:3*n] = 0.25 * qProj
      
      # SE
      xPix[3*n:4*n] = xProj + 0.5 *dxProj
      yPix[3*n:4*n] = yProj - 0.5 *dyProj
      dxPix[3*n:4*n] = 0.5 * dxProj
      dyPix[3*n:4*n] = 0.5 * dyProj
      qPix[3*n:4*n] = 0.25 * qProj
      
    else:
      xPix, dxPix, yPix, dyPix, qPix = buildPixels( x0, dx0, y0, dy0, x1, dx1, y1, dy1, q0, q1, reso)
      print("??? qPix", qPix)
    #
    initSize = qPix.size
    thetaInit = dUtil.asTheta(qPix, xPix, yPix, dxPix, dyPix )
    x = np.hstack( [x0, x1] )
    dx = np.hstack( [dx0, dx1] )
    y = np.hstack( [y0, y1] )
    dy = np.hstack( [dy0, dy1] )
    q = np.hstack( [q0, q1] )
    maxQPad = np.max( q )
    xyInfSup = dUtil.padToXYInfSup( x, y, dx, dy)
    # 
    pixTheta = dUtil.asTheta( qPix, xPix, yPix, dxPix, dyPix ) 
    # visuPix = np.copy( pixTheta )
    pixTheta, _ = EMPoissonSQR( xyInfSup, q, pixTheta, chId, 3.0, 5, qCutMode=0 ) 
    visuPix0 = np.copy( pixTheta )
    ( qLocMax, xLocMax, yLocMax, dxLocMax, dyLocMax ) = dUtil.thetaAsWMuVar( pixTheta )
    pixTheta, _ = EMPoissonSQR( xyInfSup, q, pixTheta, chId, 1.5, 60, qCutMode=1 ) 
    if (0):
      # Max selection with connected groups of pixels
      nGrp, pixToGrp = geom.getConnexComponents(  xLocMax, yLocMax, dxLocMax, dyLocMax )
      
      grpCharge = np.zeros( nGrp + 1)
      locX = np.zeros( nGrp + 1)
      locY = np.zeros( nGrp + 1)
      for g in range(1,nGrp+1):
        idx = np.where( pixToGrp == g )[0]
        grpCharge[g] = np.sum( qLocMax[idx] )
        locX[g] = np.sum( xLocMax[idx] * qLocMax[idx] )/ grpCharge[g]
        locY[g] = np.sum( yLocMax[idx] * qLocMax[idx] ) / grpCharge[g]
  
      print( "grpCharge", grpCharge)
      w = grpCharge / np.sum( grpCharge ) 
      print( "w", w)
      print("xLocMax", locX)
      #
    elif 0:
      qPix, xPix, yPix, dxPix, dyPix = dUtil.thetaAsWMuVar( pixTheta )
      xyDxyPix = dUtil.asXYdXY( xPix, yPix, dxPix, dyPix )

      pixIdx, locMaxIdx, w, locX, locY  = geom.clipOnLocalMax( xyDxyPix, qPix, hard=False)
      qPix = qPix[pixIdx] 
      xPix = xPix[pixIdx] 
      yPix = yPix[pixIdx] 
      dxPix = dxPix[pixIdx] 
      dyPix = dyPix[pixIdx] 
      pixTheta = dUtil.asTheta( qPix, xPix, yPix, dxPix, dyPix )

      pixTheta = EMPoisson( xyInfSup, q, pixTheta, chId, 100) 

      qPix, xPix, yPix, dxPix, dyPix = dUtil.thetaAsWMuVar( pixTheta )
      xyDxyPix = dUtil.asXYdXY( xPix, yPix, dxPix, dyPix )
      # (qPix, xPix, yPix, dxPix, dyPix) = geom.expandOnLocalMax( xyDxyPix, qPix )
      pixIdx, idxLocMax, w, locX, locY  = geom.clipOnLocalMax( xyDxyPix, qPix, hard=False)
      qPix = qPix[pixIdx] 
      xPix = xPix[pixIdx] 
      yPix = yPix[pixIdx] 
      dxPix = dxPix[pixIdx] 
      dyPix = dyPix[pixIdx] 
      pixTheta = dUtil.asTheta( qPix, xPix, yPix, dxPix, dyPix )
      
      pixTheta = EMPoisson( xyInfSup, q, pixTheta, chId, 100 )  
      visuPix1 = np.copy( pixTheta )

      # visuPix1 = np.copy( pixTheta )
      
      qPix, xPix, yPix, dxPix, dyPix = dUtil.thetaAsWMuVar( pixTheta )
      xyDxyPix = dUtil.asXYdXY( xPix, yPix, dxPix, dyPix )      
      pixIdx, idxLocMax, w, locX, locY  = geom.clipOnLocalMax( xyDxyPix, qPix, hard=False)

      print( "Pixel size initSize=", initSize, ", end size=", qPix.size )
      #
    elif 1:
      qPix, xPix, yPix, dxPix, dyPix = dUtil.thetaAsWMuVar( pixTheta )
      xyDxyPix = dUtil.asXYdXY( xPix, yPix, dxPix, dyPix )
      # visuPix0 = np.copy( pixTheta )

      pixIdx, locMaxIdx, w, locX, locY, dxLoc, dyLoc  = geom.clipOnLocalMax( xyDxyPix, qPix, hard=False)
      qPix = qPix[pixIdx] 
      xPix = xPix[pixIdx] 
      yPix = yPix[pixIdx] 
      dxPix = dxPix[pixIdx] 
      dyPix = dyPix[pixIdx] 
      pixTheta = dUtil.asTheta( qPix, xPix, yPix, dxPix, dyPix )


      pixTheta, _ = EMPoissonSQR( xyInfSup, q, pixTheta, chId, 1.0, 60,  qCutMode=0) 
      # visuPix0 = np.copy( pixTheta )

      # Only to get coordinates of the local max (w, mu)
      qPix, xPix, yPix, dxPix, dyPix = dUtil.thetaAsWMuVar( pixTheta )
      xyDxyPix = dUtil.asXYdXY( xPix, yPix, dxPix, dyPix )      
      pixIdx, idxLocMax, w, locX, locY, dxLoc, dyLoc  = geom.clipOnLocalMax1( xyDxyPix, qPix, hard=True)
      visuPix1 = np.copy( pixTheta )
      print( "Pixel size initSize=", initSize, ", end size=", qPix.size )

    else :
      visuPix = np.copy( pixTheta )    
      qPix, xPix, yPix, dxPix, dyPix = dUtil.thetaAsWMuVar( pixTheta )
      xyDxyPix = dUtil.asXYdXY( xPix, yPix, dxPix, dyPix )
      idxLocMax, w, locX, locY  = geom.simpleLaplacian2D( xyDxyPix, qPix)
    #
    # Filter position
    dx = np.max(dxPix) + np.min( dxPix )
    dy = np.max(dyPix) + np.min( dyPix )
    print("??? dxLoc.shape", dxLoc.shape )
    print("??? dyLoc.shape", dyLoc.shape )
    """
    eps = 1.0e-04
    n = w.size
    close = [[] for i in range(n)]
    nClose = np.zeros( n )
    for i in range(n):
      xMask = np.abs( locX[i] - locX[0:n]) < ( 3*dxLoc[i] + dxLoc[0:n] +eps) 
      yMask = np.abs( locY[i] - locY[0:n]) < ( 3*dyLoc[i] + dyLoc[0:n]+eps) 
      close[i] = np.where ( np.bitwise_and(xMask, yMask))[0]
      nClose[i] = close[i].size
    idx = np.argsort( -nClose)
    mask = np.ones( n )
    wNew = np.zeros( n )
    xNew = np.zeros( n )
    yNew = np.zeros( n )
    for k in idx:
      v = close[k] 
      wNew[k] = np.sum( w[v] * mask[v] )
      xNew[k] = np.sum( locX[v] * mask[v] * w[v] )
      yNew[k] = np.sum( locY[v] * mask[v] * w[v] )
      print( "k, xNew[k], wNew[k]", k, xNew[k], wNew[k] )
      xNew[k] = xNew[k] / wNew[k] 
      yNew[k] = yNew[k] / wNew[k] 
      mask[v] = 0;
    idx = np.where( mask == 0)[0]
    w = wNew[idx]
    locX = xNew[idx]
    locY = yNew[idx]
    """
    # 
    # Select local Max
    cutRatio = 0.01
    qCut = cutRatio * np.max( w )
    initSize = w.size
    print("last solution qCut=", qCut, "sum w", np.sum( w ))
    idx = np.where( w > qCut )[0]
    w = w[idx]
    locX = locX[ idx ]
    locY = locY[ idx ]
    dxLocMax = dxLocMax[idx]
    dyLocMax = dyLocMax[idx]
    refinedTheta = dUtil.asTheta( w, locX, locY, dxLocMax, dyLocMax)
    print( "--- > w selection: cutOff=", cutRatio, "nbr of removed peaks=", initSize - w.size)
    chisq = computeChiSq( xyInfSup, q, chId, refinedTheta )
    print( "---> chisq=", chisq)
    return ( refinedTheta, thetaInit, visuPix0, visuPix1 )

def trackLocalMax( grid, locMax, qLocMax):
  ( xOrig, yOrig) = grid["origin"]
  dxy = grid["dxy"]
  (nX, nY) = grid["nCells"]
  for (i,j) in locMax :
    x = xOrig + i*dxy
    y = xOrig + i*dxy
      
  return

def findLocalMaxWithPETV1( xyDxy0, xyDxy1, q0, q1, chId ):
    x0, y0, dx0, dy0 = dUtil.asXYdXdY( xyDxy0 )
    x1, y1, dx1, dy1 = dUtil.asXYdXdY( xyDxy1 )
    # Reso 
    xMin = min( np.min( x0 - dx0 ), np.min( x1 - dx1 ) )
    xMax = max( np.max( x0 + dx0 ), np.max( x1 + dx1 ) )
    yMin = min( np.min( y0 - dy0 ), np.min( y1 - dy1 ) )
    yMax = max( np.max( y0 + dy0 ), np.max( y1 + dy1 ) )
    dxMin = min( np.min( dx0 ), np.min( dx1 ) )
    dyMin = min( np.min( dy0 ), np.min( dy1 ) )
    reso = 0.5 * min(dxMin, dyMin )
    xOrigin = min( np.min( x0  ), np.min( x1 ) )
    yOrigin = min( np.min( y0  ), np.min( y1 ) )
    grid = {}
    grid["origin"] = ( xOrigin, yOrigin)
    grid["dxy"] = reso
    grid["nCells"] = (nX, nY)
    
    print( "findLocalMaxWithPET: reso=", reso)
    xPix, dxPix, yPix, dyPix, qPix = buildPixels( x0, dx0, y0, dy0, x1, dx1, y1, dy1, q0, q1, reso)
    thetaInit = dUtil.asTheta(qPix, xPix, yPix, dxPix, dyPix )
    x = np.hstack( [x0, x1] )
    dx = np.hstack( [dx0, dx1] )
    y = np.hstack( [y0, y1] )
    dy = np.hstack( [dy0, dy1] )
    q = np.hstack( [q0, q1] )
    xyInfSup = dUtil.padToXYInfSup( x, y, dx, dy)
    # 
    pixTheta = dUtil.asTheta( qPix, xPix, yPix, dxPix, dyPix )
    # 
    pixTheta = EMPoisson( xyInfSup, q, pixTheta, chId, 100 ) 
    ( qLocMax, xLocMax, yLocMax, dxLocMax, dyLocMax ) = dUtil.thetaAsWMuVar( pixTheta )
    
    # ??? idxLocMax, w, locX, locY  = geom.trackLocalMax( xyDxyPix, qLocMax)
    locMax = trackLocalMax( grid, locMax )
    
    
    if (0):

      # Mke groups of pixels
      nGrp, pixToGrp = geom.getConnexComponents(  xLocMax, yLocMax, dxLocMax, dyLocMax )
      
      grpCharge = np.zeros( nGrp + 1)
      locX = np.zeros( nGrp + 1)
      locY = np.zeros( nGrp + 1)
      for g in range(1,nGrp+1):
        idx = np.where( pixToGrp == g )[0]
        grpCharge[g] = np.sum( qLocMax[idx] )
        locX[g] = np.sum( xLocMax[idx] * qLocMax[idx] )/ grpCharge[g]
        locY[g] = np.sum( yLocMax[idx] * qLocMax[idx] ) / grpCharge[g]
  
      print( "grpCharge", grpCharge)
      w = grpCharge / np.sum( grpCharge ) 
      print( "w", w)
      print("xLocMax", locX)
    else:
      xyDxyPix = dUtil.asXYdXY( xLocMax, yLocMax, dxLocMax, dyLocMax )
      print("??? atk1: max qLocMax", np.max( qLocMax ) ) 

      idxLocMax, w, locX, locY  = geom.simpleLaplacian2D( xyDxyPix, qLocMax)
      print("??? atk2: max qLocMax", np.max( qLocMax ) ) 
      print("idx loc. max size", len(idxLocMax), "/", xLocMax.size)
      qPix = qLocMax[idxLocMax] 
      xPix = xLocMax[idxLocMax]
      yPix = yLocMax[idxLocMax]
      dxPix = dxLocMax[idxLocMax]
      dyPix = dyLocMax[idxLocMax]
      pixTheta = dUtil.asTheta( qPix, xPix, yPix, dxPix, dyPix )
      #
      pixTheta = EMPoisson( xyInfSup, q, pixTheta, chId, 100 ) 
      visuPix = np.copy( pixTheta )

      qPix, xPix, yPix, dxPix, dyPix = dUtil.thetaAsWMuVar( pixTheta )
      xyDxyPix = dUtil.asXYdXY( xPix, yPix, dxPix, dyPix )
      idxLocMax, w, locX, locY  = geom.simpleLaplacian2D( xyDxyPix, qPix)
      # _, _, _, _  = geom.simpleLaplacian2D( xyDxyPix, qPix)
    # Select local Max
    idx = np.where( w > 0.02 )[0]
    w = w[idx]
    locX = locX[ idx ]
    locY = locY[ idx ]
    dxLocMax = dxLocMax[idx]
    dyLocMax = dyLocMax[idx]
    refinedTheta = dUtil.asTheta( w, locX, locY, dxLocMax, dyLocMax)

    return ( refinedTheta, thetaInit, visuPix )