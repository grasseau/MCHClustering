# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np

import C.PyCWrapper as PCWrap
import Util.geometry as geom
import Util.dataTools as dUtil

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
