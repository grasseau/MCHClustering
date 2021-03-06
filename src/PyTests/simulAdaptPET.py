#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

import C.PyCWrapper as PCWrap
import Util.plot as uPlt
import Util.dataTools as tUtil

verbose = 1
# mode : cstVar (1bit)
mode = 1 
LConv = 1.0e-6
  
def EMPoisson( xyInfSup, qPad, theta, chId, nIt):
  (q, mux, muy, varx, vary) = tUtil.thetaAsWMuVar( theta)
  qPix = np.copy( q )
  nPix = qPix.size
  nPads = xyInfSup.size // 4
  nPixMax = nPads
  

  # Estimate Pad Charge (f_i)
  qPredPad = np.zeros( nPads )
  for it in range(nIt):
    # if (it >= 20):
    #  nPixMax = (nPads + 1) // 3
    if ( it % 1) == 0:
      #
      # cleaning
      # idx = np.where (qPix > 1.e-02 )[0]
      # idx = np.where (qPix > 5 )[0]
      qPixMax = np.max( qPix )
      qPixCut = min( 0.01 * np.max( qPix ), 0.2 )
      idx = np.where (qPix > qPixCut )[0]
      qPix = qPix[idx] 
      mux = mux[idx]
      muy = muy[idx]
      # qPix.size must be  less than nPixMax
      if ( qPix.size > nPixMax):
        print( "qPix.size > nPixMax=nPads", qPix.size, " > ", nPixMax)
        idx = np.argsort(-qPix)
        qPix = qPix[idx[0:nPixMax]]
        mux = mux[idx[0:nPixMax]]
        muy = muy[idx[0:nPixMax]]
      nPix = qPix.size
      prevQPix = np.copy( qPix )
    # Compute coef cij, i for pad, j for pixel
    if ( it % 1) == 0:
      theta = tUtil.asTheta(qPix, mux, muy, varx[0:nPix], vary[0:nPix] )
      Cij = PCWrap.computeCij( xyInfSup, theta, chId )
      
      # print("Cij", Cij.shape)
      # print(Cij)
    for i in range(nPads):
      qPredPad[i] = np.dot( Cij[ i, 0:nPix], qPix[0:nPix] )  #  Sum_j cij.qj
    
    # Update qPix (ie w)
    for j in range(nPix):
      # Normalization term
      norm = np.sum( Cij[:,j])
      # 
      r = np.sum( Cij[:,j]*qPad[0:nPads] / qPredPad[0:nPads] )
      qPix[j] = r * qPix[j] / norm
    #
    residu = np.abs( prevQPix - qPix )
    padResidu = np.abs( qPad - qPredPad )
    print("it=", it, ", <pixResidu>", np.sum(residu)/qPix.size, ", <padResidu>", np.sum(padResidu)/qPad.size, ", max(padResidu)", np.max(padResidu))
  print( "Total Charge", np.sum(qPad))
  print( "Total Predictited Charge", np.sum(qPredPad))
  print( "Total Pixel Charge", np.sum(qPix))
  print( "Total Cij", np.sum(Cij))
  print( "Total Ci", np.sum(Cij, axis=0))
  print( "Total Cj", np.sum(Cij, axis=1))
  
  # SVD
  # (u, s, vh) = np.linalg.svd( Cij )
  # print("s", s )
  theta = tUtil.asTheta(qPix, mux, muy, varx[0:nPix], vary[0:nPix] )

  return theta

def compactTheta( theta0, thetaMask):
  K0 = theta0.size // 5
  K = np.sum( thetaMask)
  w = np.zeros( K )
  muX = np.zeros( K )
  muY = np.zeros( K )
  varX = np.zeros( K )
  varY = np.zeros( K )
  theta = np.zeros( K*5)
  k = 0
  for k0 in range(K0):
    if thetaMask[k0]:
      w[k]    = theta0[4*K0+k0]
      muX[k]  = theta0[2*K0+k0]
      muY[k]  = theta0[3*K0+k0]
      varX[k] = theta0[k0]
      varY[k] = theta0[K0+k0]
      k+=1;
  theta[0*K:1*K] = varX
  theta[1*K:2*K] = varY
  theta[2*K:3*K] = muX
  theta[3*K:4*K] = muY
  theta[4*K:5*K] = w
  #
  return theta

def fuseThetaByW( theta, thetaMask, xyInfSup):
  precision = 1.e-4
  (w, mux, muy, varx, vary) = tUtil.thetaAsWMuVar( theta )
  (xInf, yInf, xSup, ySup )= tUtil.ungroupXYInfSup( xyInfSup )
  K = w.size
  N = xInf.size
  nMask = np.sum(thetaMask)
  wMax = np.max( w )
  wMin = np.min( w )
  # 
  print(" wMin", wMin, ", wMax", wMax)
  # absolute wCutOff
  # relative CutOff
  for k in range(K):
    if (thetaMask[k]):
      if ( w[k] < 0.1*wMax):
        w[k] = 0
        thetaMask[k] = 0

  theta = tUtil.asTheta(w, mux, muy, varx, vary)
  print("Fused theta old/new", nMask, np.sum(thetaMask) ) 
  return (theta, thetaMask)

def fuseThetaByMask( theta, thetaMask, xyInfSup):
  precision = 1.e-4
  (w, mux, muy, varx, vary) = tUtil.thetaAsWMuVar( theta )
  (xInf, yInf, xSup, ySup )= tUtil.ungroupXYInfSup( xyInfSup )
  K = w.size
  N = xInf.size
  nMask = np.sum(thetaMask)
  print("  mux", mux, ", muy", muy)
  print("  xInf ", xInf, ", xSup ", xSup)
  print("  yInf ", yInf, ", ySup ", ySup)
  # Last k in the boxes
  occupiedPad = -1 * np.ones( N, dtype=np.int16)
  for k in range(K):
    if (thetaMask[k]):
      inBoxX = np.bitwise_and( mux[k] >= (xInf - precision), mux[k] < (xSup + precision) )
      inBoxY = np.bitwise_and( muy[k] >= (yInf - precision), muy[k] < (ySup + precision) )
      inBox = np.bitwise_and(inBoxX, inBoxY)
      idx = np.where( inBox )[0]
      # Index of pad i where theta[k] belong
      i = -1
      if (idx.size > 1):
        print(" mu", k, mux[k], muy[k])
        print("idx", idx)
        print("xInf", xInf[idx])
        print("xSup", xSup[idx])
        print("yInf", yInf[idx])
        print("ySup", ySup[idx])
        input("mu[k] i several boxes match")
      elif (idx.size == 1):
        i = idx[0]
      # Test if one and only one k in boxes  
      if i != -1:
        if (occupiedPad[i] != -1):
          # Fuse Theta
          prevK = occupiedPad[i]

          mux[prevK] = 1.0/(w[prevK] + w[k]) *(mux[prevK]*w[prevK] + mux[k]*w[k])
          muy[prevK] = 1.0/(w[prevK] + w[k]) *(muy[prevK]*w[prevK] + muy[k]*w[k])
          w[prevK] += w[k]
          thetaMask[k] = 0
          w[k] = 0
          mux[k] = 0
          muy[k] = 0
        #
        else:
          occupiedPad[i] = k;
  theta = tUtil.asTheta(w, mux, muy, varx, vary)
  print("Fused theta old/new", nMask, np.sum(thetaMask) ) 
  return (theta, thetaMask)

def spacialFilterTheta( theta, dx, dy ):
  K = theta.size // 5
  if K == 0 : return
  ( w, muX, muY, varX, varY) = tUtil.thetaAsWMuVar( theta )
  newW = []
  newMuX = []
  newMuY = []
  newVarX = []
  newVarY = []
  done = np.zeros( K )
  for k0 in range(0,K):
    if not done[k0] :
      muX0 = muX[k0]
      muY0 = muY[k0]
      # for k range(k0+1,K):
      k = k0+1
      inX = np.abs( muX0 - muX[k:])  < dx
      inY = np.abs( muY0 - muY[k:])  < dy
      inXY = inX * inY
      if np.sum( inXY > 0):
        newMuX.append( muX0 )
        newMuY.append( muY0 )
        newVarX.append( varX[k0] )
        newVarY.append( varY[k0] )
        idx = np.where( inXY == 1)[0] + k
        done[idx] = 1
        # newIdx.append( idx ))
        newW.append( np.sum(w[idx]) + w[k0])
  print( newW )
  print( newMuX)
  
  w = np.array( newW )
  muX = np.array( newMuX )
  muY = np.array( newMuY )
  varX = np.array( newVarX )
  varY = np.array( newVarY )
  theta = tUtil.asTheta( w, muX, muY, varX, varY )
  return theta

def getPoint( pad0, pad1):
  epsilon = 10.0e-5
  (x0, y0, dx0, dy0) = pad0
  (x1, y1, dx1, dy1) = pad1

  x0Inf = x0 - dx0
  x0Sup = x0 + dx0
  y0Inf = y0 - dy0
  y0Sup = y0 + dy0
  #
  x1Inf = x1 - dx1
  x1Sup = x1 + dx1
  y1Inf = y1 - dy1
  y1Sup = y1 + dy1  
  
  xInf = max( x0Inf,x1Inf)
  xSup = min( x0Sup, x1Sup)
  yInf = max( y0Inf, y1Inf)
  ySup = min( y0Sup, y1Sup) 
  
  return ( 0.5*(xInf+xSup), 0.5*(yInf+ySup) )

def intersectionArea( xrInf,  xrSup,  yrInf,  yrSup, xInf,  xSup,  yInf,  ySup, z ):
  epsilon = 10.0e-5
  area = np.zeros( x.size )
  
  for j in range(xInf.shape[0]):
    xmin = max( xrInf, xInf[j] )
    xmax = min( xrSup, xSup[j] )
    xInter = ( xmin <= (xmax - epsilon) )
    ymin = max( yrInf, yInf[j] )
    ymax = min( yrSup, ySup[j] )
    yInter = ( ymin <= (ymax - epsilon))
    # intersection
    if xInter and yInter:
      area[j] = (xmax-xmin) * (ymax-ymin) / ((xSup[i] - xInf[j]) * (ySup[i] - yInf[j]) ) * z[i]
        
  return area

def intersection( refPad, pads):
  epsilon = 10.0e-5
  (xr, yr, dxr, dyr) = refPad
  (x, y, dx, dy, z) = pads
  print("???", x.size)
  mask = np.zeros( x.size )
  xrInf = xr - dxr
  xrSup = xr + dxr
  yrInf = yr - dyr
  yrSup = yr + dyr
  xInf = x - dx
  xSup = x + dx
  yInf = y - dy
  ySup = y + dy
  
  for j in range(xInf.shape[0]):
    xmin = max( xrInf, xInf[j] )
    xmax = min( xrSup, xSup[j] )
    xInter = ( xmin <= (xmax - epsilon) )
    ymin = max( yrInf, yInf[j] )
    ymax = min( yrSup, ySup[j] )
    yInter = ( ymin <= (ymax - epsilon))
    # intersection
    mask[j] = xInter and yInter
  
  return mask

def runAdaptEM( event ):
  # 
  x0, y0, dx0, dy0, cath0, saturated0, z0 = event.padCath0
  x1, y1, dx1, dy1, cath1, saturated1, z1 = event.padCath1
  print(x0.size, cath0.size, z0.size, saturated0.size)
  print(x1.size, cath1.size, z1.size, saturated1.size)
  thetai = event.theta
  ( xyDxy, cath, saturated, z ) = event.getMergedPads()  
  # Buil the projection
  (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.projectChargeOnOnePlane(
                        x0, dx0, y0, dy0, x1, dx1, y1, dy1, z0, z1)
  x = np.hstack( [x0, x1] )
  dx = np.hstack( [dx0, dx1] )
  y = np.hstack( [y0, y1] )
  dy = np.hstack( [dy0, dy1] )
  xyInfSup = tUtil.padToXYInfSup( x, y, dx, dy)
  xyInfSupProj = tUtil.padToXYInfSup( xProj, yProj, dxProj, dyProj )
  # theta init
  N = xProj.size
  w = (chA + chB) * 0.5
  norm =  1.0/ np.sum( w )
  w = w * norm
  var = np.ones (N) * 0.1

  """
  theta = tUtil.asTheta( w, xProj, yProj, var, var )
  thetaMask = np.ones( N, dtype = np.int16)
  """ 
  #
  # Set the solution
  """
  theta = np.copy(thetai)
  (w, u, v, uu, vv) = tUtil.thetaAsWMuVar( theta )
  # w[0] = 0.2
  # w[1] = 0.8
  u[0] = u[1]
  v[0] = v[1]
  theta = tUtil.asTheta(w, u, v, uu, vv)
  thetaMask = np.ones( theta.size // 5, dtype = np.int16)
  """
  #
  # Adapt theta
  saturatedProj = np.zeros( chA.size,dtype=np.int16)
  kMax = ((xyDxy.size // 4) + 1 ) // 3
  print("kMax --->", kMax)
  thetaMoved = 1
  # Compact theta if required
  # Mathieson mixture with new theta
  # zTheta = compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, chId )
  # chTheta0 = zTheta[ cath==0 ] 
  # chTheta1 = zTheta[ cath==1 ] 
  # Charge Projection
  (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.projectChargeOnOnePlane( x0, dx0, y0, dy0, 
              x1, dx1, y1, dy1, z0, z1)
  zProj = 0.5*(chA+chB)
  print("Proj size", xProj.size)
  nPix = xProj.size
  QTotal = np.sum( z )
  c = QTotal / nPix
  # qPix = np.ones( nPix ) * c
  qPix = zProj
  var = np.ones( nPix )
  xPix = xProj
  yPix = yProj
  dxPix = dxProj
  dyPix = dyProj
  
  theta = tUtil.asTheta( qPix, xPix, yPix, dxPix, dyPix) 
  converged = False
  xyDxyProj = tUtil.padToXYdXY( xProj, yProj, dxProj, dyProj)
  EMConverge = False
  while (not converged ):
    tUtil.printTheta( "theta", theta)
    """
    print("Theta-Projection")
    print("  xProj", xProj, "yProj", yProj)
    print("  dxProj", dxProj, "dyProj", dyProj)
    print("  zProj", zProj )
    print("  thetaMask", thetaMask )
    """
    # MLEM loop until thetaChanged and not converged
    # Int zProj to 1 ???
    print("theta", )
    
    
    newTheta = EMPoisson( xyInfSup, z, theta, chId, 20 ) 
    # newTheta = tUtil.asTheta( qPix, xPix, yPix, dxPix, dyPix) 
    (qPix, _, _, _, _) = tUtil.thetaAsWMuVar( newTheta)
    # xMax, yMax = tUtil.maxXYDistance( theta, thetaMask, newTheta)
    (w, _, _, _, _)    = tUtil.thetaAsWMuVar( theta)
    (wNew, _, _, _, _) = tUtil.thetaAsWMuVar( newTheta)
    print(w)
    print(wNew)
    # wDif = np.max(np.abs( wNew - w))
    # print("wDif", wDif)
    # if wDif < 1.e-00: 
    #  EMConverge = True
    EMConverge = False
    #print( "theta.size / thetaMask sum, x/yMax", newTheta.size // 5, np.sum( thetaMask), xMax, yMax )
    tUtil.printTheta( "new theta", newTheta)
    if ( EMConverge ):
      # refine pixels
      #
      # Filter 
      (wNew, xNew, yNew, dxNew, dyNew) = tUtil.thetaAsWMuVar( newTheta)
      idx = np.argsort(-wNew)
      print("idx", idx)
      idx = idx[0:kMax]
      nPix = kMax
      qPix = qPix[idx]
      print("filtere qPix", qPix) 
      xPix = xPix[idx]
      yPix = yPix[idx]
      dxPix = dxPix[idx]
      dyPix = dyPix[idx]      
      #
      """
          print("##### Refinement ##########")
          newXPix = np.zeros( nPix*4 )
          newYPix = np.zeros( nPix*4 )
          newdXPix = np.zeros( nPix*4 )
          newdYPix = np.zeros( nPix*4 )
          newXPix = np.zeros( nPix*4 )
          newQPix = np.zeros( nPix*4 )
          for p in range(nPix):
            newXPix[4*p] = xPix[p] - dxPix[p]/2
            newYPix[4*p] = yPix[p] - dyPix[p]/2
            newXPix[4*p+1] = xPix[p] + dxPix[p]/2
            newYPix[4*p+1] = yPix[p] - dyPix[p]/2   
            newXPix[4*p+2] = xPix[p] - dxPix[p]/2
            newYPix[4*p+2] = yPix[p] + dyPix[p]/2   
            newXPix[4*p+3] = xPix[p] + dxPix[p]/2
            newYPix[4*p+3] = yPix[p] + dyPix[p]/2
            newdXPix[4*p]   = dxPix[p]/2
            newdYPix[4*p]   = dyPix[p]/2
            newdXPix[4*p+1] = dxPix[p]/2
            newdYPix[4*p+1] = dyPix[p]/2   
            newdXPix[4*p+2] = dxPix[p]/2   
            newdYPix[4*p+2] = dyPix[p]/2   
            newdXPix[4*p+3] = dxPix[p]/2
            newdYPix[4*p+3] = dyPix[p]/2 
            # Charge
            newQPix[4*p]   = qPix[p]/4
            newQPix[4*p+1] = qPix[p]/4
            newQPix[4*p+2] = qPix[p]/4
            newQPix[4*p+3] = qPix[p]/4
          nPix = nPix*4
          xPix = newXPix
          yPix = newYPix
          dxPix = newdXPix
          dyPix = newdYPix
          qPix = newQPix
          newTheta = tUtil.asTheta( qPix, xPix, yPix, dxPix, dyPix)
      """  
      EMConverge = False
    answer = input("stop ?")
    if answer == 'y':
      converged = True
    else:
      converged = False
    theta=np.copy( newTheta )
    """
    # Filter theta in position (remove same pad)
    if (thetaRemoved):
      thetaChangedInPosition = True
    # Filter theta in magnitude
    if (thetaRemoved):
      thetaChangedInPosition = True
    """
  # newTheta = compactTheta( newTheta, thetaMask )
  tUtil.printTheta( "Theta", thetai) 
  tUtil.printTheta( "Final theta", theta)
  return newTheta, theta 

def compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, chId ):
  # xyInfSup = tUtil.padToXYInfSup( x, y, dx, dy )
  z = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, chId )
  return z

if __name__ == "__main__":
    
    pcWrap = PCWrap.setupPyCWrapper()
    pcWrap.initMathieson()
    
 
    chId = 2
    
    # - Theta  (-1.5 < muX/Y < 1.5)
    # - Max Charge
    # - Min Charge
    K = 1
    Nx = 2
    Ny = 3
    # Nx = 2
    # Ny = 2    
    minCh = 5.0
    maxCh = 600.0
    readObj = True
    if ( not readObj ):
      simul = tUtil.SimulCluster( Nx, Ny )
      # Build the pads
      #
      ( padCath0, padCath1, thetai) = simul.buildCluster( chId, K, minCh, maxCh, 1.1 * maxCh)
      ( padCath0, padCath1, thetai) = simul.buildCluster( chId, K, minCh, maxCh, 0.9 * maxCh)

    else:
      simul = tUtil.SimulCluster.read()
    #
    x0, y0, dx0, dy0, cath0, saturated0, z0 = simul.padCath0
    x1, y1, dx1, dy1, cath1, saturated1, z1 = simul.padCath1
    thetai = simul.theta
    ( xyDxy, cath, saturated, z ) = simul.getMergedPads()  
    simul.write()
    #
    nSaturated = np.sum(saturated)
    print("# cath0", cath0.size, "# cath1", cath1.size)
    print("# nSaturated", nSaturated)
    nFigRow = 2; nFigCol = 4
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(15, 7))
    for iRow in range( nFigRow):
      for iCol in range( nFigCol):
        ax[iRow,iCol].set_xlim( simul.gridXLimits[0], simul.gridXLimits[1])
        ax[iRow,iCol].set_ylim( simul.gridYLimits[0], simul.gridYLimits[1])

    uPlt.setLUTScale( 0, max( np.max(z0), np.max(z1) )  )
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  title="Mathieson  cath-0", doLimits=False )
    uPlt.drawPads( fig, ax[1,0], x1, y1, dx1, dy1, z1,  title="Mathieson  cath-1", doLimits=False )
    ax[0,0].plot( x0[saturated0==1], y0[saturated0==1], "o", color='black', markersize=3 )
    ax[1,0].plot( x1[saturated1==1], y1[saturated1==1], "o", color='black', markersize=3 )
    uPlt.drawPads( fig, ax[0,1], x0, y0, dx0, dy0, z0, doLimits=False, displayLUT=False, alpha=0.5)
    uPlt.drawPads( fig, ax[0,1], x1, y1, dx1, dy1, z1,  title="Mathieson  both cath", doLimits=False, alpha=0.5)    
    # uPlt.drawPads( fig, ax[1,0], x0r, y0r, dx0r, dy0r, z0r,  title="Mathieson  cath-0 - Removed pads" )
    # uPlt.drawPads( fig, ax[1,1], x1r, y1r, dx1r, dy1r, z1r,  title="Mathieson  cath-1 - Removed pads" )
    #
    uPlt.drawModelComponents( ax[0,1], thetai, color='black', pattern="+" ) 
    uPlt.drawModelComponents( ax[0,1], thetai, color='black', pattern="show w" ) 
    
    fig.suptitle( "Filtering pads to a Mathieson Mixture")
    
  
    # Projection on one plane
    (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.projectChargeOnOnePlane(
                        x0, dx0, y0, dy0, x1, dx1, y1, dy1, z0, z1)
    chProj = 0.5*(chA + chB)
    uPlt.setLUTScale( 0, 1.2 * np.max( chProj) )
    uPlt.drawPads( fig, ax[1,1], xProj, yProj, dxProj, dyProj, chProj,  title="Projection",  doLimits=False)
    uPlt.drawPads( fig, ax[1,2], xProj, yProj, dxProj, dyProj, chProj,  title="Projection",  doLimits=False)
    # uPlt.drawPads( fig, ax[0,3], xProj, yProj, dxProj, dyProj, chA,  title="Projection",  doLimits=False)
    # uPlt.drawPads( fig, ax[1,3], xProj, yProj, dxProj, dyProj, chB,  title="Projection",  doLimits=False)
    newTheta, fusedTheta = runAdaptEM( simul )
    # 
    uPlt.drawModelComponents( ax[1,1], newTheta, color='black', pattern="+" ) 
    uPlt.drawModelComponents( ax[1,1], newTheta, color='black', pattern="show w" ) 

    uPlt.drawModelComponents( ax[1,2], fusedTheta, color='black', pattern="+" ) 
    uPlt.drawModelComponents( ax[1,2], fusedTheta, color='black', pattern="show w" ) 
    
    (x, y, dx, dy) = tUtil.asXYdXdY( xyDxy )
    residual = PCWrap.computeResidual( xyDxy, z, newTheta )
    # Residual
    print("???", np.max(residual), residual)
    uPlt.setLUTScale( np.min(residual) , np.max(residual)  )
    uPlt.drawPads( fig, ax[0,2], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], residual[cath==0], doLimits=False, displayLUT=False, alpha=0.5)
    uPlt.drawPads( fig, ax[0,2], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], residual[cath==1], title="Residual", doLimits=False, alpha=0.5 ) 
    # Process
    # Connected-Components
    #
    """
    nbrGroups, padGrp = PCWrap.getConnectedComponentsOfProjPads()
    #
    print( "# of groups, pad group ", nbrGroups, padGrp, np.max(padGrp) )
    uPlt.setLUTScale( 0, np.max(padGrp)  )
    uPlt.drawPads( fig, ax[0,2], xProj, yProj, dxProj, dyProj, padGrp, alpha=1.0, title="Pad group", doLimits=False ) 
    # Process
    nbrHits = PCWrap.clusterProcess( xyDxy, cath, saturated, z, chId )
    (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
    # EM
    thetaEMFinal = PCWrap.collectThetaEMFinal()
    uPlt.drawModelComponents( ax[1,1], thetaEMFinal, color="gray", pattern='o')
    uPlt.drawModelComponents( ax[1,1], thetaResult, color='black', pattern="show w" )     
    ( w, muX, muY, _, _) = tUtil.thetaAsWMuVar( thetai)
    # print("theta0 w", w)
    # print("       muX", muX)
    # print("       muY", muY)
    ( w, muX, muY, _, _) = tUtil.thetaAsWMuVar( thetaResult)
    # print("thetaResult w", w)
    # print("            muX", muX)
    # print("            muY", muY)
    # Residual
    residual = PCWrap.collectResidual( )
    maxResidual = np.max( residual )
    minResidual = np.min( residual )  
    uPlt.setLUTScale( minResidual, maxResidual  )
    print("mi/mx residual", minResidual, maxResidual)
    print("residual.size, xProj.size", residual.size, xProj.size)
    if residual.size == xProj.size :
      uPlt.drawPads( fig, ax[1,2], xProj, yProj, dxProj, dyProj, residual, alpha=1.0, title="Residual", doLimits=False ) 
    # Residual Mathieson
    nGroup = np.max( padGrp )
    zMat = np.zeros( chProj.size )
    for g in range(nGroup):
        idx = np.where( padGrp == (g+1) )[0]
        sumProj = np.sum( chProj[idx])
        thetaIdx = np.where( thetaToGrp == (g+1) )
        (w, muX, muY, varX, varY) = tUtil.thetaAsWMuVar( thetaResult )
        wg = w[thetaIdx]
        muXg = muX[thetaIdx]
        muYg = muY[thetaIdx]
        varXg = varX[thetaIdx]
        varYg = varY[thetaIdx]
        thetaGrp = tUtil.asTheta(  wg, muXg, muYg, varXg, varYg )
        
        zg = sumProj * compute2DMathiesonMixturePadIntegrals( xProj[idx], yProj[idx], dxProj[idx], dyProj[idx], thetaGrp, chId)
        zMat[idx] = zg
    residual = chProj - zMat
    maxResidual = np.max( residual )
    minResidual = np.min( residual )
    uPlt.setLUTScale( minResidual, maxResidual  )
    uPlt.drawPads( fig, ax[1,3], xProj, yProj, dxProj, dyProj, residual, alpha=1.0, title="Residual", doLimits=False )
    # 
    # Generator
    #
    verbose = 0
    # mode : cstVar (1bit)
    mode = 1 
    LConv = 1.0e-6
    N = xProj.size
    x = np.hstack( [x0, x1])
    y = np.hstack( [y0, y1])
    dx = np.hstack( [dx0, dx1])
    dy = np.hstack( [dy0, dy1])
    z = np.hstack( [z0, z1])
    
    thetaT = seedGenerator( x, y, dx, dy, z, chId)
    # xyDxy = tUtil.asXYdXY ( xProj, yProj, dxProj, dyProj)
    # def weightedEMLoop( xyDxy, saturated, zObs, thetai, thetaMask, mode, LConvergence, verbose):
    saturated = np.zeros( N, dtype=np.int16)
    thetaMask = np.ones( N, dtype=np.int16 )
    theta, logL = PCWrap.weightedEMLoop( xyDxy, saturated, z, thetaT, thetaMask, mode, LConv, verbose ) 
    uPlt.drawModelComponents( ax[1,3], theta, color="gray", pattern='rect')

    # Filter
    thetaFilter = spacialFilterTheta( theta, dxProj[0], dyProj[0] )
    thetaMask = np.ones( thetaFilter.size // 5, dtype=np.int16 )
    # theta, logL = PCWrap.weightedEMLoop( xyDxy, saturated, chProj, thetaFilter, thetaMask, mode, LConv, verbose )     
    print("filtered theta", theta)
    uPlt.drawModelComponents( ax[1,3], theta, color="black", pattern='x')
    ( w, muX, muY, _, _) = tUtil.thetaAsWMuVar( thetai)
    print("theta0 w", w)
    print("       muX", muX)
    print("       muY", muY)
    ( w, muX, muY, _, _) = tUtil.thetaAsWMuVar( theta)
    print("thetaResult w", w)
    print("            muX", muX)
    print("            muY", muY)
    """
    plt.show()
    
    # free memory in Pad-Processing
    PCWrap.freeMemoryPadProcessing()

     