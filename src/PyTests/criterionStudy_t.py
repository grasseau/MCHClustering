#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

import C.PyCWrapper as PCWrap
import Util.plot as uPlt
import Util.dataTools as dUtil

chId = 2
verbose = 2
# mode : cstVar (1bit)
mode = 1 
LConv = 1.0e-6

if __name__ == "__main__":
        
    pcWrap = PCWrap.setupPyCWrapper()
    pcWrap.initMathieson()
    
    #
    # Build the pads
    #
    x0, y0, dx0, dy0 = dUtil.buildPads( 12, 6, -2.0, 2.0, -2.0, 2.0 )
    x1, y1, dx1, dy1 = dUtil.buildPads( 6, 12, -2.0, 2.0, -2.0, 2.0 )
    # Merge pads
    # N0 = x0.size
    # (x, y, dx, dy) = tUtil.mergePads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 )
    cath0 = np.zeros( x0.size, dtype=np.int32 )
    cath1 = np.ones( x1.size, dtype=np.int32 )
    #
    # Mathieson mixture
    #
    K = 2
    w   = np.array( [0.6, 0.4 ] )
    muX = np.array( [0.4, -0.4] )
    muY = np.array( [0.4, -0.4] )
    varX = np.array( [0.1, 0.1] )
    varY = np.array( [0.1, 0.1] )
    K = 4
    w   = np.array( [0.2, 0.4, 0.1, 0.3 ] )
    muX = np.array( [0.4, 0.4, -0.4, -0.4] )
    muY = np.array( [0.4, -0.4, 0.4, -0.4] )
    varX = np.array( [0.1, 0.1, 0.1, 0.1] )
    varY = np.array( [0.1, 0.1, 0.1, 0.1] )
    """
    K = 1
    w   = np.array( [1.0 ] )
    muX = np.array( [0.4] )
    muY = np.array( [0.4] )
    varX = np.array( [0.1] )
    varY = np.array( [0.1] )
    """
    # Filter low charge & No saturation
    ( xyDxy, cath, saturated, z ) = dUtil.buildPreCluster( w, muX, muY, 
                                                     ( x0, y0, dx0, dy0),
                                                     ( x1, y1, dx1, dy1),
                                                     chId,
                                                     0.05, 1.01 )
    print("size ", xyDxy.size, cath.size, saturated.size, z.size )
    # Projection                                                
    (x, y, dx, dy) = dUtil.asXYdXdY( xyDxy)
    (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.projectChargeOnOnePlane(
                        x[cath==0], dx[cath==0], y[cath==0], dy[cath==0], 
                        x[cath==1], dx[cath==1], y[cath==1], dy[cath==1], 
                        z[cath==0], z[cath==1]) 
    print("size proj", xProj.size, yProj.size, chA.size)
    xyDxyProj = dUtil.asXYdXY( xProj, yProj, dxProj, dyProj )
    xyInfSupProj = dUtil.padToXYInfSup( xProj, yProj, dxProj, dyProj )
    zProj = (chA + chB) * 0.5
    print("size proj", xProj.size, xyDxyProj.size, zProj.size)
    # saturated = np.zeros( (z.size), dtype=np.int16 )
    thetai = dUtil.asTheta( w, muX, muY, varX, varY)
    thetaMask = np.ones( (thetai.size // 5), dtype=np.int16 )


    # nbrHits = PCWrap.clusterProcess( xyDxy, cath, saturated, z, chId )
    theta, logL = PCWrap.weightedEMLoop( xyDxyProj, saturated, zProj, thetai, thetaMask, mode, LConv, verbose ) 
    # (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
    print("theta ???", theta)
    print("logL ???", logL)
    print("saturated", np.sum(saturated))
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    maxZ = np.max(z)
    xMin = np.min(x-dx)
    xMax = np.max(x+dx)
    yMin = np.min(y-dy)
    yMax = np.max(y+dy)
    uPlt.setLUTScale( 0, maxZ )
    (xr, yr, dxr, dyr) = dUtil.asXYdXdY( xyDxyProj)
    zDummy = np.ones( (x0.size))* maxZ
    uPlt.drawPads( fig, ax[0,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], z[cath==0], doLimits=False, alpha=1.0 )
    uPlt.drawPads( fig, ax[0,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], z[cath==1], doLimits=False, alpha=1.0 )
    uPlt.drawPads( fig, ax[1,0], xr, yr, dxr, dyr, zProj, doLimits=False, alpha=1.0 )
    uPlt.drawPads( fig, ax[1,1], xr, yr, dxr, dyr, zProj, doLimits=False, alpha=1.0 )
    for i in range(2):
      for j in range(2):
        ax[i,j].set_xlim( xMin, xMax )
        ax[i,j].set_ylim( yMin, yMax )

    plt.show()
    print ("??? sort", np.argsort( w ))
    incOrder = np.argsort( w )
    logL = np.zeros(K+1)
    RSS = np.zeros(K+1)
    crossE = np.zeros(K+1)
    for k in range(0,K):
      thetaMask = np.ones( K, dtype=np.int16 )
      thetaMask[incOrder[0:k]] = 0
      print("thetaMask", thetaMask )
      theta, lL = PCWrap.weightedEMLoop( xyDxyProj, saturated, zProj, thetai, thetaMask, mode, LConv, verbose ) 
      logL[K-k-1] = lL
      # residual = PCWrap.collectResidual()
      print("theta ???", theta)
      residual = PCWrap.computeResidual( xyDxyProj, zProj, theta)
      zPred = PCWrap.generateMixedGaussians2D( xyInfSupProj, theta) 
      print ("theta", theta)
      print("zPred", zPred)
      input( "???")
      crossE[K-k-1] = - np.sum( zProj * np.log( zPred )) 
      print( residual, RSS[K-k-1])
      RSS[K-k-1] = np.log( np.sum( residual * residual) ) * residual.size
      print( "residual", residual, RSS[K-k-1])
      print( "residual", np.log( np.sum( residual * residual)) )
      print( "residual", RSS[K-k-1] )
      # input("???")
      
    K = 5
    w   = np.array( [0.2, 0.35, 0.1, 0.25, 0.1 ] )
    muX = np.array( [0.4, 0.4, -0.4, -0.4, 0.0] )
    muY = np.array( [0.4, -0.4, 0.4, -0.4, 0.0] )
    varX = np.array( [0.1, 0.1, 0.1, 0.1, 0.1] )
    varY = np.array( [0.1, 0.1, 0.1, 0.1, 0.1] )
    thetai = dUtil.asTheta( w, muX, muY, varX, varY)
    thetaMask = np.ones( K, dtype=np.int16 )

    theta, lL = PCWrap.weightedEMLoop( xyDxyProj, saturated, zProj, thetai, thetaMask, mode, LConv, verbose ) 
    logL[K-1] = lL
    residual = PCWrap.computeResidual( xyDxyProj, zProj, theta)
    RSS[K-1] = np.log( np.sum( residual * residual) ) * residual.size
    zPred = PCWrap.generateMixedGaussians2D( xyInfSupProj, theta) 
    crossE[K-1] = - np.sum( zProj * np.log( zPred )) 
    #
    kVal = np.arange(K)+1
    print("kVal",kVal )
    print("logL",logL )
    dlL = logL[1:] - logL[0:K-1]
    bic1 = (kVal*3-1) * np.log( xyDxyProj.size // 4 )
    bic2 = (kVal) * np.log( xyDxy.size // 4 )
    bic2 = np.log(3*kVal-1) 
    dof = kVal*3-1
    n = zProj.size
    aic = 2*(dof) + 2*dof*(dof+1) / (n - dof -1)
    #plt.plot( kVal, bic1, color="blue")
    #plt.plot( kVal, 0.002*aic, color="green")
    # plt.plot( kVal, bic2, color="orange")
    plt.plot( kVal, -2*logL, color="pink")
    # plt.plot( kVal, -2*logL*n, color="pink")
    #plt.plot( kVal[1:], -2*dlL, color="orange")
    # plt.plot( kVal, aic, color="green")
    # plt.plot( kVal, -2*logL*n + aic, color="red")
    #plt.plot( kVal, RSS, color="orange")
    plt.plot( kVal, crossE, color="orange")
    plt.plot( kVal, np.log(kVal), color="green")
    #plt.plot( kVal, aic-RSS, color="red")
    
    # plt.plot( kVal, -2*logL+aic, color="red")
    print(1.0/64)
    plt.show()
    print("CrossE", crossE)
    print("CrossE", crossE + np.log( kVal))
    # free memory in Pad-Processing
    PCWrap.freeMemoryPadProcessing()
    