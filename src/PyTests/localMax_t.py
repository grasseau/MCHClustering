#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

import C.PyCWrapper as PCWrap
import Util.plot as uPlt
import utilitiesForTests as tUtil

if __name__ == "__main__":
    
    pcWrap = PCWrap.setupPyCWrapper()
    pcWrap.initMathieson()
    
 
    chId = 2
    #
    # Build the pads
    #
    x0, y0, dx0, dy0 = tUtil.buildPads( 16, 8, -2.0, 2.0, -2.0, 2.0 )
    x1, y1, dx1, dy1 = tUtil.buildPads( 8, 16, -2.0, 2.0, -2.0, 2.0 )
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
    muX = np.array( [0.5, -0.5] )
    muY = np.array( [0.5, -0.5] )
    cstVar = 0.1 * 0.1
    varX = np.array( [ cstVar, cstVar] )
    varY = np.array( [ cstVar, cstVar] )
    theta = tUtil.asTheta( w, muX, muY, varX, varY)
    xyInfSup0 = tUtil.padToXYInfSup( x0, y0, dx0, dy0)
    z0 = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup0, theta, chId )
    xyInfSup1 = tUtil.padToXYInfSup( x1, y1, dx1, dy1)
    z1 = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup1, theta, chId )
    print("sum z0 MMixture", np.sum(z0))
    print("sum z1 MMixture", np.sum(z1))
    
    (x0r, y0r, dx0r, dy0r, cath0r, z0r) = tUtil.removePads( x0, y0, dx0, dy0, cath0, z0, ( z0 < 0.05) )
    (x1r, y1r, dx1r, dy1r, cath1r, z1r) = tUtil.removePads( x1, y1, dx1, dy1, cath1, z1, ( z1 < 0.05) )
    print("sum z0 filtered MMixture", np.sum(z0r))
    print("sum z1 filtered MMixture", np.sum(z1r))
    #
  
    # Projection on one plane
    (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.projectChargeOnOnePlane(
                        x0r, dx0r, y0r, dy0r, x1r, dx1r, y1r, dy1r, z0r, z1r)
    # Laplacian/seeds
    xyDxyProj = tUtil.padToXYdXY( xProj, yProj, dxProj, dyProj)
    zProj = (chA + chB)*0.5
    N = xProj.size
    laplacian = np.zeros(N)
    theta = PCWrap.findLocalMaxWithLaplacian( xyDxyProj, zProj, laplacian)
    K = int( theta.size/5 )
    print("xyDxyProj ???", xyDxyProj)
    print("zProj", zProj)
    print("laplacian", laplacian)
    print( "theta", theta)
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    uPlt.setLUTScale( 0, np.max( zProj) )
    uPlt.drawPads( fig, ax[0,0], xProj, yProj, dxProj, dyProj, zProj,  title="Charge Projection" )
    uPlt.setLUTScale( 0, np.max( laplacian) )
    uPlt.drawPads( fig, ax[0,1], xProj, yProj, dxProj, dyProj, laplacian,  title="Laplacian" )    
    fig.suptitle( "Pad-Projection Laplacian")
    plt.show()
    
    # free memory in Pad-Processing
    PCWrap.freeMemoryPadProcessing()
    