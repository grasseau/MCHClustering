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

if __name__ == "__main__":
    
    pcWrap = PCWrap.setupPyCWrapper()
    pcWrap.initMathieson()
     
    chId = 2
    #
    # Build the pads
    #
    x0, y0, dx0, dy0 = dUtil.buildPads( 16, 8, -2.0, 2.0, -2.0, 2.0 )
    x1, y1, dx1, dy1 = dUtil.buildPads( 8, 16, -2.0, 2.0, -2.0, 2.0 )
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
    # No saturation
    ( xyDxyi, cathi, saturatedi, zi ) = dUtil.buildPreCluster( w, muX, muY, 
                                                     ( x0, y0, dx0, dy0),
                                                     ( x1, y1, dx1, dy1),
                                                     chId,
                                                     0.0, 1.01 
                                                    )
    nbrHitsi = PCWrap.clusterProcess( xyDxyi, cathi, saturatedi, zi, chId )
    (thetaResulti, thetaToGrpi) = PCWrap.collectTheta( nbrHitsi)
    #
    # With saturation
    ( xyDxy, cath, saturated, z ) = dUtil.buildPreCluster( w, muX, muY, 
                                                     ( x0, y0, dx0, dy0),
                                                     ( x1, y1, dx1, dy1),
                                                     chId,
                                                     0.05, 0.95 
                                                    )
    
    nbrHits = PCWrap.clusterProcess( xyDxy, cath, saturated, z, chId )
    (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
    print("thetaResult ???", thetaResult)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    ( xi, yi, dxi, dyi) = dUtil.asXYdXdY( xyDxyi )
    ( x, y, dx, dy) = dUtil.asXYdXdY( xyDxy )
    uPlt.setLUTScale( 0, np.max(zi) )
    uPlt.drawPads( fig, ax[0,0], xi[cathi==0], yi[cathi==0], dxi[cathi==0], dyi[cathi==0], zi[cathi==0], title="Mathieson  cath-0" )
    uPlt.drawPads( fig, ax[0,1], xi[cathi==1], yi[cathi==1], dxi[cathi==1], dyi[cathi==1], zi[cathi==1], title="Mathieson  cath-1" )
    uPlt.drawPads( fig, ax[1,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], z[cath==0], title="Mathieson & Cutoff  cath-0" )
    uPlt.drawPads( fig, ax[1,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], z[cath==1], title="Mathieson & Cutoff cath-1" )
    #
    uPlt.drawModelComponents( ax[0,0], thetaResulti, color="blue", pattern="cross")
    uPlt.drawModelComponents( ax[0,1], thetaResulti, color="blue", pattern="cross")
    uPlt.drawModelComponents( ax[1,0], thetaResult, color="black", pattern="o")
    uPlt.drawModelComponents( ax[1,1], thetaResult, color="black", pattern="o")
    fig.suptitle( "Filtering pads to a Mathieson Mixture")
    plt.show()
    
    # free memory in Pad-Processing
    PCWrap.freeMemoryPadProcessing()
    