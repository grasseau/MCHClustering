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
    muX = np.array( [0.5, -0.5] )
    muY = np.array( [0.5, -0.5] )
    #
    # To test 1 cathode plane
    x1 = np.zeros( 0 )
    y1 = np.zeros( 0 )
    dx1 = np.zeros( 0 )
    dy1 = np.zeros( 0 )
    # Filter low charge & No saturation
    ( xyDxy, cath, saturated, z ) = dUtil.buildPreCluster( w, muX, muY, 
                                                     ( x0, y0, dx0, dy0),
                                                     ( x1, y1, dx1, dy1),
                                                     chId,
                                                     0.05, 1.01 
                                                    )

    nbrHits = PCWrap.clusterProcess( xyDxy, cath, saturated, z, chId )
    (thetaResult, thetaToGrp) = PCWrap.collectTheta( nbrHits)
    print("theta ???", thetaResult)
    print("thetaGrp", thetaToGrp)
    (xyDxyResult, chResult, padToGrp) = PCWrap.collectPadsAndCharges()
    print("xyDxy ... ", xyDxyResult)
    print("charge ... ", chResult)
    print("padToGrp", padToGrp)
    
    # free memory in Pad-Processing
    PCWrap.freeMemoryPadProcessing()
    