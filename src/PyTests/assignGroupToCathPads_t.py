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
    
    (x0r, y0r, dx0r, dy0r, cath0r, z0r) = tUtil.removePads( x0, y0, dx0, dy0, cath0, z0, ( z0 < 0.005) )
    (x1r, y1r, dx1r, dy1r, cath1r, z1r) = tUtil.removePads( x1, y1, dx1, dy1, cath1, z1, ( z1 < 0.005) )
    print("sum z0 filtered MMixture", np.sum(z0r))
    print("sum z1 filtered MMixture", np.sum(z1r))
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    uPlt.setLUTScale( 0, max( np.max(z0), np.max(z0) )  )
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0,  title="Mathieson  cath-0" )
    uPlt.drawPads( fig, ax[0,1], x1, y1, dx1, dy1, z1,  title="Mathieson  cath-1" )
    uPlt.drawPads( fig, ax[1,0], x0r, y0r, dx0r, dy0r, z0r,  title="Mathieson  cath-0 - Removed pads" )
    uPlt.drawPads( fig, ax[1,1], x1r, y1r, dx1r, dy1r, z1r,  title="Mathieson  cath-1 - Removed pads" )
    #
    fig.suptitle( "Filtering pads to a Mathieson Mixture")
    plt.show()
    
  
    # Projection on one plane
    (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.projectChargeOnOnePlane(
                        x0r, dx0r, y0r, dy0r, x1r, dx1r, y1r, dy1r, z0r, z1r)
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    maxZ = max( np.max(z0), np.max(z1))
    uPlt.setLUTScale( 0, maxZ )
    uPlt.drawPads( fig, ax[0,0], x0r, y0r, dx0r, dy0r, z0r, alpha=1.0, doLimits=False, title="Mathieson cath-0" )
    ax[0,0].set_xlim( -1.0 ,  1.0 )
    ax[0,0].set_ylim( -1.0 ,  1.0 )
    uPlt.drawPads( fig, ax[0,1], x1r, y1r, dx1r, dy1r, z1r, alpha=1.0, doLimits=False, title="Mathieson cath-1" )
    ax[0,1].set_xlim( -1.0 ,  1.0 )
    ax[0,1].set_ylim( -1.0 ,  1.0 )
    maxZ = max( np.max(chA), np.max(chB))
    uPlt.setLUTScale( 0, maxZ )
    uPlt.drawPads( fig, ax[1,0], xProj, yProj, dxProj, dyProj, chA,  title="Cath-0 Projection" )
    uPlt.drawPads( fig, ax[1,1], xProj, yProj, dxProj, dyProj, chB,  title="Cath-1 Projection" )    
    fig.suptitle( "Pad-Projection")
    plt.show()
    
    # Connected-Components
    #
    nbrGroups, padGrp = PCWrap.getConnectedComponentsOfProjPads()
    #
    print( "# of groups, pad group ", nbrGroups, padGrp, np.max(padGrp) )
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
    zProj = (chA + chB) * 0.5
    uPlt.setLUTScale( 0, np.max(zProj)  )
    uPlt.drawPads( fig, ax[0], xProj, yProj, dxProj, dyProj, zProj, alpha=1.0, title="Projected charge" ) 
    uPlt.setLUTScale( 0, np.max(padGrp)  )
    uPlt.drawPads( fig, ax[1], xProj, yProj, dxProj, dyProj, padGrp, alpha=1.0, title="Pad group" ) 
    fig.suptitle( "Geometric Groups")
    plt.show()
    
    # Assign group to cath-pads
    nCath0 = cath0r.size
    nCath1 = cath1r.size
    wellSplitGroups = PCWrap.assignCathPadsToGroup( padGrp, nbrGroups, nCath0, nCath1)
    print( "wellSplitGroups", wellSplitGroups)
    cath0Pads, cath1Pads = PCWrap.copyCathToGrp(nCath0, nCath1)
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    maxZ = max( np.max(z0r), np.max(z1r))        
    uPlt.setLUTScale( 0, maxZ )
    uPlt.drawPads( fig, ax[0,0], x0r, y0r, dx0r, dy0r, z0r, alpha=1.0, doLimits=True, title="Mathieson cath-0" )
    uPlt.drawPads( fig, ax[0,1], x1r, y1r, dx1r, dy1r, z1r, alpha=1.0, doLimits=True, title="Mathieson cath-1" )
    minG = min( np.min(cath0Pads), np.min(cath1Pads))
    maxG = max( np.max(cath0Pads), np.max(cath1Pads))        
    uPlt.setLUTScale( minG, maxG )
    uPlt.drawPads( fig, ax[1,0], x0r, y0r, dx0r, dy0r, cath0Pads, alpha=1.0, doLimits=True, title="Cath-0 Groups" )
    uPlt.drawPads( fig, ax[1,1], x1r, y1r, dx1r, dy1r, cath1Pads, alpha=1.0, doLimits=True, title="Cath-1 Groups" )    
    fig.suptitle( "Cathodes pads Groups")
    plt.show()
    
    # free memory in Pad-Processing
    PCWrap.freeMemoryPadProcessing()
    