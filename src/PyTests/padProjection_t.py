#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

import C.PyCWrapper as PCWrap
import Util.dataTools as tUtil
import Util.plot as uPlt
""" ???
def buildPads( nx, ny, xMin, xMax, yMin, yMax ):
    dx = (xMax - xMin)/nx
    dy = (yMax - yMin)/ny
    x1d = np.arange( xMin, xMax - 0.1 * dx, dx) + 0.5 * dx
    y1d = np.arange( yMin, yMax - 0.1 * dy, dy) + 0.5 * dy
    x, y = np.meshgrid( x1d, y1d)
    x = x.ravel( )
    y = y.ravel( )
    #
    dx = np.ones(x.shape) * 0.5 * dx
    dy = np.ones(y.shape) * 0.5 * dy
    #
    return x, y, dx, dy
"""
if __name__ == "__main__":
    
    pcWrap = PCWrap.setupPyCWrapper()
    pcWrap.initMathieson()
    
    K = 1
    mu = [0.4, 0.5]
    
    x0, y0, dx0, dy0 = tUtil.buildPads( 8, 4, -1.0, 1.0, -1.0, 1.0 )
    x1, y1, dx1, dy1 = tUtil.buildPads( 4, 8, -1.0, 1.0, -1.0, 1.0 )

    # Compute z - Cathode 0
    chId = 2
    x0Inf = x0 - dx0 - mu[0]
    y0Inf = y0 - dy0 - mu[1]
    cath0 = np.zeros( x0.size, dtype=np.int32 )
    z0 = tUtil.compute2DPadIntegrals( x0Inf, x0Inf + 2*dx0, y0Inf, y0Inf+2*dy0, chId)
    # Compute z - Cathode 0
    x1Inf = x1 - dx1 - mu[0]
    y1Inf = y1 - dy1 - mu[1]
    cath1 = np.ones( x1.size, dtype=np.int32 )
    # pcWrap.compute2DPadIntegrals( x1Inf, x1Inf + 2*dx1, y1Inf, y1Inf+2*dy1, x1Inf.shape[0], chId, z1)
    z1 = tUtil.compute2DPadIntegrals( x1Inf, x1Inf + 2*dx1, y1Inf, y1Inf+2*dy1, chId)
    
    # Projection on one plane
    (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.projectChargeOnOnePlane(
                        x0, dx0, y0, dy0, x1, dx1, y1, dy1, z0, z1)
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    maxZ = max( np.max(z0), np.max(z1))
    uPlt.setLUTScale( 0, maxZ )
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0, alpha=1.0, title="Mathieson cath-0" )

    uPlt.drawPads( fig, ax[0,1], x1, y1, dx1, dy1, z1, alpha=1.0, doLimits=False, title="Mathieson cath-1" )
    ax[0,1].set_xlim( -1.0 ,  1.0 )
    ax[0,1].set_ylim( -1.0 ,  1.0 )
    maxZ = max( np.max(chA), np.max(chB))
    uPlt.setLUTScale( 0, maxZ )
    uPlt.drawPads( fig, ax[1,0], xProj, yProj, dxProj, dyProj, chA,  title="Cath-0 Projection" )
    uPlt.drawPads( fig, ax[1,1], xProj, yProj, dxProj, dyProj, chB,  title="Cath-1 Projection" )    
    plt.show()
    #
    PCWrap.freeMemoryPadProcessing()

    # With Alone pa
    x0, y0, dx0, dy0 = buildPads( 2, 2, -1.0, 1.0, 0.0, 1.0 )
    x1, y1, dx1, dy1 = buildPads( 2, 2, 0.0, 1.0, -1.0, 1.0 )

    # Compute z - Cathode 0
    chId = 2
    x0Inf = x0 - dx0 - mu[0]
    y0Inf = y0 - dy0 - mu[1]
    cath0 = np.zeros( x0.size, dtype=np.int32 )
    z0 = np.zeros_like(x0Inf)
    pcWrap.compute2DPadIntegrals( x0Inf, x0Inf + 2*dx0, y0Inf, y0Inf+2*dy0, x0Inf.shape[0], chId, z0)
    # Compute z - Cathode 0
    x1Inf = x1 - dx1 - mu[0]
    y1Inf = y1 - dy1 - mu[1]
    cath1 = np.ones( x1.size, dtype=np.int32 )
    z1 = np.zeros_like(x1Inf)
    pcWrap.compute2DPadIntegrals( x1Inf, x1Inf + 2*dx1, y1Inf, y1Inf+2*dy1, x1Inf.shape[0], chId, z1)
    
    # Projection on one plane
    (xProj, dxProj, yProj, dyProj, chA, chB) = PCWrap.projectChargeOnOnePlane(
                        x0, dx0, y0, dy0, x1, dx1, y1, dy1, z0, z1)
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    maxZ = max( np.max(z0), np.max(z1))
    uPlt.setLUTScale( 0, maxZ )
    uPlt.drawPads( fig, ax[0,0], x0, y0, dx0, dy0, z0, alpha=1.0, doLimits=False, title="Mathieson cath-0" )
    ax[0,0].set_xlim( -1.0 ,  1.0 )
    ax[0,0].set_ylim( -1.0 ,  1.0 )
    uPlt.drawPads( fig, ax[0,1], x1, y1, dx1, dy1, z1, alpha=1.0, doLimits=False, title="Mathieson cath-1" )
    ax[0,1].set_xlim( -1.0 ,  1.0 )
    ax[0,1].set_ylim( -1.0 ,  1.0 )
    maxZ = max( np.max(chA), np.max(chB))
    uPlt.setLUTScale( 0, maxZ )
    uPlt.drawPads( fig, ax[1,0], xProj, yProj, dxProj, dyProj, chA,  title="Cath-0 Projection" )
    uPlt.drawPads( fig, ax[1,1], xProj, yProj, dxProj, dyProj, chB,  title="Cath-1 Projection" )    
    plt.show()
    #
    PCWrap.freeMemoryPadProcessing()
    