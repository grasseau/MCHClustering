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
    
    K = 1
    mu = [0.0, 0.0]
    chId = 2
    
    x0, y0, dx0, dy0 = tUtil.buildPads( 4, 2, -1.0, 1.0, -1.0, 1.0 )
    x1, y1, dx1, dy1 = tUtil.buildPads( 2, 4, -1.0, 1.0, -1.0, 1.0 )
    
    # Merge pads
    N0 = x0.size
    (x, y, dx, dy) = tUtil.mergePads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 )
    cath = np.zeros( x.size, dtype=np.int32 )
    cath[N0:] = 1
    #
    # xyInfSup
    xInf = x - dx - mu[0]
    xSup = x + dx - mu[0]
    yInf = y - dy - mu[1]
    ySup = y + dy - mu[1]
    """
    print("x", x)
    print("y", y)
    print("xInf", xInf)
    print("xSup", xSup)
    print("yInf", yInf)
    print("ySup", ySup)
    print("cath", cath)
    """
    z = tUtil.compute2DPadIntegrals( xInf, xSup, yInf, ySup, chId )
    # print("z", z)
    print("sum z", np.sum(z))

    # Mathieson shift
    mu = [0.4, 0.5]
    chId = 0
    # xyInfSup
    xInf = x - dx - mu[0]
    xSup = x + dx - mu[0]
    yInf = y - dy - mu[1]
    ySup = y + dy - mu[1]
    
    q = tUtil.compute2DPadIntegrals( xInf, xSup, yInf, ySup, chId )
    # print("z", z)
    print("sum q", np.sum(q))
    

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    uPlt.setLUTScale( 0, np.max(z) )
    uPlt.drawPads( fig, ax[0,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], z[cath==0],  title="Mathieson (%d,%d) cath-0" % (0,0))
    uPlt.drawPads( fig, ax[0,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], z[cath==1],  title="Mathieson (%d,%d) cath-1" % (0,0))
    uPlt.setLUTScale( 0, np.max(q) )
    uPlt.drawPads( fig, ax[1,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], q[cath==0],  title="Mathieson (%3.1f, %3.1f) cath-0" % (mu[0],mu[1]) )
    uPlt.drawPads( fig, ax[1,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], q[cath==1],  title="Mathieson (%3.1f, %3.1f) cath-1" % (mu[0],mu[1]) )
    #
    plt.show()
    
    # TODO
    # Compare ch0 & ch2 with a high reso grid

    #
    # Mathieson mixture
    #
    chId = 2
    K = 2
    w   = np.array( [0.6, 0.4 ] )
    muX = np.array( [0.4, -0.4] )
    muY = np.array( [0.4, -0.4] )
    cstVar = 0.1 * 0.1
    varX = np.array( [ cstVar, cstVar] )
    varY = np.array( [ cstVar, cstVar] )
    theta = tUtil.asTheta( w, muX, muY, varX, varY)
    xyInfSup = tUtil.padToXYInfSup( x, y, dx, dy)
    zMix = PCWrap.compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, chId )
    print("sum zMix", np.sum(zMix))
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    uPlt.setLUTScale( 0, np.max(zMix) )
    uPlt.drawPads( fig, ax[0,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], zMix[cath==0],  title="Mathieson  cath-0" )
    uPlt.drawPads( fig, ax[0,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], zMix[cath==1],  title="Mathieson  cath-1" )
    #
    plt.show()
