#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt

import C.PyCWrapper as PCWrap
import utilitiesForTests as tUtil
import Util.plot as uPlt


if __name__ == "__main__":
    
    pcWrap = PCWrap.setupPyCWrapper()
    pcWrap.initMathieson()
    
    K = 1
    w   = np.array( [1.0] )
    muX = np.array( [0.5] )
    muY = np.array( [0.6] )
    varX = np.array( [0.1 * 0.1] )
    varY = np.array( [0.1 * 0.1] )
    chId = 2
    
    x0, y0, dx0, dy0 = tUtil.buildPads( 4, 2, -1.0, 1.0, -1.0, 1.0 )
    x1, y1, dx1, dy1 = tUtil.buildPads( 2, 4, -1.0, 1.0, -1.0, 1.0 ) 
    
    #
    # Merge pads
    N0 = x0.size
    (x, y, dx, dy) = tUtil.mergePads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 )
    cath = np.zeros( x.size, dtype=np.int32 )
    cath[N0:] = 1
    #
    # Compute charge on the pads
    #
    # xyInf
    xInf = x - dx - muX[0]
    xSup = x + dx - muX[0]
    yInf = y - dy - muY[0]
    ySup = y + dy - muY[0]
    
    z = tUtil.compute2DPadIntegrals( xInf, xSup, yInf, ySup, chId )
    print("z", z)
    print("sum z", np.sum(z))    
    
    # 
    # Test Gaussian EM
    # 
    verbose = 2
    # mode : cstVar (1bit)
    mode = 1 
    LConv = 1.0e-6
    N = x.size
    thetai = tUtil.asTheta( w, muX, muY, varX, varY)
    xyDxy  = tUtil.asXYdXY( x, y, dx, dy)
    xyInfSup = tUtil.padToXYInfSup( x, y, dx, dy )
    zGauss = PCWrap.generateMixedGaussians2D( xyInfSup, thetai )
    print( "zGauss min/max/sum", min(zGauss), max(zGauss), sum(zGauss))
    print( "zGauss", zGauss )
    theta = PCWrap.weightedEMLoop( xyDxy, z, thetai, mode, LConv, verbose ) 
    print("theta", theta)
    # 
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    uPlt.setLUTScale( 0, np.max(z) )
    uPlt.drawPads( fig, ax[0,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], z[cath==0],  title="Mathieson (%d,%d)" % (0,0))
    uPlt.drawPads( fig, ax[0,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], z[cath==1],  title="Mathieson (%d,%d)" % (0,0))
    uPlt.setLUTScale( 0, np.max(zGauss) )
    uPlt.drawPads( fig, ax[1,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], zGauss[cath==0],  title="Gaussian (%d,%d)" % (0,0))
    uPlt.drawPads( fig, ax[1,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], zGauss[cath==1],  title="Gaussian (%d,%d)" % (0,0))    
    plt.show()
