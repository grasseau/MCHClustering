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


if __name__ == "__main__":
    
    pcWrap = PCWrap.setupPyCWrapper()
    pcWrap.initMathieson()
    
    K = 1
    w   = np.array( [1.0] )
    muX = np.array( [0.5] )
    muY = np.array( [0.5] )
    varX = np.array( [0.1 * 0.1] )
    varY = np.array( [0.2 * 0.2] )
    """
    w   = np.array( [0.5, 0.5] )
    muX = np.array( [0.5, 0.5] )
    muY = np.array( [0.5, 0.5] )
    varX = np.array( [0.1 * 0.1, 0.1 * 0.1] )
    varY = np.array( [0.2 * 0.2, 0.2 * 0.2] )
    """
    chId = 1
    
    x, y, dx, dy = tUtil.buildPads( 10, 10, -0.2, 1.2, -0.2, 1.2 )
    
    #
    # Merge pads
    N = x.size
    # (x, y, dx, dy) = tUtil.mergePads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 )
    cath = np.zeros( x.size, dtype=np.int32 )
    #
    # Compute charge on the pads
    #
    # xyInfM
    xInfM = x - dx - muX[0]
    xSupM = x + dx - muX[0]
    yInfM = y - dy - muY[0]
    ySupM = y + dy - muY[0]
    
    # Compute Mathieson
    z = tUtil.compute2DPadIntegrals( xInfM, xSupM, yInfM, ySupM, chId )
    print("sum z", np.sum(z))    
    
    # 
    # Test Gaussian EM
    # 
    verbose = 2
    # mode : cstVar (1bit)
    mode = 0 
    LConv = 1.0e-6
    N = x.size

    thetaRef = tUtil.asTheta( w, muX, muY, varX, varY)
    """
    muXi = np.array( [0.5, 0.5] )
    muYi = np.array( [0.5, 0.5] )
    """
    wi   = np.array( [0.6, 0.4] )
    muXi = np.array( [0.6, 0.4] )
    muYi = np.array( [0.6, 0.4] )
    varXi = np.array( [0.1 * 0.1, 0.1 * 0.1] )
    varYi = np.array( [0.2 * 0.2, 0.2 * 0.2] )

    thetai = tUtil.asTheta( wi, muXi, muYi, varXi, varYi)
    # Set variance to approximate Mathienson fct
    PCWrap.setMathiesonVarianceApprox( chId, thetai)

    xyDxy  = tUtil.asXYdXY( x, y, dx, dy)
    xyInfSup = tUtil.padToXYInfSup( x, y, dx, dy )
    zi = PCWrap.generateMixedGaussians2D( xyInfSup, thetai )
    tUtil.printTheta("theta initial", thetai)
    print( "zi min/max/sum", min(zi), max(zi), sum(zi))
    # print( "zi", zi )
    saturated = np.zeros( x.size, dtype=np.int16 )
    thetaMask = np.ones( 1, dtype=np.int16  )
    theta, logL = PCWrap.weightedEMLoop( xyDxy, saturated, z, thetai, thetaMask, mode, LConv, verbose ) 
    # theta = PCWrap.weightedEMLoop( xyDxy, z, thetai, mode, LConv, verbose ) 
    tUtil.printTheta("theta final", theta)
    # 
    zf = PCWrap.generateMixedGaussians2D( xyInfSup, theta )
    print( "zf min/max/sum", min(zf), max(zf), sum(zf))

    residual = z - zf
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    uPlt.setLUTScale( 0, np.max(z) )
    uPlt.drawPads( fig, ax[0,0], x, y, dx, dy, z,  title="Mathieson (%d,%d)" % (0,0))
    uPlt.setLUTScale( 0, np.max(zi) )
    uPlt.drawPads( fig, ax[0,1], x, y, dx, dy, zi,  title="Init. Gaussian (%d,%d)" % (0,0))
    uPlt.setLUTScale( 0, np.max(zf) )
    uPlt.drawPads( fig, ax[1,0], x, y, dx, dy, zf,  title="Final Gaussian (%d,%d)" % (0,0))
    uPlt.setLUTScale( np.min(residual), np.max(residual) )
    uPlt.drawPads( fig, ax[1,1], x, y, dx, dy, residual,  title="Resudual (%d,%d)" % (0,0))
    # uPlt.drawPads( fig, ax[0,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], z[cath==1],  title="Mathieson (%d,%d)" % (0,0))
    # uPlt.drawPads( fig, ax[1,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], zGauss[cath==0],  title="Gaussian (%d,%d)" % (0,0))
    # uPlt.drawPads( fig, ax[1,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], zGauss[cath==1],  title="Gaussian (%d,%d)" % (0,0))    
    plt.show()
