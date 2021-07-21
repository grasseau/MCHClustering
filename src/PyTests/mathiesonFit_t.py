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
    
    K = 1
    w   = np.array( [1.0] )
    muX = np.array( [0.4] )
    muY = np.array( [0.5] )
    chId = 2
    
    x0, y0, dx0, dy0 = tUtil.buildPads( 4, 2, -1.0, 1.0, -1.0, 1.0 )
    x1, y1, dx1, dy1 = tUtil.buildPads( 2, 4, -1.0, 1.0, -1.0, 1.0 ) 
    
    # Merge pads
    N0 = x0.size
    x = np.hstack( [x0, x1] )
    y = np.hstack( [y0, y1] )    
    dx = np.hstack( [dx0, dx1] )
    dy = np.hstack( [dy0, dy1] )
    cath = np.zeros( x.size, dtype=np.int16 )
    cath[N0:] = 1
    #
    # xyInf
    xInf = x - dx - muX[0]
    xSup = x + dx - muX[0]
    yInf = y - dy - muY[0]
    ySup = y + dy - muY[0]
    
    print("x", x)
    print("y", y)
    print("xInf", xInf)
    print("xSup", xSup)
    print("yInf", yInf)
    print("ySup", ySup)
    print("cath", cath)
    
    z = tUtil.compute2DPadIntegrals( xInf, xSup, yInf, ySup, chId )
    print("z", z)
    print("sum z", np.sum(z))
    
    # Mathieson Fitting
    N = x.size
    thetai = tUtil.asTheta( w, muX, muY)
    xyDxy  = tUtil.asXYdXY( x, y, dx, dy)
    # zCathTotalCharge = np.zeros( 2)
    # zCathTotalCharge[0] = np.sum( z )
    
    (thetaf, khi2, pError) = PCWrap.fitMathieson( thetai, xyDxy, cath, z,
                         chId, verbose=1, doJacobian=0, doKhi=0, doStdErr=0)

    print("thetaf", thetaf)
    print("khi2", khi2)
    print("pError", pError)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    uPlt.setLUTScale( 0, np.max(z) )
    uPlt.drawPads( fig, ax[0,0], x[cath==0], y[cath==0], dx[cath==0], dy[cath==0], z[cath==0],  title="Mathieson (%d,%d)" % (0,0))
    uPlt.drawPads( fig, ax[0,1], x[cath==1], y[cath==1], dx[cath==1], dy[cath==1], z[cath==1],  title="Mathieson (%d,%d)" % (0,0))
    plt.show()
    print("Hello World")
