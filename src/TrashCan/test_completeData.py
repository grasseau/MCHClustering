#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 30, 2020 4:46:52 PM$"

import numpy as np
import matplotlib.pyplot as plt

import GaussianEM2D as EM
import plotUtil as plu

if __name__ == "__main__":
    # 
    N = 5
    delta = 1.0/N
    # w0, mu0, sig0, wi, mui, sigi = EM.set1pop(verbose=True)
    w0, mu0, sig0, wi, mui, sigi = EM.set4popId(verbose=True)

    var0 = sig0*sig0
    vari = sigi*sigi

    dx = delta
    dy = delta
    EM.setGridInfo( Nx=N+1, Ny=N+1, XOrig=0.0, YOrig=1.0, dx=delta, dy=delta)
    xy = np.mgrid[ 0: 1.0+delta: delta, 0: 1.0+delta: delta].reshape(2,-1)
    print "??? xy0.shape", xy.shape
    z0 = EM.generateMixedGaussians2D( xy,  w0, mu0, var0)
    print "zi minmax", np.min(z0), np.max(z0)
    # Plot
    colors = plu.getColorMap()
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(13, 7) )
    maxZ0 = np.max(z0)
    maxZ = 0.0 # np.max(zc)

    dx0 = EM.GridInfo['dx']
    dy0 = EM.GridInfo['dy']
    dxy = np.zeros( xy.shape, dtype = EM.DataType)
    dxy[0,:] = dx0 * 0.5
    dxy[1,:] = dy0 * 0.5
    plu.setLUTScale( 0, max( maxZ0, maxZ) )
    plu.drawPads(ax[0], xy[0, :], xy[1, :], dxy[0, :], dxy[1, :], z0[:],  title="Input array")
    # (wf, muf, varf) = EM.weightedEMLoop(xy, dx, dy, z0, wi, mui, vari, plotEvery=1)

    # Test 1 : theta  theta0
    xyc, zc = EM.completeData(xy, z0,  wi, mui, vari, Check = True)
    dxy = np.zeros( xyc.shape, dtype = EM.DataType)
    dxy[0,:] = dx0 * 0.5
    dxy[1,:] = dy0 * 0.5
    print "zc.shape", zc.shape
    plu.setLUTScale( 0, max( maxZ0, maxZ) )
    plu.drawPads(ax[1], xyc[0, :], xyc[1, :], dxy[0, :], dxy[1, :], zc[:],  title="New grid")

    # Test 1 : mu top left
    mui[0] = [0.8, 0.8] 
    xyc, zc = EM.completeData(xy, z0,  wi, mui, vari, Check = True)
    dxy = np.zeros( xyc.shape, dtype = EM.DataType)
    dxy[0,:] = dx0 * 0.5
    dxy[1,:] = dy0 * 0.5
    print "zc.shape", zc.shape
    plu.setLUTScale( 0, max( maxZ0, maxZ) )
    plu.drawPads(ax[2], xyc[0, :], xyc[1, :], dxy[0, :], dxy[1, :], zc[:],  title="New grid")

    plu.displayLUT( ax[3] )

    plt.show()
    print "Hello World";
