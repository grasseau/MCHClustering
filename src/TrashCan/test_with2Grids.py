#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 30, 2020 4:46:52 PM$"

import numpy as np
import GaussianEM2D as EM

if __name__ == "__main__":
    # 
    N = 5
    delta = 1.0/N
    # Model
    # w0, mu0, sig0, wi, mui, sigi = EM.set4pop(verbose=True)
    w0, mu0, sig0, wi, mui, sigi = EM.set1popId(verbose=True)
    var0 = sig0*sig0
    vari = sigi*sigi

    # Grids
    dx = delta
    dy = delta
    grid1 = EM.setGridInfo( Nx=N+1, Ny=N+1, XOrig=0.0, YOrig=0.0, dx=delta, dy=delta)
    xy1 = np.mgrid[ 0: 1.0+delta: delta, 0: 1.0+delta: delta].reshape(2,-1)
    print "??? xy1.shape", xy1.shape
    #
    grid2 =EM.setGridInfo( Nx=N+1, Ny=N+1, XOrig=0.1, YOrig=0.1, dx=delta, dy=delta)
    xy2 = np.mgrid[ 0.1: 1.0+delta+0.1: delta, 0.1: 1.0+delta+0.1: delta].reshape(2,-1)
    print "??? xy2.shape", xy2.shape
    xy = np.concatenate( (xy1, xy2), axis=1)
    print "??? xy.shape", xy.shape

    # Generate the 2 distributions "the Truth"
    z1 = EM.generateMixedGaussians2D( xy1,  w0, mu0, var0)
    z2 = EM.generateMixedGaussians2D( xy2,  w0, mu0, var0)
    z = np.concatenate( (z1, z2), axis=0)
    print "??? z.shape", z.shape


    (wf, muf, varf) = EM.weightedEMLoopWith2Grids(grid1, grid2, xy, z, wi, mui, vari, plotEvery=20)
    print "Hello World";
