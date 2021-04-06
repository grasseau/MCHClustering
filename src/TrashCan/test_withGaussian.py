#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 30, 2020 4:46:52 PM$"

import numpy as np
import GaussianEM2D as EM

if __name__ == "__main__":
    # 
    N = 5
    delta = 1.0/N
    w0, mu0, sig0, wi, mui, sigi = EM.set1pop(verbose=True)
    var0 = sig0*sig0
    vari = sigi*sigi

    dx = delta
    dy = delta
    EM.setGridInfo( Nx=N+1, Ny=N+1, XOrig=0.0, YOrig=1.0, dx=delta, dy=delta)
    xy = np.mgrid[ 0: 1.0+delta: delta, 0: 1.0+delta: delta].reshape(2,-1)
    print "??? xy0.shape", xy.shape
    zi = EM.generateMixedGaussians2D( xy,  w0, mu0, var0)
    print "zi minmax", np.min(zi), np.max(zi)
    (wf, muf, varf) = EM.weightedEMLoop(xy, dx, dy, zi, wi, mui, vari, plotEvery=1)
    print "Hello World";
