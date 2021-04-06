#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 30, 2020 4:46:52 PM$"

import GaussianEM2D as EM
import numpy as np

# TODO ??? : Shift of grids must be taken into account too

if __name__ == "__main__":
    # Model
    # w0, mu0, sig0, wi, mui, sigi = EM.set4pop(verbose=True)
    w0, mu0, sig0, wi, mui, sigi = EM.set1popId(verbose=True)
    var0 = sig0*sig0
    vari = sigi*sigi
    #
    # Set EM Mode
    EM.setDiscretized(True)
    #
    # Grid 1
    Nx = 13
    Ny = 8
    deltaX = 1.0/Nx
    # deltaX = 1.0
    deltaY = 1.0 / Ny
    grid1 = EM.setGridInfo( Nx=Nx+1, Ny=Ny+1, XOrig=0.0, YOrig=0.0, dx=deltaX, dy=deltaY)
    xy1 = np.mgrid[ 0: 1.0+0.5*deltaX: deltaX, 0: 1.0+0.5*deltaY: deltaY].reshape(2,-1)
    print "??? xy1.shape", xy1.shape
    print xy1
    #
    # Grid 2
    #
    Nx = 8
    Ny = 13
    deltaX = 1.0/Nx
    # deltaX = 1.0
    deltaY = 1.0 / Ny
    # shiftX = 0.1
    # shiftY = 0.1
    shiftX = 0.15
    shiftY = 0.2
    grid2 =EM.setGridInfo( Nx=Nx+1, Ny=Ny+1, XOrig=shiftX, YOrig=shiftY, dx=deltaX, dy=deltaY)
    xy2 = np.mgrid[ shiftX: 1.0+deltaX*0.5+shiftX: deltaX, shiftY: 1.0+deltaY*0.5+shiftY: deltaY].reshape(2,-1)
    print "??? xy2.shape", xy2.shape
    xy = np.concatenate( (xy1, xy2), axis=1)
    print "??? xy.shape", xy.shape

    # Generate the 2 distributions "the Truth"
    dxy = np.array( [ grid1['dx'], grid1['dy'] ], dtype= EM.InternalType)
    z1 = EM.generateMixedGaussians2D( xy1,  dxy, w0, mu0, var0,  normalize=False)
    print "??? xy1.shape", xy1.shape
    print "??? dxy.shape", dxy.shape
    dxy = np.array( [ grid2['dx'], grid2['dy'] ], dtype=EM.InternalType)
    print "??? dxy.shape", dxy.shape
    z2 = EM.generateMixedGaussians2D( xy2,  dxy, w0, mu0, var0, normalize=False)
    print "??? z1.shape", z1.shape
    print "??? z2.shape", z2.shape
    #z1 = z1 * 0.5
    #z2 = z2 * 0.5
    z = np.concatenate( (z1, z2), axis=0)
    print "??? z.shape", z.shape

    (wf, muf, varf) = EM.weightedEMLoopWith2Grids( grid1, grid2, xy, z, wi, mui, vari, dataCompletion=True, plotEvery=20)
    print "Hello World";
