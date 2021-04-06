#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 30, 2020 4:46:52 PM$"

import GaussianEM2Dv4 as EM
import numpy as np

import matplotlib.pyplot as plt
import plotUtil as plu

if __name__ == "__main__":
    # Model
    w0, mu0, sig0, wi, mui, sigi = EM.GMModel.set1popId(verbose=True)
    var0 = sig0*sig0
    vari = sigi*sigi
    #
    # Set EM Mode
    EM.setDiscretized(True)
    plu.getColorMap()
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(13, 7) )

    #
    # Grid 1
    Nx = 10
    Ny = 10
    deltaX = 1.0/Nx
    # deltaX = 1.0
    deltaY = 1.0 / Ny
    grid1 = EM.setGridInfo( Nx=Nx+1, Ny=Ny+1, XOrig=0.0, YOrig=0.0, dx=deltaX, dy=deltaY)
    xy1 = np.mgrid[ 0: 1.0+deltaX: deltaX, 0: 1.0+deltaY: deltaY].reshape(2,-1)
    print("??? xy1.shape", xy1.shape)
    print (xy1)
    dxy=np.array( [deltaX, deltaY] )
    z = EM.generateMixedGaussians2D( xy1, dxy, w0, mu0, var0 )
    print( "z sum", np.sum(z) )
    EM.setDiscretized(False)
    zmid = EM.generateMixedGaussians2D( xy1, dxy, w0, mu0, var0 )
    zmax = np.copy( zmid)
    zmin = np.copy( zmid)
    xyt = np.zeros( xy1.shape )
    xyt[0,:] = xy1[0,:] - 0.5 * deltaX
    xyt[1,:] = xy1[1,:] - 0.5 * deltaY
    zmm = EM.generateMixedGaussians2D( xyt, dxy, w0, mu0, var0 )
    zmax = np.maximum( zmm, zmax )
    zmin = np.minimum( zmm, zmin )
    xyt[0,:] = xy1[0,:] + 0.5 * deltaX
    xyt[1,:] = xy1[1,:] - 0.5 * deltaY
    zpm = EM.generateMixedGaussians2D( xyt, dxy, w0, mu0, var0 )
    zmax = np.maximum( zpm, zmax )
    zmin = np.minimum( zpm, zmin )
    xyt[0,:] = xy1[0,:] - 0.5 * deltaX
    xyt[1,:] = xy1[1,:] + 0.5 * deltaY
    zmp = EM.generateMixedGaussians2D( xyt, dxy, w0, mu0, var0 )
    zmax = np.maximum( zmp, zmax )
    zmin = np.minimum( zmp, zmin )
    xyt[0,:] = xy1[0,:] + 0.5 * deltaX
    xyt[1,:] = xy1[1,:] + 0.5 * deltaY
    zpp = EM.generateMixedGaussians2D( xyt, dxy, w0, mu0, var0 )
    zmax = np.maximum( zpp, zmax )
    zmin = np.minimum( zpp, zmin )
    cst =  (dxy[0] * dxy[1])
    zmax = zmax * cst
    zmin = zmin * cst
    zmid = zmid * cst

    print("zmax-normalized sum", np.sum(zmax) )
    print("zmin-normalized sum", np.sum(zmin))
    plu.setLUTScale( 0,  max ( np.max(z), np.max(zmin), np.max(zmax) ) )

    plu.drawPads( ax[0,0], xy1[0],  xy1[1], 0.5*dxy[0], 0.5*dxy[1], z,  title="Integral")
    plu.drawPads( ax[0,1], xy1[0],  xy1[1], 0.5*dxy[0], 0.5*dxy[1], zmid,  title="Gaussian value x ds")
    plu.drawPads( ax[0,2], xy1[0],  xy1[1], 0.5*dxy[0], 0.5*dxy[1], zmin,  title="Min Gauss. value x ds")
    plu.drawPads( ax[0,3], xy1[0],  xy1[1], 0.5*dxy[0], 0.5*dxy[1], zmax,  title="Max Gauss. value x ds")

    plu.displayLUT( ax[0,4] )
    plt.show()
    #
    # Grid 2
    #
    Nx = 2
    Ny = 2
    deltaX = 1.0/Nx
    # deltaX = 1.0
    deltaY = 1.0 / Ny
    # shiftX = 0.1
    # shiftY = 0.1
    shiftX = 0.0
    shiftY = 0.0
    grid2 =EM.setGridInfo( Nx=Nx+1, Ny=Ny+1, XOrig=shiftX, YOrig=shiftY, dx=deltaX, dy=deltaY)
    xy2 = np.mgrid[ shiftX: 1.0+deltaX*0.5+shiftX: deltaX, shiftY: 1.0+deltaY*0.5+shiftY: deltaY].reshape(2,-1)
    print("??? xy2.shape", xy2.shape)
    xy = np.concatenate( (xy1, xy2), axis=1)

    # Generate the 2 distributions "the Truth"
    dxy = np.array( [ grid1['dx'], grid1['dy'] ], dtype= EM.InternalType)
    z1 = EM.generateMixedGaussians2D( xy1,  dxy, w0, mu0, var0,  normalize=False)
    dxy = np.array( [ grid2['dx'], grid2['dy'] ], dtype=EM.InternalType)
    z2 = EM.generateMixedGaussians2D( xy2,  dxy, w0, mu0, var0, normalize=False)
    #z1 = z1 * 0.5
    #z2 = z2 * 0.5
    z = np.concatenate( (z1, z2), axis=0)
    print( "??? z.shape", z.shape)

    #  (wf, muf, varf) = EM.weightedEMLoopWith2Grids( grid1, grid2, xy, z, wi, mui, vari, dataCompletion=False, plotEvery=1)
    # print "Hello World";
