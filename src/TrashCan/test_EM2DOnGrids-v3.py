#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 30, 2020 4:46:52 PM$"

import GaussianEM2Dv3 as EM
import numpy as np

# TODO ??? : Shift of grids must be taken into account too

if __name__ == "__main__":
    #
    # Model
    #
    # w0, mu0, sig0, wi, mui, sigi = EM.set4pop(verbose=True)
    model = EM.GMModel()
    w0, mu0, sig0, wi, mui, sigi = model.set1popId(verbose=True)
    var0 = sig0*sig0
    vari = sigi*sigi
    #
    # Set EM Mode in Discretised Gaussian mode
    #
    discretizedPDF = False
    em = EM.EM2D( discretizedPDF )
    grids = []
    #
    # Data/grids
    #
    # Grid 1
    Nx = 1
    Ny = 3
    startX = 0.0
    startY = 0.5
    grid1 = EM.Grid2D( Nx, Ny,  wi, mui, vari, startX=startX, startY=startY)
    print "sum z1", grid1.zSum
    grids.append( grid1 )
    # ??? grid1 = EM.setGridInfo( Nx=Nx+1, Ny=Ny+1, XOrig=0.0, YOrig=0.0, dx=deltaX, dy=deltaY)
    # ??? xy1 = np.mgrid[ 0: 1.0+0.5*deltaX: deltaX, 0: 1.0+0.5*deltaY: deltaY].reshape(2,-1)
    #
    # Grid 2
    #
    Nx = 4
    Ny = 1
    # shiftX = 0.1
    # shiftY = 0.1
    # shiftX = 0.5
    # shiftY = 0.0
    #  ???  grid2 =EM.setGridInfo( Nx=Nx+1, Ny=Ny+1, XOrig=shiftX, YOrig=shiftY, dx=deltaX, dy=deltaY)
    # ??? xy2 = np.mgrid[ shiftX: 1.0+deltaX*0.5+shiftX: deltaX, shiftY: 1.0+deltaY*0.5+shiftY: deltaY].reshape(2,-1)
    # grid2 = EM.Grid2D( Nx, Ny,  wi, mui, vari, startX = shiftX,  startY = shiftY, endX = 1.0 + shiftX, endY= 1.0 + shiftY )
    startX = 0.5
    startY = 0.0
    grid2 = EM.Grid2D( Nx, Ny,  wi, mui, vari, startX = startX,  startY = startY )
    print "sum z2", grid2.zSum
    grids.append( grid2 )

    (wf, muf, varf) = em.weightedEMLoopOnGrids( grids,  wi, mui, vari, dataCompletion=False, plotEvery=10)
