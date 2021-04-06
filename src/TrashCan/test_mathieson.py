#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 30, 2020 4:46:52 PM$"

import GaussianEM2Dv4 as EM
import numpy as np

import matplotlib.pyplot as plt
import plotUtil as plu
import CurveFiting as CF
import Mathieson

def anyClusters():
 
    # Model
    w0, mu0, sig0, wi, mui, sigi = EM.GMModel.set2popId(verbose=True)
    input("next")
    w0[0] = 0.6
    w0[1] = 0.4
    mu0[0][0] = 0.8
    mu0[0][1] = 0.8
    mu0[1][0] = 0.0
    mu0[1][1] = 0.4 
    mui[0][0] = 0.8
    mui[0][1] = 0.8
    mui[1][0] = 0.0
    mui[1][1] = 0.4   
    # sig0[0] = 0.3
    sig0[0][0] = 0.1
    sig0[0][1] = 0.1
    sig0[1][0] = 0.1
    sig0[1][1] = 0.1
    var0 = sig0*sig0
    vari = sigi*sigi
    #
    # Set EM Mode
    em = EM.EM2D(True)
    CF.setCstSig( np.array( [0.1, 0.1]) )
    
    plu.getColorMap()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )
 
    xl= [-0.5, 0.0, 0.5, 1.0, 1.5,  -0.5, 0.0, 0.5, 1.0, 1.5     ] #, 2.5 ] #, 0.8,]
    yl= [0.8, 0.8, 0.8, 0.8, 0.8,  0.4, 0.4, 0.4, 0.4, 0.4 ] # , 0.8] #, .5 ]
    cath = np.array( [0, 0, 0, 0, 0,   0, 0, 0, 0, 0]) #, 0]) #, 1] )
    dxl = [0.25, 0.25, 0.25, 0.25, 0.25,   0.25, 0.25, 0.25, 0.25, 0.25   ] # , 0.5] # , 0.2 ]
    dyl = [0.2, 0.2, 0.2, 0.2, 0.2,    0.2, 0.2, 0.2, 0.2, 0.2 ] #, 
    x = np.array( [xl] )
    y = np.array( [yl] )
    dx = np.array( [dxl] )
    dy = np.array( [dyl] )
    xy = np.vstack( [x, y])
    dxy = np.vstack( [dx, dy])
    allx = [xy, dxy]
    print(xy)
    print(dxy)
    z = np.zeros( (x.size) )
    for k in range(2):
      z += w0[k] * EM.computeDiscretizedGaussian2D( xy, dxy, mu0[k], var0[k] )

    zMax = np.max( z )
    plu.setLUTScale( 0, zMax )
    idx0 = np.where( cath==0 )
    idx1 = np.where( cath==1 )
    w, mu, var = EM.simpleProcessEMCluster( x[0], y[0], dx[0], dy[0], cath, z, w0, mu0, var0, cstVar=True , discretizedPDF=True )
    print( "mu", mu)
    print( "sig", np.sqrt(var) ) 

    wi = np.array( [0.5, 0.5] )
    """
    shape, dtype = check_func('leastsq', 'func', err_fct2, x0, (xy,dxy, z, 2), n)
    print("shape, n", shape, n)
    c = check_gradient(err_fct2, jac_fct2, x0, args=(xy,dxy, z, 2))
    print("check derivates", c)
    res = opt.leastsq( err_fct2, x0=x0, args=(xy,dxy, z, 2), Dfun=jac_fct2) #, **kwargs)
    """
    CF.clusterFit( CF.err_fct2, wi, mui, vari, xy , dxy, z, jacobian=CF.jac_fct2)
    CF.clusterFit( CF.err_fct2, wi, mui, vari, xy , dxy, z)
    input("next")
    # res = least_squares( func, x0_rosenbrock)   
    # print(res)
    xInf, xSup, yInf, ySup = plu.getPadBox(x, y, dx, dy )
    print( xInf, xSup, yInf, ySup )
    if( xy[0][idx0].size != 0 ):
      plu.drawPads( ax[0,0], xy[0][idx0],  xy[1][idx0], dxy[0][idx0], dxy[1][idx0], z[idx0],  alpha=0.5, title="Integral")
    if( xy[0][idx1].size != 0 ):
      plu.drawPads( ax[0,0], xy[0][idx1],  xy[1][idx1], dxy[0][idx1], dxy[1][idx1], z[idx1],  alpha=0.5, title="Integral")
    ax[0,0].set_xlim( xInf, xSup)
    ax[0,0].set_ylim( yInf, ySup)
    
    xp = np.linspace(xInf, xSup, num=20)
    yp = np.ones( xp.shape)*0.8
    xyp = np.vstack( [xp, yp] )
    zp = EM.computeGaussian2D(xyp, mu0[0], var0[0] )
    ax[0,1].plot(xp, zp)
    plt.show()
    return

def oneCluster():
 
    # Model
    #w0, mu0, sig0, wi, mui, sigi = EM.GMModel.set1popId(verbose=True)
    w0 = np.array([1.0])
    mu0 = np.array( [[0.8, 0.8]] )
    mui = np.array( [[0.5, 0.5]] )
    sig0 = np.array( [[0.1, 0.1]] )
    sigi = np.array( [[0.1, 0.1]] )
    var0 = sig0*sig0
    vari = sigi*sigi
    #
    # Set EM Mode
    em = EM.EM2D(True)
    CF.setCstSig( np.array( [0.1, 0.1]) )
    
    plu.getColorMap()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )
 
    xl= [-0.5, 0.0, 0.5, 1.0, 1.5,  -0.5, 0.0, 0.5, 1.0, 1.5     ] #, 2.5 ] #, 0.8,]
    yl= [0.8, 0.8, 0.8, 0.8, 0.8,  0.4, 0.4, 0.4, 0.4, 0.4 ] # , 0.8] #, .5 ]
    cath = np.array( [0, 0, 0, 0, 0,   0, 0, 0, 0, 0]) #, 0]) #, 1] )
    dxl = [0.25, 0.25, 0.25, 0.25, 0.25,   0.25, 0.25, 0.25, 0.25, 0.25   ] # , 0.5] # , 0.2 ]
    dyl = [0.2, 0.2, 0.2, 0.2, 0.2,    0.2, 0.2, 0.2, 0.2, 0.2 ] #, 
    x = np.array( [xl] )
    y = np.array( [yl] )
    dx = np.array( [dxl] )
    dy = np.array( [dyl] )
    xy = np.vstack( [x, y])
    dxy = np.vstack( [dx, dy])
    allx = [xy, dxy]
    print(xy)
    print(dxy)
    z = np.zeros( (x.size) )
    K = w0.size
    for k in range(K):
      print("???, mu0[k]", mu0[k])
      z += w0[k] * EM.computeDiscretizedGaussian2D( xy, dxy, mu0[k], var0[k] )

    zMax = np.max( z )
    plu.setLUTScale( 0, zMax )
    idx0 = np.where( cath==0 )
    idx1 = np.where( cath==1 )
    w, mu, var = EM.simpleProcessEMCluster( x[0], y[0], dx[0], dy[0], cath, z, w0, mu0, var0, cstVar=True , discretizedPDF=True )
    print( "mu", mu)
    print( "sig", np.sqrt(var) ) 

    wi = np.array( [1.0] )
    """
    shape, dtype = check_func('leastsq', 'func', err_fct2, x0, (xy,dxy, z, 2), n)
    print("shape, n", shape, n)
    c = check_gradient(err_fct2, jac_fct2, x0, args=(xy,dxy, z, 2))
    print("check derivates", c)
    res = opt.leastsq( err_fct2, x0=x0, args=(xy,dxy, z, 2), Dfun=jac_fct2) #, **kwargs)
    """
    CF.clusterFit( CF.err_fct2, wi, mui, vari, xy , dxy, z, jacobian=CF.jac_fct2)
    CF.clusterFit( CF.err_fct2, wi, mui, vari, xy , dxy, z)
    input("next")
    # res = least_squares( func, x0_rosenbrock)   
    # print(res)
    xInf, xSup, yInf, ySup = plu.getPadBox(x, y, dx, dy )
    print( xInf, xSup, yInf, ySup )
    if( xy[0][idx0].size != 0 ):
      plu.drawPads( ax[0,0], xy[0][idx0],  xy[1][idx0], dxy[0][idx0], dxy[1][idx0], z[idx0],  alpha=0.5, title="Integral")
    if( xy[0][idx1].size != 0 ):
      plu.drawPads( ax[0,0], xy[0][idx1],  xy[1][idx1], dxy[0][idx1], dxy[1][idx1], z[idx1],  alpha=0.5, title="Integral")
    ax[0,0].set_xlim( xInf, xSup)
    ax[0,0].set_ylim( yInf, ySup)
    
    xp = np.linspace(xInf, xSup, num=20)
    yp = np.ones( xp.shape)*0.8
    xyp = np.vstack( [xp, yp] )
    zp = EM.computeGaussian2D(xyp, mu0[0], var0[0] )
    ax[0,1].plot(xp, zp)
    plt.show()
    return

def mathieson1D( x, y , dx, dy ):
    N = 10
    dx = np.ones( (N))*0.1
    dy = np.ones( (N))*0.2
    x = np.zeros( (N) )
    y = np.zeros( (N) )
    for i in range(N):
      x[i] = 1.0 + 2*i*dx[i]
      y[i] = 1.0 + 2*i*dy[i]
    print( x )
    print( y )
    mu = np.array([ 2.0, 3.0 ])
    mat = Mathieson.Mathieson(0)
    ch = mat.computeMathieson1DIntegral( x-dx - mu[0], x+dx -mu[0])
    print(ch)
    print( np.sum(ch) )
    plt.plot( x, ch, "o" )
    plt.show()
    
    return
      
if __name__ == "__main__":
    N = 3
    dx = np.ones( (2*N))*0.2
    dy = np.ones( (2*N))*0.2
    x = np.zeros( (2*N) )
    y = np.zeros( (2*N) )
    dx = np.array( [ 0.2, 0.2, 0.2, 0.6, 0.6, 0.6 ])    
    x = np.array([ 1.1, 1.5, 1.9, 1.5, 1.5, 1.5 ])
    dy = np.array( [ 0.6, 0.6, 0.6, 0.2, 0.2, 0.2 ])    
    y = np.array([ 1.5, 1.5, 1.5, 1.1, 1.5, 1.9  ])
    """
    for j in range(N):
      for i in range(N):
        x[i+j*N] = 1.0 + 2*i*dx[i+j*N]
        y[i+j*N] = 1.0 + 2*i*dy[i+j*N]
    """
    print( x )
    print( y )
    mu = np.array([ 1.4, 1.4 ])
    mat = Mathieson.Mathieson(0, 2)
    ch = mat.computeMathieson2DIntegral( x-dx - mu[0], x+dx -mu[0], y-dy - mu[1], y+dy -mu[1])
    print(ch)
    print( np.sum(ch) )
    wGrp = np.array([1.0])
    muGrp = np.array([[1.0, 1.0]])
    mType = 0
    npxy = np.array( [x, y] )
    npdxy = np.array( [dx, dy] )
    wFit, muFit, varFit = CF.clusterFitMath( wGrp, muGrp, mType, npxy , npdxy, ch, 1, jacobian=None)
    print( wFit, muFit)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )
    plu.setLUTScale( 0.0, np.max( ch) ) 
    plu.drawPads( ax[0,0], x, y, dx, dy, ch, title="", yTitle="", alpha=0.5)

    # plt.plot( x, ch, "o" )
    plt.show()
    
    