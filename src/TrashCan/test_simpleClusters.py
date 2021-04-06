#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 30, 2020 4:46:52 PM$"

import GaussianEM2Dv4 as EM
import numpy as np

import matplotlib.pyplot as plt
import plotUtil as plu
from scipy.optimize import curve_fit

fixed1Sig = np.array([ [0.1, 0.1] ]) 
fixed1Var =  fixed1Sig * fixed1Sig
fixed2Sig = np.array([ [0.1, 0.1], [0.1, 0.1] ]) 
fixed2Var =  fixed2Sig * fixed2Sig

y0Store = []

"""
I have a fitting function which has the form:

def fit_func(x_data, a, b, c, N)

where a, b, c are lists of lenth N, every entry of which is a variable parameter to be optimized in scipy.optimize.curve_fit(), and N is a fixed number used for loop index control.

Following this question I think I am able to fix N, but I currently am calling curve_fit as follows:

params_0 = [a_init, b_init, c_init]
popt, pcov = curve_fit(lambda x, a, b, c: fit_func(x, a, b, c, N), x_data, y_data, p0=params_0)

I get an error: lambda() takes exactly Q arguments (P given)

############################# Sol ##################
def wrapper_fit_func(x, N, *args):
    a, b, c = list(args[0][:N]), list(args[0][N:2*N]), list(args[0][2*N:3*N])
    return fit_func(x, a, b, c, N)

popt, pcov = curve_fit(lambda x, *params_0: wrapper_fit_func(x, N, params_0), x, y, p0=params_0)
"""

# def compactFunc2( inputs, w, mux, muy ):
def compactFunc2( inputs, params ):
# def compactFunc2( **kwargs ):
  """
  inputs[0] [ , 0]: x
  inputs[0] [ , 1]: y
  inputs[1] [ , 0]: dx
  inputs[1] [ , 1]: dy
  params[k, 0] : weights 
  params[k, 1] :mux
  params[k, 2] :muy
  """
  """
  # Iterating over the Python kwargs dictionary
  for arg in kwargs.values():
    print(arg)
  """
  
  print("params", params)
  w = params[0:2]
  mux = params[2:4]
  muy = params[6:8]
  
  print("w", w)
  print("mux", mux)
  print("muy", muy)
  z = 0.0
  w  = np.array( [w] )
  mu = np.array( [[mux, muy]] )
  print("mu shape", mu.shape)
  K = w.size
  for k in range(K):
    z += w[k] * EM.computeDiscretizedGaussian2D( inputs[0], inputs[1], mu[k,:], fixed1Var[k,:] )
  return z

def fit_func( xy, dxy,  w, mux, muy):
# def compactFunc2( **kwargs ):
  """
  inputs[0] [ , 0]: x
  inputs[0] [ , 1]: y
  inputs[1] [ , 0]: dx
  inputs[1] [ , 1]: dy
  params[k, 0] : weights 
  params[k, 1] :mux
  params[k, 2] :muy
  """
  """
  # Iterating over the Python kwargs dictionary
  for arg in kwargs.values():
    print(arg)
  """
  
  print  
  print("w", w)
  
  s =  np.sum( np.array( w ))
  w.append( 1.0 - s )
  w  = np.array( w )
  print( s, w )
  mu = np.array( [mux, muy] ).T
  # print("mu shape", mu.shape)
  print("mu0", mu[0])
  print("mu1", mu[1])
  print( " x ", xy[0])
  print( " y ", xy[1])
  input("next")
  print(xy) 
  print(dxy)

  # print( " dxy ", dxy)
  z = 0.0
  K = w.size
  for k in range(K):
    z +=  w[k] * EM.computeDiscretizedGaussian2D( xy, dxy, mu[k,:], fixed2Var[k,:] )
    print("k, w, mu, z", k, w[k], mu[k,:], z)
  r = (y0Store - z)
  idx = np.argmax(r)
  print ("res", np.sum(np.abs(r)), np.min(r), np.max( r ))
  print("argmax th, mesured, x, y", y0Store[idx], z[idx], xy[0,idx], xy[1, idx], 
    w[0] * EM.computeDiscretizedGaussian2D( xy[:,idx].reshape(2,-1), dxy[:,idx].reshape(2,-1), mu[0,:], fixed2Var[0,:] )
    + w[1] * EM.computeDiscretizedGaussian2D( xy[:,idx].reshape(2,-1), dxy[:,idx].reshape(2,-1), mu[1,:], fixed2Var[1,:] )
  )
  wPenal = 1.0 -  np.sum( w )
  zPenal = z 
  return zPenal

def compactFunc( inputs, w, mux, muy ):
  """
  inputs[0] [ , 0]: x
  inputs[0] [ , 1]: y
  inputs[1] [ , 0]: dx
  inputs[1] [ , 1]: dy
  params[k, 0] : weights 
  params[k, 1] :mux
  params[k, 2] :muy
  """
  print("w", w)
  print("mux", mux)
  print("muy", muy)
  z = 0.0
  w  = np.array( [w] )
  mu = np.array( [[mux, muy]] )
  print("mu shape", mu.shape)
  K = w.size
  for k in range(K):
    z += w[k] * EM.computeDiscretizedGaussian2D( inputs[0], inputs[1], mu[k,:], fixed1Var[k,:] )
  return z

def func( xy, mux, muy, varx, vary ):
    # print("mu=", mux, muy)
    # print("var=", varx, vary)
    
    mu = np.array( [mux, muy] )
    var = np.array( [varx, vary] )
    z = EM.computeDiscretizedGaussian2D( xy[0], xy[1], mu, var )
    # z = EM.computeGaussian2D( xy[0], mu, var )
    return z

def simpleCluster():
    # Model
    w0, mu0, sig0, wi, mui, sigi = EM.GMModel.set1popId(verbose=True)
    mu0[0][0] = 0.8
    mu0[0][1] = 0.8
    # sig0[0] = 0.3
    var0 = sig0*sig0
    vari = sigi*sigi
    #
    # Set EM Mode
    em = EM.EM2D(True)
    
    plu.getColorMap()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )
    """
    xl= [0., 0., 0., -0.33, +0.33]
    yl= [0., 0.5, -0.5, 0.,0.]
    cath = np.array( [0, 0, 0, 1, 1] )
    dxl = [0.5, 0.5, 0.5, 0.33, 0.33]
    dyl = [0.25, 0.25, 0.25, 0.5, 0.5]
    """
    """
    xl= [0.5, 0.8]
    yl= [0.8, 0.5 ]
    cath = np.array( [0, 1] )
    dxl = [0.5, 0.2 ]
    dyl = [0.2, 0.5, ]
    """
    xl= [0.5, -0.5, 1.5] #, 2.5 ] #, 0.8,]
    yl= [0.8, 0.8, 0.8] # , 0.8] #, .5 ]
    cath = np.array( [0, 0, 0]) #, 0]) #, 1] )
    dxl = [0.5, 0.5, 0.5] # , 0.5] # , 0.2 ]
    dyl = [0.2, 0.2, 0.2] #, 0.2] #, 0.5 ]
    x = np.array( [xl] )
    y = np.array( [yl] )
    dx = np.array( [dxl] )
    dy = np.array( [dyl] )
    xy = np.vstack( [x, y])
    dxy = np.vstack( [dx, dy])
    allx = [xy, dxy]
    print(xy)
    print(dxy)
    z = EM.computeDiscretizedGaussian2D( xy, dxy, mu0[0], var0[0] )
    # z = EM.computeGaussian2D(xy, mu0[0], var0[0] )
    cst = np.sum(z[0:4])
    u = z[0:4] / cst
    print("??? u", u)
    print("??? z", u * x[0][0:4 ] )
    print("??? sum", z, np.dot( u, x[0][0:4 ]) )
    zMax = np.max( z )
    plu.setLUTScale( 0, zMax )
    idx0 = np.where( cath==0 )
    idx1 = np.where( cath==1 )
    w, mu, var = EM.simpleProcessEMCluster( x[0], y[0], dx[0], dy[0], cath, z, w0, mu0, var0, cstVar=True , discretizedPDF=True )
    print( "mu", mu)
    print( "sig", np.sqrt(var) ) 

    zf = EM.computeDiscretizedGaussian2D( xy, dxy, mu[0], var[0] )
    print("xy", xy[0, :])
    print("z", z)
    print("zf", zf)
    """
        p0 = [ np.array( [[1.0]] ),       # w
              np.array( [[0.75, 0.85]] )], # mu
    """
    popt, pcov = curve_fit(compactFunc, allx, z,
        bounds= (
                  # [ np.array([0.0]), np.array([-1.0]), np.array([0.0]) ], 
                  # [ np.array([1.0]), np.array([1.5]), np.array([1.0]) ]  
                  np.array([ 0.0, -1, 0.0 ] ) , np.array( [ 1.0, 1.5, 1.0 ] )
                #  [np.array( [[0.0]] ), np.array( [[-1, 0.0]] )],   # Inf
                #  [np.array( [[1.0]] ), np.array( [[1.5, 1.0]] ) ]  # Sup
                 )
        )
    print(popt)
    print(pcov)
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


def wrapper(x, *args): #take a list of arguments and break it down into two lists for the fit function to understand
    # print( "wrap xy", x)
    xy = x[0]
    dxy =x[1]
    
    # print( "wrap xy.shape", xy.shape)
    # print( "wrap xy", xy)
    # print( "wrap params", args)
    N = int( (len(args)+1)/3 )
    w = list(args[0:N-1])
    mux = args[1*N-1:2*N-1]
    muy = list(args[2*N-1:3*N-1])
    # Normalize w
    K = N
    """
    kSum = 0.0
    for k in range(K):
      test = kSum + w[k]
      if test >= 1.0:
        w[k] = 1.0 - kSum
        kSum = 1.0
      else:
        kSum = test
    """
    return fit_func(xy, dxy, w, mux, muy )

def packParameters( w, mu):
    K = w.size
    pack = list( w[0:-1] )
    pack.extend( list(mu[:, 0]) )
    pack.extend( list(mu[:, 1]) )
    # print("pack", pack)
    return pack
    
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
    
    plu.getColorMap()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )
    """
    xl= [0., 0., 0., -0.33, +0.33]
    yl= [0., 0.5, -0.5, 0.,0.]
    cath = np.array( [0, 0, 0, 1, 1] )
    dxl = [0.5, 0.5, 0.5, 0.33, 0.33]
    dyl = [0.25, 0.25, 0.25, 0.5, 0.5]
    """
    """
    xl= [0.5, 0.8]
    yl= [0.8, 0.5 ]
    cath = np.array( [0, 1] )
    dxl = [0.5, 0.2 ]
    dyl = [0.2, 0.5, ]
    """
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
    print(xy)
    print(dxy)
    for k in range(2):
      z += w0[k] * EM.computeDiscretizedGaussian2D( xy, dxy, mu0[k], var0[k] )
      print("k, w, mu, z", k, mu0[k,:], z)
    input("???")
    global y0Store
    y0Store = np.copy( z )
    # z = EM.computeGaussian2D(xy, mu0[0], var0[0] )
    cst = np.sum(z[0:4])
    u = z[0:4] / cst
    print("??? u", u)
    print("??? z", u * x[0][0:4 ] )
    print("??? sum", z, np.dot( u, x[0][0:4 ]) )
    zMax = np.max( z )
    plu.setLUTScale( 0, zMax )
    idx0 = np.where( cath==0 )
    idx1 = np.where( cath==1 )
    w, mu, var = EM.simpleProcessEMCluster( x[0], y[0], dx[0], dy[0], cath, z, w0, mu0, var0, cstVar=True , discretizedPDF=True )
    print( "mu", mu)
    print( "sig", np.sqrt(var) ) 

    zf = EM.computeDiscretizedGaussian2D( xy, dxy, mu[0], var[0] )
    print("xy", xy[0, :])
    print("z", z)
    print("zf", zf)
    """
        p0 = [ np.array( [[1.0]] ),       # w
              np.array( [[0.75, 0.85]] )], # mu
    """
    limits = (
                  # [ np.array([0.0]), np.array([-1.0]), np.array([0.0]) ], 
                  # [ np.array([1.0]), np.array([1.5]), np.array([1.0]) ]  
                  # Ok
                  # np.array([ 0.0, -1, 0.0 ] ) , np.array( [ 1.0, 1.5, 1.0 ] )
                  [ 0.0,     -1, -1, 0.0, 0.0 ] , [ 1.0    , 1.5, 1.5, 1.0, 1.0 ]
                  #[ 0.0, -1, 0.0, -1, 0.0 ] , [ 1.0, 1.5, 1.0, 1.5, 1.0] 
                #  [np.array( [[0.0]] ), np.array( [[-1, 0.0]] )],   # Inf
                #  [np.array( [[1.0]] ), np.array( [[1.5, 1.0]] ) ]  # Sup
                 )
    for b in limits:
      print("??? b ", b)
      # lb, ub = [ np.asarray(b, dtype=float) for b in limits]
      lb = np.asarray(b, dtype=float)
    """
    popt, pcov = curve_fit( compactFunc2, allx, z,
                    bounds= limits, p0 = [0.5, 0.5, 0.1, 0.1, 0.6, 0.6]
                 )
    """
    wi = np.array( [0.5, 0.5] )
    # p0 = [ wi, mui ]
    #p0 = [ 0.6, 0.4, mu0[0,0], mu0[1,0], mu0[0,1], mu0[1,1] ]
    p0 = packParameters( w0, mu0)
    print("??? w0, mu0", w0, mu0)
    input("next")
    popt, pcov = curve_fit(lambda allx, *p0: wrapper(allx, *p0), allx, z, p0=p0, bounds=limits) #call with lambda function

    print(popt)
    print(pcov)
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

def twoClusters():
 
    # Model
    w0, mu0, sig0, wi, mui, sigi = EM.GMModel.set2popId(verbose=True)
    mu0[0][0] = 0.8
    mu0[0][1] = 0.8
    mu0[1][0] = 0.0
    mu0[1][1] = 0.4    
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
    
    plu.getColorMap()
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )
    """
    xl= [0., 0., 0., -0.33, +0.33]
    yl= [0., 0.5, -0.5, 0.,0.]
    cath = np.array( [0, 0, 0, 1, 1] )
    dxl = [0.5, 0.5, 0.5, 0.33, 0.33]
    dyl = [0.25, 0.25, 0.25, 0.5, 0.5]
    """
    """
    xl= [0.5, 0.8]
    yl= [0.8, 0.5 ]
    cath = np.array( [0, 1] )
    dxl = [0.5, 0.2 ]
    dyl = [0.2, 0.5, ]
    """
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
      z += EM.computeDiscretizedGaussian2D( xy, dxy, mu0[k], var0[k] )
    # z = EM.computeGaussian2D(xy, mu0[0], var0[0] )
    cst = np.sum(z[0:4])
    u = z[0:4] / cst
    print("??? u", u)
    print("??? z", u * x[0][0:4 ] )
    print("??? sum", z, np.dot( u, x[0][0:4 ]) )
    zMax = np.max( z )
    plu.setLUTScale( 0, zMax )
    idx0 = np.where( cath==0 )
    idx1 = np.where( cath==1 )
    w, mu, var = EM.simpleProcessEMCluster( x[0], y[0], dx[0], dy[0], cath, z, w0, mu0, var0, cstVar=True , discretizedPDF=True )
    print( "mu", mu)
    print( "sig", np.sqrt(var) ) 

    zf = EM.computeDiscretizedGaussian2D( xy, dxy, mu[0], var[0] )
    print("xy", xy[0, :])
    print("z", z)
    print("zf", zf)
    """
        p0 = [ np.array( [[1.0]] ),       # w
              np.array( [[0.75, 0.85]] )], # mu
    """
    limits = (
                  # [ np.array([0.0]), np.array([-1.0]), np.array([0.0]) ], 
                  # [ np.array([1.0]), np.array([1.5]), np.array([1.0]) ]  
                  # Ok
                  # np.array([ 0.0, -1, 0.0 ] ) , np.array( [ 1.0, 1.5, 1.0 ] )
                  [ 0.0, 0.0, -1, -1, 0.0, 0.0 ] , [ 1.0, 1.0, 1.5, 1.5, 1.0, 1.0 ]
                  #[ 0.0, -1, 0.0, -1, 0.0 ] , [ 1.0, 1.5, 1.0, 1.5, 1.0] 
                #  [np.array( [[0.0]] ), np.array( [[-1, 0.0]] )],   # Inf
                #  [np.array( [[1.0]] ), np.array( [[1.5, 1.0]] ) ]  # Sup
                 )
    for b in limits:
      print("??? b ", b)
      # lb, ub = [ np.asarray(b, dtype=float) for b in limits]
      lb = np.asarray(b, dtype=float)
    popt, pcov = curve_fit( compactFunc2, allx, z,
                    bounds= limits, p0 = [0.5, 0.5, 0.1, 0.1, 0.6, 0.6]
                 )
    print(popt)
    print(pcov)
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

if __name__ == "__main__":
    anyClusters()
    """
    #
    # Grid 1
    Nx = 10
    Ny = 10
    deltaX = 1.0/Nx
    # deltaX = 1.0
    deltaY = 1.0 / Ny
    grid1 = EM.setGridInfo( Nx=Nx+1, Ny=Ny+1, XOrig=0.0, YOrig=0.0, dx=deltaX, dy=deltaY)
    xy1 = np.mgrid[ 0: 1.0+deltaX: deltaX, 0: 1.0+deltaY: deltaY].reshape(2,-1)
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
    """
    
    simpleProcessEMCluster( x, y, dx, dy, cathode, charge, wi, mui, vari, cstVar=False)
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
