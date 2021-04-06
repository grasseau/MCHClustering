#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 30, 2020 4:46:52 PM$"

import GaussianEM2Dv4 as EM
import numpy as np

import matplotlib.pyplot as plt
import plotUtil as plu
import scipy.optimize as opt
# from scipy.optimize import leastsq

import Mathieson

DisplayFittingInfo = False

# ??? Obsolete : var sent in fct parameters
fixedSig = np.array( [0.1, 0.1] ) 
fixedVar =  fixedSig * fixedSig

def setCstSig( sig ):
  global fixedSig
  global fixedVar 
  fixedSig = sig 
  fixedVar =  fixedSig * fixedSig
  return

def setCstVar( var ):
  global fixedSig
  global fixedVar 
  fixedVar = var
  fixedSig = np.sqrt( var)
  return

def check_func(checker, argname, thefunc, x0, args, numinputs,
                output_shape=None):
    res = np.atleast_1d(thefunc(*((x0[:numinputs],) + args)))
    if (output_shape is not None) and (shape(res) != output_shape):
        if (output_shape[0] != 1):
            if len(output_shape) > 1:
                if output_shape[1] == 1:
                    return shape(res)
            msg = "%s: there is a mismatch between the input and output " \
                  "shape of the '%s' argument" % (checker, argname)
            func_name = getattr(thefunc, '__name__', None)
            if func_name:
                msg += " '%s'." % func_name
            else:
                msg += "."
            msg += 'Shape should be %s but it is %s.' % (output_shape, np.shape(res))
            raise TypeError(msg)
    if np.issubdtype(res.dtype, np.inexact):
        dt = res.dtype
    else:
        dt = dtype(float)
    return np.shape(res), dt

def check_gradient(fcn, Dfcn, x0, args=(), col_deriv=0):
    """Perform a simple check on the gradient for correctness.
    """

    x = np.atleast_1d(x0)
    n = len(x)
    x = x.reshape((n,))
    fvec = np.atleast_1d(fcn(x, *args))
    m = len(fvec)
    fvec = fvec.reshape((m,))
    ldfjac = m
    fjac = np.atleast_1d(Dfcn(x, *args))
    fjac = fjac.reshape((m, n))
    if col_deriv == 0:
        fjac = np.transpose(fjac)

    xp = np.zeros((n,), float)
    err = np.zeros((m,), float)
    fvecp = None
    opt._minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 1, err)

    fvecp = np.atleast_1d(fcn(xp, *args))
    fvecp = fvecp.reshape((m,))
    opt._minpack._chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 2, err)

    good = (np.prod(np.greater(err, 0.5), axis=0))

    return (good, err)

def jac_fct2( p,  xy, dxy, zObs, K, cstVar):
  """
  Jac[0:n, 0:nParams]
  """
  w = np.zeros( (K))
  w[0:K-1] = np.array( p[0:K-1] ) 
  wSum = np.sum( w )
  wLast = 1.0 - wSum
  w[K-1] = wLast
  
  mu = np.array( p[ K-1:3*K-1] ).reshape( K, 2)
 
  # print( "w", w)
  # print( "mu", mu)
  
  Jac = np.zeros( ( xy.shape[1], 3*K-1))
  wPenal = 1.0 - wSum
  for k in range(K-1):
    Jac[:,k]=  EM.computeDiscretizedGaussian2D( xy, dxy, mu[k,:], cstVar[:] ) - wPenal
      
  jdx = K-1 
  for k in range(K):
    Jac[:,jdx] =  w[k] * EM.pderivateOfDiscretizedGaussian2D( "mux", xy, dxy, mu[k,:], cstVar[:] )
    Jac[:,jdx+1] = w[k] * EM.pderivateOfDiscretizedGaussian2D( "muy", xy, dxy, mu[k,:], cstVar[:] )
    jdx += 2
  #
  print("mu", mu)
  print("Jac", Jac)
  # input( "Next")

  return Jac

def err_fct2( p,  xy, dxy, zObs, K, cstVar ):
 
  w = np.zeros( (K))
  w[0:K-1] = np.array( p[0:K-1] ) 
  wLast = 1.0 - np.sum( w )
  w[K-1] = wLast
  mu = np.array( p[ K-1:3*K-1] ).reshape( K, 2)
  

  z = 0.0
  for k in range(K):
    z +=  w[k] * EM.computeDiscretizedGaussian2D( xy, dxy, mu[k,:], cstVar[:] )
    # print("k, w, mu, z", k, w[k], mu[k,:], z)  
  res = zObs -z
  wPenal = 1. - np.sum( w )
  # input("next")
  # input( "Next")
  print( "w", w)
  print( "mu", mu)
  print("res+wPenal", res+wPenal)
  print("wPenal", wPenal)
  return res + wPenal

def clusterFit( err_fct, wi, mui, vari, xy , dxy, z, jacobian=None):
  """
  w0, mu0, var0: initial guess
  xy, dxy, z : pad
  Warning : var not a parameter
  """
  K = wi.shape[0]
  # Obsolete: argument added in fct
  setCstVar(vari)
  # Test argument
  # Later ... TODO
  print("???", wi)
  # Parameter build
  p0 = np.zeros( (3*K-1))
  # w
  p0[0:K-1] = wi[0:K-1]
  # mux, mux
  kIdx = K-1
  for k in range(K):
    p0[kIdx:kIdx+2 ] = mui[k]
    kIdx += 2
  print("ClusterFit: p0", p0)
  
  # Normalisation
  zSum = np.sum( z )
  z = z / zSum
  #
  # Check function (shape)
  n = len( np.asarray(p0).flatten() )
  shape, dtype = check_func('leastsq', 'func', err_fct, p0, (xy,dxy, z, K, vari), n)
  print("ClusterFit: shape check, n", shape, n)
  # Check derivative
  if jacobian is not None:
    chk = check_gradient(err_fct, jacobian, p0, args=(xy,dxy, z, K, vari))
    print("ClusterFit: check derivates", chk) 
  
  # Fiting
  res = opt.leastsq( err_fct, x0=p0, args=(xy,dxy, z, K, vari), Dfun=jacobian, full_output=1 ) #
  print( "ClusterFit: result", res)
  param = res[0]
  # w
  wf = np.zeros( (K) )
  wf[0:K-1] = param[0:K-1]
  wf[K-1] = 1.0 - np.sum( wf )
  # mu
  muf = np.array( param[ K-1:3*K-1] ).reshape( K, 2)
  # var
  varf = vari
  return wf, muf, varf

def err_mathieson( p,  xy, dxy, zObs, cath, zCathTotalCharge, K, mathObj ):
 
  w = np.zeros( (K))
  w[0:K-1] = np.array( p[0:K-1] ) 
  wLast = 1.0 - np.sum( w )
  w[K-1] = wLast
  mu = np.array( p[ K-1:3*K-1] ).reshape( K, 2)
  cathWeight = (cath==0) * zCathTotalCharge[0] + (cath==1) * zCathTotalCharge[1]
  z = np.zeros( xy.shape[1] )
  z_k = np.zeros((K))
  for k in range(K):
    # z +=  w[k] * EM.computeDiscretizedGaussian2D( xy, dxy, mu[k,:], cstVar[:] )
    # print("k, w, mu, z", k, w[k], mu[k,:], z)  
    # xInf = mu[k,0] - xy[0] - dxy[0]  
    xInf = xy[0] - dxy[0] - mu[k,0] 
    xSup = xInf + 2.0 * dxy[0] 
    yInf = xy[1] - dxy[1] - mu[k,1]
    ySup = yInf + 2.0 * dxy[1]
    ss = cathWeight * mathObj.computeMathieson2DIntegral( xInf, xSup, yInf, ySup )
    z_k[k] = np.sum( ss )
    z += w[k] * ss
  res = zObs -z
  zCath0 = np.sum( z[cath==0] )
  zCath1 = np.sum( z[cath==1] )
  
  # wPenal = 1. - np.sum( w )
  wPenal = np.abs( 1.0 - np.sum( w ) )
  chPenal = np.abs( 1.0 - np.sum( z ) )
  # wPenal = 0.0
  # input("next")
  # input( "Next")
  """
  print( "sum z", np.sum(z))
  print( "w", w)
  print( "z_k", z_k[k])
  print("zCath0/1", zCath0, zCath1)
  print("zCathTotalCharge", zCathTotalCharge)
  """
  # print( "mu", mu)
  # print( "mux - x", xy[0] - mu[0, 0]) 
  # print( "muy - y", xy[1] - mu[0, 1]) 
  # print("zObs", zObs)
  # print("z", z)
  """
  print("res", res)
  print("wPenal", wPenal)
  """
  cathPenal = np.zeros( (2) )
  cathPenal[0] = np.abs(zCathTotalCharge[0] - zCath0) 
  cathPenal[1] = np.abs(zCathTotalCharge[1] - zCath1)
  # print("res+wPenal+cathPenal", res*(1.+cathPenal) + wPenal)
  # print("wPenal/chPenal", wPenal, chPenal)
  # print( "sum of ^2", np.dot( res,res ) )
  # return res + wPenal + chPenal
  print("wPenal", wPenal)
  print("catPenal", cathPenal)
  print("residuals :", res*(1.+cathPenal[0]*(cath==0)+cathPenal[1]*(cath==1))  + wPenal)
  return res*(1.+cathPenal[0]+cathPenal[1]) + wPenal
  #return res*(1.+cathPenal[0]*(cath==0)+cathPenal[1]*(cath==1))  + wPenal

def err_mathieson0( p,  xy, dxy, zObs, K, mathObj ):
 
  w = np.zeros( (K))
  w[0:K-1] = np.array( p[0:K-1] ) 
  wLast = 1.0 - np.sum( w )
  w[K-1] = wLast
  mu = np.array( p[ K-1:3*K-1] ).reshape( K, 2)
  

  z = np.zeros( xy.shape[1] )
  z_k = np.zeros((K))
  for k in range(K):
    # z +=  w[k] * EM.computeDiscretizedGaussian2D( xy, dxy, mu[k,:], cstVar[:] )
    # print("k, w, mu, z", k, w[k], mu[k,:], z)  
    # xInf = mu[k,0] - xy[0] - dxy[0]  
    xInf = xy[0] - dxy[0] - mu[k,0] 
    xSup = xInf + 2.0 * dxy[0] 
    yInf = xy[1] - dxy[1] - mu[k,1]
    ySup = yInf + 2.0 * dxy[1]
    ss = mathObj.integralWeight * mathObj.computeMathieson2DIntegral( xInf, xSup, yInf, ySup )
    z_k[k] = np.sum( ss )
    z += w[k] * ss
  res = zObs -z
  # wPenal = 1. - np.sum( w )
  wPenal = np.abs( 1.0 - np.sum( w ) )
  chPenal = np.abs( 1.0 - np.sum( z ) )
  # wPenal = 0.0
  # input("next")
  # input( "Next")
  print( "sum z", np.sum(z))
  print( "w", w)
  print( "z_k", z_k[k])
  # print( "mu", mu)
  # print( "mux - x", xy[0] - mu[0, 0]) 
  # print( "muy - y", xy[1] - mu[0, 1]) 
  # print("zObs", zObs)
  # print("z", z)
  print("res", res)
  print("wPenal", wPenal)
  print("res+wPenal+chPenal", res+wPenal+chPenal)
  # print("wPenal/chPenal", wPenal, chPenal)
  # print( "sum of ^2", np.dot( res,res ) )
  # return res + wPenal + chPenal
  return res + wPenal 

def clusterFitMath( wi, mui, mType, xy , dxy, z, cath, jacobian=None):
  """
  w0, mu0, var0: initial guess
  xy, dxy, z : pad
  Warning : var not a parameter
  """
  K = wi.shape[0]
  # Test argument
  # Later ... TODO
  # Parameter build
  p0 = np.zeros( (3*K-1))
  # w
  p0[0:K-1] = wi[0:K-1]
  # mux, mux
  kIdx = K-1
  for k in range(K):
    p0[kIdx:kIdx+2 ] = mui[k]
    kIdx += 2
  print("ClusterFit: p0", p0)
  
  # Not used
  # Normalisation
  # zSum = np.sum( z )
  # z = z / zSum
  # Set the parameter set
  # chWeight = 1.0
  # if nbrOfCath == 2:
  #  chWeight = 0.5
  # Not used
  matObj = Mathieson.Mathieson(mType, 0.5)
  #
  # Cathodes contributions
  zCath = [ np.sum( z[cath==0] ), np.sum( z[cath==1] ) ]

  #
  # Check function (shape)
  n = len( np.asarray(p0).flatten() )
  shape, dtype = check_func('leastsq', 'func', err_mathieson, p0, (xy,dxy, z, cath, zCath, K, matObj), n)
  # print("ClusterFit: shape check, n", shape, n)
  if shape[0] < n:
    input("Fit: Nbre parameters > nbr of data")
    vari = np.ones( (K,2) ) * 0.1
    return wi, mui, vari
  # Check derivative
  if jacobian is not None:
    chk = check_gradient(err_fct, jacobian, p0, args=(xy,dxy, z, K))
    print("ClusterFit: check derivates", chk) 
  
  # Fiting
  res = opt.leastsq( err_mathieson, x0=p0, args=(xy,dxy, z, cath, zCath, K, matObj), Dfun=None, full_output=1 ) #
  if DisplayFittingInfo : print( "ClusterFit: result", res)
  param = res[0]
  # w
  wf = np.zeros( (K) )
  wf[0:K-1] = param[0:K-1]
  wf[K-1] = 1.0 - np.sum( wf )
  # mu
  muf = np.array( param[ K-1:3*K-1] ).reshape( K, 2)
  # var
  # varf = vari
  varf = np.ones( (K,2) ) * 0.1
  return wf, muf, varf

def clusterFitMath0( wi, mui, mType, xy , dxy, z, nbrOfCath, jacobian=None):
  """
  w0, mu0, var0: initial guess
  xy, dxy, z : pad
  Warning : var not a parameter
  """
  K = wi.shape[0]
  # Test argument
  # Later ... TODO
  # Parameter build
  p0 = np.zeros( (3*K-1))
  # w
  p0[0:K-1] = wi[0:K-1]
  # mux, mux
  kIdx = K-1
  for k in range(K):
    p0[kIdx:kIdx+2 ] = mui[k]
    kIdx += 2
  print("ClusterFit: p0", p0)
  
  # Normalisation
  zSum = np.sum( z )
  z = z / zSum
  # Set the parameter set
  chWeight = 1.0
  if nbrOfCath == 2:
    chWeight = 0.5
  matObj = Mathieson.Mathieson(mType, chWeight)
  #
  # Check function (shape)
  n = len( np.asarray(p0).flatten() )
  shape, dtype = check_func('leastsq', 'func', err_mathieson0, p0, (xy,dxy, z, K, matObj), n)
  # print("ClusterFit: shape check, n", shape, n)
  if shape[0] < n:
    input("Fit: Nbre parameters > nbr of data")
    vari = np.ones( (K,2) ) * 0.1
    return wi, mui, vari
  # Check derivative
  if jacobian is not None:
    chk = check_gradient(err_fct, jacobian, p0, args=(xy,dxy, z, K))
    print("ClusterFit: check derivates", chk) 
  
  # Fiting
  res = opt.leastsq( err_mathieson, x0=p0, args=(xy,dxy, z, K, matObj), Dfun=None, full_output=1 ) #
  if DisplayFittingInfo : print( "ClusterFit: result", res)
  param = res[0]
  # w
  wf = np.zeros( (K) )
  wf[0:K-1] = param[0:K-1]
  wf[K-1] = 1.0 - np.sum( wf )
  # mu
  muf = np.array( param[ K-1:3*K-1] ).reshape( K, 2)
  # var
  # varf = vari
  varf = np.ones( (K,2) ) * 0.1
  return wf, muf, varf

if __name__ == "__main__":
    print("Hello !")
  