#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt
import math
import TrashCan.Mathieson as mat

import C.PyCWrapper as PCWrap
import Util.plot as uPlt
import Util.dataTools as tUtil

def vErf(x):
    y = np.zeros( x.shape )
    for i in range( x.shape[0]):
        y[i] = math.erf( x[i] )
    return y

def computeGaussian1D( x,  mu=0.0, var=1.0):
    # print ( "???", __name__, x[-1], x.shape )
    # print "x", x
    TwoPi = 2 * np.pi
    SqrtTwoPi = np.sqrt( TwoPi )
    sig = np.sqrt(var)
    u = (x - mu) / sig
    # print "u", u
    u = - 0.5 * u * u
    cst = 1.0 / ( sig * SqrtTwoPi)
    y = cst * np.exp( u )
    return y

def gaussianIntegral(x): 
    mu0 = 0.0
    var0 = 1.0
    sig0 = np.sqrt( var0 )
    cstx = 1.0 / ( np.sqrt(2.0)*sig0 )
    integral = vErf( (x - mu0) * cstx )
    return integral

    
class TabulatedChargeIntegration:

  # Spline implementation of the book "Numerical Analysis" - 9th edition
  # Richard L Burden, J Douglas Faires
  # Section 3.5, p. 146
  # Restrictions : planed with a regular sampling (dx = cst)
  # spline(x) :[-inf, +inf] -> [-1/2, +1/2]
  # Error < 7.0 e-11 for 1001 sampling between [0, 3.0]
  def __init__(self, x, f, dx, lDerivate, rDerivate ):
    self.nTabulations = x.size
    N = x.size
    self.a = np.copy( f )
    self.b = np.zeros(N)
    self.c = np.zeros(N)
    self.d = np.zeros(N)
    self.dx = dx

    # Step 1 
    # for (i = 0; i < n - 1; ++i) h[i] = x[i + 1] - x[i];
    #for i in range(0, N-1):
    #  self.h[i] = x[i+1] - x[i];
    # h = x[0:N-1] = x[1:N] - x[0:N-1]
    h = self.dx

    # Step 2 
    # for (i = 1; i < n-1; ++i)
    #    A[i] = 3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1];
    # A[1:N-1] = 3 * (a[2:N] - a[1:N-1]) / h[1:N-1]  -  3 * (a[1:N-1] - a[0:N-2]) / h[0:N-2]];

    # Step 2 & 3 
    alpha = np.zeros(N)
    # alpha[0] = 3.0 / self.h[0] * (f[1] - f[0]) - 3*lDerivate
    alpha[0] = 3.0 / h * (f[1] - f[0]) - 3*lDerivate
    # alpha[N-1] =  3*rDerivate - 3.0 / self.h[N-2] * (f[N-1] - f[N-2])
    alpha[N-1] =  3*rDerivate - 3.0 / h * (f[N-1] - f[N-2])
    # for (i = 1; i < n-1; ++i)
    for i in range(1, N-1):
      # alpha[i] = 3.0/self.h[i] * (f[i+1] - f[i]) - 3.0/self.h[i-1] * (f[i] - f[i-1]);
      alpha[i] = 3.0/h * (f[i+1] - f[i]) - 3.0/h * (f[i] - f[i-1]);
    
    # Step 4 to 6 solve a tridiagonal linear system

    # Step 4
    l = np.zeros(N)
    mu = np.zeros(N)
    z = np.zeros(N)
    # l[0] = 2 * self.h[0]
    l[0] = 2 * h
    mu[0] = 0.5
    z[0] = alpha[0] / l[0]
    
    # Step 5
    # for (i = 1; i < n - 1; ++i) {
    for i in range(1, N-1):
      # l[i] = 2 * (x[i+1] - x[i-1]) - self.h[i-1] * mu[i - 1];
      # mu[i] = self.h[i] / l[i];
      # z[i] = (alpha[i] - self.h[i-1]*z[i-1]) / l[i];
      l[i] = 2 * (x[i+1] - x[i-1]) - h * mu[i-1];
      mu[i] = h / l[i];
      z[i] = (alpha[i] - h*z[i-1]) / l[i];
    
    # Step 6 & 7
    # l[N-1] = self.h[N-2]*(2.0-mu[N-2])
    # z[N-1] = (alpha[N-1] - self.h[N-2]*z[N-2]) / l[N-1]
    l[N-1] = h*(2.0-mu[N-2])
    z[N-1] = (alpha[N-1] - h*z[N-2]) / l[N-1]

    self.c[N-1] = z[N-1]
    # for (j = n - 2; j >= 0; --j) {
    for j in range(N-2, -1, -1):
      self.c[j] = z[j] - mu[j] * self.c[j+1]
      # self.b[j] = (f[j+1]-f[j]) / self.h[j] - self.h[j]/3.0 * (self.c[j+1] + 2*self.c[j]) 
      # self.d[j] = (self.c[j+1]-self.c[j]) / (3 * self.h[j])
      self.b[j] = (f[j+1]-f[j]) / h - h/3.0 * (self.c[j+1] + 2*self.c[j]) 
      self.d[j] = (self.c[j+1]-self.c[j]) / (3 * h)
    
  def splineAtanTanh( self, x ):
      a = self.a
      b = self.b
      c = self.c
      d = self.d
      N = self.nTabulations
      
      signX = np.where( x >= 0, 1.0, -1.0 )
      # unsigned x
      uX = x * signX
      # 0.49999999724624 
      # 0.499999996965014 point precedent f0(2OO-1)
      # 0.49999999724624 f0(200)
      # 0.499999997245073 f(200-1)
      # 0.499999997232819 y[200-1]
      # 0.49999999748923 y[200]
      # 0.       890
      # 0.49999999724624 f0(200)
      np.set_printoptions(precision=15)
      # print("???  x / self.dx", x / self.dx)
      cst = 1.0 / self.dx
      u = np.trunc( uX * cst + self.dx*0.1)
      # idx = u.astype(np.int)
      idx = np.int32(u)
      # print("??? idx ", idx)
      idx = np.where( idx >= N, N-1, idx)
      h = np.where( idx < N-1, uX - idx * self.dx, 0)
      # h = x - idx * self.dx
      # print("??? idx filter large indexes", idx)
      
      print ("uX ",  uX)
      print ("h ",  h)
      print ("f(x0) ",  a[idx])
      print ("df|dx0",  h*( b[idx] + h*( c[idx] + h *(d[idx]))))
      print ("f, ",  a[idx] + h*( b[idx] + h*( c[idx] + h *(d[idx]))))
      f = signX * (a[idx] + h*( b[idx] + h*( c[idx] + h *(d[idx]))))
      return f
    
if __name__ == "__main__":
    
    #pcWrap = PCWrap.setupPyCWrapper()
    #pcWrap.initMathieson()

    xPrecision = 1.0e-3
    xLimit = 3.0
    N = int(xLimit / xPrecision) + 1
    x = np.linspace(0.0, xLimit, N)
    dxVar = x[1:] - x[0:-1]
    print("Verify sampling N, xPrecision, dxMin, dxMax", N, xPrecision, np.min(dxVar), np.max(dxVar))
    dx = xPrecision
    mat0 = mat.Mathieson( 0, 1.0 )
    leftDerivate = 2.0 * mat0.curK4x * mat0.curSqrtK3x * mat0.curK2x * mat0.curInvPitch
    print("leftDerivate", leftDerivate)
    # leftDerivate = 2.77
    y = mat0.computeAtanTanh( x)
    tf = TabulatedChargeIntegration(x, y, dx, leftDerivate, 0.0)


    """
    m = int( N/2 )
    print("N", N, x.size )
    print("x ", x[0], x[1], x[2], '...', x[m-1], x[m], x[m+1], "...", x[-3], x[-2], x[-1] )
    print("\n")
    print("maxErr", np.max(np.abs(f-y)) )
    """

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    # Spline at sampling points
    f = tf.splineAtanTanh(x)
    ax[0,0].plot( x, y)
    ax[0,0].scatter( x, f, marker='x', color="red")
    # , markersize=4)
    ax[0,0].set_ylabel( "atan(tanh(x0)) and spline(x0) [in red]")   
    #
    ax[0,1].scatter( x, f-y, marker='x')
    ax[0,1].set_ylabel( "atan(tanh(x0)) - spline(x0)")    

    
    # Far away points
    print("--------------------")
    x1 = x + 0.0095
    y1 = mat0.computeAtanTanh( x1)
    f1 = tf.splineAtanTanh(x1) 
    print("y1", y1)
    ax[1,0].scatter( x1, f1-y1, marker='x')
    ax[1,0].set_ylabel( "atan(tanh(x1)) - spline(x1)")
    print("--------------------")
    # RND
    xrnd = (np.random.ranf(20*N) * 2  - 1.0) * (xLimit + 1.0)
    frnd = tf.splineAtanTanh(xrnd)
    yrnd = mat0.computeAtanTanh(xrnd)
    #
    ax[1,1].scatter( xrnd, frnd-yrnd, marker='x')
    ax[1,1].set_ylabel( "atan(tanh(rnd)) - spline(rnd)")
    # relative error
    # ax[1,1].scatter( x1[1:], (f1[1:]-y1[1:]) / y1[1:] )
    #
    print("maxErr f1", np.max(np.abs(f1-y1)) )
    print( "convergence last point y, dy ",  y1[-1], np.max(np.abs(f1[-1]-y1[-1])))
    np.set_printoptions(precision=15)
    print( "f(x) x=[0, ..,9]", mat0.computeAtanTanh( np.arange(10.0)) - 0.5 )
    
    print("FIRST POINT")
    tf.splineAtanTanh( np.array([0.0]) )
    print("Function",  mat0.computeAtanTanh( np.array([0.0])) )
    print("Last POINT")
    tf.splineAtanTanh( np.array([2.0]) )
    print("Function",  mat0.computeAtanTanh( np.array([2.0])) )
    print("Outer POINT")
    tf.splineAtanTanh( np.array([15.0]) )    
    print("Function",  mat0.computeAtanTanh( np.array([15.0])) )
    
    xx = np.arange(6.0)
    print("xx",  xx )
    print("f(xx) - 0.5",  mat0.computeAtanTanh( xx ) - 0.5)
    
    plt.show()
