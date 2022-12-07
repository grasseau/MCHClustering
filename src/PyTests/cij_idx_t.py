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

def getI( x, xMin, dxMin ):
  u = int( (x - xMin) / dxMin + 0.5)
  return u

if __name__ == "__main__":
  x = np.arange(2)/2 + 0.25

  b  = np.arange(7)/7 + 1.0/7.0
  db = np.ones(7)*0.5/7.0
    
  x  = np.array([0.25, 0.75, 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286])
  dx = np.array([0.25, 0.25, 0.07142857, 0.07142857, 0.07142857, 0.07142857, 0.07142857, 0.07142857])
       
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
