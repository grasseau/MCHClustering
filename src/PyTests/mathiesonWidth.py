#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib.pyplot as plt
import math
import TrashCan.Mathieson as mat
"""
import C.PyCWrapper as PCWrap
import Util.plot as uPlt
import Util.dataTools as tUtil
"""


    
if __name__ == "__main__":
    
    #pcWrap = PCWrap.setupPyCWrapper()
    #pcWrap.initMathieson()
    """
    xPrecision = 1.0e-3
    xLimit = 3.0
    N = int(xLimit / xPrecision) + 1
    x = np.linspace(0.0, xLimit, N)
    dxVar = x[1:] - x[0:-1]
    print("Verify sampling N, xPrecision, dxMin, dxMax", N, xPrecision, np.min(dxVar), np.max(dxVar))
    dx = xPrecision
    """
    mat0 = mat.Mathieson( 0, 1.0 )
    x = np.arange(-0.15, 0.20, 0.05)
    print("x/y", x )
    print("chId 1..2")
    prim = mat0.primitive( x, axe=0 )
    print("  xPrim ", prim )
    prim = mat0.primitive( x, axe=1 )
    print("  yPrim", prim )
    x0 = np.array( [0.0] )
    fx = mat0.mathieson( x0, axe=0)
    fy = mat0.mathieson( x0, axe=1)
    print("mathieson max x/y:", fx, ", ", fy)

    mat1 = mat.Mathieson( 1, 1.0 )
    print("chId 3..10")
    prim = mat1.primitive( x, axe=0 )
    print("  xPrim", prim )
    prim = mat1.primitive( x, axe=1 )
    print("  yPrim", prim )
    fx = mat1.mathieson( x0, axe=0)
    fy = mat1.mathieson( x0, axe=1)
    print("mathieson max x/y:", fx, ", ", fy)

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
"""