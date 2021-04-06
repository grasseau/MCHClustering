#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 20, 2020 7:52:53 AM$"

import sys, traceback
import operator
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

import IO
import plotUtil as plu
import GaussianEM2Dv4 as EM

SimplePrecision = 5.0e-5
    
def processEMCluster( clusters, i ):
    discretizedPDF = False
    em = EM.EM2D( discretizedPDF )
    # ???
    grids = []
    grid0 = EM.Grid2D( 3, 3)
    grids.append( grid0 )
    grid1 = EM.Grid2D( 3, 3)
    grids.append( grid1 )

    print("########### Cluster i=", i)
    
    if ( clusters.x[i].shape[0] == 0): return ([],[],[])
    x = clusters.x[i]
    y = clusters.y[i]
    dx = clusters.dx[i]
    dy = clusters.dy[i]
    #
    # Check if the undelying grid is regular
    #
    cathIdx0 = np.where( clusters.cathode[i] ==0 )
    x0 = x[cathIdx0]
    y0 = y[cathIdx0]
    dx0 =dx[cathIdx0]
    dy0 =dy[cathIdx0]
    cathIdx1 = np.logical_not( cathIdx0 )
    x1 = x[cathIdx1]
    y1 = y[cathIdx1]
    dx1 =dx[cathIdx1]
    dy1 =dy[cathIdx1]
    x0, y0, dx0, dy0, x1, y1, dx1, dy1 = computeOverlapingPads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 )
    
    c0 = clusters.charge[i][cathIdx]
    c0 = c0 / (4*dx0*dy0)
    l0 = x0.shape[0]
    if l0 !=0:
      checkRegularGrid( x0, dx0 )
      checkRegularGrid( y0, dy0 )
    cathIdx = np.where( clusters.cathode[i] == 1 )
    x1 = x[cathIdx]
    y1 = y[cathIdx]
    dx1 =dx[cathIdx]
    dy1 =dy[cathIdx]
    c1 = clusters.charge[i][cathIdx]
    c1 = c1 / (4*dx1*dy1)
    l1 = x1.shape[0]
    if l1 !=0:
      checkRegularGrid( x1, dx[cathIdx] )
      checkRegularGrid( y1, dy[cathIdx] )
    xm, xsig = plu.getBarycenter( x0, dx0, x1, dx1, c0, c1)
    ym, ysig = plu.getBarycenter( y0, dy0, y1, dy1, c0, c1)
    print( "Barycenter [x, y, sigx, sigy]", xm, ym, xsig, ysig )
    # Normalization
    ds = dx0 * dy0 * 4
    c0 = c0 / ds
    s = np.sum( c0)
    c0 = c0 /s
    ds = dx1 * dy1 * 4
    c1 = c1 / ds
    s = np.sum( c1)
    c1 = c1 /s
    xy = []; dxy = []; z = []
    print "xy shape:", np.array( [x1, y1]).shape
    xy.append( np.array( [x0, y0] ) )
    xy.append( np.array( [x1, y1]) )
    dxy.append( np.array( [dx0, dy0]) )
    dxy.append( np.array( [dx1, dy1] ) )
    z.append( np.array( c0 ) )
    z.append( np.array( c1) )
    for g in range(2):
      grids[g].setGridValues( xy[g], dxy[g], z[g] )
    wi = np.array( [1.0], EM.InternalType )
    mui = np.array( [[xm,ym]], EM.InternalType )
    vari = np.array( [[xsig*xsig, ysig*ysig]], EM.InternalType )
    (wf, muf, varf) = em.weightedEMLoopOnGrids( grids,  wi, mui, vari, dataCompletion=False, plotEvery=0)
    print "sum charge each cathodes", np.sum(c0), np.sum(c1) 
    print "Cathodes charge input", clusters.cFeatures["ChargeCath0"][i], clusters.cFeatures["ChargeCath1"][i]
    print "Clustering/Fitting X, Y ", clusters.cFeatures["X"][i], clusters.cFeatures["Y"][i]
    print( "Barycenter [x, y, sigx, sigy]", xm, ym, xsig, ysig )
    return wf, muf, varf, xm, xsig, ym, ysig

def simpleProcessEMCluster( cl, wi, mui, vari ):
    discretizedPDF = False
    em = EM.EM2D( discretizedPDF )
    # ???
    grids = []
    grid0 = EM.Grid2D( 3, 3)
    grids.append( grid0 )
    grid1 = EM.Grid2D( 3, 3)
    grids.append( grid1 )
    
    if ( cl.x.shape[0] == 0): return ([],[],[])
    x = cl.x
    y = cl.y
    dx = cl.dx
    dy = cl.dy
    #
    # Check if the undelying grid is regular
    #
    cathIdx0 = np.where( cl.cathode ==0 )
    x0 = x[cathIdx0]
    y0 = y[cathIdx0]
    dx0 =dx[cathIdx0]
    dy0 =dy[cathIdx0]
    cathIdx1 = np.where( cl.cathode == 1 )
    x1 = x[cathIdx1]
    y1 = y[cathIdx1]
    dx1 =dx[cathIdx1]
    dy1 =dy[cathIdx1]
    # x0, y0, dx0, dy0, x1, y1, dx1, dy1 = computeOverlapingPads( x0, y0, dx0, dy0, x1, y1, dx1, dy1 )
    """
    c0 = cl.charge[cathIdx0]
    c0 = c0 / (4*dx0*dy0)
    l0 = x0.shape[0]
    """
    """
    if l0 !=0:
      checkRegularGrid( x0, dx0 )
      checkRegularGrid( y0, dy0 )
    """
    """
    cathIdx = np.where( cl.cathode == 1 )
    x1 = x[cathIdx]
    y1 = y[cathIdx]
    dx1 =dx[cathIdx]
    dy1 =dy[cathIdx]
    c1 = cl.charge[cathIdx]
    c1 = c1 / (4*dx1*dy1)
    l1 = x1.shape[0]
    """
    """
    if l1 !=0:
      checkRegularGrid( x1, dx[cathIdx] )
      checkRegularGrid( y1, dy[cathIdx] )
    """
    """
    xm, xsig = plu.getBarycenter( x0, dx0, x1, dx1, c0, c1)
    ym, ysig = plu.getBarycenter( y0, dy0, y1, dy1, c0, c1)
    print( "Barycenter [x, y, sigx, sigy]", xm, ym, xsig, ysig )
    """
    # Normalization
    # ds = dx0 * dy0 * 4
    # c0 = c0 / ds
    s = np.sum( cl.charge[cathIdx0])
    c0 = cl.charge[cathIdx0] / s
    # ds = dx1 * dy1 * 4
    # c1 = c1 / ds
    s = np.sum( cl.charge[cathIdx1])
    c1 = cl.charge[cathIdx1] / s
    xy = []; dxy = []; z = []
    xy.append( np.array( [x0, y0] ) )
    xy.append( np.array( [x1, y1]) )
    dxy.append( np.array( [dx0, dy0]) )
    dxy.append( np.array( [dx1, dy1] ) )
    z.append( np.array( c0 ) )
    z.append( np.array( c1) )
    for g in range(2):
      grids[g].setGridValues( xy[g], dxy[g], z[g] )
    """
    wi = np.array( [1.0], EM.InternalType )
    mui = np.array( [[xm,ym]], EM.InternalType )
    vari = np.array( [[xsig*xsig, ysig*ysig]], EM.InternalType )
    """
    # (wf, muf, varf) = em.weightedEMLoopOnGrids( grids,  wi, mui, vari, dataCompletion=False, plotEvery=0)
    (wf, muf, varf) = em.weightedEMLoopOnGrids( grids,  wi, mui, vari, dataCompletion=False, cstVar=True, plotEvery=0)
    print "sum charge each cathodes", np.sum(c0), np.sum(c1) 
    return wf, muf, varf

def drawOneCluster( ax, cluster, mode="all", title=["Cathode 0", "Cathode 1", "Both"], noYTicksLabels=True):
        if title == None :
            title = ["", "", ""]
        # Graph limits
        xSup = np.max( cluster.x + cluster.dx )
        ySup = np.max( cluster.y + cluster.dy )
        xInf = np.min( cluster.x - cluster.dx )
        yInf = np.min( cluster.y - cluster.dy )
        #
        minCh =  np.min( cluster.charge )
        maxCh =  np.max( cluster.charge )
        # Set Lut scale
        plu.setLUTScale( 0, maxCh)
        #
        if mode == "all":
            c0Idx = np.where( cluster.cathode == 0)
            c1Idx = np.where( cluster.cathode == 1)
            plu.drawPads(ax[0],  cluster.x[c0Idx], cluster.y[c0Idx], cluster.dx[c0Idx], cluster.dy[c0Idx], cluster.charge[c0Idx],
                        title=title[0], alpha=1.0, noYTicksLabels=False, doLimits=False)
            plu.drawPads(ax[1],  cluster.x[c1Idx], cluster.y[c1Idx], cluster.dx[c1Idx], cluster.dy[c1Idx], cluster.charge[c1Idx],
                        title=title[1], alpha=1.0, noYTicksLabels=True, doLimits=False)
            plu.drawPads(ax[2],  cluster.x[c0Idx], cluster.y[c0Idx], cluster.dx[c0Idx], cluster.dy[c0Idx], cluster.charge[c0Idx],
                        title="", alpha=1.0, doLimits=False)
            plu.drawPads(ax[2],  cluster.x[c1Idx], cluster.y[c1Idx], cluster.dx[c1Idx], cluster.dy[c1Idx], cluster.charge[c1Idx],
                        title=title[2], alpha=0.5, noYTicksLabels=True, doLimits=False)
        elif mode == "superposition":
            c0Idx = np.where( cluster.cathode == 0)
            c1Idx = np.where( cluster.cathode == 1)
            plu.drawPads(ax[0],  cluster.x[c0Idx], cluster.y[c0Idx], cluster.dx[c0Idx], cluster.dy[c0Idx], cluster.charge[c0Idx],
                        title="", alpha=1.0, doLimits=False, noYTicksLabels=noYTicksLabels)
            plu.drawPads(ax[0],  cluster.x[c1Idx], cluster.y[c1Idx], cluster.dx[c1Idx], cluster.dy[c1Idx], cluster.charge[c1Idx],
                        title=title[0], alpha=0.5, doLimits=False, noYTicksLabels=noYTicksLabels)
                        
        return
    
def drawOneCSCluster( ax, csClusters, cID, mode="all", title=["Cathode 0", "Cathode 1", "Both"]):
        if title == None :
            title = ["", "", ""]
        # Graph limits
        xSup = np.max( csClusters.x[cID] + csClusters.dx[cID] )
        ySup = np.max( csClusters.y[cID] + csClusters.dy[cID] )
        xInf = np.min( csClusters.x[cID] - csClusters.dx[cID] )
        yInf = np.min( csClusters.y[cID] - csClusters.dy[cID] )
        #
        minCh =  np.min( csClusters.charge[cID] )
        maxCh =  np.max( csClusters.charge[cID] )
        # Set Lut scale
        plu.setLUTScale( 0, maxCh)
        #
        c0Idx = np.where( csClusters.cathode[cID] == 0)
        c1Idx = np.where( csClusters.cathode[cID] == 1)
        x0 = csClusters.x[cID][c0Idx]
        x1 = csClusters.x[cID][c1Idx]
        dx0 = csClusters.dx[cID][c0Idx]
        dx1 = csClusters.dx[cID][c1Idx]
        y0 = csClusters.y[cID][c0Idx]
        y1 = csClusters.y[cID][c1Idx]
        dy0 = csClusters.dy[cID][c0Idx]
        dy1 = csClusters.dy[cID][c1Idx]
        ch0 = csClusters.charge[cID][c0Idx]
        ch1 = csClusters.charge[cID][c1Idx]
        if mode == "all":
            plu.drawPads(ax[0], x0, y0, dx0, dy0, ch0,
                        title=title[0], alpha=1.0, noYTicksLabels=False, doLimits=False)
            plu.drawPads(ax[1], x1, y1, dx1, dy1, ch1,
                        title=title[1], alpha=1.0, noYTicksLabels=True, doLimits=False)
            plu.drawPads(ax[2], x0, y0, dx0, dy0, ch0,
                        title="", alpha=1.0, doLimits=False)
            plu.drawPads(ax[2], x1, y1, dx1, dy1, ch1,
                        title=title[2], alpha=0.5, noYTicksLabels=True, doLimits=False)
        elif mode == "superposition":
            plu.drawPads(ax[0], x0, y0, dx0, dy0, ch0,
                        title="", alpha=1.0, doLimits=False)
            plu.drawPads(ax[0], x1, y1, dx1, dy1, ch1,
                        title=title[0], alpha=0.5, noYTicksLabels=True, doLimits=False)
        return
    
def displayOneCluster( ax, tracksObj, tID, cID):
    colors = plu.getColorMap()
    cl = tracksObj.tracks[tID][cID]
    drawOneCluster( ax[0:3], cl, mode="all", title=None )
    MCTracks = tracksObj.MCTracks
    c0Idx = np.where( cl.cathode == 0)
    c1Idx = np.where( cl.cathode == 1)
    nocolor = np.ones( cl.x.shape )
    plu.drawPads(ax[3],  cl.x[c0Idx], cl.y[c0Idx], cl.dx[c0Idx], cl.dy[c0Idx], nocolor[c0Idx],
                title="", alpha=0.5, noYTicksLabels=True, doLimits=False )
    plu.drawPads(ax[3],  cl.x[c1Idx], cl.y[c1Idx], cl.dx[c1Idx], cl.dy[c1Idx], nocolor[c1Idx],
                title="", alpha=0.5, doLimits=False)


    # EM loop

    wf, muf, varf, xm, xsig, ym, ysig = EM.processEMClusterV4( 
                            cl.x[c0Idx], cl.y[c0Idx], cl.dx[c0Idx], cl.dy[c0Idx], cl.charge[c0Idx],
                            cl.x[c1Idx], cl.y[c1Idx], cl.dx[c1Idx], cl.dy[c1Idx], cl.charge[c1Idx],
                            cID )
    # Barycenter
    w = np.array([1.0])
    mu = np.array( [ [xm, ym] ] )
    var = np.array([[xsig, ysig]])
    plu.drawModelComponents( ax[2], w, mu, var, color='black', pattern="rect" )
    # ax[ky,2].plot( muf[0][0], muf[0][1], "o", color="black")
    # ax[ky,2].plot( xm, ym, "x", color="blue")
    #
    # EM
    plu.drawModelComponents( ax[2], wf, muf, varf, color='black', pattern="cross" )

    #
    # Overlapping
    #

    x0, y0 , dx0, dy0, ch0 = plu.computeOverlapingPads(  cl.x[c0Idx], cl.y[c0Idx], cl.dx[c0Idx], cl.dy[c0Idx], cl.charge[c0Idx],
                            cl.x[c1Idx], cl.y[c1Idx], cl.dx[c1Idx], cl.dy[c1Idx], cl.charge[c1Idx] )
    plu.drawPads( ax[4], x0, y0, dx0, dy0, ch0,
                title="", alpha=1.0, noYTicksLabels=True, doLimits=False)
    plu.drawPads( ax[6], x0, y0, dx0, dy0, ch0,
                title="", alpha=0.5, noYTicksLabels=True, doLimits=False)

    x1, y1, dx1, dy1, ch1 = plu.computeOverlapingPads(  cl.x[c1Idx], cl.y[c1Idx], cl.dx[c1Idx], cl.dy[c1Idx], cl.charge[c1Idx],
                            cl.x[c0Idx], cl.y[c0Idx], cl.dx[c0Idx], cl.dy[c0Idx], cl.charge[c0Idx] )
    plu.drawPads( ax[5], x1, y1, dx1, dy1, ch1,
                title="", alpha=1.0, noYTicksLabels=True, doLimits=False)
    plu.drawPads( ax[6], x1, y1, dx1, dy1, ch1,
                title="", alpha=0.5, noYTicksLabels=True, doLimits=False)

    # Draw MC clusters
    # ???
    print "track t=", tID, len( MCTracks ), MCTracks[tID].x.shape
    mc = MCTracks[tID]
    mu = np.array( [[mc.x[cID], mc.y[cID]]] )
    var = np.array([[0.0, 0.0]])
    plu.drawModelComponents( ax[6], w, mu, var, color='black', pattern="o" )
    plu.drawModelComponents( ax[2], w, mu, var, color='black', pattern="o" )

    # EM loop
    wf, muf, varf, xm, xsig, ym, ysig = EM.processEMClusterV4( 
        x0, y0, dx0, dy0, ch0, 
        x1, y1, dx1, dy1, ch1, 
        cID )
    # Barycenter
    mu = np.array( [ [xm, ym] ] )
    var = np.array([[xsig, ysig]])
    plu.drawModelComponents( ax[6], w, mu, var, color='black', pattern="rect" )
    # ax[ky,2].plot( muf[0][0], muf[0][1], "o", color="black")
    # ax[ky,2].plot( xm, ym, "x", color="blue")
    # EM
    plu.drawModelComponents( ax[6], wf, muf, varf, color='black', pattern="cross" )

    # Graph limits
    xSup = np.max( cl.x + cl.dx )
    ySup = np.max( cl.y + cl.dy )
    xInf = np.min( cl.x - cl.dx )
    yInf = np.min( cl.y - cl.dy )
    print cl.x
    print "xSup,ySup, xInf, yInf :", xSup,ySup, xInf, yInf 
    for kx in range( len(ax) -1):
      ax[kx].set_xlim( xInf,  xSup )
      ax[kx].set_ylim( yInf,  ySup)
    plu.displayLUT( ax[-1], colors )
    xMargin = 0.25
    fontSize = 10
    (ev, trackID, partCode, clusterID, nPads) = tracksObj.getClusterInfos( tID, cID ) 
    text = "Ev. " + str(ev)
    ax[-1].text( xMargin, 0.8, text, fontsize=fontSize )
    text = "Track:" + str(trackID)
    ax[-1].text( xMargin, 0.65, text, fontsize=fontSize )
    text = "Cluster:" + str(clusterID)
    ax[-1].text( xMargin, 0.5, text, fontsize=fontSize )
    return

def displayTracks( tracks ):
    #
    nTracks = len( tracks.tracks )
    nFigCol = 8
    nFigRow = 4
    nFig = nFigCol * nFigRow

    # Color Map
    # ???
    cID = 0   
    for t in range(nTracks):
        nClusters = len(tracks.tracks[t])
        clusters = tracks.tracks[t]
        N = int( np.ceil( float(len(clusters)) / nFigRow ) )
        cID = 0
        for k in range(N):
          fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
          fig.suptitle('From TrackRef', fontsize=16)
          #
          for ky in range(nFigRow):
            """
            print ('x', clusters.x[cID])
            print ('dx', clusters.dx[cID])
            print ('y', clusters.y[cID])
            print ('dy', clusters.dy[cID])
            print ("charge", clusters.charge[cID] )
            print ("cathode", clusters.cathode[cID] )
            """
            #
            # Y
            """
            minDy = np.min( clusters.dy[cID] )
            maxDy = np.max( clusters.dy[cID] )
            # if np.abs( maxDy - minDy ) > 10.0e-5: print( "---- > 2 dy"); exit()
            minY = np.min( clusters.y[cID] )
            maxY = np.max( clusters.y[cID] )
            sizeY = int( np.round( (maxY - minY) / minDy)) + 1
            """
            # ???????????????
            if (cID < nClusters):
              displayOneCluster( ax[ky,:], tracks, t, cID )
            cID = cID +1
          plt.show()
      #
    return

def displayTracksByHashCode( tracksObj ):
    #
    nTracks = len( tracksObj.tracks )
    nFigCol = 8
    nFigRow = 4
    nFig = nFigCol * nFigRow
    # Sort by 
    length = []
    for key in tracksObj.clusterHash.keys():
        length.append( ( key, len(tracksObj.clusterHash[key]) ) )
    orderedKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    # Color Map
    # ???
    
    # Loop on the "same" cluster
    for key, l in orderedKeys:
        sameClusters = tracksObj.clusterHash[key]
        print "key, number of identical Clusters", key,  l
        tRef = sameClusters[0][0]
        cRef = sameClusters[0][1]
        print "tRef, cRef =", tRef, cRef
        fig, ax = plt.subplots(nrows=1, ncols=nFigCol, figsize=(17, 7) )
        displayOneCluster( ax[:], tracksObj, tRef, cRef )
        nbRecoClusters = 0
        for t, c in sameClusters:
            #fig, ax = plt.subplots(nrows=1, ncols=nFigCol, figsize=(17, 7) )
            #displayOneCluster( ax[:], tracksObj, t, c )
            if t != tRef or c != cRef:
                print "NOT THE SAME TRACK/CLUSTER t=", t,"/", tRef, ", c=",c ,"/", cRef
                (ev, mcLabel, partCode, iCl, nPads) = tracksObj.getClusterInfos( t, c)
                print "ev=", ev, ", mcLabel=", mcLabel, ", code=", partCode, ", clID=", iCl, ", nPads=", nPads
            # Bounds of the Reco Cluster
            xMin = np.min( tracksObj.tracks[t][c].x - tracksObj.tracks[t][c].dx) 
            xMax = np.max( tracksObj.tracks[t][c].x + tracksObj.tracks[t][c].dx) 
            yMin = np.min( tracksObj.tracks[t][c].y - tracksObj.tracks[t][c].dy) 
            yMax = np.max( tracksObj.tracks[t][c].y + tracksObj.tracks[t][c].dy) 
            xMC = tracksObj.MCTracks[t].x[c]
            yMC = tracksObj.MCTracks[t].y[c]
            if (xMC >= xMin) and (xMC <= xMax) and  (yMC >= yMin) and (yMC <= yMax):
                nbRecoClusters += 1
            print "xmc, ymc", xMC, yMC
            w = np.array( [ 1.0 ] )
            mu = np.array( [ [ xMC, yMC] ] )
            var = np.array([[0.0, 0.0]])
            plu.drawModelComponents( ax[2], w, mu, var, color='black', pattern="o" )
        print "nbRecoClusters=",nbRecoClusters
        plt.show()
    return
def compareTracksWithServerClusters( tracksObj, sClusters ):
    #
    nTracks = len( tracksObj.tracks )
    nFigCol = 8
    nFigRow = 4
    nFig = nFigCol * nFigRow
    # Sort by 
    length = []
    for key in tracksObj.clusterHash.keys():
        length.append( ( key, len(tracksObj.clusterHash[key]) ) )
    orderedKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    length = []
    for key in sClusters.clusterHash.keys():
        length.append( ( key, len(sClusters.clusterHash[key]) ) )
    sOrderedKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    # Color Map
    # ???
    
    # Loop on the "same" cluster
    for key, l in orderedKeys:
        sameClusters = tracksObj.clusterHash[key]
        print "key, number of identical Clusters", key,  l
        tRef = sameClusters[0][0]
        cRef = sameClusters[0][1]
        print "tRef, cRef =", tRef, cRef
        fig, ax = plt.subplots(nrows=1, ncols=nFigCol, figsize=(17, 7) )
        displayOneCluster( ax[:], tracksObj, tRef, cRef )
        nbRecoClusters = 0
        for t, c in sameClusters:
            #fig, ax = plt.subplots(nrows=1, ncols=nFigCol, figsize=(17, 7) )
            #displayOneCluster( ax[:], tracksObj, t, c )
            if t != tRef or c != cRef:
                print "NOT THE SAME TRACK/CLUSTER t=", t,"/", tRef, ", c=",c ,"/", cRef
                (ev, mcLabel, partCode, iCl, nPads) = tracksObj.getClusterInfos( t, c)
                print "ev=", ev, ", mcLabel=", mcLabel, ", code=", partCode, ", clID=", iCl, ", nPads=", nPads
            # Bounds of the Reco Cluster
            xMin = np.min( tracksObj.tracks[t][c].x - tracksObj.tracks[t][c].dx) 
            xMax = np.max( tracksObj.tracks[t][c].x + tracksObj.tracks[t][c].dx) 
            yMin = np.min( tracksObj.tracks[t][c].y - tracksObj.tracks[t][c].dy) 
            yMax = np.max( tracksObj.tracks[t][c].y + tracksObj.tracks[t][c].dy) 
            xMC = tracksObj.MCTracks[t].x[c]
            yMC = tracksObj.MCTracks[t].y[c]
            if (xMC >= xMin) and (xMC <= xMax) and  (yMC >= yMin) and (yMC <= yMax):
                nbRecoClusters += 1
            print "xmc, ymc", xMC, yMC
            w = np.array( [ 1.0 ] )
            mu = np.array( [ [ xMC, yMC] ] )
            var = np.array([[0.0, 0.0]])
            plu.drawModelComponents( ax[2], w, mu, var, color='black', pattern="o" )
        print "nbRecoClusters=",nbRecoClusters

        # ClusterServer
        t, c = sameClusters[0]
        if sClusters.clusterHash.has_key(key):
          sameCSClusters = sClusters.clusterHash[key]
            

            
          for cs in sameCSClusters:
            #fig, ax = plt.subplots(nrows=1, ncols=nFigCol, figsize=(17, 7) )
            #displayOneCluster( ax[:], tracksObj, t, c )
            """
            if t != tRef or c != cRef:
                print "NOT THE SAME TRACK/CLUSTER t=", t,"/", tRef, ", c=",c ,"/", cRef
                (ev, mcLabel, partCode, iCl, nPads) = tracksObj.getClusterInfos( t, c)
                print "ev=", ev, ", mcLabel=", mcLabel, ", code=", partCode, ", clID=", iCl, ", nPads=", nPads
            """
            # Bounds of the Reco Cluster
            """
            xMC = tracksObj.MCTracks[t].x[c]
            yMC = tracksObj.MCTracks[t].y[c]
            if (xMC >= xMin) and (xMC <= xMax) and  (yMC >= yMin) and (yMC <= yMax):
                nbRecoClusters += 1
            print "xmc, ymc", xMC, yMC
            """
            print "???", cs
            
            xCS = sClusters.cFeatures["X"][cs]
            yCS = sClusters.cFeatures["Y"][cs]
            dxCS = sClusters.cFeatures["ErrorX"][cs]
            dyCS = sClusters.cFeatures["ErrorY"][cs]
            w = np.array( [ 1.0 ] )
            mu = np.array( [ [ xCS, yCS] ] )
            var = np.array([[dxCS, dyCS]])
            plu.drawModelComponents( ax[2], w, mu, var, color='red', pattern='diam')        
        
        plt.show()
    return

def processTracksWithServerClusters( tracksObj, sClusters ):
    #
    nTracks = len( tracksObj.tracks )
    nFigCol = 4
    nFigRow = 2
    nFig = nFigCol * nFigRow
    # Sort by 
    length = []
    for key in tracksObj.clusterHash.keys():
        length.append( ( key, len(tracksObj.clusterHash[key]) ) )
    orderedTrackKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    length = []
    for key in sClusters.clusterHash.keys():
        length.append( ( key, len(sClusters.clusterHash[key]) ) )
    sOrderedKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    # Color Map
    # ???
    
    # Loop on the "same" cluster
    for key, l in orderedTrackKeys:
        sameClusters = tracksObj.clusterHash[key]
        print "key, number of identical Clusters", key,  l
        tRef = sameClusters[0][0]
        cRef = sameClusters[0][1]
        print "tRef, cRef =", tRef, cRef
        fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
        cl = tracksObj.tracks[tRef][cRef]
        # drawOneCluster( ax[0, :], cl, mode="all", title=["TrackRef C0", "TrackRef C1", "TrackRef C0 & C1"] )
        # drawOneCluster( ax[0, :],  cl, mode="superposition", title=["TrackRef C0 & C1"], noYTicksLabels=False)
        drawOneCluster( ax[0, 0:], cl, mode="superposition", title=["TrackRef MC hits"], noYTicksLabels=False)
        drawOneCluster( ax[1, 0:], cl, mode="superposition", title=["TrackRef maxima ($\\theta_i$)"], noYTicksLabels=False)
        drawOneCluster( ax[1, 1:], cl, mode="superposition", title=["TrackRef EM process ($\\mu_f$)"] )
        drawOneCluster( ax[1, 2:], cl, mode="superposition", title=["TrackRef EM process ($w_f$)"] )
        drawOneCluster( ax[1, 3:], cl, mode="superposition", title=["TrackRef filtered ($\\theta_f$)"] )
        # Process
        wi, mui, vari = findLocalMaxWithSubstraction( cl )
        wf, muf, varf = simpleProcessEMCluster( cl, wi, mui, vari )
        wff, muff, varff = filterModel( wf, muf, varf )
        """
        print "wi", wi
        print 'mui', mui
        print 'vari', vari
        """
        plu.drawModelComponents( ax[1, 0], wi, mui, vari, pattern="diam")
        plu.drawModelComponents( ax[1, 1], wf, muf, varf, pattern="diam")
        plu.drawModelComponents( ax[1, 2], wf, muf, varf, pattern="show w")
        plu.drawModelComponents( ax[1, 3], wff, muff, varff, pattern="diam", color='red')
        
        xInf = np.min( tracksObj.tracks[tRef][cRef].x - tracksObj.tracks[tRef][cRef].dx) 
        xSup = np.max( tracksObj.tracks[tRef][cRef].x + tracksObj.tracks[tRef][cRef].dx) 
        yInf = np.min( tracksObj.tracks[tRef][cRef].y - tracksObj.tracks[tRef][cRef].dy) 
        ySup = np.max( tracksObj.tracks[tRef][cRef].y + tracksObj.tracks[tRef][cRef].dy) 
        for ky in range( nFigRow ):
          for kx in range( nFigCol ):
            ax[ky, kx].set_xlim( xInf,  xSup )
            ax[ky, kx].set_ylim( yInf,  ySup )
        (evRef, mcLabel, partCode, iCl, nPadsRef) = tracksObj.getClusterInfos( tRef, cRef)
        nbMCOutside = 0
        xyMCs = []
        for t, c in sameClusters:
            #fig, ax = plt.subplots(nrows=1, ncols=nFigCol, figsize=(17, 7) )
            #displayOneCluster( ax[:], tracksObj, t, c )
            (ev, mcLabel, partCode, iCl, nPads) = tracksObj.getClusterInfos( t, c)
            if ev != evRef or nPads != nPadsRef:
                print "NOT THE SAME TRACK/CLUSTER t=", t,"/", tRef, ", c=",c ,"/", cRef
                print "ev=", ev, ", mcLabel=", mcLabel, ", code=", partCode, ", clID=", iCl, ", nPads=", nPads
            # Bounds of the Reco Cluster
            xMin = np.min( tracksObj.tracks[t][c].x - tracksObj.tracks[t][c].dx) 
            xMax = np.max( tracksObj.tracks[t][c].x + tracksObj.tracks[t][c].dx) 
            yMin = np.min( tracksObj.tracks[t][c].y - tracksObj.tracks[t][c].dy) 
            yMax = np.max( tracksObj.tracks[t][c].y + tracksObj.tracks[t][c].dy) 
            if xMin != xInf or xMax != xSup or yMin != yInf or yMax != ySup :
                print "WARNING: xyInfSupRef", xInf, xSup, yInf, ySup
                print "WARNING: xyMinMax", xMin, xMax, yMin, yMax
                
            
            xMC= tracksObj.MCTracks[t].x[c]
            yMC = tracksObj.MCTracks[t].y[c]
            if not ( (xMC >= xMin) and (xMC <= xMax) and  (yMC >= yMin) and (yMC <= yMax)):
                nbMCOutside += 1
            
            xyMCs.append( np.array( [xMC, yMC]) )
            # ??? yMCs.append( yMC )
            """
            w = np.array( [ 1.0 ] )
            mu = np.array( [ [ xMC, yMC] ] )
            var = np.array([[0.0, 0.0]])
            plu.drawModelComponents( ax[2], w, mu, var, color='black', pattern="o" )
            """
        if nbMCOutside != 0 :
          print "WARNING: ", nbMCOutside, "/", len(xyMCs), "MC particle outside of the cluster"
        l = len( xyMCs )
        w = np.ones( l )
        mu = np.array( xyMCs )
        var = np.zeros( (l,2))
        print "??? mu shape", mu.shape, w.shape, var.shape

        plu.drawModelComponents( ax[0,0], w, mu, var, color='black', pattern="o" )
        # plu.drawModelComponents( ax[0,2], w, mu, var, color='black', pattern="o" )
        plu.drawModelComponents( ax[0,2], w, mu, var, color='black', pattern="o" )
        plu.drawModelComponents( ax[1,3], w, mu, var, color='black', pattern="o" )

        # ClusterServer
        t, c = sameClusters[0]
        if sClusters.clusterHash.has_key(key):
          sameCSClusters = sClusters.clusterHash[key]
          csRef = sameCSClusters[0]
          drawOneCSCluster( ax[0, 1:], sClusters, csRef, mode="superposition", title=["ClusterServer Reco hits"] )
          drawOneCSCluster( ax[0, 2:], sClusters, csRef, mode="superposition", title=["ClusterServer Reco & MC hits"] )
          
          # Draws points in the same chamber ans region
          chamberRef =  sClusters.id["ChamberID"][csRef]
          xCh = []; yCh = []
          dxCh = []; dyCh = []
          for k, chamb in enumerate(sClusters.id["ChamberID"]):
            if chamb == chamberRef:
              xCh.append( sClusters.cFeatures["X"][k] )
              yCh.append( sClusters.cFeatures["Y"][k] )
              dxCh.append( sClusters.cFeatures["ErrorX"][k] )
              dyCh.append( sClusters.cFeatures["ErrorY"][k] )
          nCh = len(xCh)
          wCh = np.ones( nCh )
          muCh = np.array( [xCh,yCh] ).T
          varCh = np.array( [dxCh,dyCh] ).T
          print "??? len(xCh), shape", len(xCh), muCh.shape

          plu.drawModelComponents( ax[0, 1], wCh, muCh, varCh, color='black', pattern='cross')        
          plu.drawModelComponents( ax[0, 2], wCh, muCh, varCh, color='black', pattern='cross')        
          
          w = []; mu = []; var =[]
          for cs in sameCSClusters:
            # Bounds of the Reco Cluster
            print "???", cs
            print "CS event", sClusters.id["Event"][cs]
            print "CS DE", sClusters.id["DetectElemID"][cs]
            print "CS Chamber", sClusters.id["ChamberID"][cs]
            print "Chamber len", len(sClusters.id["ChamberID"])
            print " x lenth", len(sClusters.cFeatures["X"])
            xCS = sClusters.cFeatures["X"][cs]
            yCS = sClusters.cFeatures["Y"][cs]
            dxCS = sClusters.cFeatures["ErrorX"][cs]
            dyCS = sClusters.cFeatures["ErrorY"][cs]
            w.append( 1.0 )
            mu.append( np.array( [xCS, yCS]))
            var.append( np.array( [dxCS, dyCS]))
          #
          w = np.array( w )
          mu = np.array( mu )
          var = np.array( var)
          plu.drawModelComponents( ax[0, 1], w, mu, var, color='red', pattern='diam')        
          plu.drawModelComponents( ax[0, 2], w, mu, var, color='red', pattern='diam')        
        
        plt.show()
    return

def detailTheEMModel( tracksObj, sClusters ):
    #
    nTracks = len( tracksObj.tracks )
    nFigCol = 4
    nFigRow = 2
    nFig = nFigCol * nFigRow
    # Sort by 
    length = []
    for key in tracksObj.clusterHash.keys():
        length.append( ( key, len(tracksObj.clusterHash[key]) ) )
    orderedTrackKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    length = []
    for key in sClusters.clusterHash.keys():
        length.append( ( key, len(sClusters.clusterHash[key]) ) )
    sOrderedKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    # Color Map
    # ???
    
    # Loop on the "same" cluster
    for key, l in orderedTrackKeys:
        sameClusters = tracksObj.clusterHash[key]
        print "key, number of identical Clusters", key,  l
        tRef = sameClusters[0][0]
        cRef = sameClusters[0][1]
        print "tRef, cRef =", tRef, cRef
        fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
        cl = tracksObj.tracks[tRef][cRef]
        # drawOneCluster( ax[0, :], cl, mode="all", title=["TrackRef C0", "TrackRef C1", "TrackRef C0 & C1"] )
        # drawOneCluster( ax[0, :],  cl, mode="superposition", title=["TrackRef C0 & C1"], noYTicksLabels=False)
        drawOneCluster( ax[0, 0:], cl, mode="superposition", title=["TrackRef MC hits"], noYTicksLabels=False)
        drawOneCluster( ax[1, 0:], cl, mode="superposition", title=["TrackRef maxima ($\\theta_i$)"], noYTicksLabels=False)
        drawOneCluster( ax[1, 1:], cl, mode="superposition", title=["TrackRef EM process ($\\mu_f$)"] )
        drawOneCluster( ax[1, 2:], cl, mode="superposition", title=["TrackRef EM process ($w_f$)"] )
        drawOneCluster( ax[1, 3:], cl, mode="superposition", title=["TrackRef filtered ($\\theta_f$)"] )
        # Process
        wi, mui, vari = findLocalMaxWithSubstraction( cl, ax=ax[0, 1:3], maxIter=1 ) # ax[0, 1:3]
        vari = np.ones( mui.shape ) * 0.4 * 0.4
        wf, muf, varf = simpleProcessEMCluster( cl, wi, mui, vari )
        wff, muff, varff = filterModel( wf, muf, varf )
        wfff, mufff, varfff = simpleProcessEMCluster( cl, wff, muff, varff )
        
        """
        print "wi", wi
        print 'mui', mui
        print 'vari', vari
        """
        plu.drawModelComponents( ax[1, 0], wi, mui, vari, pattern="diam")
        plu.drawModelComponents( ax[1, 1], wf, muf, varf, pattern="diam")
        plu.drawModelComponents( ax[1, 2], wf, muf, varf, pattern="show w")
        plu.drawModelComponents( ax[1, 3], wff, muff, varff, pattern="diam", color='red')
        plu.drawModelComponents( ax[0, 3], wfff, mufff, varfff, pattern="diam", color='red')
        
        xInf = np.min( tracksObj.tracks[tRef][cRef].x - tracksObj.tracks[tRef][cRef].dx) 
        xSup = np.max( tracksObj.tracks[tRef][cRef].x + tracksObj.tracks[tRef][cRef].dx) 
        yInf = np.min( tracksObj.tracks[tRef][cRef].y - tracksObj.tracks[tRef][cRef].dy) 
        ySup = np.max( tracksObj.tracks[tRef][cRef].y + tracksObj.tracks[tRef][cRef].dy) 
        for ky in range( nFigRow ):
          for kx in range( nFigCol ):
            ax[ky, kx].set_xlim( xInf,  xSup )
            ax[ky, kx].set_ylim( yInf,  ySup )
        (evRef, mcLabel, partCode, iCl, nPadsRef) = tracksObj.getClusterInfos( tRef, cRef)
        #
        # MC hits
        #
        nbMCOutside = 0
        xyMCs = []
        for t, c in sameClusters:
            #fig, ax = plt.subplots(nrows=1, ncols=nFigCol, figsize=(17, 7) )
            #displayOneCluster( ax[:], tracksObj, t, c )
            (ev, mcLabel, partCode, iCl, nPads) = tracksObj.getClusterInfos( t, c)
            if ev != evRef or nPads != nPadsRef:
                print "NOT THE SAME TRACK/CLUSTER t=", t,"/", tRef, ", c=",c ,"/", cRef
                print "ev=", ev, ", mcLabel=", mcLabel, ", code=", partCode, ", clID=", iCl, ", nPads=", nPads
            # Bounds of the Reco Cluster
            xMin = np.min( tracksObj.tracks[t][c].x - tracksObj.tracks[t][c].dx) 
            xMax = np.max( tracksObj.tracks[t][c].x + tracksObj.tracks[t][c].dx) 
            yMin = np.min( tracksObj.tracks[t][c].y - tracksObj.tracks[t][c].dy) 
            yMax = np.max( tracksObj.tracks[t][c].y + tracksObj.tracks[t][c].dy) 
            if xMin != xInf or xMax != xSup or yMin != yInf or yMax != ySup :
                print "WARNING: xyInfSupRef", xInf, xSup, yInf, ySup
                print "WARNING: xyMinMax", xMin, xMax, yMin, yMax
            #    
            xMC= tracksObj.MCTracks[t].x[c]
            yMC = tracksObj.MCTracks[t].y[c]
            if not ( (xMC >= xMin) and (xMC <= xMax) and  (yMC >= yMin) and (yMC <= yMax)):
                nbMCOutside += 1
            #
            xyMCs.append( np.array( [xMC, yMC]) )
            # ??? yMCs.append( yMC )
        #
        if nbMCOutside != 0 :
          print "WARNING: ", nbMCOutside, "/", len(xyMCs), "MC particle outside of the cluster"
        l = len( xyMCs )
        w = np.ones( l )
        mu = np.array( xyMCs )
        var = np.zeros( (l,2))
        # Draw in graphs
        plu.drawModelComponents( ax[0,0], w, mu, var, color='black', pattern="o" )
        # plu.drawModelComponents( ax[0,2], w, mu, var, color='black', pattern="o" )
        plu.drawModelComponents( ax[0,2], w, mu, var, color='black', pattern="o" )
        plu.drawModelComponents( ax[0,3], w, mu, var, color='black', pattern="o" )
        plu.drawModelComponents( ax[1,3], w, mu, var, color='black', pattern="o" )
        
        # ClusterServer
        # 
        # Removed
        
        plt.show()
    return

def checkTracksWithServerClusters( tracksObj, sClusters ):
    #
    nTracks = len( tracksObj.tracks )
    # Sort by 
    length = []
    for key in tracksObj.clusterHash.keys():
        length.append( ( key, len(tracksObj.clusterHash[key]) ) )
    orderedTrackKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    length = []
    for key in sClusters.clusterHash.keys():
        length.append( ( key, len(sClusters.clusterHash[key]) ) )
    sOrderedKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    # Key proximity
    # keys = tracksObj.clusterHash.keys()
    keys = sClusters.clusterHash.keys()
    print " TrackRef keys number", len(keys)
    keys = sorted( keys)
    nbrKeyEqual = 0
    for k, key1 in enumerate(keys):
      for key2 in keys[k+1:]:
        pos = 0
        while key1[pos] == key2[pos]:
          pos += 1
          if pos >= len(key1):
            break
        if pos > 17:
          if key1 != key2: 
            print key1
            print key2
          else:
            nbrKeyEqual +=1
    print nbrKeyEqual
    
    """
    # Loop on the "same" cluster
    for key, l in orderedTrackKeys:
        sameClusters = tracksObj.clusterHash[key]
        print "key, number of identical Clusters", key,  l
        tRef = sameClusters[0][0]
        cRef = sameClusters[0][1]
        print "tRef, cRef =", tRef, cRef
        cl = tracksObj.tracks[tRef][cRef]
        (evRef, mcLabel, partCode, iCl, nPadsRef) = tracksObj.getClusterInfos( tRef, cRef)
        nbMCOutside = 0
        xyMCs = []
            
        xMC= tracksObj.MCTracks[t].x[c]
        yMC = tracksObj.MCTracks[t].y[c]

        # ClusterServer
        t, c = sameClusters[0]
        if sClusters.clusterHash.has_key(key):
          sameCSClusters = sClusters.clusterHash[key]
          csRef = sameCSClusters[0]
            xCS = sClusters.cFeatures["X"][cs]
            yCS = sClusters.cFeatures["Y"][cs]
            dxCS = sClusters.cFeatures["ErrorX"][cs]
            dyCS = sClusters.cFeatures["ErrorY"][cs]
    """
    return

def statTracksByHashCode( tracksObj ):
    #
    nTracks = len( tracksObj.tracks )
    # Sort by 
    length = []
    for key in tracksObj.clusterHash.keys():
        length.append( ( key, len(tracksObj.clusterHash[key]) ) )
    orderedKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    
    # Loop on the "same" cluster
    minCharge = []
    maxCharge = []
    meanCharge = []
    clusterCharge = []
    clusterNbrOfPads = []
    for key, l in orderedKeys:
        sameClusters = tracksObj.clusterHash[key]
        nbRecoClusters = 0
        for t, c in sameClusters:
            # Bounds of the Reco Cluster
            xMin = np.min( tracksObj.tracks[t][c].x - tracksObj.tracks[t][c].dx) 
            xMax = np.max( tracksObj.tracks[t][c].x + tracksObj.tracks[t][c].dx) 
            yMin = np.min( tracksObj.tracks[t][c].y - tracksObj.tracks[t][c].dy) 
            yMax = np.max( tracksObj.tracks[t][c].y + tracksObj.tracks[t][c].dy) 
            xMC = tracksObj.MCTracks[t].x[c]
            yMC = tracksObj.MCTracks[t].y[c]
            if (xMC >= xMin) and (xMC <= xMax) and  (yMC >= yMin) and (yMC <= yMax):
                nbRecoClusters += 1
                tStore = t; cStore = c
        # print "nbRecoClusters=",nbRecoClusters
        if (nbRecoClusters == 1):
           minCharge.append( np.min( tracksObj.tracks[tStore][cStore].charge ))
           maxCharge.append( np.max( tracksObj.tracks[tStore][cStore].charge ))
           meanCharge.append( np.mean( tracksObj.tracks[tStore][cStore].charge ))
           
           clusterCharge.append( np.sum( tracksObj.tracks[tStore][cStore].charge ))
           clusterNbrOfPads.append( tracksObj.tracks[tStore][cStore].x.shape[0] )
        if (nbRecoClusters == 0):
            print "not recognized", sameClusters

    plt.plot( clusterNbrOfPads, clusterCharge, 'o')
    plt.show()
    plt.plot( clusterCharge, minCharge,'o', color='blue')
    plt.plot( clusterCharge, maxCharge,'o', color='red')
    plt.plot( clusterCharge, meanCharge,'o', color='green')
    plt.show()

      #
    return

def findLocalMaxWithDerivative( cl ):
    x = []; dx = []
    y = []; dy = []
    ch = []
    print cl.x
    c0Idx = np.where( cl.cathode == 0)
    c1Idx = np.where( cl.cathode == 1)
    x.append(cl.x[c0Idx])
    x.append(cl.x[c1Idx])
    dx.append(cl.dx[c0Idx])
    dx.append(cl.dx[c1Idx])    
    y.append(cl.y[c0Idx])
    y.append(cl.y[c1Idx])
    dy.append(cl.dy[c0Idx])
    dy.append(cl.dy[c1Idx])
    ch.append(cl.charge[c0Idx])
    ch.append(cl.charge[c1Idx])  

    # Extract neighbours
    eps = 1.0e-10
    neigh = []
    for cath in range(len(x)):
      neighCath = []
      for i in range( x[cath].shape[0]):
        xMask = np.abs( x[cath][i] - x[cath] ) <= ( (1.0 + eps) * dx[cath][i] + dx[cath] )
        yMask = np.abs( y[cath][i] - y[cath] ) <= ( (1.0 + eps) * dy[cath][i] + dy[cath] )
        neighCath.append( np.where( np.bitwise_and(xMask, yMask) ) ) 
      neigh.append( neighCath )
 
    print "neigh", neigh
    # Get the max
    locMax = []
    for cath in range(len(x)):
      locMaxCath = []
      mask = np.ones( ch[cath].shape )
      while np.sum(mask) != 0 :
        maxIdx = np.argmax( ch[cath] * mask  )
        print "sum mask", np.sum(mask), ",cath=", cath
        print "  max ch", ch[cath][maxIdx]
        print "  neigh idx", neigh[cath][maxIdx]
        print "  neigh values", ch[cath][neigh[cath][maxIdx]]
        
        # Set zero mask on the neighboring and itself
        mask[ neigh[cath][maxIdx] ] = 0;
        locMaxCath.append( maxIdx) 
      locMax.append( locMaxCath)
    
    xLMax = [None]*2
    dxLMax = [None]*2
    yLMax = [None]*2
    dyLMax = [None]*2
    chLMax = [None]*2
    for cath in range(len(x)):
        idx = np.array( locMax[cath] )
        xLMax[cath] =  x[cath][ idx ]
        yLMax[cath] =  y[cath][ idx ]
        dxLMax[cath] =  dx[cath][ idx ]
        dyLMax[cath] =  dy[cath][ idx ]
        chLMax[cath] =  ch[cath][ idx ]
        
    return xLMax, dxLMax, yLMax, dyLMax, chLMax 

def filterModel( w0, mu0, var0 ):
    w = np.copy(w0)
    mu = np.copy(mu0)
    var = np.copy(var0)
    K = mu.shape[0]
    sig = np.sqrt( var )
    for k in range( K ):
      if w[k] != 0:
        for l in range(k+1, K):
          xDelta = max( sig[k][0], sig[l][0] ) * 0.5
          yDelta = max( sig[k][1], sig[l][1] ) * 0.5
          # Test if positions are close
          if (np.abs(mu[k,0] - mu[l, 0]) < xDelta) and (np.abs(mu[k,1] - mu[l, 1]) < yDelta) :
            if ( w[k] >= w[l]) :
              w[k] += w[l]
              w[l] = 0.0
            else:
              w[l] += w[k]
              w[k] = 0.0
    #          
    return w, mu, var

def findLocalMaxWithSubstraction( cl, ax=None, maxIter=200 ):
    x = []; dx = []
    y = []; dy = []
    ch = []
    c0Idx = np.where( cl.cathode == 0)
    c1Idx = np.where( cl.cathode == 1)
    x.append(cl.x[c0Idx])
    x.append(cl.x[c1Idx])
    dx.append(cl.dx[c0Idx])
    dx.append(cl.dx[c1Idx])    
    y.append(cl.y[c0Idx])
    y.append(cl.y[c1Idx])
    dy.append(cl.dy[c0Idx])
    dy.append(cl.dy[c1Idx])
    ch.append( np.copy( cl.charge[c0Idx] ))
    ch.append( np.copy( cl.charge[c1Idx] ))  

    # sig0 = np.array( [0.35, 0.35] )
    sig0 = np.array( [0.4, 0.4] )
    var0 = sig0 * sig0
    mu = np.zeros(2)
    
    # Extract neighbours
    eps = 1.0e-10
    neigh = []
    for cath in range(len(x)):
      neighCath = []
      for i in range( x[cath].shape[0]):
        xMask = np.abs( x[cath][i] - x[cath] ) <= ( (1.0 + eps) * dx[cath][i] + dx[cath] )
        yMask = np.abs( y[cath][i] - y[cath] ) <= ( (1.0 + eps) * dy[cath][i] + dy[cath] )
        neighCath.append( np.where( np.bitwise_and(xMask, yMask) ) ) 
      neigh.append( neighCath )

    # Get the max
    locMax = []
    mask = [ np.ones( ch[0].shape ), np.ones( ch[1].shape ) ]
    maxVal = [0., 0.]
    wa = []
    mua = []
    vara = []
    nIter = 0
    while ( np.sum(mask[0][:]) + np.sum(mask[1][:] ) > 0.01 ) and nIter < maxIter :

        print "Min filter", np.sum( mask[0][:] ), np.sum( mask[1][:] )
        
        # Find the max
        # Rq: Max location Can be refined if 
        # there is an intersection between the 2 pads
        maxVal[0] = np.max( ch[0][:] * mask[0][:]  )
        maxVal[1] = np.max( ch[1][:] * mask[1][:]  )
        print "maxVal", maxVal[0], maxVal[1]
        if( maxVal[1] > maxVal[0]):
          maxCath = 1
        else:
          maxCath = 0
        maxIdx =np.argmax( ch[maxCath][:]*mask[maxCath][:] )
        
        # Refine mu
        # Compute mu (barycenter of the charge)
        idx = neigh[maxCath][maxIdx]
        u = ch[maxCath][idx]*x[maxCath][idx]*mask[maxCath][idx]
        mux = np.sum(u) / np.sum( ch[maxCath][idx]*mask[maxCath][idx] )
        u = ch[maxCath][idx]*y[maxCath][idx]*mask[maxCath][idx]
        muy = np.sum(u) / np.sum( ch[maxCath][idx]*mask[maxCath][idx] )
        
        print "cath, mu, x, y ", maxCath, mux, muy, x[maxCath][maxIdx], y[maxCath][maxIdx]
        print "xmin, xmax, ymin, ymax ", np.min( x[maxCath] ), np.max( x[maxCath] ), \
                    np.min( y[maxCath] ), np.max( y[maxCath] )
        
        # mu[0] = x[maxCath][maxIdx]
        # mu[1] = y[maxCath][maxIdx]
        mu[0] = mux
        mu[1] = muy
        #
        # Remove the gaussian
        #
        """ 
        # sig = cst
        w = ch[maxCath][maxIdx] * EM.TwoPi * sig0[0] * sig0[1]
        print "y, cst", ch[maxCath][maxIdx], EM.TwoPi * sig0[0] * sig0[1]
        """
        var = np.array([ 1.0 * dx[maxCath][maxIdx], 1.0 *dy[maxCath][maxIdx]])
        var = var * var
        vara.append( var )
        # w = ch[maxCath][maxIdx] * EM.TwoPi * dx[maxCath][maxIdx] * dy[maxCath][maxIdx]
        muTmp = np.array( [ mu ] ).T
        dxyTmp = np.array([ [dx[maxCath][maxIdx], dy[maxCath][maxIdx] ]]).T

        xyMax = np.array( [ [ x[maxCath][maxIdx], y[maxCath][maxIdx] ] ] ).T 
        dxyMax = np.array( [ [dx[maxCath][maxIdx], dy[maxCath][maxIdx]] ] ).T
        integral = EM.computeDiscretizedGaussian2D( xyMax, dxyMax, mu, var )
        w = ch[maxCath][maxIdx] / integral
        print "ch. max, integral at max, w", ch[maxCath][maxIdx], integral, w
        print "mu=", mu
        wa.append( w )
        mua.append( np.copy(mu) )

        for cath in range(2):
            print "cath=", cath
            xy = [ x[cath], y[cath] ]
            dxy = [ dx[cath], dy[cath] ]
            print "  maxIdx", maxIdx
            # print "  max ch, w, gauss(max):", ch[maxCath][maxIdx], w,  EM.computeGaussian2D( mu, mu, var )
            print "  max ch, w, gauss(max):", ch[maxCath][maxIdx], w, EM.computeDiscretizedGaussian2D( muTmp, dxyTmp, mu, var )
            xyMax = np.array( [ [ x[maxCath][maxIdx], y[maxCath][maxIdx] ] ] ).T 
            dxyMax = np.array( [ [dx[maxCath][maxIdx], dy[maxCath][maxIdx]] ] ).T
#           # print " value at index max :", w*EM.computeGaussian2D(xyMax, mu, var)
            print " value at index max :", w*EM.computeDiscretizedGaussian2D(xyMax, dxyMax, mu, var)
#            print "  gauss : ",  w*EM.computeGaussian2D(xy, mu, var)
            xyTmp = np.array( xy )
            dxyTmp = np.array( dxy)
            print "  gauss : ",  w*EM.computeDiscretizedGaussian2D( xyTmp, dxyTmp, mu, var)
            print "??? x", x
            print "??? x", y
            print "??? x", mu
            print "??? x", var

            
            # print "  charges :", ch[cath][:] 
#            ch[cath][:] = ch[cath][:] - mask[cath][:]* w * EM.computeGaussian2D(xy, mu, var )
            ch[cath][:] = ch[cath][:] - mask[cath][:]* w * EM.computeDiscretizedGaussian2D(xyTmp, dxyTmp, mu, var )
            # print "  charges :", ch[cath][:] 
            # print "sum mask", np.sum( ch[0][:]*mask[0][:] >= 5 ), np.sum( ch[1][:]*mask[1][:] >= 5 )
            # print "dx ", dx[cath]
            # print "dy ", dy[cath]
        wi = np.array(wa)
        s = np.sum( wi)
        wi = wi / s
        mui = np.array( mua )
        vari = np.array(vara)
        # wf, muf, varf = simpleProcessEMCluster( cl, wi, mui, vari )
        
        mask[0][:] = ( ch[0][:]*mask[0][:] >= 5 )
        mask[1][:] = ( ch[1][:]*mask[1][:] >= 5 )
        # print "mask 0", mask[0]
        # print "mask 1", mask[1]
        
    if ax is not None :
      plu.drawPads(ax[0],  x[0], y[0], dx[0], dy[0], np.where( ch[0] > 0, ch[0], 0),
                title="", alpha=1.0, noYTicksLabels=True, doLimits=False )   
      plu.drawPads(ax[1],  x[1], y[1], dx[1], dy[1], np.where( ch[1] > 0, ch[1], 0),
                title="", alpha= 1.0, noYTicksLabels=True, doLimits=False )
    
    print "wi ", wi
    print "mui ", mui
    print "vari", vari

    return np.array( wi ) , np.array( mui ), np.array( vari )

def processTracksByHashCode( tracksObj ):
    #
    nTracks = len( tracksObj.tracks )
    # Sort by 
    length = []
    for key in tracksObj.clusterHash.keys():
        length.append( ( key, len(tracksObj.clusterHash[key]) ) )
    orderedKeys = sorted(length, key=operator.itemgetter(1), reverse=True )
    
    # Loop on the "same" cluster
    for key, l in orderedKeys:
        sameClusters = tracksObj.clusterHash[key]
        print "KEY=", key
        (t,c) = sameClusters[0]
        aCluster = copy.deepcopy( tracksObj.tracks[t][c] )
        # xMax, dxMax, yMax, dyMax, chMax = findLocalMaxWithDerivatives( aCluster )
        
        nFigRow = 2
        nFigCol = 4
        fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(15, 7) )
        fig.suptitle('Local Max', fontsize=16)
        # Graph limits
        cluster = tracksObj.tracks[t][c]
        xSup = np.max( cluster.x + cluster.dx )
        ySup = np.max( cluster.y + cluster.dy )
        xInf = np.min( cluster.x - cluster.dx )
        yInf = np.min( cluster.y - cluster.dy )
        for ky in range( nFigRow):
          for kx in range( nFigCol -1):
            ax[ky, kx].set_xlim( xInf,  xSup )
            ax[ky, kx].set_ylim( yInf,  ySup)
            
        drawOneCluster( ax[0, 0:3], tracksObj.tracks[t][c] )
        drawOneCluster( ax[1, 0], tracksObj.tracks[t][c], mode="superposition" )
        """
        for k in range( len(wi)):
            plu.drawModelComponents( ax[2], wi[k], mui[k], vari[k])
        """
        wi, mui, vari = findLocalMaxWithSubstraction( aCluster )
        wf, muf, varf = simpleProcessEMCluster( aCluster, wi, mui, vari )
        print "wi", wi
        print 'mui', mui
        print 'vari', vari
        plu.drawModelComponents( ax[0, 2], wi, mui, vari, pattern="diam")
        plu.drawModelComponents( ax[1, 2], wf, muf, varf, pattern="diam")
        plu.drawModelComponents( ax[1, 1], wf, muf, varf, pattern="show w")
        
 
        """
        plu.drawPads(ax[3],  xMax[0], yMax[0], dxMax[0], dyMax[0], chMax[0],
                title="", alpha=1.0, noYTicksLabels=True, doLimits=True )
        plu.drawPads(ax[5],  xMax[0], yMax[0], dxMax[0], dyMax[0], chMax[0],
                title="", alpha=0.5, noYTicksLabels=True, doLimits=False )
        plu.drawPads(ax[4],  xMax[1], yMax[1], dxMax[1], dyMax[1], chMax[1],
                title="", alpha=1.0, noYTicksLabels=True, doLimits=True )        
        plu.drawPads(ax[5],  xMax[1], yMax[1], dxMax[1], dyMax[1], chMax[1],
                title="", alpha=0.5, noYTicksLabels=True, doLimits=False )   
        """

        plt.show()
        

      #
    return

def printClusters( clusters, pClusters, pixels ):
    print( "shape cluster, pre-cluster, pixels", len(clusters.x), len(pClusters.x) )
    if ( len(clusters.x)  !=  len(pClusters.x) ):
        print( "displayClusters: Clusters and pre-Clusters have diff. size")
        # exit()
    # Assign event ID to clusters 
    n = len( clusters.id["DetectElemID"] )
    print n
    evID = 0 
    old = clusters.id["DetectElemID"][0]/100
    clusters.id["Event"][0] = evID
    for i in range(1, n):
     new = clusters.id["DetectElemID"][i] /100
     if (new < old):
         print "i found", i
         evID += 1
     clusters.id["Event"][i] = evID
     old = new
    print "nev=", evID 
    nEv = evID+1
    
    print "Cl Events:", clusters.id["Event"][0:22]
    print "Cl ServerID:", clusters.id["cServerID"][0:22]
    print "Cl DE Id:", clusters.id["DetectElemID"][0:22]
    print "Cl Ch Id:", clusters.id["ChamberID"][0:22]
    x = clusters.cFeatures["X"]
    dx = clusters.cFeatures["ErrorX"]
    y = clusters.cFeatures["Y"]
    dy = clusters.cFeatures["ErrorY"]
    print "preCl Events:", pClusters.id["Event"][0:22]
    print "preCl ServerID:", pClusters.id["cServerID"][0:22]
    print "preCl preClusterID :", pClusters.id["preClusterID"][0:22]
    print "preCl DE Id:", pClusters.id["DetectElemID"][0:22]
    print "preCl Ch Id:", pClusters.id["ChamberID"][0:22]
    """
    for i in range(n):
      if (pClusters.id["Event"][i] != clusters.id["Event"][i]):
        print "problem i=", i, pClusters.id["Event"][i], clusters.id["Event"][i]
        print "Cl DE Id:", clusters.id["DetectElemID"][i-1:i+20]
        print "preCl DE Id:", pClusters.id["DetectElemID"][i-1:i+20]
    """
    pcEv = np.array(pClusters.id["Event"])
    csEv = np.array(clusters.id["Event"])
    pcDEId = np.array(pClusters.id["DetectElemID"])
    csDEId = np.array(clusters.id["DetectElemID"])
    
    for ev in range(nEv):
        pcidx = np.where( pcEv == ev)
        csidx = np.where( csEv == ev)
        if np.all( pcDEId[pcidx] != csDEId[csidx] ):
            print "ev=", ev, ", pCluster DEId=", pcDEId[pcidx]
            print "ev=", ev, ", ServCluster DEId=", csDEId[csidx]
            pc = 0; cs = 0
            npc = pcidx[0].shape[0]
            ncs = csidx[0].shape[0]
            done = ((cs == ncs) and (pc == npc))

            while ( not done ):
                if ( pcDEId[pcidx[0][pc]] < csDEId[csidx[0][cs]] ):
                    print "pre-cl < cluster serv.:", pcDEId[pcidx[0][pc]], csDEId[csidx[0][cs]]
                    pc += 1
                elif ( pcDEId[pcidx[0][pc]] > csDEId[csidx[0][cs]] ):
                    print "pre-cl >cluster serv.:", pcDEId[pcidx[0][pc]], csDEId[csidx[0][cs]]                    
                    cs +=1
                else:
                    # equal
                    cs +=1; pc +=1
                done = ((cs == ncs) and (pc == npc))
                    
    return

def printClustersV2( clusters, pClusters, pixels, tracks ):
    """ 
    Deals with Tracks and cluster[serveur] or pClusters
    """
    print( "shape cluster, tracks, pixels", len(tracks.x), len(pClusters.x) )
    if ( len(clusters.x)  !=  len(pClusters.x) ):
        print( "displayClusters: Clusters and pre-Clusters have diff. size")
        # exit()
    # Assign event ID to clusters 
    n = len( clusters.id["DetectElemID"] )
    print n
    evID = 0 
    old = clusters.id["DetectElemID"][0]/100
    clusters.id["Event"][0] = evID
    for i in range(1, n):
     new = clusters.id["DetectElemID"][i] /100
     if (new < old):
         print "i found", i
         evID += 1
     clusters.id["Event"][i] = evID
     old = new
    print "nev=", evID 
    nEv = evID+1
    
    print "Cl Events:", len( clusters.id["Event"]), clusters.id["Event"][0:22]
    print "Tr Events:", len( tracks.id["Event"]), tracks.id["Event"][0:22]
    clEv = np.array( clusters.id["Event"] )
    trEv = np.array( tracks.id["Event"] )
    nEv = max( np.max(clEv), np.max(trEv))
    
    for ev in range(nEv):
        clIdx = np.where( clEv == ev )
        trIdx = np.where( trEv == ev )
        print "Cl ev=", ev, "[",
        for i in clIdx[0]:
            print clusters.id["DetectElemID"][i],
            # print "ev=", ev, clusters.id["DetectElemID"][i for i in clIdx[0]]
        print "]" 
        #print "ev=", ev, tracks.id["DetectElemID"][trIdx]
        print "Tr ev=", ev, "["
        for i in trIdx[0]:
            print tracks.id["DetectElemID"][i]
            # print "ev=", ev, clusters.id["DetectElemID"][i for i in clIdx[0]]
        print "]"   
        
        for icl in clIdx[0]:
            DEId = clusters.id["DetectElemID"][icl]
            print "trIdx", trIdx
            for itr in trIdx[0]:
                print "itr", itr, tracks.id["DetectElemID"][itr]
                npDEId = np.array( tracks.id["DetectElemID"][itr])
                sameDEId = np.where( npDEId == DEId )
                print "ev=", ev, "cluster DEid=", DEId, "tr same DEId", sameDEId
                for id in sameDEId[0].tolist():
                    print "id=", id
                    print "Cluster X,Y from clsrv", clusters.cFeatures["X"][icl], clusters.cFeatures["Y"][icl]
                    print "Cluster X,Y from tr", tracks.cFeatures["X"][id], tracks.cFeatures["Y"][id]
                    
            # print "ev=", ev, clusters.id["DetectElemID"][i for i in clIdx[0]]
        print "]"     
    
    """
    print "Cl ServerID:", clusters.id["cServerID"][0:22]
    print "Cl DE Id:", clusters.id["DetectElemID"][0:22]
    print "Cl Ch Id:", clusters.id["ChamberID"][0:22]
    x = clusters.cFeatures["X"]
    dx = clusters.cFeatures["ErrorX"]
    y = clusters.cFeatures["Y"]
    dy = clusters.cFeatures["ErrorY"]
    print "preCl Events:", pClusters.id["Event"][0:22]
    print "preCl ServerID:", pClusters.id["cServerID"][0:22]
    print "preCl preClusterID :", pClusters.id["preClusterID"][0:22]
    print "preCl DE Id:", pClusters.id["DetectElemID"][0:22]
    print "preCl Ch Id:", pClusters.id["ChamberID"][0:22]
    """
    """
    for i in range(n):
      if (pClusters.id["Event"][i] != clusters.id["Event"][i]):
        print "problem i=", i, pClusters.id["Event"][i], clusters.id["Event"][i]
        print "Cl DE Id:", clusters.id["DetectElemID"][i-1:i+20]
        print "preCl DE Id:", pClusters.id["DetectElemID"][i-1:i+20]
    """
    """
    pcEv = np.array(pClusters.id["Event"])
    csEv = np.array(clusters.id["Event"])
    pcDEId = np.array(pClusters.id["DetectElemID"])
    csDEId = np.array(clusters.id["DetectElemID"])
    
    for ev in range(nEv):
        pcidx = np.where( pcEv == ev)
        csidx = np.where( csEv == ev)
        if np.all( pcDEId[pcidx] != csDEId[csidx] ):
            print "ev=", ev, ", pCluster DEId=", pcDEId[pcidx]
            print "ev=", ev, ", ServCluster DEId=", csDEId[csidx]
            pc = 0; cs = 0
            npc = pcidx[0].shape[0]
            ncs = csidx[0].shape[0]
            done = ((cs == ncs) and (pc == npc))

            while ( not done ):
                if ( pcDEId[pcidx[0][pc]] < csDEId[csidx[0][cs]] ):
                    print "pre-cl < cluster serv.:", pcDEId[pcidx[0][pc]], csDEId[csidx[0][cs]]
                    pc += 1
                elif ( pcDEId[pcidx[0][pc]] > csDEId[csidx[0][cs]] ):
                    print "pre-cl >cluster serv.:", pcDEId[pcidx[0][pc]], csDEId[csidx[0][cs]]                    
                    cs +=1
                else:
                    # equal
                    cs +=1; pc +=1
                done = ((cs == ncs) and (pc == npc))
    """             
    return

def checkRegularGrid( u, du):
        # Check if the undelying grid is regular

        # Check du
        duMin = np.min(du)
        duMax = np.max(du)
        if ( duMax - duMin) > SimplePrecision:
            print("processClusters: ABORT multiple du [min, max]", duMin, duMax)
            exit()
        # Check u interval
        s = u.astype( np.float32)
        s = np.sort( s )
        s = np.unique( s)
        if s.shape[0] > 1:
          ds = s[1:] - s[0:-1]
          print ("s, ds", s, ds)
          dsMin = np.min(ds)
          dsMax = np.max(ds)
          if ( (dsMax - dsMin) / dsMax ) > SimplePrecision:
            print("processClusters: ABORT multiple ds [min, max]", dsMin, dsMax, (dsMax - dsMin)/dsMax )
            exit()
          if np.abs( ( ds[0] - 2.0 * du[0] )/ds[0] ) > SimplePrecision:
              print("processClusters: ABORT different ds, du", ds, du, np.abs( ( ds[0] - 2.0 * du[0] )/ds[0] ) )
              exit()
        return True


def processClusters( clusters ):
    N = len(clusters.x)
    for i in range(N):
        print("i", i)
        if ( clusters.x[i].shape[0] == 0): continue
        x = clusters.x[i]
        y = clusters.y[i]
        dx = clusters.dx[i]
        dy = clusters.dy[i]
        #
        # Check if the undelying grid is regular
        #
        cathIdx = np.where( clusters.cathode[i] ==0 )
        x0 = x[cathIdx]
        y0 = y[cathIdx]
        dx0 = dx[cathIdx]
        dy0 = dy[cathIdx]
        c0 = clusters.charge[i][cathIdx]
        l0 = x0.shape[0]
        if l0 !=0:
          checkRegularGrid( x0, dx[cathIdx] )
          checkRegularGrid( y0, dy[cathIdx] )
        cathIdx = np.where( clusters.cathode[i] == 1 )
        x1 = x[cathIdx]
        y1 = y[cathIdx]
        dx1 = dx[cathIdx]
        dy1 = dy[cathIdx]
        c1 = clusters.charge[i][cathIdx]
        l1 = x1.shape[0]
        if l1 !=0:
          checkRegularGrid( x1, dx[cathIdx] )
          checkRegularGrid( y1, dy[cathIdx] )
        xm, xsig = plu.getBarycenter( x0, dx0, x1, dx1, c0, c1)
        ym, ysig = plu.getBarycenter( y0, dy0, y1, dy1, c0, c1)
        print( "Barycenter [x, y, sigx, sigy]", xm, ym, xsig, ysig )
        
        
if __name__ == "__main__":
    """
    pixs = IO.TrackInfo(fileName="clusterPixDump-v3.dat")
    pads = IO.TrackInfo(fileName="clusterPadDump-v3.dat")
    """
    sClusters = IO.TrackInfo(fileName="clusterServerDump.dat")
    sClusters.read(mode="ClusterServer")
    sClusters.hashClusters()
    
    tracksMC = IO.TrackRef(fileName="MCTrackRefDump.dat")
    tracksMC.read()
    tracksMC.hashClusters()
    tracksMC.extractdXdY()

    # printClustersV2( clusters, pads, pixs, tracksMC )
    # displayClusters( clusters, pads, pixs )
    # processClusters( clusters )
    print "Tracks", len( tracksMC.tracks )
    # Test overlapping
    """
    t = 0
    c = 0
    cl = tracksMC.tracks[t][c]
    c0Idx = np.where( cl.cathode == 0)
    c1Idx = np.where( cl.cathode == 1)
    computeOverlapingPads(  cl.x[c0Idx], cl.y[c0Idx], cl.dx[c0Idx], cl.dy[c0Idx], 
                            cl.x[c1Idx], cl.y[c1Idx], cl.dx[c1Idx], cl.dy[c1Idx] )
    """
    # displayTracksByHashCode( tracksMC )
    # statTracksByHashCode( tracksMC )
    # processTracksByHashCode(tracksMC)
    # checkTracksWithServerClusters( tracksMC, sClusters )
    detailTheEMModel( tracksMC, sClusters )
    # processTracksWithServerClusters( tracksMC, sClusters )
