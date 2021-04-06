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
"""
    Float_t fKx2;                      ///< Mathieson Kx2
    Float_t fKx4;                      ///< Mathieson Kx4 = Kx1/Kx2/Sqrt(Kx3)  
    Float_t fSqrtKy3;                  ///< Mathieson Sqrt(Ky3)
    Float_t fKy2;                      ///< Mathieson Ky2
    Float_t fKy4;                      ///< Mathieson Ky4 = Ky1/Ky2/Sqrt(Ky3)
  // The Mathieson function 
  Double_t ux1=fSqrtKx3*TMath::TanH(fKx2*xi1);
  Double_t ux2=fSqrtKx3*TMath::TanH(fKx2*xi2);
  
  Double_t uy1=fSqrtKy3*TMath::TanH(fKy2*yi1);
  Double_t uy2=fSqrtKy3*TMath::TanH(fKy2*yi2);
  
  
  return Float_t(4.*fKx4*(TMath::ATan(ux2)-TMath::ATan(ux1))*
                 fKy4*(TMath::ATan(uy2)-TMath::ATan(uy1)));
"""
import MuonDetector as md
import IOv5 as IO
import util as utl
import plotUtil as plu
import GaussianEM2Dv4 as EM
import CurveFiting as CF

SimplePrecision = 5.0e-5
InputWarning = False

class localMaxParam:
    # Ch 1-2
    x = np.array( [0.25, 0.22 ])
    var1 = x * x
    # Ch 2-10
    x = np.array( [0.3, 0.245 ])
    var3 = x * x
    
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
    
    if clusters.x[i].shape[0] == 0 : 
       return ([],[],[])
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
    print( "xy shape:", np.array( [x1, y1]).shape)
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
    print( "sum charge each cathodes", np.sum(c0), np.sum(c1)) 
    print( "Cathodes charge input", clusters.cFeatures["ChargeCath0"][i], clusters.cFeatures["ChargeCath1"][i])
    print( "Clustering/Fitting X, Y ", clusters.cFeatures["X"][i], clusters.cFeatures["Y"][i])
    print( "Barycenter [x, y, sigx, sigy]", xm, ym, xsig, ysig )
    return wf, muf, varf, xm, xsig, ym, ysig

def drawOneCluster( ax, x, y, dx, dy, cathode, charge,  mode="both", title=["Cathode 0", "Cathode 1", "Both"], noYTicksLabels=False):
        if title == None :
            title = ["", "", ""]
        # Graph limits
        xSup = np.max( x + dx )
        ySup = np.max( y + dy )
        xInf = np.min( x - dx )
        yInf = np.min( y - dy )
        #
        minCh =  np.min( charge )
        maxCh =  np.max( charge )
        # Set Lut scale
        plu.setLUTScale( 0, maxCh)
        #
        if mode == "cath0":
            c0Idx = np.where( cathode == 0)
            plu.drawPads(ax,  x[c0Idx], y[c0Idx], dx[c0Idx], dy[c0Idx], charge[c0Idx],
                        title=title[0], alpha=1.0, noYTicksLabels=False, doLimits=False)
        if mode == "cath1":
            c1Idx = np.where( cathode == 1)
            plu.drawPads(ax,  x[c1Idx], y[c1Idx], dx[c1Idx], dy[c1Idx], charge[c1Idx],
                        title=title[1], alpha=1.0, noYTicksLabels=True, doLimits=False)
        elif mode == "both":
            c0Idx = np.where( cathode == 0)
            c1Idx = np.where( cathode == 1)
            plu.drawPads(ax,  x[c0Idx], y[c0Idx], dx[c0Idx], dy[c0Idx], charge[c0Idx],
                        title="", alpha=1.0, doLimits=False, noYTicksLabels=noYTicksLabels)
            plu.drawPads(ax,  x[c1Idx], y[c1Idx], dx[c1Idx], dy[c1Idx], charge[c1Idx],
                        title=title[2], alpha=0.5, doLimits=True, noYTicksLabels=noYTicksLabels)
                        
        return 
    
    
def displayOneClusterByCathode( ax, tracksObj, tID, cID):
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
    print("track t=", tID, len( MCTracks ), MCTracks[tID].x.shape)
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
    print( cl.x )
    print( "xSup,ySup, xInf, yInf :", xSup,ySup, xInf, yInf )
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
            print('x', clusters.x[cID])
            print('dx', clusters.dx[cID])
            print('y', clusters.y[cID])
            print('dy', clusters.dy[cID])
            print("charge", clusters.charge[cID] )
            print("cathode", clusters.cathode[cID] )
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
        print( "key, number of identical Clusters", key,  l)
        tRef = sameClusters[0][0]
        cRef = sameClusters[0][1]
        print( "tRef, cRef =", tRef, cRef)
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
        print("wi", wi)
        print('mui', mui)
        print( 'vari', vari
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
                print("NOT THE SAME TRACK/CLUSTER t=", t,"/", tRef, ", c=",c ,"/", cRef)
                print( "ev=", ev, ", mcLabel=", mcLabel, ", code=", partCode, ", clID=", iCl, ", nPads=", nPads )
            # Bounds of the Reco Cluster
            xMin = np.min( tracksObj.tracks[t][c].x - tracksObj.tracks[t][c].dx) 
            xMax = np.max( tracksObj.tracks[t][c].x + tracksObj.tracks[t][c].dx) 
            yMin = np.min( tracksObj.tracks[t][c].y - tracksObj.tracks[t][c].dy) 
            yMax = np.max( tracksObj.tracks[t][c].y + tracksObj.tracks[t][c].dy) 
            if xMin != xInf or xMax != xSup or yMin != yInf or yMax != ySup :
                print( "WARNING: xyInfSupRef", xInf, xSup, yInf, ySup)
                print( "WARNING: xyMinMax", xMin, xMax, yMin, yMax)
                
            
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
          print( "WARNING: ", nbMCOutside, "/", len(xyMCs), "MC particle outside of the cluster")
        l = len( xyMCs )
        w = np.ones( l )
        mu = np.array( xyMCs )
        var = np.zeros( (l,2))
        print( "??? mu shape", mu.shape, w.shape, var.shape)

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
          print( "??? len(xCh), shape", len(xCh), muCh.shape )

          plu.drawModelComponents( ax[0, 1], wCh, muCh, varCh, color='black', pattern='cross')        
          plu.drawModelComponents( ax[0, 2], wCh, muCh, varCh, color='black', pattern='cross')        
          
          w = []; mu = []; var =[]
          for cs in sameCSClusters:
            # Bounds of the Reco Cluster
            print( "???", cs )
            print( "CS event", sClusters.id["Event"][cs] )
            print( "CS DE", sClusters.id["DetectElemID"][cs] )
            print( "CS Chamber", sClusters.id["ChamberID"][cs] )
            print( "Chamber len", len(sClusters.id["ChamberID"]) )
            print( " x lenth", len(sClusters.cFeatures["X"] ) )
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
 
def processEMOnPreClusters( preClusters, mcObj, ev, emObj, rejectPadFactor=0.5):
  #
  dTmp = []
  nbrOfPreClusters = len( preClusters.padId[ev] )
  print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
  assign = []
  allDMin = []
  evTP = 0; evFP = 0 ; evFN = 0
  evNbrOfHits = 0
  # Create events storage
  if ev >= len(emObj.w):
    for e in range(len(emObj.w), ev+1):
      emObj.w.append([])
      emObj.mu.append([])
      emObj.var.append([])
  #
  xShort0 = np.empty( shape=(0, 0) ); yShort0 = np.empty( shape=(0, 0) ); dxShort0 = np.empty( shape=(0, 0) ); dyShort0 = np.empty( shape=(0, 0) ); chShort0 = np.empty( shape=(0, 0) );
  xShort1 = np.empty( shape=(0, 0) ); yShort1 = np.empty( shape=(0, 0) ); dxShort1 = np.empty( shape=(0, 0) ); dyShort1 = np.empty( shape=(0, 0) ); chShort1 = np.empty( shape=(0, 0) );
  for pc in range(nbrOfPreClusters):
    xi = preClusters.padX[ev][pc]
    dxi = preClusters.padDX[ev][pc]
    yi = preClusters.padY[ev][pc]
    dyi = preClusters.padDY[ev][pc]
    cathi = preClusters.padCath[ev][pc]
    chi = preClusters.padCharge[ev][pc]
    chIds = preClusters.padChId[ev][pc]
    DEIds = preClusters.padDEId[ev][pc]
    DEIds = np.unique( DEIds )
    chId = np.unique( chIds )[0]
    xy = [ xi, yi]
    dxy = [ dxi, dyi]
    if (chId < 3):
      var = localMaxParam.var1
    else:
      var = localMaxParam.var3
    dxMax = np.max( dxi)
    dyMax = np.max( dyi)
    dxyMax = max( dxMax, dyMax)
    print("dxyMax", dxyMax)
    if dxyMax < 0.5:
      w0, mu0, var0 = findLocalMaxWithSubstraction( xi, yi, dxi, dyi, cathi, chi, var, fitParameter=15.0 )
      xShort = np.empty( shape=(0, 0) )
    else:
      print( "xi", xi)
      idx = np.where( cathi == 0)
      x0 = xi[idx]
      y0 = yi[idx]
      dx0 = dxi[idx]
      dy0 = dyi[idx]
      ch0 = chi[idx]
      idx = np.where( cathi == 1)
      x1 = xi[idx]
      y1 = yi[idx]
      dx1 = dxi[idx]
      dy1 = dyi[idx]
      ch1 = chi[idx]
      if ch0.size != 0:
        z0 = np.max( ch0 )
      else:
        z0 = 0.0
      if ch1.size != 0:
        z1 = np.max( ch1 )
      else:
        z1 = 0
      zMax = max( z0, z1  )
      if (z0 == 0) and (z1 == 0):
        print("WARNING: no pads")
        input("next")
      """
      fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 7) )
      plu.setLUTScale( 0.0, zMax ) 
      plu.drawPads(ax[0,0], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,1], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,2], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=0.5, doLimits=True)
      plu.drawPads(ax[0,2], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=0.5, doLimits=True)
      """
      xShort0, yShort0, dxShort0, dyShort0, chShort0, chShort1 = plu.shorteningPads( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1)
      # xShort1, yShort1, dxShort1, dyShort1, chShort1 = plu.shorteningPads( x1, y1, dx1, dy1, ch1, 
      #                                                  xShort0, yShort0, dxShort0, dyShort0, chShort0)
      xShort1 = np.copy( xShort0 )
      dxShort1 = np.copy( dxShort0 )
      yShort1 = np.copy( yShort0 )
      dyShort1 = np.copy( dyShort0 )
      xShort = np.hstack( [xShort0, xShort1])
      yShort = np.hstack( [yShort0, yShort1])
      dxShort = np.hstack( [dxShort0, dxShort1])
      dyShort = np.hstack( [dyShort0, dyShort1])
      chShort = np.hstack( [chShort0, chShort1])
      cathShort = np.hstack( [np.zeros( xShort0.shape[0], dtype=np.int ), np.ones( xShort1.shape[0], dtype=np.int ) ])
      print("xShort0", xShort0)
      print("xShort1", xShort1)
      if xShort.size != 0:
        w0, mu0, var0 = findLocalMaxWithSubstraction( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, fitParameter=15.0 )
        print(" sum ch0, cShort0", np.sum(ch0), np.sum( chShort0 ) )
        print(" sum ch1, cShort1", np.sum(ch1), np.sum( chShort1 ) )
        """
        plu.drawPads(ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,2], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=0.5, doLimits=True)
        plu.drawPads(ax[1,2], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=0.5, doLimits=True)
        plt.show()
        """
      else:
        w0 = np.empty( shape=(0, 0) )
        m0 = []
        var0 = []
      #
    #
    if ( w0.size != 0 ):
      if xShort.size != 0:
        xu = xShort; yu = yShort; dxu = dxShort; dyu = dyShort; cathu = cathShort; chu = chShort
      else:
        xu = xi; yu = yi; dxu = dxi; dyu = dyi; cathu = cathi; chu = chi 
      wEM, muEM, varEM = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, w0, mu0, var0, cstVar=True )
      wf, muf, varf = filterModelWithMag( wEM, muEM, varEM )
      wFinal, muFinal, varFinal = EM.simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, wf, muf, varf, cstVar=True )
      K = wFinal.shape[0]
      cutOff = 1.e-02 / K
      while (np.any( wFinal < cutOff ) ):
        idx = np.where( wFinal >= cutOff )
        wFinal = wFinal[idx]
        muFinal = muFinal[idx]
        varFinal = varFinal[idx]
        wFinal, muFinal, varFinal = EM.simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, wFinal, muFinal, varFinal, cstVar=True )
        K = wFinal.shape[0]
        cutOff = 1.e-02 / K
      if (wFinal.size < 4):
        npxy = np.array( [xi, yi] )
        npdxy = np.array( [dxi, dyi] )
        if (chId < 3):
          varCst = localMaxParam.var1 
        else:
          varCst = localMaxParam.var3
        K = wFinal.size
        # varCst = np.tile( varCst, (K, 1))
        print(varCst)
        print(muFinal)
        wFit, muFit, varFit = CF.clusterFit( CF.err_fct2, wFinal, muFinal, varCst, npxy , npdxy, chi, jacobian=CF.jac_fct2)
        wFinal = wFit
        muFinal = muFit
        varFinal = varCst = np.tile( varCst, (K, 1))
      #
      print(wFinal)
      print(muFinal)
      idx = np.where( wFinal != 0.0)
      wFinal = wFinal[idx]
      muFinal = muFinal[idx]
      varFinal = varFinal[idx]
      print(" w0.size", w0.size)
      print(" wf.size", wf.size)
      print(" wFinal.size", wFinal.size)
      print(" # reco Hits", preClusters.rClusterX[ev][pc].size)
      # input("next")

      #    
      #
      emObj.w[ev].append(wFinal)
      emObj.mu[ev].append(muFinal)
      emObj.var[ev].append(varFinal)
      #
      # Assign Cluster hits to MC hits
      print(wFinal)
      print(muFinal)
      x0 = muFinal[:,0]
      y0 = muFinal[:,1]
      spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
      rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
      rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
      spanBox = plu.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                             preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
      match, nbrOfHits, TP, FP, FN, dMin, mcHitsInvolved = matchMCTrackHits(x0, y0, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
      if (match > 0.33):
        assign.append( (pc, match, nbrOfHits, TP, FP,FN, mcHitsInvolved ) )
        evTP += TP; evFP += FP; evFN += FN
        allDMin.append( dMin )
        evNbrOfHits += nbrOfHits
      # Debug
      """
      if pc == 26:
        a = input("PreCluster="+str(pc))
      """
      #
    else:
      wFinal = w0
      muFinal = mu0
      varFinal = var0
    # if wFinal.size > 0 or wFinal.size == 0 : 
    if wFinal.size > 0 and chId > 4 : 
      print("match, dMin", match, dMin)
      if wFinal.size == 1:
        dTmp.extend( list( dMin) )
        u = np.array( dTmp)
        print( "dMin average", np.sum(u)/ u.size )
      fig, ax = displayAPrecluster(xy, dxy, cathi, chi, wFinal, muFinal, varFinal, mcObj, preClusters, ev, pc, DEIds )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var, ax[1] )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var )
      if( xShort.size !=0 ):
        plu.drawPads( ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, title="Integral")
        plu.drawPads( ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1,  alpha=1.0, title="Integral")
      #
      plu.drawModelComponents( ax[0, 1], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[0, 2], wEM, muEM, varEM, color='red', pattern="o" )
      plu.drawModelComponents( ax[0, 2], wFinal, muFinal, varFinal, color='red', pattern="o" )
      plu.drawModelComponents( ax[1, 0], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[1, 1], wEM, muEM, varEM, color='red', pattern="cross" )
      plu.drawModelComponents( ax[1, 2], wf, muf, varf, color='red', pattern="cross" )
      plu.drawModelComponents( ax[1, 3], wFinal, muFinal, varFinal, color='red', pattern="cross" )
      frame = plu.getPadBox(xy[0], xy[1], dxy[0], dxy[1] )
      drawMCHitsInFrame ( ax[0,0], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[0,1], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[0,2], frame, mcObj, ev, DEIds )      # input("Next cluster")
      title = "Ev=" +str(ev)+ ", pc=" + str(pc) + ", DEId="+ str(DEIds)
      # fig.inftitle( title )
      title += str( np.min( chi) ) + "<=" + str(np.max( chi)) 
      fig.suptitle( title )
      plt.show()
  # end loop pc
  print( "# hits=", evNbrOfHits, "TP, FP, FN = (", evTP, evFP, evFN, "), assigned preClusters", len(assign),"/", nbrOfPreClusters  ) 
  return evNbrOfHits, evTP, evFP, evFN, allDMin, assign

def processEMOnPreClustersV3( preClusters, mcObj, ev, emObj, rejectPadFactor=0.5):
  # with cath projection on one plane
  # and connex coponents
  dTmp = []
  nbrOfPreClusters = len( preClusters.padId[ev] )
  print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
  assign = []
  allDMin = []
  evTP = 0; evFP = 0 ; evFN = 0
  evNbrOfHits = 0
  # Create events storage
  if ev >= len(emObj.w):
    for e in range(len(emObj.w), ev+1):
      emObj.w.append([])
      emObj.mu.append([])
      emObj.var.append([])
  #
  xShort0 = np.empty( shape=(0, 0) ); yShort0 = np.empty( shape=(0, 0) ); dxShort0 = np.empty( shape=(0, 0) ); dyShort0 = np.empty( shape=(0, 0) ); chShort0 = np.empty( shape=(0, 0) );
  xShort1 = np.empty( shape=(0, 0) ); yShort1 = np.empty( shape=(0, 0) ); dxShort1 = np.empty( shape=(0, 0) ); dyShort1 = np.empty( shape=(0, 0) ); chShort1 = np.empty( shape=(0, 0) );
  for pc in range(nbrOfPreClusters):
    print("###")
    print("### New Pre Cluster", pc,"/", ev)
    print("###")
    xi = preClusters.padX[ev][pc]
    dxi = preClusters.padDX[ev][pc]
    yi = preClusters.padY[ev][pc]
    dyi = preClusters.padDY[ev][pc]
    cathi = preClusters.padCath[ev][pc]
    chi = preClusters.padCharge[ev][pc]
    chIds = preClusters.padChId[ev][pc]
    DEIds = preClusters.padDEId[ev][pc]
    DEIds = np.unique( DEIds )
    chId = np.unique( chIds )[0]
    xy = [ xi, yi]
    dxy = [ dxi, dyi]
    if (chId < 3):
      var = localMaxParam.var1
    else:
      var = localMaxParam.var3
    dxMax = np.max( dxi)
    dyMax = np.max( dyi)
    dxyMax = max( dxMax, dyMax)
    print("dxyMax", dxyMax)
    wFinal = []
    muFinal = []
    varFinal = []
    if True:
      # print( "xi", xi)
      idx = np.where( cathi == 0)
      x0 = xi[idx]
      y0 = yi[idx]
      dx0 = dxi[idx]
      dy0 = dyi[idx]
      ch0 = chi[idx]
      idx = np.where( cathi == 1)
      x1 = xi[idx]
      y1 = yi[idx]
      dx1 = dxi[idx]
      dy1 = dyi[idx]
      ch1 = chi[idx]
      if ch0.size != 0:
        z0 = np.max( ch0 )
      else:
        z0 = 0.0
      if ch1.size != 0:
        z1 = np.max( ch1 )
      else:
        z1 = 0
      zMax = max( z0, z1  )
      if (z0 == 0) and (z1 == 0):
        print("WARNING: no pads")
        input("next")
      plu.setLUTScale( 0.0, zMax ) 
      print("sum ch0, ch1 ", np.sum( ch0), np.sum( ch1))
      if ( abs( np.sum( ch0) - np.sum( ch1)) >= 1.0e-05 ):
        print( "Not symetric charge")
        input("next")
      """
      plu.drawPads(ax[0,0], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,1], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,2], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=0.5, doLimits=True)
      plu.drawPads(ax[0,2], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=0.5, doLimits=True)
      """
      xShort0, yShort0, dxShort0, dyShort0, chShort0, chShort1 = plu.shorteningPads( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1)

      if xShort0.size != 0:
        nGrp, groups = plu.getConnexComponents(xShort0, yShort0, dxShort0, dyShort0)
        ##############
        for g in range(1, nGrp+1):
          print("### Group ", g, "/", nGrp)
          grpIdx =np.where( groups == g )
          xShg0 = xShort0[ grpIdx ]
          xShg1 = np.copy( xShg0 )
          dxShg0 = dxShort0[ grpIdx ]
          dxShg1 = np.copy( dxShg0 )
          yShg0 = yShort0[ grpIdx ]
          yShg1 = np.copy( yShg0 )
          dyShg0 = dyShort0[ grpIdx ]
          dyShg1 = np.copy( dyShg0 )
          xShort = np.hstack( [xShg0, xShg1])
          yShort = np.hstack( [yShg0, yShg1])
          dxShort = np.hstack( [dxShg0, dxShg1])
          dyShort = np.hstack( [dyShg0, dyShg1])
          chShg0 = chShort0[ grpIdx ]
          chShg1 = chShort1[ grpIdx ]
          chShort = np.hstack( [chShg0, chShg1])
          cathShort = np.hstack( [np.zeros( xShg0.shape[0], dtype=np.int ), np.ones( xShg1.shape[0], dtype=np.int ) ])
          # print("xShort0", xShort0)
          # print("xShort1", xShort1)          
          w0, mu0, var0 = findLocalMaxWithSubstraction( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, mcObj, preClusters, ev, pc, DEIds,fitParameter=15.0, chId=chId )
          print(" sum ch0, cShort0", np.sum(ch0), np.sum( chShort0 ) )
          print(" sum ch1, cShort1", np.sum(ch1), np.sum( chShort1 ) )          
          xu = xShort; yu = yShort; dxu = dxShort; dyu = dyShort; cathu = cathShort; chu = chShort
          wEM, muEM, varEM = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, w0, mu0, var0, cstVar=True )
          wf, muf, varf = filterModelWithMag( wEM, muEM, varEM )
          wGrp, muGrp, varGrp = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, wf, muf, varf, cstVar=True )
          K = wGrp.shape[0]
          cutOff = 1.e-02 / K
          while (np.any( wGrp < cutOff ) ):
            idx = np.where( wGrp >= cutOff )
            wGrp = wGrp[idx]
            muGrp = muGrp[idx]
            varGrp = varGrp[idx]
            wGrp, muGrp, varGrp = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, wGrp, muGrp, varGrp, cstVar=True )
            K = wGrp.shape[0]
            cutOff = 1.e-02 / K
          #
          # LMS
          """
          if (wFinal.size < 0):
            npxy = np.array( [xu, yu] )
            npdxy = np.array( [dxu, dyu] )
            if (chId < 3):
              varCst = localMaxParam.var1 
            else:
              varCst = localMaxParam.var3
            K = wFinal.size
            # varCst = np.tile( varCst, (K, 1))
            print(varCst)
            print(muFinal)
            wFit, muFit, varFit = CF.clusterFit( CF.err_fct2, wFinal, muFinal, varCst, npxy , npdxy, chu, jacobian=CF.jac_fct2)
            wFinal = wFit
            muFinal = muFit
            varFinal = varCst = np.tile( varCst, (K, 1))
          """
          wFinal.append( wGrp )
          muFinal.append( muGrp )
          varFinal.append( varGrp )
        # End group loop
        wFinal = np.concatenate( wFinal )
        muFinal = np.concatenate( muFinal )
        varFinal = np.concatenate( varFinal )
        # input("next")

        """
        plu.drawPads(ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,2], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=0.5, doLimits=True)
        plu.drawPads(ax[1,2], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=0.5, doLimits=True)
        plt.show()
        """
      else:
        w0 = np.empty( shape=(0, 0) )
        m0 = []
        var0 = []
      #
    #
    # TP, FN, FP, ... analysis
    #
    if ( w0.size != 0 ):
      idx = np.where( wFinal != 0.0)
      wFinal = wFinal[idx]
      muFinal = muFinal[idx]
      varFinal = varFinal[idx]
      print(" w0.size", w0.size)
      print(" wf.size", wf.size)
      print(" wFinal.size", wFinal.size)
      print(" # reco Hits", preClusters.rClusterX[ev][pc].size)
      # input("next")

      #    
      #
      emObj.w[ev].append(wFinal)
      emObj.mu[ev].append(muFinal)
      emObj.var[ev].append(varFinal)
      #
      # Assign Cluster hits to MC hits
      print(wFinal)
      print(muFinal)
      x0 = muFinal[:,0]
      y0 = muFinal[:,1]
      spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
      rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
      rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
      spanBox = plu.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                             preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
      match, nbrOfHits, TP, FP, FN, dMin, mcHitsInvolved = matchMCTrackHits(x0, y0, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
      if (match > 0.33):
        assign.append( (pc, match, nbrOfHits, TP, FP,FN, mcHitsInvolved ) )
        evTP += TP; evFP += FP; evFN += FN
        allDMin.append( dMin )
        evNbrOfHits += nbrOfHits
      # Debug
      """
      if pc == 26:
        a = input("PreCluster="+str(pc))
      """
      #
    else:
      wFinal = w0
      muFinal = mu0
      varFinal = var0
    #
    # Display
    #
    # if wFinal.size > 0 or wFinal.size == 0 : 
    if wFinal.size > 0 and chId > 4 : 
      fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 7) )
      print("match, dMin", match, dMin)
      if wFinal.size == 1:
        dTmp.extend( list( dMin) )
        u = np.array( dTmp)
        print( "dMin average", np.sum(u)/ u.size )
      displayAPrecluster(ax, xy, dxy, cathi, chi, wFinal, muFinal, varFinal, mcObj, preClusters, ev, pc, DEIds )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var, ax[1] )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var )
      # w0, mu0, var0 = findLocalMaxWithLaplacian( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, ax=ax[1,:] )
      plu.setLUTScale( 0.0, zMax ) 

      plu.drawPads( ax[0,3], xShort0, yShort0, dxShort0, dyShort0, chShort0, doLimits=False, alpha=1.0, )
      plu.drawModelComponents( ax[0, 3], wFinal, muFinal, varFinal, color='red', pattern="o" )

      if( True and xShort.size !=0 ):
        plu.drawPads( ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, )
        if xShort1.size !=0 :
          plu.drawPads( ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1,  alpha=1.0, )
        plu.drawPads( ax[1,2], xShort0, yShort0, dxShort0, dyShort0, chShort1 - chShort0,  alpha=1.0, )
        plu.drawPads( ax[1,3], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, )
      #
      if True:
        plu.drawModelComponents( ax[0, 1], w0, mu0, var0, color='red', pattern="o" )
        plu.drawModelComponents( ax[0, 2], wEM, muEM, varEM, color='red', pattern="o" )
        plu.drawModelComponents( ax[0, 3], wFinal, muFinal, varFinal, color='red', pattern="o" )
        plu.drawModelComponents( ax[1, 0], w0, mu0, var0, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 1], wEM, muEM, varEM, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 2], wf, muf, varf, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 3], wFinal, muFinal, varFinal, color='red', pattern="cross" )
        ax[0,0].set_title("MC/Reco")
        ax[0,1].set_title("Theta0")
        ax[0,2].set_title("EM") 
        ax[0,3].set_title("Theta Final") 
        ax[1,0].set_title("Theta0") 
        ax[1,1].set_title("EM") 
        ax[1,2].set_title("Filter") 
        ax[1,3].set_title("Theta Final")
      else:
        ax[1,0].set_title("1st peak")
        
      frame = plu.getPadBox(xy[0], xy[1], dxy[0], dxy[1] )
      drawMCHitsInFrame ( ax[1,0], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,1], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,2], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,3], frame, mcObj, ev, DEIds )      # input("Next cluster")
      title = "Ev=" +str(ev)+ ", pc=" + str(pc) + ", DEId="+ str(DEIds)
      # fig.inftitle( title )
      title += str( np.min( chi) ) + "<=" + str(np.max( chi)) 
      fig.suptitle( title )
      plt.show()
  # end loop pc
  print( "# hits=", evNbrOfHits, "TP, FP, FN = (", evTP, evFP, evFN, "), assigned preClusters", len(assign),"/", nbrOfPreClusters  ) 
  return evNbrOfHits, evTP, evFP, evFN, allDMin, assign

def processEMOnPreClustersV4( preClusters, mcObj, ev, emObj, rejectPadFactor=0.5):
  # with cath projection on one plane
  # and connex coponents
  dTmp = []
  nbrOfPreClusters = len( preClusters.padId[ev] )
  print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
  assign = []
  allDMin = []
  evTP = 0; evFP = 0 ; evFN = 0
  evNbrOfHits = 0
  # Create events storage
  if ev >= len(emObj.w):
    for e in range(len(emObj.w), ev+1):
      emObj.w.append([])
      emObj.mu.append([])
      emObj.var.append([])
  #
  xShort0 = np.empty( shape=(0, 0) ); yShort0 = np.empty( shape=(0, 0) ); dxShort0 = np.empty( shape=(0, 0) ); dyShort0 = np.empty( shape=(0, 0) ); chShort0 = np.empty( shape=(0, 0) );
  xShort1 = np.empty( shape=(0, 0) ); yShort1 = np.empty( shape=(0, 0) ); dxShort1 = np.empty( shape=(0, 0) ); dyShort1 = np.empty( shape=(0, 0) ); chShort1 = np.empty( shape=(0, 0) );
  for pc in range(nbrOfPreClusters):
  # for pc in range(150, 151):
    print("###")
    print("### New Pre Cluster", pc,"/", ev)
    print("###")
    xi = preClusters.padX[ev][pc]
    dxi = preClusters.padDX[ev][pc]
    yi = preClusters.padY[ev][pc]
    dyi = preClusters.padDY[ev][pc]
    cathi = preClusters.padCath[ev][pc]
    chi = preClusters.padCharge[ev][pc]
    chIds = preClusters.padChId[ev][pc]
    DEIds = preClusters.padDEId[ev][pc]
    DEIds = np.unique( DEIds )
    chId = np.unique( chIds )[0]
    xy = [ xi, yi]
    dxy = [ dxi, dyi]
    if (chId < 3):
      var = localMaxParam.var1
    else:
      var = localMaxParam.var3
    dxMax = np.max( dxi)
    dyMax = np.max( dyi)
    dxyMax = max( dxMax, dyMax)
    wFinal = []
    muFinal = []
    varFinal = []
    wSave = np.empty(shape=[0])
    muSave = np.empty(shape=[0])
    varSave = np.empty(shape=[0])   
    # Print cluster info
    print("# DEIds", DEIds)
    print("# Nbr of pads:", xi.size)
    print("# Saturated pads", np.sum( preClusters.padSaturated[ev][pc]))
    print("# Calibrated pads", np.sum( preClusters.padCalibrated[ev][pc]))
    if( np.sum( preClusters.padSaturated[ev][pc]) != 0):
        input("pb with saturated pads")
    if ( xi.size !=  np.sum( preClusters.padCalibrated[ev][pc])) :
        input("pb with calibrated pads")
    idx = np.where( cathi == 0 )
    nz0 = idx[0].size
    zz0 = np.sum( chi[idx] )
    print("# Charge sum on cath 0", np.sum( chi[idx]) )
    idx = np.where( cathi == 1 )
    nz1 = idx[0].size
    zz1 = np.sum( chi[idx] )
    print("# Charge sum on cath 1", np.sum( chi[idx]) )
    if ( np.abs( 2 * (zz1 - zz0)/(zz1+zz0)) > 0.02 ):
      print("### Warning: different charge on the 2 cathodes", zz0, zz1)
      # input("next")
    chargeOnCath = [ zz0, zz1 ]
    onlyOneCath = ((nz0 == 0) or (nz1 == 0), (nz1 == 0)*1 )
    if onlyOneCath[0]: print("# Warning: only one cathode")  
    print("??? onlyOneCath", onlyOneCath, nz0 ,  nz1)
    if True:
      print( "xi", xi)
      print( "yi", yi)
      idx = np.where( cathi == 0)
      x0 = xi[idx]
      y0 = yi[idx]
      dx0 = dxi[idx]
      dy0 = dyi[idx]
      ch0 = chi[idx]
      idx = np.where( cathi == 1)
      x1 = xi[idx]
      y1 = yi[idx]
      dx1 = dxi[idx]
      dy1 = dyi[idx]
      ch1 = chi[idx]
      if ch0.size != 0:
        z0 = np.max( ch0 )
      else:
        z0 = 0.0
      if ch1.size != 0:
        z1 = np.max( ch1 )
      else:
        z1 = 0
      zMax = max( z0, z1  )
      if (z0 == 0) and (z1 == 0):
        print("WARNING: no pads")
        input("next")
      plu.setLUTScale( 0.0, zMax ) 
      """
      plu.drawPads(ax[0,0], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,1], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,2], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=0.5, doLimits=True)
      plu.drawPads(ax[0,2], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=0.5, doLimits=True)
      """
      if not onlyOneCath[0]:
        (xShort0, yShort0, dxShort0, dyShort0, chShort0, chShort1,
        mapIJToK, mapKToIJ, IinterJ, JinterI ) = plu.shorteningPads( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1)
        # chShort Mean (take care to the proption 0.5) ???
        # The both sums should be equal
        # chShortM = 0.5 * (chShort0 + chShort1) 
        chShortM =  (chShort0 + chShort1) 
        print("# short/cathode 0 & 1", np.sum(chShort0), np.sum( ch0), np.sum(chShort1), np.sum(ch1))
        if( np.abs( np.sum(chShort0)- np.sum(ch0)) + np.abs( np.sum(chShort1)- np.sum(ch1) )  > 0.1 ):
            input("Charge lost !")
        if (chShort0 != chShort1).all():
          print("chShort0", chShort0)
          print("chShort1", chShort1)
          # unused saveChShort0 = np.copy( chShort0)
          # unused ??? chShort0 = (chShort0 + chShort1)*0.5
          print("sum (chShort0 + chShort1)", np.sum(chShortM) )
          # input("chShorts differ")
      else:
        print("No shortening")
        xShort0 = xi; dxShort0 = dxi; yShort0 = yi; dyShort0 = dyi; chShort0 = chi
        chShotM = chShort0
      # print( "??? xShort0", xShort0)
      if xShort0.size != 0 :
        if not onlyOneCath[0]:
          nGrp, groups = plu.getConnexComponents(xShort0, yShort0, dxShort0, dyShort0)
          # unused chShort0 = np.copy( saveChShort0)
          # Get groups without 1-Pad clusters and maping to cathodes
          nGrp1, groups1, wellSplit, padGrpCath0, padGrpCath1 = plu.associateGrpPads( nGrp, groups, chShort0, chShort1, ch0, ch1,  mapIJToK, mapKToIJ, IinterJ, JinterI )
          idx0 = np.where( cathi == 0)[0] 
          idx1 = np.where( cathi == 1)[0]
          padGrpCath0 = np.array( padGrpCath0 )
          padGrpCath1 = np.array( padGrpCath1 )
          # Check
          nCath0 = 0; nCath1 = 0
          for gg in range(nGrp1+1):
            iii = np.where(padGrpCath0==gg)
            jjj = np.where(padGrpCath1==gg)
            nCath0 += iii[0].size
            nCath1 += jjj[0].size
            print("group", gg, "nbr associated Pads cath0/1", iii[0].size, jjj[0].size)
          origCath0 = np.sum( cathi==0 )
          origCath1 = np.sum( cathi==1 )
          if ( nCath0 != origCath0  or nCath1 != origCath1):
             print( "Original Cath0/1", origCath0, origCath1, "associated cath0/1", nCath0, nCath1)
             if InputWarning : input( "Pb nbr of associated pads ")
          if (padGrpCath0.size != idx0.size) or (padGrpCath1.size != idx1.size):
             print( "padGrpCath0",  padGrpCath0.size , padGrpCath0)
             print( "padGrpCath1",  padGrpCath1.size , padGrpCath1)
             print( "idx0",  idx0.size , idx0 )
             print( "idx1",  idx1.size , idx1 )
             # input( "Different size with association")
          if ( nGrp != nGrp1 or not wellSplit.any() ):
              print("nGrp", nGrp, nGrp1 )
              print("wellSplit", wellSplit) 
              print("padGrpCath0", padGrpCath0) 
              print("padGrpCath1", padGrpCath1)
              # print("saveChShort0", chShort0)
              print("chShort1", chShort1)
              # input("next")
          nGrp =nGrp1
          groups = groups1
          # unused chShort0 = (chShort0 + chShort1) * 0.5
        else:
           # Only one cathode
           nGrp, groups = plu.getConnexComponents(xShort0, yShort0, dxShort0, dyShort0)
           wellSplit = [True for i in range(nGrp+1)]
           idx0 = np.where( cathi == 0)[0] 
           idx1 = np.where( cathi == 1)[0]
           # TODO : to optimize ... a hammer !
           # nGrp, groups, wellSplit, padGrpCath0, padGrpCath1 = plu.associateGrpPads( nGrp, groups, chShort0, chShort1, ch0, ch1,  mapIJToK, mapKToIJ, IinterJ, JinterI )
           if ch0.size == 0:
              padGrpCath0 = np.zeros( 0 )
              padGrpCath1 = groups
           else:
              padGrpCath1 = np.zeros( 0 )
              padGrpCath0 = groups
           print("??? groups", groups)
           print("??? padGrpCath0/1", padGrpCath0, padGrpCath1)
        ##############
        # To ccheck sumof charge
        sum0 = 0; sum1 = 0
        for g in range(1, nGrp+1):
          print("### Group ", g, "/", nGrp)
          grpIdx =np.where( groups == g )
          if ( grpIdx[0].size > 1 ):
              print("??? grpIdx size", grpIdx[0].size )
              xShg0 = xShort0[ grpIdx ]
              xShg1 = np.copy( xShg0 )
              dxShg0 = dxShort0[ grpIdx ]
              dxShg1 = np.copy( dxShg0 )
              yShg0 = yShort0[ grpIdx ]
              yShg1 = np.copy( yShg0 )
              dyShg0 = dyShort0[ grpIdx ]
              dyShg1 = np.copy( dyShg0 )
              xShort = np.hstack( [xShg0, xShg1])
              yShort = np.hstack( [yShg0, yShg1])
              dxShort = np.hstack( [dxShg0, dxShg1])
              dyShort = np.hstack( [dyShg0, dyShg1])
              chShg0 = chShort0[ grpIdx ]
              chShg1 = chShort1[ grpIdx ]
              chShort = np.hstack( [chShg0, chShg1])
              cathShort = np.hstack( [np.zeros( xShg0.shape[0], dtype=np.int ), np.ones( xShg1.shape[0], dtype=np.int ) ])
              # print("xShort0", xShort0)
              # print("xShort1", xShort1)          
              # w0, mu0, var0 = findLocalMaxWithSubstraction( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, mcObj, preClusters, ev, pc, DEIds,fitParameter=15.0, chId=chId )
              # print("??? xShort0", xShort0.shape, xShort0 )
              # print("??? yShort0", yShort0.shape, yShort0 )
              print("??? chShortM", chShortM[grpIdx].shape, chShortM[grpIdx] )
              #print("??? cathSh", cathShort.shape, cathShort)
              # w0, mu0, var0, lapl = findLocalMaxWithLaplacian( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, graphTest= (chId < 0) ) # , mcObj, preClusters, ev, pc, DEIds,fitParameter=15.0, chId=chId )
              # w0, mu0, var0, lapl = findLocalMaxWithLaplacian( xShort0, yShort0, dxShort0, dyShort0, chShort0, var, graphTest= (chId < 0), verbose=True ) # , mcObj, preClusters, ev, pc, DEIds,fitParameter=15.0, chId=chId )
              w0, mu0, var0, lapl = findLocalMaxWithLaplacian( xShg0, yShg0, dxShg0, dyShg0, chShortM[grpIdx], var, graphTest= (chId < 0), verbose=True ) # , mcObj, preClusters, ev, pc, DEIds,fitParameter=15.0, chId=chId )
              print("w0, mu0, var0", w0, mu0, var0)
              print(" sum ch0, cShort0", np.sum(ch0), np.sum( chShort0 ) )
              print(" sum ch1, cShort1", np.sum(ch1), np.sum( chShort1 ) )          
              xu = xShort; yu = yShort; dxu = dxShort; dyu = dyShort; cathu = cathShort; chu = chShort
              if w0.size != 0:
                # print("shapes", xu.shape, chu.shape, cathu.shape)
                # print("xu",xu)
                # print("dxu",dxu)
                # print("yu",yu)
                # print("chu",chu)
                # print("cathu", cathu) 
                wEM, muEM, varEM = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, w0, mu0, var0, cstVar=True )
                print("# wEM, muEM, varEM", wEM, muEM, varEM)
                wf, muf, varf = filterModelWithMag( wEM, muEM, varEM )
                print("# wf, muf, varf", wf, muf, varf)
                wGrp, muGrp, varGrp = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, wf, muf, varf, cstVar=True )
                K = wGrp.shape[0]
                cutOff = 1.e-02 / K
                while (np.any( wGrp < cutOff ) ):
                  input("Path used ???")
                  idx = np.where( wGrp >= cutOff )
                  wGrp = wGrp[idx]
                  muGrp = muGrp[idx]
                  varGrp = varGrp[idx]
                  wGrp, muGrp, varGrp = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, wGrp, muGrp, varGrp, cstVar=True )
                  K = wGrp.shape[0]
                  cutOff = 1.e-02 / K
                #
                # LMS
                #
                print("End EM mu", muGrp)
                # if (nGrp == 1):
                print ("group", g, "/", nGrp )
                # print("groups ???", groups)
                # print("cathi ???", cathi)

                
                print("# wellSplit:", wellSplit)
                padInGrp =  np.where( padGrpCath0==g)[0].size + np.where( padGrpCath1==g )[0].size
                nbrOfSeeds = wGrp.size
                ratioPadPerSeed = padInGrp / nbrOfSeeds
                print("# nbrPads in Group", padInGrp)
                print("# ratioPadPerSeed:", ratioPadPerSeed)
                if wellSplit[g] == 0 :
                  input("Not well split")
                if ( wellSplit[g] and ratioPadPerSeed < 11 and nbrOfSeeds < 10):
                  print ("idx0", idx0)
                  print("??? padGrpCath0", padGrpCath0.size, padGrpCath0 )
                  print("??? padGrpCath1", padGrpCath1.size, padGrpCath1 ) 
                  grpToCathIdx0 = idx0[ np.where( padGrpCath0==g ) ]
                  grpToCathIdx1 = idx1[ np.where( padGrpCath1==g ) ]
                  print("??? grpToCathIdx", grpToCathIdx0,  grpToCathIdx1)
                  sum0 += np.sum( chi[grpToCathIdx0] ) 
                  sum1 += np.sum( chi[grpToCathIdx1] ) 
                  wSave = np.copy( wGrp )
                  muSave = np.copy( muGrp )
                  varSave = np.copy( varGrp )
                  # Case on initial pads (2 planes)
                  # npxy = np.array( [xi, yi] )
                  # npdxy = np.array( [dxi, dyi] )
                  # print("??? xi ", xi )
                  # print("??? xi group", xi[grpToCathIdx0], xi[grpToCathIdx1] )
                  # print("??? xi group", yi[grpToCathIdx0], yi[grpToCathIdx1] )
                  xv = np.hstack( [xi[grpToCathIdx0], xi[grpToCathIdx1]] )
                  yv = np.hstack( [yi[grpToCathIdx0], yi[grpToCathIdx1]] )
                  dxv = np.hstack( [dxi[grpToCathIdx0], dxi[grpToCathIdx1]] )
                  dyv = np.hstack( [dyi[grpToCathIdx0], dyi[grpToCathIdx1]] )
                  npxy = np.array( [xv, yv ] )
                  npdxy = np.array( [dxv, dyv] )
                  cathv = np.hstack( [cathi[grpToCathIdx0], cathi[grpToCathIdx1]] )
                  chv = np.hstack( [chi[grpToCathIdx0], chi[grpToCathIdx1]] )
                  # Case 1D pads 
                  # npxy = np.array( [xu, yu] )
                  # npdxy = np.array( [dxu, dyu] )
                  if (chId < 3):
                    varCst = 1.2 * 1.2 * localMaxParam.var1 
                    mType = 0
                  else:
                    varCst =  1.2 * 1.2 * localMaxParam.var3
                    mType = 1
                  if np.unique(cathi).size == 1:
                    nbrOfCath = 1
                  else:
                    nbrOfCath = 2
                  K = wGrp.size
                  # varCst = np.tile( varCst, (K, 1))
                  # wFit, muFit, varFit = CF.clusterFit( CF.err_fct2, wGrp, muGrp, varCst, npxy , npdxy, chi, jacobian=CF.jac_fct2)
                  # Case 2 cathodes
                  # wFit, muFit, varFit = CF.clusterFitMath0( wGrp, muGrp, mType, npxy , npdxy, chi, nbrOfCath, jacobian=None)
                  # 2 cathodes constrains
                  # wFit, muFit, varFit = CF.clusterFitMath( wGrp, muGrp, mType, npxy , npdxy, chi, cathi, jacobian=None)
                  wFit, muFit, varFit = CF.clusterFitMath( wGrp, muGrp, mType, npxy , npdxy, chv, cathv, jacobian=None)
                  print("# Mathieson fitting")
                  # input("Mathieson fitting, next")
                  #
                  # Case one projected plane
                  # Take care ??? : since there are 2 x 1D superposed plane, nbrOfCath = 2
                  # wFit, muFit, varFit = CF.clusterFitMath( wGrp, muGrp, mType, npxy , npdxy, chu, nbrOfCath, jacobian=None)
                  # 
                  wGrp = wFit
                  muGrp = muFit
                  varGrp = varCst = np.tile( varCst, (K, 1))
                  print("End Fit mu", muFit)
                # End LMS
                wFinal.append( wGrp )
                muFinal.append( muGrp )
                varFinal.append( varGrp )
                if muGrp.shape[1] != 2:
                  print("?? muGrp", muGrp )
                  input("Bad shape")
              else:
                print("No solution in this group")
          # end size of grp !=1
          if sum0 != 0 and ( (sum0 != np.sum( chi[cathi==0] )) or (sum1 != np.sum( chi[cathi==1] )) ):
              print ("sum0", sum0, np.sum( chi[cathi==0] ))
              print ("sum1", sum1, np.sum( chi[cathi==1] ))
              # input("Ch sum differs")
          else:
            # Only one pad ... skipp
            # wFinal.append( np.array([1.0]) )
            # muFinal.append( np.array( [ [xShort0[0],yShort0[0] ] ] ) )
            # varFinal.append( np.array( [var] ) )
            # print("???", varFinal)
            if InputWarning: input("Group/cluster whith one pad")
        # End group loop
        # print("??? wFinal", wFinal)
        # print("??? muFinal", muFinal)
        if ( len(wFinal) != 0 ):
         wFinal = np.concatenate( wFinal )
         muFinal = np.concatenate( muFinal )
         varFinal = np.concatenate( varFinal )
        # print("??? muFinal", muFinal)
        # input("next")

        """
        plu.drawPads(ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,2], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=0.5, doLimits=True)
        plu.drawPads(ax[1,2], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=0.5, doLimits=True)
        plt.show()
        """
      else:
        w0 = np.empty( shape=(0, 0) )
        m0 = []
        var0 = []
      #
    #
    # TP, FN, FP, ... analysis
    #
    wFinal = np.array( wFinal )
    if ( wFinal.size != 0 ):
      idx = np.where( wFinal != 0.0)
      wFinal = wFinal[idx]
      muFinal = muFinal[idx]
      varFinal = varFinal[idx]
      print(" w0.size", w0.size)
      print(" wf.size", wf.size)
      print(" wFinal.size", wFinal.size)
      print(" # reco Hits", preClusters.rClusterX[ev][pc].size)
      # input("next")

      #    
      #
      emObj.w[ev].append(wFinal)
      emObj.mu[ev].append(muFinal)
      emObj.var[ev].append(varFinal)
      #
      # Assign Cluster hits to MC hits
      print(wFinal)
      print(muFinal)
      x0 = muFinal[:,0]
      y0 = muFinal[:,1]
      spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
      rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
      rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
      spanBox = plu.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                             preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
      match, nbrOfHits, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved = matchMCTrackHits(x0, y0, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
      if (dxMin.size != 0):
        maxDxMin = np.max( dxMin )
        maxDyMin = np.max( dyMin )
      else:
        maxDxMin = 1.0
        maxDyMin = 1.0
      if (match > 0.33):
        assign.append( (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) )
        evTP += TP; evFP += FP; evFN += FN
        allDMin.append( dMin )
        evNbrOfHits += nbrOfHits
        print( "TP, FP, FN", TP, FP, FN)
      # Debug
      """
      if pc == 26:
        a = input("PreCluster="+str(pc))
      """
      #
    else:
      wFinal = w0
      muFinal = mu0
      varFinal = var0
    #
    # Display
    #
    # if wFinal.size < 0 : 
    # if wFinal.size > 0 and pc == 238: 
    # if wFinal.size > 0 and wFinal.size < 4 : 
    # if wFinal.size > 0 and wFinal.size < 3 and chId > 2 : 
    # if wFinal.size < 0 or wFinal.size == 0 : 
    # if wFinal.size > 0 and chId > 4 :
    # if maxDxMin > 0.07 or maxDyMin > 0.05 :
    if maxDxMin > 0.1 or maxDyMin > 0.07 :
      print("maxDxMin,maxDyMin",maxDxMin,maxDyMin)
      fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 7) )
      print("match, dMin", match, dMin)
      if wFinal.size == 1:
        dTmp.extend( list( dMin) )
        u = np.array( dTmp)
        print( "dMin average", np.sum(u)/ u.size )
      displayAPrecluster(ax, xy, dxy, cathi, chi, wFinal, muFinal, varFinal, mcObj, preClusters, ev, pc, DEIds )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var, ax[1] )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var )
      # w0, mu0, var0 = findLocalMaxWithLaplacian( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, ax=ax[1,:] )
      plu.setLUTScale( 0.0, zMax ) 
      if xShort0.size != 0:
        plu.drawPads( ax[0,3], xShort0, yShort0, dxShort0, dyShort0, chShort0,  doLimits=False, alpha=1.0, )
      plu.drawModelComponents( ax[0, 3], wFinal, muFinal, varFinal, color='red', pattern="o" )
      if (wSave.size != 0 ):
        plu.drawModelComponents( ax[0, 3], wSave, muSave, varSave, color='green', pattern="+" )

      if( True and xShort0.size !=0 ):
        plu.drawPads( ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, )

        """
        if xShort1.size !=0 :
          plu.drawPads( ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1,  alpha=1.0, )
        """
        plu.setLUTScale( np.min( lapl ), np.max(lapl) ) 
        # print("??? xShort", xu)
        # print("??? lapl", lapl)
        plu.drawPads( ax[1,1], xShort[cathShort==0], yShort[cathShort==0], dxShort[cathShort==0], dyShort[cathShort==0], lapl,  alpha=1.0, )
        plu.setLUTScale( 0.0, zMax ) 

        # plu.drawPads( ax[1,2], xShort0, yShort0, dxShort0, dyShort0, chShort1 - chShort0,  alpha=1.0, )
        plu.drawPads( ax[1,3], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, )
      #
      if True:
        plu.drawModelComponents( ax[0, 1], w0, mu0, var0, color='red', pattern="o" )
        plu.drawModelComponents( ax[0, 2], wEM, muEM, varEM, color='red', pattern="o" )
        plu.drawModelComponents( ax[0, 3], wFinal, muFinal, varFinal, color='red', pattern="o" )
        plu.drawModelComponents( ax[1, 0], w0, mu0, var0, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 1], wEM, muEM, varEM, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 2], wf, muf, varf, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 3], wFinal, muFinal, varFinal, color='red', pattern="cross" )
        ax[0,0].set_title("MC/Reco")
        ax[0,1].set_title("Theta0")
        ax[0,2].set_title("EM") 
        ax[0,3].set_title("Theta Final") 
        ax[1,0].set_title("Theta0") 
        ax[1,1].set_title("EM") 
        ax[1,2].set_title("Filter") 
        ax[1,3].set_title("Theta Final")
      else:
        ax[1,0].set_title("1st peak")
        
      frame = plu.getPadBox(xy[0], xy[1], dxy[0], dxy[1] )
      drawMCHitsInFrame ( ax[0,0], frame, mcObj, ev, DEIds )
      drawMCHitsInFrame ( ax[0,1], frame, mcObj, ev, DEIds )
      drawMCHitsInFrame ( ax[0,2], frame, mcObj, ev, DEIds )
      drawMCHitsInFrame ( ax[0,3], frame, mcObj, ev, DEIds )      
      drawMCHitsInFrame ( ax[1,0], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,1], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,2], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,3], frame, mcObj, ev, DEIds )      # input("Next cluster")
      title = "Ev=" +str(ev)+ ", pc=" + str(pc) + ", DEId="+ str(DEIds)
      # fig.inftitle( title )
      title += str( np.min( chi) ) + "<=" + str(np.max( chi)) 
      fig.suptitle( title )
      plt.show()
  # end loop pc
  print( "# hits=", evNbrOfHits, "TP, FP, FN = (", evTP, evFP, evFN, "), assigned preClusters", len(assign),"/", nbrOfPreClusters  ) 
  return evNbrOfHits, evTP, evFP, evFN, allDMin, assign

def processEMOnPreClustersV5( preClusters, mcObj, ev, emObj, rejectPadFactor=0.5):
  # with cath projection on one plane
  # and connex coponents
  dTmp = []
  nbrOfPreClusters = len( preClusters.padId[ev] )
  print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
  assign = []
  allDMin = []
  evTP = 0; evFP = 0 ; evFN = 0
  evNbrOfHits = 0
  # Create events storage
  if ev >= len(emObj.w):
    for e in range(len(emObj.w), ev+1):
      emObj.w.append([])
      emObj.mu.append([])
      emObj.var.append([])
  #
  xShort0 = np.empty( shape=(0, 0) ); yShort0 = np.empty( shape=(0, 0) ); dxShort0 = np.empty( shape=(0, 0) ); dyShort0 = np.empty( shape=(0, 0) ); chShort0 = np.empty( shape=(0, 0) );
  xShort1 = np.empty( shape=(0, 0) ); yShort1 = np.empty( shape=(0, 0) ); dxShort1 = np.empty( shape=(0, 0) ); dyShort1 = np.empty( shape=(0, 0) ); chShort1 = np.empty( shape=(0, 0) );
  for pc in range(nbrOfPreClusters):
  # for pc in range(150, 151):
    print("###")
    print("### New Pre Cluster", pc,"/", ev)
    print("###")
    xi = preClusters.padX[ev][pc]
    dxi = preClusters.padDX[ev][pc]
    yi = preClusters.padY[ev][pc]
    dyi = preClusters.padDY[ev][pc]
    cathi = preClusters.padCath[ev][pc]
    chi = preClusters.padCharge[ev][pc]
    chIds = preClusters.padChId[ev][pc]
    DEIds = preClusters.padDEId[ev][pc]
    DEIds = np.unique( DEIds )
    chId = np.unique( chIds )[0]
    xy = [ xi, yi]
    dxy = [ dxi, dyi]
    if (chId < 3):
      var = localMaxParam.var1
    else:
      var = localMaxParam.var3
    dxMax = np.max( dxi)
    dyMax = np.max( dyi)
    dxyMax = max( dxMax, dyMax)
    wFinal = []
    muFinal = []
    varFinal = []
    wSave = np.empty(shape=[0])
    muSave = np.empty(shape=[0])
    varSave = np.empty(shape=[0])   
    # Print cluster info
    print("# DEIds", DEIds)
    print("# Nbr of pads:", xi.size)
    print("# Saturated pads", np.sum( preClusters.padSaturated[ev][pc]))
    print("# Calibrated pads", np.sum( preClusters.padCalibrated[ev][pc]))
    # if( np.sum( preClusters.padSaturated[ev][pc]) != 0):
    #    input("pb with saturated pads")
    if ( xi.size !=  np.sum( preClusters.padCalibrated[ev][pc])) :
        input("pb with calibrated pads")
    idx = np.where( cathi == 0 )
    nz0 = idx[0].size
    zz0 = np.sum( chi[idx] )
    print("# Charge sum on cath 0", np.sum( chi[idx]) )
    idx = np.where( cathi == 1 )
    nz1 = idx[0].size
    zz1 = np.sum( chi[idx] )
    print("# Charge sum on cath 1", np.sum( chi[idx]) )
    if ( np.abs( 2 * (zz1 - zz0)/(zz1+zz0)) > 0.02 ):
      print("### Warning: different charge on the 2 cathodes", zz0, zz1)
      # input("next")
    chargeOnCath = [ zz0, zz1 ]
    onlyOneCath = ((nz0 == 0) or (nz1 == 0), (nz1 == 0)*1 )
    if onlyOneCath[0]: print("# Warning: only one cathode")  
    if True:
      print( "xi", xi)
      print( "yi", yi)
      idx = np.where( cathi == 0)
      x0 = xi[idx]
      y0 = yi[idx]
      dx0 = dxi[idx]
      dy0 = dyi[idx]
      ch0 = chi[idx]
      idx = np.where( cathi == 1)
      x1 = xi[idx]
      y1 = yi[idx]
      dx1 = dxi[idx]
      dy1 = dyi[idx]
      ch1 = chi[idx]
      if ch0.size != 0:
        z0 = np.max( ch0 )
      else:
        z0 = 0.0
      if ch1.size != 0:
        z1 = np.max( ch1 )
      else:
        z1 = 0
      zMax = max( z0, z1  )
      if (z0 == 0) and (z1 == 0):
        print("WARNING: no pads")
        input("next")
      plu.setLUTScale( 0.0, zMax ) 
      """
      plu.drawPads(ax[0,0], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,1], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,2], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=0.5, doLimits=True)
      plu.drawPads(ax[0,2], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=0.5, doLimits=True)
      """
      if not onlyOneCath[0]:
        (xShort0, yShort0, dxShort0, dyShort0, chShort0, chShort1,
        mapIJToK, mapKToIJ, IinterJ, JinterI ) = plu.shorteningPads( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1)
        # chShort Mean (take care to the proption 0.5) ???
        # The both sums should be equal
        # chShortM = 0.5 * (chShort0 + chShort1) 
        chShortM =  (chShort0 + chShort1) 
        print("# short/cathode 0 & 1", np.sum(chShort0), np.sum( ch0), np.sum(chShort1), np.sum(ch1))
        if( np.abs( np.sum(chShort0)- np.sum(ch0)) + np.abs( np.sum(chShort1)- np.sum(ch1) )  > 0.1 ):
            input("Charge lost !")
    
      else:
        print("No shortening")
        xShort0 = xi; dxShort0 = dxi; yShort0 = yi; dyShort0 = dyi; chShort0 = chi
        chShortM = chShort0
      if xShort0.size != 0 :
        if not onlyOneCath[0]:
          nGrp, groups = plu.getConnexComponents(xShort0, yShort0, dxShort0, dyShort0)
          # unused chShort0 = np.copy( saveChShort0)
          # Get groups without 1-Pad clusters and maping to cathodes
          nGrp1, groups1, wellSplit, padGrpCath0, padGrpCath1 = plu.associateGrpPads( nGrp, groups, chShort0, chShort1, ch0, ch1,  mapIJToK, mapKToIJ, IinterJ, JinterI )
          idx0 = np.where( cathi == 0)[0] 
          idx1 = np.where( cathi == 1)[0]
          padGrpCath0 = np.array( padGrpCath0 )
          padGrpCath1 = np.array( padGrpCath1 )
          # Check
          nCath0 = 0; nCath1 = 0
          for gg in range(nGrp1+1):
            iii = np.where(padGrpCath0==gg)
            jjj = np.where(padGrpCath1==gg)
            nCath0 += iii[0].size
            nCath1 += jjj[0].size
            print("group", gg, "nbr associated Pads cath0/1", iii[0].size, jjj[0].size)
          origCath0 = np.sum( cathi==0 )
          origCath1 = np.sum( cathi==1 )
          if ( nCath0 != origCath0  or nCath1 != origCath1):
             print( "Original Cath0/1", origCath0, origCath1, "associated cath0/1", nCath0, nCath1)
             if InputWarning : input( "Pb nbr of associated pads ")
          if (padGrpCath0.size != idx0.size) or (padGrpCath1.size != idx1.size):
             print( "padGrpCath0",  padGrpCath0.size , padGrpCath0)
             print( "padGrpCath1",  padGrpCath1.size , padGrpCath1)
             print( "idx0",  idx0.size , idx0 )
             print( "idx1",  idx1.size , idx1 )
             # input( "Different size with association")
          if ( nGrp != nGrp1 or not wellSplit.any() ):
              print("nGrp", nGrp, nGrp1 )
              print("wellSplit", wellSplit) 
              print("padGrpCath0", padGrpCath0) 
              print("padGrpCath1", padGrpCath1)
              # print("saveChShort0", chShort0)
              print("chShort1", chShort1)
              # input("next")
          nGrp =nGrp1
          groups = groups1
          # unused chShort0 = (chShort0 + chShort1) * 0.5
        else:
           # Only one cathode
           nGrp, groups = plu.getConnexComponents(xShort0, yShort0, dxShort0, dyShort0)
           wellSplit = [True for i in range(nGrp+1)]
           idx0 = np.where( cathi == 0)[0] 
           idx1 = np.where( cathi == 1)[0]
           # TODO : to optimize ... a hammer !
           # nGrp, groups, wellSplit, padGrpCath0, padGrpCath1 = plu.associateGrpPads( nGrp, groups, chShort0, chShort1, ch0, ch1,  mapIJToK, mapKToIJ, IinterJ, JinterI )
           if ch0.size == 0:
              padGrpCath0 = np.zeros( 0 )
              padGrpCath1 = groups
           else:
              padGrpCath1 = np.zeros( 0 )
              padGrpCath0 = groups
           print("??? groups", groups)
           print("??? padGrpCath0/1", padGrpCath0, padGrpCath1)
        ##############
        # To check sum of charge
        sum0 = 0; sum1 = 0
        for g in range(1, nGrp+1):
          grpIdx =np.where( groups == g )
          print("### Group ", g, "/", nGrp, "grp size=", grpIdx[0].size)
          if ( grpIdx[0].size > 1 ):
              xShg0 = xShort0[ grpIdx ]
              xShg1 = np.copy( xShg0 )
              dxShg0 = dxShort0[ grpIdx ]
              dxShg1 = np.copy( dxShg0 )
              yShg0 = yShort0[ grpIdx ]
              yShg1 = np.copy( yShg0 )
              dyShg0 = dyShort0[ grpIdx ]
              dyShg1 = np.copy( dyShg0 )
              xShort = np.hstack( [xShg0, xShg1])
              yShort = np.hstack( [yShg0, yShg1])
              dxShort = np.hstack( [dxShg0, dxShg1])
              dyShort = np.hstack( [dyShg0, dyShg1])
              chShg0 = chShort0[ grpIdx ]
              chShg1 = chShort1[ grpIdx ]
              chShort = np.hstack( [chShg0, chShg1])
              cathShort = np.hstack( [np.zeros( xShg0.shape[0], dtype=np.int ), np.ones( xShg1.shape[0], dtype=np.int ) ])
              # print("xShort0", xShort0)
              # print("xShort1", xShort1)          
              # w0, mu0, var0 = findLocalMaxWithSubstraction( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, mcObj, preClusters, ev, pc, DEIds,fitParameter=15.0, chId=chId )
              # print("??? xShort0", xShort0.shape, xShort0 )
              # print("??? yShort0", yShort0.shape, yShort0 )
              #print("??? cathSh", cathShort.shape, cathShort)
              # w0, mu0, var0, lapl = findLocalMaxWithLaplacian( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, graphTest= (chId < 0) ) # , mcObj, preClusters, ev, pc, DEIds,fitParameter=15.0, chId=chId )
              # w0, mu0, var0, lapl = findLocalMaxWithLaplacian( xShort0, yShort0, dxShort0, dyShort0, chShort0, var, graphTest= (chId < 0), verbose=True ) # , mcObj, preClusters, ev, pc, DEIds,fitParameter=15.0, chId=chId )
              w0, mu0, var0, lapl = findLocalMaxWithLaplacian( xShg0, yShg0, dxShg0, dyShg0, chShortM[grpIdx], 
                                                               var, graphTest= (chId < 0), verbose=True ) # , mcObj, preClusters, ev, pc, DEIds,fitParameter=15.0, chId=chId )
              print("w0, mu0, var0", w0, mu0, var0)
              EM.printGM("# w0, mu0, var0", w0, mu0, var0 )
              print(" sum ch0, cShort0", np.sum(ch0), np.sum( chShort0 ) )
              print(" sum ch1, cShort1", np.sum(ch1), np.sum( chShort1 ) )          
              xEM = xShort; yEM = yShort; dxEM = dxShort; dyEM = dyShort; cathEM = cathShort; chEM = chShort
              if w0.size != 0:
                # print("shapes", xu.shape, chu.shape, cathu.shape)
                # print("xu",xu)
                # print("dxu",dxu)
                # print("yu",yu)
                # print("chu",chu)
                # print("cathu", cathu)
                wEM, muEM, varEM = EM.simpleProcessEMCluster( xEM, yEM, dxEM, dyEM, cathEM, chEM, w0, mu0, var0, cstVar=True )
                # print("# wEM, muEM, varEM", wEM, muEM, varEM)
                EM.printGM("# wEM, muEM, varEM", wEM, muEM, varEM )
                wf, muf, varf = filterModelWithMag( wEM, muEM, varEM )
                # print("# wf, muf, varf", wf, muf, varf)
                EM.printGM("# wf, muf, varf", wf, muf, varf )
                wGrp, muGrp, varGrp = EM.simpleProcessEMCluster( xEM, yEM, dxEM, dyEM, cathEM, chEM, wf, muf, varf, cstVar=True )
                EM.printGM("# wGrp, muGrp, varGrp", wGrp, muGrp, varGrp )
                K = wGrp.shape[0]
                cutOff = 1.e-02 / K
                while (np.any( wGrp < cutOff ) ):
                  input("Path used ???")
                  idx = np.where( wGrp >= cutOff )
                  wGrp = wGrp[idx]
                  muGrp = muGrp[idx]
                  varGrp = varGrp[idx]
                  wGrp, muGrp, varGrp = EM.simpleProcessEMCluster( xEM, yEM, dxEM, dyEM, cathEM, chEM, wGrp, muGrp, varGrp, cstVar=True )
                  K = wGrp.shape[0]
                  cutOff = 1.e-02 / K
                print("End EM mu", muGrp)
                print ("group", g, "/", nGrp )
                #
                # LMS
                #
                print("# wellSplit:", wellSplit)
                padInGrp =  np.where( padGrpCath0==g)[0].size + np.where( padGrpCath1==g )[0].size
                nbrOfSeeds = wGrp.size
                ratioPadPerSeed = padInGrp / nbrOfSeeds
                print("# nbrPads in Group", padInGrp)
                print("# ratioPadPerSeed:", ratioPadPerSeed)
                if not wellSplit[g]:
                  input("Not well split")

                if ( wellSplit[g] and ratioPadPerSeed < 11 and nbrOfSeeds < 10):
                  # print ("idx0", idx0)
                  # print("??? padGrpCath0", padGrpCath0.size, padGrpCath0 )
                  # print("??? padGrpCath1", padGrpCath1.size, padGrpCath1 ) 
                  grpToCathIdx0 = idx0[ np.where( padGrpCath0==g ) ]
                  grpToCathIdx1 = idx1[ np.where( padGrpCath1==g ) ]
                  # print("??? grpToCathIdx", grpToCathIdx0,  grpToCathIdx1)
                  sum0 += np.sum( chi[grpToCathIdx0] ) 
                  sum1 += np.sum( chi[grpToCathIdx1] ) 
                  wSave = np.copy( wGrp )
                  muSave = np.copy( muGrp )
                  varSave = np.copy( varGrp )
                  # Case on initial pads (2 planes)
                  # Map 
                  xv = np.hstack( [xi[grpToCathIdx0], xi[grpToCathIdx1]] )
                  yv = np.hstack( [yi[grpToCathIdx0], yi[grpToCathIdx1]] )
                  dxv = np.hstack( [dxi[grpToCathIdx0], dxi[grpToCathIdx1]] )
                  dyv = np.hstack( [dyi[grpToCathIdx0], dyi[grpToCathIdx1]] )
                  xyFit = np.array( [xv, yv ] )
                  dxyFit = np.array( [dxv, dyv] )
                  cathFit = np.hstack( [cathi[grpToCathIdx0], cathi[grpToCathIdx1]] )
                  chFit = np.hstack( [chi[grpToCathIdx0], chi[grpToCathIdx1]] )
                  # Case 1D pads 
                  # npxy = np.array( [xu, yu] )
                  # npdxy = np.array( [dxu, dyu] )
                  if (chId < 3):
                    varCst = 1.2 * 1.2 * localMaxParam.var1 
                    mType = 0
                  else:
                    varCst =  1.2 * 1.2 * localMaxParam.var3
                    mType = 1
                  if np.unique(cathi).size == 1:
                    nbrOfCath = 1
                  else:
                    nbrOfCath = 2
                  K = wGrp.size
                  # varCst = np.tile( varCst, (K, 1))
                  # wFit, muFit, varFit = CF.clusterFit( CF.err_fct2, wGrp, muGrp, varCst, npxy , npdxy, chi, jacobian=CF.jac_fct2)
                  # Case 2 cathodes
                  # wFit, muFit, varFit = CF.clusterFitMath0( wGrp, muGrp, mType, npxy , npdxy, chi, nbrOfCath, jacobian=None)
                  # 2 cathodes constrains
                  # wFit, muFit, varFit = CF.clusterFitMath( wGrp, muGrp, mType, npxy , npdxy, chi, cathi, jacobian=None)
                  wFit, muFit, varFit = CF.clusterFitMath( wGrp, muGrp, mType, xyFit , dxyFit, chFit, cathFit, jacobian=None)
                  print("# Mathieson fitting")
                  # input("Mathieson fitting, next")
                  #
                  # Case one projected plane
                  # Take care ??? : since there are 2 x 1D superposed plane, nbrOfCath = 2
                  # wFit, muFit, varFit = CF.clusterFitMath( wGrp, muGrp, mType, npxy , npdxy, chu, nbrOfCath, jacobian=None)
                  # 
                  wGrp = wFit
                  muGrp = muFit
                  varGrp = varCst = np.tile( varCst, (K, 1))
                  print("End Fit mu", muFit)
                # End LMS
                wFinal.append( wGrp )
                muFinal.append( muGrp )
                varFinal.append( varGrp )
                if muGrp.shape[1] != 2:
                  print("?? muGrp", muGrp )
                  input("Bad shape")
              else:
                print("No solution in this group")
                input("Group with size 0")
          # end size of grp !=1
          if sum0 != 0 and ( (sum0 != np.sum( chi[cathi==0] )) or (sum1 != np.sum( chi[cathi==1] )) ):
              print ("sum0", sum0, np.sum( chi[cathi==0] ))
              print ("sum1", sum1, np.sum( chi[cathi==1] ))
              # input("Ch sum differs")
          else:
            # Only one pad ... skipp
            # wFinal.append( np.array([1.0]) )
            # muFinal.append( np.array( [ [xShort0[0],yShort0[0] ] ] ) )
            # varFinal.append( np.array( [var] ) )
            # print("???", varFinal)
            if InputWarning: input("Group/cluster whith one pad")
        # End group loop
        # print("??? wFinal", wFinal)
        # print("??? muFinal", muFinal)
        if ( len(wFinal) != 0 ):
         wFinal = np.concatenate( wFinal )
         muFinal = np.concatenate( muFinal )
         varFinal = np.concatenate( varFinal )
        # print("??? muFinal", muFinal)
        # input("next")

        """
        plu.drawPads(ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,2], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=0.5, doLimits=True)
        plu.drawPads(ax[1,2], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=0.5, doLimits=True)
        plt.show()
        """
      else:
        w0 = np.empty( shape=(0, 0) )
        m0 = []
        var0 = []
      #
    # if True
    #
    # TP, FN, FP, ... analysis
    #
    wFinal = np.array( wFinal )
    if ( wFinal.size != 0 ):
      idx = np.where( wFinal != 0.0)
      wFinal = wFinal[idx]
      muFinal = muFinal[idx]
      varFinal = varFinal[idx]
      print(" w0.size", w0.size)
      print(" wf.size", wf.size)
      print(" wFinal.size", wFinal.size)
      print(" # reco Hits", preClusters.rClusterX[ev][pc].size)
      # input("next")

      #    
      #
      emObj.w[ev].append(wFinal)
      emObj.mu[ev].append(muFinal)
      emObj.var[ev].append(varFinal)
      #
      # Assign Cluster hits to MC hits
      print(wFinal)
      print(muFinal)
      x0 = muFinal[:,0]
      y0 = muFinal[:,1]
      spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
      rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
      rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
      spanBox = plu.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                             preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
      match, nbrOfHits, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved = matchMCTrackHits(x0, y0, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
      if (dxMin.size != 0):
        maxDxMin = np.max( dxMin )
        maxDyMin = np.max( dyMin )
      else:
        maxDxMin = 1.0
        maxDyMin = 1.0
      if (match > 0.33):
        assign.append( (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) )
        evTP += TP; evFP += FP; evFN += FN
        allDMin.append( dMin )
        evNbrOfHits += nbrOfHits
        print( "TP, FP, FN", TP, FP, FN)
      # Debug
      """
      if pc == 26:
        a = input("PreCluster="+str(pc))
      """
      #
    else:
      wFinal = w0
      muFinal = mu0
      varFinal = var0
    #
    # Display
    #
    ws = np.sum( wellSplit)
    # if wFinal.size < 0 : 
    # if wFinal.size > 0 and pc == 238: 
    # if wFinal.size > 0 and wFinal.size < 4 : 
    # if wFinal.size > 0 and wFinal.size < 3 and chId > 2 : 
    # if wFinal.size < 0 or wFinal.size == 0 : 
    # if wFinal.size > 0 and chId > 4 :
    # if maxDxMin > 0.07 or maxDyMin > 0.05 :
    #if maxDxMin > 0.1 or maxDyMin > 0.07 :
    # if ws != (wellSplit.size):
    if wFinal.size > 0 : 
      print( "??? wellSpli", ws, wellSplit.size )
      print("maxDxMin,maxDyMin",maxDxMin,maxDyMin)
      fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 7) )
      print("match, dMin", match, dMin)
      if wFinal.size == 1:
        dTmp.extend( list( dMin) )
        u = np.array( dTmp)
        print( "dMin average", np.sum(u)/ u.size )
      displayAPrecluster(ax, xy, dxy, cathi, chi, wFinal, muFinal, varFinal, mcObj, preClusters, ev, pc, DEIds )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var, ax[1] )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var )
      # w0, mu0, var0 = findLocalMaxWithLaplacian( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, ax=ax[1,:] )
      plu.setLUTScale( 0.0, zMax ) 
      if xShort0.size != 0:
        plu.drawPads( ax[0,3], xShort0, yShort0, dxShort0, dyShort0, chShort0,  doLimits=False, alpha=1.0, )
      plu.drawModelComponents( ax[0, 3], wFinal, muFinal, varFinal, color='red', pattern="o" )
      if (wSave.size != 0 ):
        plu.drawModelComponents( ax[0, 3], wSave, muSave, varSave, color='green', pattern="+" )

      if( True and xShort0.size !=0 ):
        plu.drawPads( ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, )

        """
        if xShort1.size !=0 :
          plu.drawPads( ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1,  alpha=1.0, )
        """
        plu.setLUTScale( np.min( lapl ), np.max(lapl) ) 
        # print("??? xShort", xu)
        # print("??? lapl", lapl)
        plu.drawPads( ax[1,1], xShort[cathShort==0], yShort[cathShort==0], dxShort[cathShort==0], dyShort[cathShort==0], lapl,  alpha=1.0, )
        plu.setLUTScale( 0.0, zMax ) 

        # plu.drawPads( ax[1,2], xShort0, yShort0, dxShort0, dyShort0, chShort1 - chShort0,  alpha=1.0, )
        plu.drawPads( ax[1,3], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, )
      #
      if True:
        plu.drawModelComponents( ax[0, 1], w0, mu0, var0, color='red', pattern="o" )
        plu.drawModelComponents( ax[0, 2], wEM, muEM, varEM, color='red', pattern="o" )
        plu.drawModelComponents( ax[0, 3], wFinal, muFinal, varFinal, color='red', pattern="o" )
        plu.drawModelComponents( ax[1, 0], w0, mu0, var0, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 1], wEM, muEM, varEM, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 2], wf, muf, varf, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 3], wFinal, muFinal, varFinal, color='red', pattern="cross" )
        ax[0,0].set_title("MC/Reco")
        ax[0,1].set_title("Theta0")
        ax[0,2].set_title("EM") 
        ax[0,3].set_title("Theta Final") 
        ax[1,0].set_title("Theta0") 
        ax[1,1].set_title("EM") 
        ax[1,2].set_title("Filter") 
        ax[1,3].set_title("Theta Final")
      else:
        ax[1,0].set_title("1st peak")
        
      frame = plu.getPadBox(xy[0], xy[1], dxy[0], dxy[1] )
      drawMCHitsInFrame ( ax[0,0], frame, mcObj, ev, DEIds )
      drawMCHitsInFrame ( ax[0,1], frame, mcObj, ev, DEIds )
      drawMCHitsInFrame ( ax[0,2], frame, mcObj, ev, DEIds )
      drawMCHitsInFrame ( ax[0,3], frame, mcObj, ev, DEIds )      
      drawMCHitsInFrame ( ax[1,0], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,1], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,2], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,3], frame, mcObj, ev, DEIds )      # input("Next cluster")
      title = "Ev=" +str(ev)+ ", pc=" + str(pc) + ", DEId="+ str(DEIds)
      # fig.inftitle( title )
      title += str( np.min( chi) ) + "<=" + str(np.max( chi)) 
      fig.suptitle( title )
      plt.show()
  # end loop pc
  print( "# hits=", evNbrOfHits, "TP, FP, FN = (", evTP, evFP, evFN, "), assigned preClusters", len(assign),"/", nbrOfPreClusters  ) 
  return evNbrOfHits, evTP, evFP, evFN, allDMin, assign



def processEMOnPreClustersV2( preClusters, mcObj, ev, emObj, rejectPadFactor=0.5):
  # with cath projection on one plane
  dTmp = []
  nbrOfPreClusters = len( preClusters.padId[ev] )
  print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
  assign = []
  allDMin = []
  evTP = 0; evFP = 0 ; evFN = 0
  evNbrOfHits = 0
  # Create events storage
  if ev >= len(emObj.w):
    for e in range(len(emObj.w), ev+1):
      emObj.w.append([])
      emObj.mu.append([])
      emObj.var.append([])
  #
  xShort0 = np.empty( shape=(0, 0) ); yShort0 = np.empty( shape=(0, 0) ); dxShort0 = np.empty( shape=(0, 0) ); dyShort0 = np.empty( shape=(0, 0) ); chShort0 = np.empty( shape=(0, 0) );
  xShort1 = np.empty( shape=(0, 0) ); yShort1 = np.empty( shape=(0, 0) ); dxShort1 = np.empty( shape=(0, 0) ); dyShort1 = np.empty( shape=(0, 0) ); chShort1 = np.empty( shape=(0, 0) );
  for pc in range(nbrOfPreClusters):
    xi = preClusters.padX[ev][pc]
    dxi = preClusters.padDX[ev][pc]
    yi = preClusters.padY[ev][pc]
    dyi = preClusters.padDY[ev][pc]
    cathi = preClusters.padCath[ev][pc]
    chi = preClusters.padCharge[ev][pc]
    chIds = preClusters.padChId[ev][pc]
    DEIds = preClusters.padDEId[ev][pc]
    DEIds = np.unique( DEIds )
    chId = np.unique( chIds )[0]
    xy = [ xi, yi]
    dxy = [ dxi, dyi]
    if (chId < 3):
      var = localMaxParam.var1
    else:
      var = localMaxParam.var3
    dxMax = np.max( dxi)
    dyMax = np.max( dyi)
    dxyMax = max( dxMax, dyMax)
    print("dxyMax", dxyMax)

    if True:
      print( "xi", xi)
      idx = np.where( cathi == 0)
      x0 = xi[idx]
      y0 = yi[idx]
      dx0 = dxi[idx]
      dy0 = dyi[idx]
      ch0 = chi[idx]
      idx = np.where( cathi == 1)
      x1 = xi[idx]
      y1 = yi[idx]
      dx1 = dxi[idx]
      dy1 = dyi[idx]
      ch1 = chi[idx]
      if ch0.size != 0:
        z0 = np.max( ch0 )
      else:
        z0 = 0.0
      if ch1.size != 0:
        z1 = np.max( ch1 )
      else:
        z1 = 0
      zMax = max( z0, z1  )
      if (z0 == 0) and (z1 == 0):
        print("WARNING: no pads")
        input("next")
      plu.setLUTScale( 0.0, zMax ) 
      """
      plu.drawPads(ax[0,0], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,1], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=1.0, doLimits=True)
      plu.drawPads(ax[0,2], x0, y0, dx0, dy0, ch0, title="", yTitle="", alpha=0.5, doLimits=True)
      plu.drawPads(ax[0,2], x1, y1, dx1, dy1, ch1, title="", yTitle="", alpha=0.5, doLimits=True)
      """
      xShort0, yShort0, dxShort0, dyShort0, chShort0, chShort1 = plu.shorteningPads( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1)
      xShort1 = np.copy( xShort0 )
      dxShort1 = np.copy( dxShort0 )
      yShort1 = np.copy( yShort0 )
      dyShort1 = np.copy( dyShort0 )
      xShort = np.hstack( [xShort0, xShort1])
      yShort = np.hstack( [yShort0, yShort1])
      dxShort = np.hstack( [dxShort0, dxShort1])
      dyShort = np.hstack( [dyShort0, dyShort1])
      chShort = np.hstack( [chShort0, chShort1])
      cathShort = np.hstack( [np.zeros( xShort0.shape[0], dtype=np.int ), np.ones( xShort1.shape[0], dtype=np.int ) ])
      print("xShort0", xShort0)
      print("xShort1", xShort1)
      if xShort.size != 0:

        w0, mu0, var0 = findLocalMaxWithSubstraction( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, fitParameter=15.0 )
        print(" sum ch0, cShort0", np.sum(ch0), np.sum( chShort0 ) )
        print(" sum ch1, cShort1", np.sum(ch1), np.sum( chShort1 ) )
        """
        plu.drawPads(ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=1.0, doLimits=True)
        plu.drawPads(ax[1,2], xShort0, yShort0, dxShort0, dyShort0, chShort0, title="", yTitle="", alpha=0.5, doLimits=True)
        plu.drawPads(ax[1,2], xShort1, yShort1, dxShort1, dyShort1, chShort1, title="", yTitle="", alpha=0.5, doLimits=True)
        plt.show()
        """
      else:
        w0 = np.empty( shape=(0, 0) )
        m0 = []
        var0 = []
      #
    #
    if ( w0.size != 0 ):
    #  if xShort.size != 0:
      xu = xShort; yu = yShort; dxu = dxShort; dyu = dyShort; cathu = cathShort; chu = chShort
    #  else:
    #    xu = xi; yu = yi; dxu = dxi; dyu = dyi; cathu = cathi; chu = chi 
      wEM, muEM, varEM = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, w0, mu0, var0, cstVar=True )
      wf, muf, varf = filterModelWithMag( wEM, muEM, varEM )
      wFinal, muFinal, varFinal = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, wf, muf, varf, cstVar=True )
      K = wFinal.shape[0]
      cutOff = 1.e-02 / K
      while (np.any( wFinal < cutOff ) ):
        idx = np.where( wFinal >= cutOff )
        wFinal = wFinal[idx]
        muFinal = muFinal[idx]
        varFinal = varFinal[idx]
        wFinal, muFinal, varFinal = EM.simpleProcessEMCluster( xu, yu, dxu, dyu, cathu, chu, wFinal, muFinal, varFinal, cstVar=True )
        K = wFinal.shape[0]
        cutOff = 1.e-02 / K
      # LMS
      if (wFinal.size < 0):
        npxy = np.array( [xu, yu] )
        npdxy = np.array( [dxu, dyu] )
        if (chId < 3):
          varCst = localMaxParam.var1 
        else:
          varCst = localMaxParam.var3
        K = wFinal.size
        wFit, muFit, varFit = CF.clusterFit( CF.err_fct2, wFinal, muFinal, varCst, npxy , npdxy, chu, jacobian=CF.jac_fct2)
        wFinal = wFit
        muFinal = muFit
        varFinal = varCst = np.tile( varCst, (K, 1))
      #
      print(wFinal)
      print(muFinal)
      idx = np.where( wFinal != 0.0)
      wFinal = wFinal[idx]
      muFinal = muFinal[idx]
      varFinal = varFinal[idx]
      print(" w0.size", w0.size)
      print(" wf.size", wf.size)
      print(" wFinal.size", wFinal.size)
      print(" # reco Hits", preClusters.rClusterX[ev][pc].size)
      # input("next")

      #    
      #
      emObj.w[ev].append(wFinal)
      emObj.mu[ev].append(muFinal)
      emObj.var[ev].append(varFinal)
      #
      # Assign Cluster hits to MC hits
      print(wFinal)
      print(muFinal)
      x0 = muFinal[:,0]
      y0 = muFinal[:,1]
      spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
      rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
      rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
      spanBox = plu.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                             preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
      match, nbrOfHits, TP, FP, FN, dMin, mcHitsInvolved = matchMCTrackHits(x0, y0, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
      if (match > 0.33):
        assign.append( (pc, match, nbrOfHits, TP, FP,FN, mcHitsInvolved ) )
        evTP += TP; evFP += FP; evFN += FN
        allDMin.append( dMin )
        evNbrOfHits += nbrOfHits
      # Debug
      """
      if pc == 26:
        a = input("PreCluster="+str(pc))
      """
      #
    else:
      wFinal = w0
      muFinal = mu0
      varFinal = var0
    # if wFinal.size > 0 or wFinal.size == 0 : 
    if wFinal.size > 0 and chId > 4 : 
      fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 7) )
      print("match, dMin", match, dMin)
      if wFinal.size == 1:
        dTmp.extend( list( dMin) )
        u = np.array( dTmp)
        print( "dMin average", np.sum(u)/ u.size )
      displayAPrecluster(ax, xy, dxy, cathi, chi, wFinal, muFinal, varFinal, mcObj, preClusters, ev, pc, DEIds )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var, ax[1] )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var )
      w0, mu0, var0 = findLocalMaxWithLaplacian( xShort, yShort, dxShort, dyShort, cathShort, chShort, var, ax=ax[1,:] )
      plu.setLUTScale( 0.0, zMax ) 

      plu.drawPads( ax[0,3], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, )

      if( False and xShort.size !=0 ):
        plu.drawPads( ax[1,0], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, )
        plu.drawPads( ax[1,1], xShort1, yShort1, dxShort1, dyShort1, chShort1,  alpha=1.0, )
        plu.drawPads( ax[1,2], xShort0, yShort0, dxShort0, dyShort0, chShort1 - chShort0,  alpha=1.0, )
        plu.drawPads( ax[1,3], xShort0, yShort0, dxShort0, dyShort0, chShort0,  alpha=1.0, )
      #
      if False:
        plu.drawModelComponents( ax[0, 1], w0, mu0, var0, color='red', pattern="o" )
        plu.drawModelComponents( ax[0, 2], wEM, muEM, varEM, color='red', pattern="o" )
        plu.drawModelComponents( ax[0, 3], wFinal, muFinal, varFinal, color='red', pattern="o" )
        plu.drawModelComponents( ax[1, 0], w0, mu0, var0, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 1], wEM, muEM, varEM, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 2], wf, muf, varf, color='red', pattern="cross" )
        plu.drawModelComponents( ax[1, 3], wFinal, muFinal, varFinal, color='red', pattern="cross" )
        ax[0,0].set_title("MC/Reco")
        ax[0,1].set_title("Theta0")
        ax[0,2].set_title("EM") 
        ax[0,3].set_title("Theta Final") 
        ax[1,0].set_title("Theta0") 
        ax[1,1].set_title("EM") 
        ax[1,2].set_title("Filter") 
        ax[1,3].set_title("Theta Final")
      else:
        ax[1,0].set_title("1st peak")
        
      frame = plu.getPadBox(xy[0], xy[1], dxy[0], dxy[1] )
      drawMCHitsInFrame ( ax[1,0], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,1], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,2], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[1,3], frame, mcObj, ev, DEIds )      # input("Next cluster")
      title = "Ev=" +str(ev)+ ", pc=" + str(pc) + ", DEId="+ str(DEIds)
      # fig.inftitle( title )
      title += str( np.min( chi) ) + "<=" + str(np.max( chi)) 
      fig.suptitle( title )
      plt.show()
  # end loop pc
  print( "# hits=", evNbrOfHits, "TP, FP, FN = (", evTP, evFP, evFN, "), assigned preClusters", len(assign),"/", nbrOfPreClusters  ) 
  return evNbrOfHits, evTP, evFP, evFN, allDMin, assign

def fitLocalMaxWithSubtraction( preClusters, mcObj, ev, clusterSize, mcClusterSize, recoClusterSize, rejectPadFactor=0.5, fitParameter=5.0):
  #
  dTmp = []
  nbrOfPreClusters = len( preClusters.padId[ev] )
  print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
  """ ???
  clusterSize = []
  mcClusterSize = []
  recoClusterSize = []
  """
  assign = []
  allDMin = []
  evTP = 0; evFP = 0 ; evFN = 0
  evNbrOfHits = 0
  #
  for pc in range(nbrOfPreClusters):
    xi = preClusters.padX[ev][pc]
    dxi = preClusters.padDX[ev][pc]
    yi = preClusters.padY[ev][pc]
    dyi = preClusters.padDY[ev][pc]
    cathi = preClusters.padCath[ev][pc]
    chi = preClusters.padCharge[ev][pc]
    chIds = preClusters.padChId[ev][pc]
    DEIds = preClusters.padDEId[ev][pc]
    DEIds = np.unique( DEIds )
    chId = np.unique( chIds )[0]
    xy = [ xi, yi]
    dxy = [ dxi, dyi]
    if (chId < 3):
      var = localMaxParam.var1
    else:
      var = localMaxParam.var3
    w0, mu0, var0 = findLocalMaxWithSubstraction( xi, yi, dxi, dyi, cathi, chi, var, fitParameter= fitParameter)
    # wEM, muEM, varEM = EM.simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, w0, mu0, var0, cstVar=True )
    # wf, muf, varf = filterModelWithMag( wEM, muEM, varEM )
    # wFinal, muFinal, varFinal = EM.simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, wf, muf, varf, cstVar=True )
    wFinal = w0
    muFinal = mu0
    varFinal = var0
    print(" wFinal.size", wFinal.size)
    print(" # reco Hits", preClusters.rClusterX[ev][pc].size)
    # input("next")

    #    
    #
    # Assign Cluster hits to MC hits
    x0 = muFinal[:,0]
    y0 = muFinal[:,1]
    spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
    rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
    rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
    spanBox = plu.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                             preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
    match, nbrOfHits, TP, FP, FN, dMin, mcHitsInvolved = matchMCTrackHits(x0, y0, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
    if (match > 0.33):
      assign.append( (pc, match, nbrOfHits, TP, FP,FN, mcHitsInvolved ) )
      evTP += TP; evFP += FP; evFN += FN
      allDMin.append( dMin )
      evNbrOfHits += nbrOfHits
      clusterSize.append( wFinal.size )
      nHits = len( mcHitsInvolved )
      # print( "mcHitsInvolved", mcHitsInvolved ) 
      # print( "# of mcHits", nHits)
      mcClusterSize.append( nHits )
      recoClusterSize.append( preClusters.rClusterX[ev][pc].size )
    # print(match)
      
    #
    
    if wFinal.size < 0 : 
      print("match, dMin", match, dMin)
      fig, ax = displayAPrecluster(xy, dxy, cathi, chi, wFinal, muFinal, varFinal, mcObj, preClusters, ev, pc, DEIds )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var, ax[1] )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var )
      plu.drawModelComponents( ax[0, 1], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[0, 2], wFinal, muFinal, varFinal, color='red', pattern="o" )
      plu.drawModelComponents( ax[1, 0], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[1, 1], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[1, 2], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[1, 3], w0, mu0, var0, color='red', pattern="cross" )
      frame = plu.getPadBox(xy[0], xy[1], dxy[0], dxy[1] )
      drawMCHitsInFrame ( ax[0,0], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[0,1], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[0,2], frame, mcObj, ev, DEIds )      # input("Next cluster")
      title = "Ev=" +str(ev)+ ", pc=" + str(pc) + ", DEId="+ str(DEIds)
      # fig.inftitle( title )
      title += str( np.min( chi) ) + "<=" + str(np.max( chi)) 
      fig.suptitle( title )
      plt.show()
  # end loop pc
  print( "# hits=", evNbrOfHits, "TP, FP, FN = (", evTP, evFP, evFN, "), assigned preClusters", len(assign),"/", nbrOfPreClusters  ) 
  return clusterSize, mcClusterSize, recoClusterSize

def processFitingThePreClusters( preClusters, mcObj, ev, emObj, rejectPadFactor=0.5):
  #
  nbrOfPreClusters = len( preClusters.padId[ev] )
  print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
  assign = []
  allDMin = []
  evTP = 0; evFP = 0 ; evFN = 0
  evNbrOfHits = 0
  # Create events storage
  if ev >= len(emObj.w):
    for e in range(len(emObj.w), ev+1):
      emObj.w.append([])
      emObj.mu.append([])
      emObj.var.append([])
  #
  for pc in range(nbrOfPreClusters):
    xi = preClusters.padX[ev][pc]
    dxi = preClusters.padDX[ev][pc]
    yi = preClusters.padY[ev][pc]
    dyi = preClusters.padDY[ev][pc]
    cathi = preClusters.padCath[ev][pc]
    chi = preClusters.padCharge[ev][pc]
    chIds = preClusters.padChId[ev][pc]
    chIds = preClusters.padChId[ev][pc]
    chId = np.unique( chIds )[0]
    DEIds = preClusters.padDEId[ev][pc]
    DEIds = np.unique( DEIds )
    
    if (chId < 3):
      var = localMaxParam.var1 
    else:
      var = localMaxParam.var3
    w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var )
    print("0000000000 w0", w0.shape, w0)
    print("0000000000 w0", mu0.shape, mu0)

    wFilt, muFilt, varFilt = filterModelWithMag( w0, mu0, var0 )
    # wEM, muEM, varEM = simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, w0, mu0, var0, cstVar=True )
    # wFit, muFit, varFit = CF.clusterFit( CF.err_fct2, w0, mu0, var0, xy , dxy, chi, jacobian=CF.jac_fct2)
    print("w0", w0)
    print("mu0", mu0)
    print("var0", var0)
    # TODO : ????
    w0 = w0.ravel()
    xy = np.vstack( [[xi], [yi]])
    dxy = np.vstack( [[dxi], [dyi]])
    print("wFilt", wFilt)
    print("muFilt", muFilt)
    print("varFilt", varFilt)
    wFit, muFit, varFit = CF.clusterFit( CF.err_fct2, wFilt, muFilt, var, xy , dxy, chi, jacobian=CF.jac_fct2)
    print("wFit", wFit)
    print("muFit", muFit)
    print("varFit", varFit)

    # CF.clusterFit( CF.err_fct2, wi, mui, vari, xy , dxy, z)
    # wf, muf, varf = filterModel( wFit, muFit, varFit )
    # wFinal, muFinal, varFinal = simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, wf, muf, varf, cstVar=True )
    wFinal, muFinal, varFinal = wFit, muFit, varFit
    
    idx = np.where( wFinal != 0.0)
    wFinal = wFinal[idx]
    muFinal = muFinal[idx]
    # varFinal = varFinal[idx]
    varx = []
    for k in range(wFinal.shape[0]):
      varx.append( var )
    varx = np.array( varx )
    if w0.size > 0 : 
      fig, ax = displayAPrecluster(xy, dxy, cathi, chi, wFit, muFit, varx, mcObj, preClusters, ev, pc, DEIds, )
      w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var, ax[1] )
      # w0, mu0, var0 = findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi, var )
      plu.drawModelComponents( ax[0, 1], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[0, 2], wFinal, muFinal, varx, color='red', pattern="o" )
      plu.drawModelComponents( ax[1, 0], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[1, 1], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[1, 2], w0, mu0, var0, color='red', pattern="cross" )
      plu.drawModelComponents( ax[1, 3], w0, mu0, var0, color='red', pattern="cross" )
      frame = plu.getPadBox(xy[0], xy[1], dxy[0], dxy[1] )
      drawMCHitsInFrame ( ax[0,0], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[0,1], frame, mcObj, ev, DEIds )      # input("Next cluster")
      drawMCHitsInFrame ( ax[0,2], frame, mcObj, ev, DEIds )      # input("Next cluster")
      title = "Ev=" +str(ev)+ ", pc=" + str(pc) + ", DEId="+ str(DEIds)
      # fig.inftitle( title )
      title += str( np.min( chi) ) + "<=" + str(np.max( chi)) 
      fig.suptitle( title )
      plt.show()
    #
    emObj.w[ev].append(wFinal)
    emObj.mu[ev].append(muFinal)
    emObj.var[ev].append(varFinal)
    #
    # Assign Cluster hits to MC hits
    print(wFinal)
    print(muFinal)
    x0 = muFinal[:,0]
    y0 = muFinal[:,1]
    spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
    rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
    rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
    spanBox = plu.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                             preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
    match, nbrOfHits, TP, FP, FN, dMin, mcHitsInvolved = matchMCTrackHits(x0, y0, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
    if (match > 0.33):
      assign.append( (pc, match, nbrOfHits, TP, FP,FN, mcHitsInvolved ) )
      evTP += TP; evFP += FP; evFN += FN
      allDMin.append( dMin )
      evNbrOfHits += nbrOfHits
    # Debug
    """
    if pc == 26:
      a = input("PreCluster="+str(pc))
    """
    #
  # end loop pc
  print( "# hits=", evNbrOfHits, "TP, FP, FN = (", evTP, evFP, evFN, "), assigned preClusters", len(assign),"/", nbrOfPreClusters  ) 
  return evNbrOfHits, evTP, evFP, evFN, allDMin, assign

def testFindLocalMax( preClusters, mcObj, evRange ):
  #
  nMax = [ 0 for _ in range(11) ]
  nClusters = [ 0 for _ in range(11) ]
  for ev in evRange :
    nbrOfPreClusters = len( preClusters.padId[ev] )
    print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
    # Create events storage
    """
    if ev >= len(emObj.w):
      for e in range(len(emObj.w), ev+1):
        emObj.w.append([])
        emObj.mu.append([])
        emObj.var.append([])
    """
    #
    for pc in range(nbrOfPreClusters):
      xi = preClusters.padX[ev][pc]
      dxi = preClusters.padDX[ev][pc]
      yi = preClusters.padY[ev][pc]
      dyi = preClusters.padDY[ev][pc]
      cathi = preClusters.padCath[ev][pc]
      chi = preClusters.padCharge[ev][pc]
      chIds = preClusters.padChId[ev][pc]
      chId = np.unique( chIds )[0]
      if (chId < 5):
        # var= np.array( [0.8, 0.8 ])
        var = localMaxParam.var1
      else:
        var = localMaxParam.var5
        # var= np.array( [10.0, 10.0 ])
          
      w0, mu0, var0 = findLocalMaxWithSubstraction( xi, yi, dxi, dyi, cathi, chi, var )
      # wEM, muEM, varEM = simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, w0, mu0, var0, cstVar=True )
      # wf, muf, varf = filterModel( wEM, muEM, varEM )
      # wFinal, muFinal, varFinal = simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, wf, muf, varf, cstVar=True )
      nMax[chId] += w0.shape[0]
      nClusters[chId] += 1
    # preCluster loop
  # ev loop
  for c in range(1,11):
    print("ChId=", c, "nbre de max=", nMax[c], "nbre de Clusters=", nClusters[c])
  nm = np.sum( np.array( nMax) )
  nc = np.sum( np.array( nClusters) )
  print( "nbre total de max=", nm, "nbre de Clusters=", nc )
  return

def worstCasesOld( measure, mcObj, recoObj ):
  maxFP = 0
  maxFN = 0
  allTP = []
  allFP = []
  allFN = []
  allPC = []
  evNbrOfHits, evTP, evFP, evFN, allDMin, assign = measure
  for m in assign :
    (pc, match, nbrOfHits, TP, FP,FN ) = m
    allTP.append( TP )
    allFP.append( FP )
    allFN.append( FN )
    allPC.append( pc )
  print(len(allFP), allFP)
  allTP = np.array( allTP, dtype=int )
  allFP = np.array( allFP, dtype=int )
  allFN = np.array( allFN, dtype=int)
  allPC = np.array( allPC, dtype=int)
  FPIdx = np.argsort( allFP )
  FNIdx = np.argsort( allFN + allFP )
  nFigRow = 1
  nFigCol = 1
  ev = 5

  n = len(allFP)
  for k in range(n-5, n):
    pc = allPC[FPIdx[k]]
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
    x = recoObj.padX[ev][pc]
    dx = recoObj.padDX[ev][pc]
    y = recoObj.padY[ev][pc]
    dy = recoObj.padDY[ev][pc]    
    ( xMin, xMax, yMin, yMax ) = plu.getPadBox( x, y, dx, dy )
    drawOneCluster(ax, x, y, dx, dy,
        recoObj.padCath[ev][pc], recoObj.padCharge[ev][pc], noYTicksLabels=False )
    xLen = xMax -xMin
    yLen = yMax -yMin
    ax.set_xlim( xMin, xMax )
    ax.set_ylim( yMin, yMax )
    #
    # Reco
    #
    nClusters = recoObj.rClusterX[ev][pc].shape[0]
    w = np.ones( (nClusters) )
    mu = np.array( [recoObj.rClusterX[ev][pc], recoObj.rClusterY[ev][pc]] ).T
    var = np.zeros( (nClusters, 2))
    plu.drawModelComponents( ax, w, mu, var, color='red', pattern="o" )
    #
    # MC
    #
    DEIds = np.unique( recoObj.rClusterDEId[ev][pc] )
    ChIds = np.unique( recoObj.rClusterChId[ev][pc] )
    if ChIds.shape[0] != 1:
        print("Must have only 1 chamber")
        exit()
    chId = ChIds[0]
    
    x = []
    y = []
    charge = []       
    for deid in DEIds:
      print("??? deid", deid), 
      for tidx, tid in enumerate(mcObj.trackId[ev]):
        flag0 = ( mcObj.trackDEId[ev][tidx] == deid )
        flag1 = ( mcObj.trackCharge[ev][tidx] > 0 )
        idx = np.where( np.bitwise_and( flag0, flag1 ) )
        x.append( mcObj.trackX[ev][tidx][idx])
        y.append( mcObj.trackY[ev][tidx][idx])
        charge.append( mcObj.trackCharge[ev][tidx][idx])
    x = np.concatenate( x ).ravel()
    y = np.concatenate( y ).ravel()
    charge = np.concatenate( charge ).ravel()
    print("??? mc points", x.shape, DEIds)
    flag0 = np.bitwise_and( (x >= xMin), (x < xMax) )
    flag1 = np.bitwise_and( (y >= yMin), (y < yMax) )
    flags = np.bitwise_and( flag0, flag1)
    print( "sum(flags)",np.sum(flags))
    #
    nClusters = x.shape[0]
    w = np.ones( (nClusters) )
    mu = np.array( [x, y] ).T
    var = np.zeros( (nClusters, 2))
    plu.drawModelComponents( ax, w, mu, var, color='black', pattern="+" )
    
    fig.suptitle('Ev='+str(ev)+", pc="+str(pc) , fontsize=16)  
    print( "FP / FN / pc", allTP[FPIdx[k]],allFP[FPIdx[k]], allFN[FPIdx[k]], allPC[FPIdx[k]])

    plt.show()
  #  
  n = len(allFN)
  for k in range(n-5, n):
    pc = allPC[FNIdx[k]]
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
    x = recoObj.padX[ev][pc]
    dx = recoObj.padDX[ev][pc]
    y = recoObj.padY[ev][pc]
    dy = recoObj.padDY[ev][pc]    
    ( xMin, xMax, yMin, yMax ) = plu.getPadBox( x, y, dx, dy )
    drawOneCluster(ax, x, y, dx, dy,
        recoObj.padCath[ev][pc], recoObj.padCharge[ev][pc], noYTicksLabels=False )
    ax.set_xlim( xMin, xMax )
    ax.set_ylim( yMin, yMax )    
    #
    # Reco
    #
    nClusters = recoObj.rClusterX[ev][pc].shape[0]
    w = np.ones( (nClusters) )
    mu = np.array( [recoObj.rClusterX[ev][pc], recoObj.rClusterY[ev][pc]] ).T
    var = np.zeros( (nClusters, 2))
    plu.drawModelComponents( ax, w, mu, var, color='red', pattern="o" )
    #
    # MC
    #
    DEIds = np.unique( recoObj.rClusterDEId[ev][pc] )
    ChIds = np.unique( recoObj.rClusterChId[ev][pc] )
    if ChIds.shape[0] != 1:
        print("Must have only 1 chamber")
        exit()
    chId = ChIds[0]
    
    x = []
    y = []
    charge = []       
    for deid in DEIds:
      print("??? deid", deid), 
      for tidx, tid in enumerate(mcObj.trackId[ev]):
        flag0 = ( mcObj.trackDEId[ev][tidx] == deid )
        flag1 = ( mcObj.trackCharge[ev][tidx] > 0 )
        idx = np.where( np.bitwise_and( flag0, flag1 ) )
        x.append( mcObj.trackX[ev][tidx][idx])
        y.append( mcObj.trackY[ev][tidx][idx])
        charge.append( mcObj.trackCharge[ev][tidx][idx])
    x = np.concatenate( x ).ravel()
    y = np.concatenate( y ).ravel()
    charge = np.concatenate( charge ).ravel()
    print("??? mc points", x.shape, DEIds)
    flag0 = np.bitwise_and( (x >= xMin), (x < xMax) )
    flag1 = np.bitwise_and( (y >= yMin), (y < yMax) )
    flags = np.bitwise_and( flag0, flag1)
    print( "sum(flags)",np.sum(flags))
    #
    nClusters = x.shape[0]
    w = np.ones( (nClusters) )
    mu = np.array( [x, y] ).T
    var = np.zeros( (nClusters, 2))
    plu.drawModelComponents( ax, w, mu, var, color='black', pattern="+" )
    
    fig.suptitle('Ev='+str(ev)+", pc="+str(pc) , fontsize=16)  
    print( "FP / FN / pc", allTP[FNIdx[k]],allFP[FNIdx[k]], allFN[FNIdx[k]], allPC[FNIdx[k]])
    plt.show()
  return

def drawRecoClusterHits( ax, recoObj, ev, pc, newRecoObj):
  nClusters = recoObj.rClusterX[ev][pc].shape[0]
  w = np.ones( (nClusters) )
  mu = np.array( [recoObj.rClusterX[ev][pc], recoObj.rClusterY[ev][pc]] ).T
  var = np.zeros( (nClusters, 2))
  plu.drawModelComponents( ax, w, mu, var, color='red', pattern="o" )

  if len(newRecoObj.mu) != 0 :
    w = newRecoObj.w[ev][pc]
    mu = newRecoObj.mu[ev][pc]
    var = newRecoObj.var[ev][pc]
    print("w", w)
    print("mu", mu)
    print("sig", np.sqrt(var) )
    plu.drawModelComponents( ax, w, mu, var, color='black', pattern="o" )
  #
  return

def drawPreClusterPads( ax, recoObj, ev, pc):
    x = recoObj.padX[ev][pc]
    dx = recoObj.padDX[ev][pc]
    y = recoObj.padY[ev][pc]
    dy = recoObj.padDY[ev][pc]    
    frame = plu.getPadBox( x, y, dx, dy )
    ( xMin, xMax, yMin, yMax ) = plu.getPadBox( x, y, dx, dy )
    drawOneCluster(ax, x, y, dx, dy,
        recoObj.padCath[ev][pc], recoObj.padCharge[ev][pc], noYTicksLabels=False )
    xLen = xMax -xMin
    yLen = yMax -yMin
    ax.set_xlim( xMin, xMax )
    ax.set_ylim( yMin, yMax )
    return frame

def drawMCHitsInFrame ( ax, frame, mcObj, ev, DEIds ):

    (xMin, xMax, yMin, yMax) = frame
    x = []
    y = []
    charge = []       
    for deid in DEIds:
      for tidx, tid in enumerate(mcObj.trackId[ev]):
        flag0 = ( mcObj.trackDEId[ev][tidx] == deid )
        flag1 = ( mcObj.trackCharge[ev][tidx] > 0 )
        idx = np.where( np.bitwise_and( flag0, flag1 ) )
        x.append( mcObj.trackX[ev][tidx][idx])
        y.append( mcObj.trackY[ev][tidx][idx])
        charge.append( mcObj.trackCharge[ev][tidx][idx])
    x = np.concatenate( x ).ravel()
    y = np.concatenate( y ).ravel()
    charge = np.concatenate( charge ).ravel()
    flag0 = np.bitwise_and( (x >= xMin), (x < xMax) )
    flag1 = np.bitwise_and( (y >= yMin), (y < yMax) )
    flags = np.bitwise_and( flag0, flag1)
    idx = np.where( flags )
    x = x[idx]
    y = y[idx]
    # print( "sum(flags)",np.sum(flags))
    #
    nClusters = x.shape[0]
    w = np.ones( (nClusters) )
    mu = np.array( [x, y] ).T
    var = np.zeros( (nClusters, 2))
    plu.drawModelComponents( ax, w, mu, var, color='black', pattern="+" )
    #
    return

def statOnEvents( measure, mcObj, recoObj, emObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP = []
  allFP = []
  allFN = []
  allDMin = []
  allPC = []
  allEv = []
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    print( "ev = ", ev, "all matched hits TP, FP, FN", evTP, evFP, evFN)
    allTP.append( evTP )
    allFP.append( evFP )
    allFN.append( evFN )
    allDMin.append( np.concatenate( evDMin ).ravel() )
  #
  allTP = np.array( allTP, dtype=int )
  allFP = np.array( allFP, dtype=int )
  allFN = np.array( allFN, dtype=int )
  # print(allDMin)
  print( "All ev. matched hits TP, FP, FN", np.sum(allTP), np.sum(allFP), np.sum(allFN) )
  allDMin = np.concatenate( allDMin ).ravel()
  nValues = allDMin.size
  aver = float( np.sum( allDMin ) ) / nValues
  print("Distances", aver, np.max( allDMin ) )
  n, bins, patches  = plt.hist(allDMin, bins=100, range=None)

  plt.xlabel('distances (cm)')
  plt.ylabel('# of occurences')
  plt.title('Histogram of the distances between Reco & MC clusters')
  #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
  # plt.xlim(40, 160)
  # plt.ylim(0, 0.03)
  plt.grid(True)
  plt.show()
  return

def refineOneHitClusters( measure, mcObj, preClusters, evRange, nGaussians=1 ):
  var = [ [] for _ in range(11) ]
  res = [ [] for _ in range(11) ]
  ClusterId = [ [] for _ in range(11) ]
  minCharge = 0*11
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    for im, m in enumerate(assign) :
      (pc, match, nbrOfHits, TP, FP, FN, mcHitsInvolved ) = m
      if (TP == 1) and (FP==0) and (FN==0):
        chIds = preClusters.padChId[ev][pc] 
        diffChIds = np.unique(chIds)
        if diffChIds.size != 1 : 
          print("several or empy Chambers")
          print(m)
          print(chIds)
          input("Warning on Chamber")
          continue
        #
        chId = chIds[0]
        xi = preClusters.padX[ev][pc]
        dxi = preClusters.padDX[ev][pc]
        yi = preClusters.padY[ev][pc]
        dyi = preClusters.padDY[ev][pc]
        cathi = preClusters.padCath[ev][pc]
        chi = preClusters.padCharge[ev][pc]

        x0 = np.copy( xi[cathi==0] )
        dx0 = np.copy( dxi[cathi==0] )
        y0 = np.copy( yi[cathi==0] )
        dy0 = np.copy( dyi[cathi==0] )
        ch0 = np.copy( chi[cathi==0] )
        x1 = np.copy( xi[cathi==1] )
        dx1 = np.copy( dxi[cathi==1] )
        y1 = np.copy( yi[cathi==1] )
        dy1 = np.copy( dyi[cathi==1] )
        ch1 = np.copy( chi[cathi==1] )
        
        print("??? x0", x0)
        print("??? dx0", dx0)        
        # Edge reduction
        x0, y0, dx0, dy0, ch0 = plu.computeOverlapingPads( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1 )
        print("??? x0", x0)
        print("??? dx0", dx0)
        x1, y1, dx1, dy1, ch1 = plu.computeOverlapingPads( x1, y1, dx1, dy1, ch1, x0, y0, dx0, dy0, ch0 )        
        cathii = np.hstack( [ np.zeros( x0.shape, dtype=np.int), np.ones( x1.shape, dtype=np.int)] )
        xii = np.hstack( [x0, x1] )
        dxii = np.hstack( [dx0, dx1] )
        yii = np.hstack( [y0, y1] )
        dyii = np.hstack( [dy0, dy1] )
        chii = np.hstack( [ch0, ch1] )
        # drawOneCluster( ax[1], xii, yii, dxii, dyii, cathii, chii) 
        # plt.show()
        
        # No edge reduction
        # xii = xi; yii = yi; dxii=dxi; dyii = dyi;  cathii = cathi; chii = chi;
        
        w, mux, maxCath, idxMax = computeBarycenter( xii, yii, dxii, dyii, cathii, chii )
        mu0 = np.array ( [ mux] )
        var0 = np.array( [ [0.16, 0.16] ] )
        w0 = np.array( [1.0] )        
        wEM, muEM, varEM = simpleProcessEMCluster( xii, yii, dxii, dyii, cathii, chii, w0, mu0, var0, cstVar=False )

        # wf, muf, varf = filterModel( wEM, muEM, varEM )
        # wFinal, muFinal, varFinal = simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, wEM, muEM, varEM )
        print("Chamber", chId)
        print("w", wEM)
        print("mu", muEM )
        print("sig", np.sqrt( varEM ) )
        print("x recoHit", preClusters.rClusterX[ev][pc])
        print("y recoHit", preClusters.rClusterY[ev][pc])
        xErr = np.abs( muEM[0][0] - preClusters.rClusterX[ev][pc])
        yErr = np.abs( muEM[0][1] - preClusters.rClusterY[ev][pc])
        print("Error", xErr, yErr, np.sqrt( xErr*xErr + yErr*yErr))
        print( "distance min", evDMin[im])
        if varEM.shape[0] == 1:
          var[chId].append( varEM[0] )
      # One Hit
    # pc loop
  # ev loop
  fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 7) )
  print(ax.shape, ax)
  for ch in range(1,11):
    v = np.vstack( var[ch] )
    print("??? shape v", v.shape)
    iRow = (ch - 1) // 5
    iCol = (ch - 1) % 5
    xp = np.sqrt(v[:, 0])
    yp = np.sqrt(v[:, 1])
    # ax[iRow, iCol].plot( x, y, "+" )
    print( iRow, iCol)
    ax[iRow, iCol].plot( xp, yp, "+" )
  plt.show( )

  return

def selectOneHitClusters( measure, emObj,mcObj, preClusters, evRange, nGaussians=1 ):
  var = [ [] for _ in range(11) ]
  res = [ [] for _ in range(11) ]
  ClusterId = [ [] for _ in range(11) ]
  minCharge = 0*11
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    for im, m in enumerate(assign) :
      print("???", m)
      (pc, match, nbrOfHits, TP, FP, FN, mcHitsInvolved ) = m
      if (TP == 1) and (FP==0) and (FN==0):
        chIds = preClusters.rClusterChId[ev][pc] 
        diffChIds = np.unique(chIds)
        if diffChIds.size != 1 : 
          print("several or empy Chambers")
          print(m)
          print(chIds)
          input("Warning on Chamber")
          continue
        #
        chId = chIds[0]
        xi = preClusters.padX[ev][pc]
        dxi = preClusters.padDX[ev][pc]
        yi = preClusters.padY[ev][pc]
        dyi = preClusters.padDY[ev][pc]
        cathi = preClusters.padCath[ev][pc]
        chi = preClusters.padCharge[ev][pc]
        # w0, mu0, var0 = findLocalMaxWithSubstraction( xi, yi, dxi, dyi, cathi, chi )

        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 7) )
        # drawOneCluster( ax[0], xi, yi, dxi, dyi, cathi, chi) 

        # Figure frame
        (xInf, xSup, yInf, ySup ) = plu.getPadBox( xi, yi, dxi, dyi)
        x0 = np.copy( xi[cathi==0] )
        dx0 = np.copy( dxi[cathi==0] )
        y0 = np.copy( yi[cathi==0] )
        dy0 = np.copy( dyi[cathi==0] )
        ch0 = np.copy( chi[cathi==0] )
        x1 = np.copy( xi[cathi==1] )
        dx1 = np.copy( dxi[cathi==1] )
        y1 = np.copy( yi[cathi==1] )
        dy1 = np.copy( dyi[cathi==1] )
        ch1 = np.copy( chi[cathi==1] )
        cath0 = np.zeros( x0.shape, dtype=np.int)
        cath1 = np.ones( x1.shape, dtype=np.int)
        
        # Edge reduction
        x0, y0, dx0, dy0, ch0 = plu.computeOverlapingPads( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1 )
        print("??? x0", x0)
        print("??? dx0", dx0)
        # plu.drawPads( ax[1], x0, y0, dx0, dy0, ch0) 
        # plt.show()
        x1, y1, dx1, dy1, ch1 = plu.computeOverlapingPads( x1, y1, dx1, dy1, ch1, x0, y0, dx0, dy0, ch0 )
        
        cathii = np.hstack( [ np.zeros( x0.shape, dtype=np.int), np.ones( x1.shape, dtype=np.int)] )
        xii = np.hstack( [x0, x1] )
        dxii = np.hstack( [dx0, dx1] )
        yii = np.hstack( [y0, y1] )
        dyii = np.hstack( [dy0, dy1] )
        chii = np.hstack( [ch0, ch1] )
        # drawOneCluster( ax[1], xii, yii, dxii, dyii, cathii, chii) 
        # plt.show()
        
        # No edge reduction
        # xii = xi; yii = yi; dxii=dxi; dyii = dyi;  cathii = cathi; chii = chi;
        w, mux, maxCath, idxMax = computeBarycenter( xii, yii, dxii, dyii, cathii, chii )
        mu0 = np.array ( [ mux] )
        var0 = np.array( [ [0.16, 0.16] ] )
        w0 = np.array( [1.0] )        
        print("??? mu0", mux)
        print("??? xi", xi)
        print("??? charge", chi[idxMax])
        wEM, muEM, varEM = EM.simpleProcessEMCluster( xii, yii, dxii, dyii, cathii, chii, w0, mu0, var0, cstVar=False )

        # wf, muf, varf = filterModel( wEM, muEM, varEM )
        # wFinal, muFinal, varFinal = simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, wEM, muEM, varEM )
        print("Chamber", chId)
        print("w", wEM)
        print("mu", muEM )
        print("sig", np.sqrt( varEM ) )
        print("x recoHit", preClusters.rClusterX[ev][pc])
        print("y recoHit", preClusters.rClusterY[ev][pc])
        xErr = np.abs( muEM[0][0] - preClusters.rClusterX[ev][pc])
        yErr = np.abs( muEM[0][1] - preClusters.rClusterY[ev][pc])
        print("Error", xErr, yErr, np.sqrt( xErr*xErr + yErr*yErr))
        print( "distance min", evDMin[im])
        print
        if np.any(xErr > 0.1) or np.any(yErr > 0.1):
          fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7) )
          drawOneCluster( ax[0, 0], x0, y0, dx0, dy0, cath0, ch0, mode="cath0") 
          drawOneCluster( ax[0, 1], x1, y1, dx1, dy1, cath1, ch1, mode="cath1") 
          drawOneCluster( ax[1, 0], xi, yi, dxi, dyi, cathi, chi) 
          drawOneCluster( ax[1, 1], xii, yii, dxii, dyii, cathii, chii)
          plu.drawModelComponents( ax[1,1], wEM, muEM, varEM, color='red', pattern="x" )
          # Plot preClusters
          l = preClusters.rClusterX[ev][pc].shape[0]
          wd = np.ones( l )
          mud = np.array( [ preClusters.rClusterX[ev][pc], preClusters.rClusterY[ev][pc]] ).T
          vard = np.zeros( (l,2))
          plu.drawModelComponents( ax[0,0], wd, mud, vard, color='black', pattern="o" )
          plu.drawModelComponents( ax[0,1], wd, mud, vard, color='black', pattern="o" )
          plu.drawModelComponents( ax[1,1], wd, mud, vard, color='black', pattern="o" )
          
          # MC plot
          wMC = []
          muMC = []
          varMC = []
          for mchit in mcHitsInvolved:
            (tidx, idx) = mchit
            for ik in idx:
              print("??? mc pos", mcData.trackX[ev][tidx][ik])
              print("??? mc pos", mcData.trackY[ev][tidx][ik])              
              muMC.append( np.array( [mcData.trackX[ev][tidx][ik], mcData.trackY[ev][tidx][ik] ]) )
              wMC.append( 1.0 )
              varMC.append( np.ones( (2) ))
          if len( wMC) != 0 :
            wMC = np.array( wMC )
            muMC = np.vstack( muMC )
            varMC = np.vstack( varMC )
            plu.drawModelComponents( ax[0,0], wMC, muMC, varMC, color='black', pattern="+" )            
            plu.drawModelComponents( ax[0,1], wMC, muMC, varMC, color='black', pattern="+" )            
            plu.drawModelComponents( ax[1,1], wMC, muMC, varMC, color='black', pattern="+" )            
          plu.drawModelComponents( ax[1,1], emObj.w[ev][pc], emObj.mu[ev][pc], emObj.var[ev][pc],
                                    color='black', pattern="x" )
          for i in range(2):
            for j in range(2):
              ax[i,j].set_xlim( xInf, xSup )
              ax[i,j].set_ylim( yInf, ySup )
          #
          plt.show()
        # input("A unique hit")
      # One Hit
    # pc loop
  # ev loop
  return


def statWithOneHitClusters( measure, mcObj, preClusters, evRange, nGaussians=1, displayDEIds=False ):
  var = [ [] for _ in range(11) ]
  res = [ [] for _ in range(11) ] 
  d = []
  sig = []
  allChIds = []
  allDEIds = []
  DX = []
  ClusterId = [ [] for _ in range(11) ]
  minCharge = 0*11
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    for im, m in enumerate(assign) :
      (pc, match, nbrOfHits, TP, FP, FN, mcHitsInvolved ) = m
      if (TP == 1) and (FP==0) and (FN==0):
        chIds = preClusters.rClusterChId[ev][pc] 
        diffChIds = np.unique(chIds)
        if diffChIds.size != 1 : 
          print("several or empy Chambers")
          print(m)
          print(chIds)
          print(DEIds)
          # input("Warning on Chamber")
          continue
        #
        xi  = preClusters.padX[ev][pc]
        dxi = preClusters.padDX[ev][pc]
        yi  = preClusters.padY[ev][pc]
        dyi = preClusters.padDY[ev][pc]
        cathi = preClusters.padCath[ev][pc]
        chi   = preClusters.padCharge[ev][pc]
        DEIds = preClusters.padDEId[ev][pc]
        DEIds = np.unique( DEIds )
        chIds = preClusters.padChId[ev][pc]
        chIds = np.unique( chIds )
        var = np.array( [0.1, 0.1])
        var = var * var
        if xi.size == 0 or yi.size == 0 or chi[cathi==0].size == 0 or chi[cathi==1].size == 0:
          continue
        w0, mu0, var0 = findLocalMaxWithSubstraction( xi, yi, dxi, dyi, cathi, chi, var )
        w0 = w0[0:1]
        mu0  = mu0[0:1, :]
        var0 = var0[0:1, :]
        wEM, muEM, varEM = EM.simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, w0, mu0, var0, cstVar=False )
        # wf, muf, varf = filterModel( wEM, muEM, varEM )
        # wFinal, muFinal, varFinal = simpleProcessEMCluster( xi, yi, dxi, dyi, cathi, chi, wf, muf, varf, cstVar=True )
        # print(wEM)
        # print(muEM)
        # print(varEM)
        # print( "distance min", evDMin[im])
        sig.append( np.sqrt(varEM[0]) )
        allChIds.append( chIds[0] )
        allDEIds.append( DEIds[0] )
        d.append( evDMin[im] )
        DX.append( np.unique( dxi ) )
      # One Hit
    # pc loop
  # ev loop
  allDEIds = np.array( allDEIds)
  DEIdSet = list( np.unique( allDEIds ) )
  # print('DEIdSet', len(DEIdSet), DEIdSet)
  # ChIdSet = list( np.unique( allDEIds ) )
  # print('DEIdSet', len(DEIdSet), DEIdSet)
  sig = np.array( sig )
  DX = np.array( DX )
  input("Next")
  if displayDEIds :
    for k, did in enumerate(DEIdSet):
        if ( (k % 10) == 0 ):
          fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(17, 7) )
        #
        idx = np.where( allDEIds == did )
        # print(idx)
        # print(sig[idx])

        u = sig[idx,:][0]
        iRow = int( ( k % 10 ) // 5 )
        iCol = k % 5 
        n, bins, patches = ax[iRow, iCol].hist(u[:,0], 50)    
        ax[iRow, iCol].set_title( str(did) )
        # print( "DX", did, DX[idx] )
        if ( k % 10 == 9 ):
          plt.show()
      # 
    plt.show()
  else:
    chids = np.array( allChIds )
    sig = np.array( sig )
    print("sig shape", sig.shape )
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(17, 7) )
    for k in range(1,11):
      idx = np.where( chids == k)
      u = sig[idx,:][0]
      iRow = int( (k-1) // 5 )
      iCol = (k-1) % 5 
      n, bins, patches = ax[iRow, iCol].hist(u[:,0], 25)
      print("??? n", n)
      nMax = np.max( n )
      if k == 1 or k == 2:
        ax[iRow, iCol].plot( [0.25, 0.25], [0, nMax], "red", "--" )
      else:
        ax[iRow, iCol].plot( [0.3, 0.3], [0, nMax], "red", "--" )
    plt.show()
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(17, 7) )
    for k in range(1,11):
      idx = np.where( chids == k)
      u = sig[idx,:][0]
      iRow = int( (k-1) // 5 )
      iCol = (k-1) % 5 
      n, bins, patches = ax[iRow, iCol].hist(u[:,1], 25)
      print("??? n", n)
      nMax = np.max( n )
      if k == 1 or k == 2:
        ax[iRow, iCol].plot( [0.22, 0.22], [0, nMax], "red", "--" )
      else:
        ax[iRow, iCol].plot( [0.245, 0.245], [0, nMax], "red", "--" )
    plt.show()
  return

def statOnClusterSize( measure, mcObj, recoObj, emObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP = [0]*11
  allFP = [0]*11
  allFN = [0]*11
  allDMin = [ [] for _ in range(11) ]
  allSizes = [ [] for _ in range(11) ]
  allPC = []*11
  allEv = []*11 
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    for k, m in enumerate( assign ) :
      (pc, match, nbrOfHits, TP, FP,FN, dxMin, dyMin, mcHitsInvolved ) = m
      # print( "??? ", recoObj.rClusterX[ev][pc].size, len(mcHitsInvolved) )
      """
      if len(mcHitsInvolved) != 1:
         print( "??? len pb", mcHitsInvolved )
         exit()
      """
      # ( trackIdx, hitIdx) =  mcHitsInvolved[0]
      chIds = recoObj.padChId[ev][pc] 
      diffChIds = np.unique(chIds)
      if diffChIds.size != 1 : 
        print("several Chambers")
        print(m)
        print(chIds)
        input("Warning on Chamber")
        continue
      chId = chIds[0]
      # print( "??? # reco/mc", recoObj.rClusterX[ev][pc].size, len(mcHitsInvolved) )
      # print("??? dMin", evDMin[k])
      # print(evDMin)
      clSize = len(mcHitsInvolved)
      if clSize  > 60:
        print (evDMin[k])
        print( "??? # reco/fullEM/mc", recoObj.rClusterX[ev][pc].size, len(mcHitsInvolved) )
        input("next")  
      allDMin[chId].append( evDMin[k]  )
      allSizes[chId].append( np.ones(evDMin[k].shape)*clSize)
      # input("next")
  nFigRow = 2
  nFigCol = 5
  fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(10, 7) )
  for ch in range(1,11):
    print( "ch=", ch, "All matched hits TP, FP, FN", allTP[ch], allFP[ch], allFN[ch] )
    y = np.concatenate( allDMin[ch] ).ravel() 
    x = np.concatenate( allSizes[ch] ).ravel() 
    nValues = x.size
    fX = (ch - 1) // nFigCol
    fY = (ch - 1) % nFigCol
    ax[fX, fY].plot( x, y,".")
    if (nValues != 0):
      aver = float( np.sum( y ) ) / nValues
      print("Distances", aver )
    xMin = np.min(x)
    xMax = np.max(x)
    
    ax[fX, fY].plot( [ xMin, xMax], [aver, aver], color="r" )
    xSet = np.unique( x )
    for k in xSet:
      yk = y[ np.where( x == k)]
      averk = float( np.sum( yk ) ) / yk.size
      ax[fX, fY].plot( [ k ], [averk], "o", color="r" )
    
    ax[fX, fY].plot( [xMin, xMax], [0.07, 0.07], "--", color="green" )
    print("xSet ", xSet)
    if fX == 1 :
      ax[fX, fY].set_xlabel('# mcHits')
    if fY == 0 :
      ax[fX, fY].set_ylabel('distances (cm)')
    ax[fX, fY].set_ylim( 0.0, 0.4 )
  # fig.title('Hit accuracy versus cluster size (# of MC Hits)')
  plt.show()
  return allSizes, allDMin

def compareClusterSizes( rCSizes, rDMin, emCSizes, emDMin):
  nFigRow = 2
  nFigCol = 3
  fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(10, 7) )
  for s in range(5):
    ch = 2*s + 1
    y1 = np.concatenate( rDMin[ch] ).ravel()
    y2 = np.concatenate( rDMin[ch+1] ).ravel()
    yr = np.hstack( [y1, y2 ] )
    x1 = np.concatenate( rCSizes[ch] ).ravel()
    x2 = np.concatenate( rCSizes[ch+1] ).ravel()
    xr = np.hstack( [x1, x2 ] )
    y1 = np.concatenate( emDMin[ch] ).ravel()
    y2 = np.concatenate( emDMin[ch+1] ).ravel()
    yem = np.hstack( [y1, y2 ] )
    x1 = np.concatenate( emCSizes[ch] ).ravel()
    x2 = np.concatenate( emCSizes[ch+1] ).ravel()
    xem = np.hstack( [x1, x2 ] )
    fX = (s) // nFigCol
    fY = (s) % nFigCol
    """
    nValues = x.size
    ax[fX, fY].plot( x, y,".")
    if (nValues != 0):
      aver = float( np.sum( y ) ) / nValues
      print("Distances", aver )
    """
    print(xr.shape)
    print(xr)
    x1 = np.min( xr ) 
    x2 =  np.min( xem )
    xMin = min( x1, x2 )
    x1 = np.max( xr ) 
    x2 =  np.max( xem )
    xMax = max( x1, x2 )
    
    # ax[fX, fY].plot( [ xMin, xMax], [aver, aver], color="r" )
    # reco
    xSet = np.unique( xr )
    for k in xSet:
      yk = yr[ np.where( xr == k)]
      averk = float( np.sum( yk ) ) / yk.size
      ax[fX, fY].plot( [ k ], [averk], "o", color="b" )
    # em 
    xSet = np.unique( xem )
    for k in xSet:
      yk = yem[ np.where( xem == k)]
      averk = float( np.sum( yk ) ) / yk.size
      ax[fX, fY].plot( [ k ], [averk], "o", color="r" )
      
    
    ax[fX, fY].plot( [xMin, xMax], [0.07, 0.07], "--", color="green" )
    print("xSet ", xSet)
    if fX == 1 :
      ax[fX, fY].set_xlabel('# mcHits')
    if fY == 0 :
      ax[fX, fY].set_ylabel('distances (cm)')
  # fig.title('Hit accuracy versus cluster size (# of MC Hits)')
  plt.show()
  return
  
def statOnChambers( measure, mcObj, recoObj, emObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP = [0]*11
  allFP = [0]*11
  allFN = [0]*11
  allDMin = [ [] for _ in range(11) ]
  allPC = []*11
  allEv = []*11 
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    for m in assign :
      (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin  ) = m
      chIds = recoObj.padChId[ev][pc] 
      diffChIds = np.unique(chIds)
      if diffChIds.size != 1 : 
        print("several Chambers")
        print(m)
        print(chIds)
        input("Warning on Chamber")
        continue
      chId = diffChIds[0]
      allTP[chId] += TP 
      allFP[chId] += FP 
      allFN[chId] += FN
      allDMin[chId].append( np.concatenate( evDMin ).ravel() )
  
  for ch in range(1,11):
    print( "ch=", ch, "All matched hits TP, FP, FN", allTP[ch], allFP[ch], allFN[ch] )
    allDMin[ch] = np.concatenate( allDMin[ch] ).ravel()
    nValues = allDMin[ch].size
    if (nValues != 0):
      aver = float( np.sum( allDMin[ch] ) ) / nValues
      print("Distances", aver, np.max( allDMin[ch] ) )
      n, bins, patches  = plt.hist(allDMin[ch], bins=100, range=None)
      plt.xlabel('distances (cm)')
      plt.ylabel('# of occurences')
      plt.title('Histogram of the distances between Reco & MC clusters')
      plt.grid(True)
      plt.show()
  return

def statOnDistances( measure, mcObj, recoObj, emObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP = [0]*11
  allFP = [0]*11
  allFN = [0]*11
  allDMin = [ [] for _ in range(11) ]
  allDxMin = [ [] for _ in range(11) ]
  allDyMin = [ [] for _ in range(11) ]
  allPC = []*11
  allEv = []*11 
  totalnValues = 0
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    for m in assign :
      (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) = m
      # print("PreCluster pc, DEId", pc, recoObj.rClusterDEId[ev][pc], recoObj.rClusterX[ev][pc])
      # print("mcHitsInvolved", mcHitsInvolved)
      # for mch in mcHitsInvolved:
      #    tid, idx = mch
      #    print("MC part, DEId", mcObj.trackParticleId[ev][tid], mcObj.trackDEId[ev][tid], mcObj.trackX[ev][tid])
      # input('next')
      chIds = recoObj.padChId[ev][pc] 
      diffChIds = np.unique(chIds)
      if diffChIds.size != 1 : 
        print("several Chambers")
        print(m)
        print(chIds)
        input("Warning on Chamber")
        continue
      chId = diffChIds[0]
      allTP[chId] += TP 
      allFP[chId] += FP 
      allFN[chId] += FN
      allDxMin[chId].append(dxMin)
      allDyMin[chId].append(dyMin)
      allDMin[chId].append( np.concatenate( evDMin ).ravel() )
  # ev loop


  U = np.empty( shape=(0) )
  V = np.empty( shape=(0) )
  # X
  fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(13, 7) ) 
  for ch in range(1,11,2):
    print( "ch=", ch, "All matched hits TP, FP, FN", allTP[ch], allFP[ch], allFN[ch] )
    allDMin[ch] = np.concatenate( allDMin[ch] ).ravel()
    dxMin = np.concatenate(allDxMin[ch]).ravel()
    dxMin = np.hstack( [dxMin, np.concatenate(allDxMin[ch+1]).ravel()])
    dyMin = np.concatenate(allDyMin[ch]).ravel()
    dyMin = np.hstack( [dyMin, np.concatenate(allDyMin[ch+1]).ravel()])
    # dyMin = np.concatenate(allDyMin[ch]).ravel()
    nValues = dxMin[ch].size
    totalnValues += nValues 
    U = np.hstack( [ U, dxMin ])
    V = np.hstack( [ V, dyMin ])
    # aver = float( np.sum( allDMin[ch] ) ) / nValues
    # print("Distances", aver, np.max( allDMin[ch] ) )
    jp = (ch - 1)  // 2
    ax[0,jp].grid(True)
    ax[1,jp].grid(True)

    n, bins, patches  = ax[0, jp].hist(dxMin, bins=100, range= (-0.5, 0.5))
    n, bins, patches  = ax[1, jp].hist(dyMin, bins=100, range= (-0.1, 0.1))
    # X Std dev
    xMean = np.mean( dxMin)
    xStd = np.std( dxMin )
    t = r'$\sigma=%.3f$' % xStd
    plu.setText( ax[0,jp], (0.6, 0.9), t, ha='left', fontsize=10)
    t = r'St %1d' % (jp+1)
    plu.setText( ax[0,jp], (0.1, 0.9), t, ha='left', fontsize=11)
    # Y Std dev
    yMean = np.mean( dyMin)
    yStd = np.std( dyMin )
    t = r'$\sigma=%.3f$' % yStd
    plu.setText( ax[1,jp], (0.6, 0.9), t, ha='left', fontsize=10)
    print( "mean / std", xMean, xStd)
    t = r'St %1d' % (jp+1)
    plu.setText( ax[1,jp], (0.1, 0.9), t, ha='left', fontsize=11)
    
  ax[0,0].set_ylabel( "Hist. of X residues" )
  ax[1,0].set_ylabel( "Hist. of Y residues" )
  label = "Residues (cm)"
  ax[1,0].set_xlabel( label )
  ax[1,1].set_xlabel( label )
  ax[1,2].set_xlabel( label )
  ax[1,3].set_xlabel( label )
  ax[1,4].set_xlabel( label )

  plt.suptitle('Histogram of the residues between Reco & MC clusters')
  plt.show()
  # All chambers      plt.show()

  print("???", totalnValues)
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13, 7) ) 
  n, bins, patches  = ax[0,0].hist(U, bins=100, range=(-0.5, 0.5))
  n, bins, patches  = ax[0,1].hist(V, bins=100, range=(-0.1, 0.1))
  xMean = np.mean( U )
  xStd = np.std( U )
  t = r'$\sigma=%.3f$' % xStd
  plu.setText( ax[0,0], (0.8, 0.9), t, ha='left', fontsize=10)
  xMean = np.mean( V )
  xStd = np.std( V )
  t = r'$\sigma=%.3f$' % xStd
  plu.setText( ax[0,1], (0.8, 0.9), t, ha='left', fontsize=10)
  ax[0,0].set_title( "Histogram of X residues" )
  ax[0,1].set_title( "Histogram of Y residues" )
  ax[0,0].set_xlabel( "X residues (cm)" )
  ax[0,1].set_xlabel( "Y residues (cm)" )
  plt.show()
  return

def statOnDistancesMuons( measure, mcObj, recoObj, emObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP = [0]*11
  allFP = [0]*11
  allFN = [0]*11
  allDMin = [ [] for _ in range(11) ]
  allDxMin = [ [] for _ in range(11) ]
  allDyMin = [ [] for _ in range(11) ]
  fakeMuonsDMin = [ [] for _ in range(11) ]
  allPC = []*11
  allEv = []*11 
  totalnValues = 0
  muonDetected = 0
  muonNotDetected = 0
  for ev in evRange:
    muonNotDetectedPerEv = 0
    muonDetectedPerEv = 0
    evNbrOfHits, evTP, evFP, evFN, evDMin, assign = measure[ev]
    for m in assign :
      (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) = m
      # print("PreCluster pc, DEId", pc, recoObj.rClusterDEId[ev][pc], recoObj.rClusterX[ev][pc])
      # print("mcHitsInvolved", mcHitsInvolved)
      # for mch in mcHitsInvolved:
      #    tid, idx = mch
      #    print("MC part, DEId", mcObj.trackParticleId[ev][tid], mcObj.trackDEId[ev][tid], mcObj.trackX[ev][tid])
      # input('next')
      chIds = recoObj.padChId[ev][pc] 
      diffChIds = np.unique(chIds)
      if diffChIds.size != 1 : 
        print("several Chambers")
        print(m)
        print(chIds)
        input("Warning on Chamber")
        continue
      chId = diffChIds[0]
      allTP[chId] += TP 
      allFP[chId] += FP 
      allFN[chId] += FN
      allDyMin[chId].append(dyMin)
      allDMin[chId].append( np.concatenate( evDMin ).ravel() )
      # Debug for muons
      # print("pc=", pc, "match", match )
      # print("dxMin size", dxMin.size)
      #print("assign MC Clusters size", len(mcHitsInvolved) )
      #
      # Assuming that one hit per track
      if tfMatrix.shape[1] != len(mcHitsInvolved):
          input("Pb matrix shape")
      # if tfMatrix.shape[0] != recoObj.rClusterX[ev][pc].shape[0]:
      #    input("Pb matrix shape")
      nbrOfHits = 0
      particles = []
      isMuons=[]
      tracks = []
      tracksIdx = []
      # print("mcHitsInvolved", mcHitsInvolved)
      #
      # Find muon in MC data
      for tHits in mcHitsInvolved:
        (t, idx ) = tHits
        particles.append( mcObj.trackParticleId[ev][t] )
        isMuons.append( abs( mcObj.trackParticleId[ev][t]) == 13)
        tracks.append( t )
        tracksIdx.append( idx[0] )
        #for i in idx:    
        nbrOfHits += idx.size
        # if idx.size != 1:
        #  print("hit per track", idx.size )
        #  input("More than 1 hit per track")
      isMuons = np.array( isMuons, dtype=np.int )
      particles = np.array( particles, dtype=np.int )
      tracks = np.array( tracks, dtype=np.int )
      tracksIdx = np.array( tracksIdx, dtype=np.int )
      if nbrOfHits != len(mcHitsInvolved):
          input("One track with more than 1 hit in the cluster: not treated")

      # Verifying that tfMatrix[nReco, nMC]
      print("tfMatrix shape", tfMatrix.shape)
      # print(" sum on rows", np.sum(tfMatrix, axis=0).size)
      # print(" sum on columns", np.sum(tfMatrix, axis=1).size)
      selectedHits = np.where( np.sum(tfMatrix, axis=1) == 1)[0]
      # print("selectedHits", selectedHits.size, selectedHits)
      dxyMinTotfMatrixRow = selectedHits
      # tfMatrixRowToDxyMinIdx = np.zeros( tfMatrix.shape[0])
      # tfMatrixRowToDxyMinIdx = selectedHits
      # print("# Hits in Reco", recoObj.rClusterX[ev][pc].shape)
      # print( "isMuons", isMuons)
      # print( "particles", particles)
      # 
      # Column index in tfMatrix (j for MC mesures)
      muonJdx = np.where(isMuons  == 1)[0]
      # print("muonJdx", muonJdx)
      # tfMuons = tfMatrix[:, isMuons == 1]
      # print("tfMuons", tfMuons.size, tfMuons.shape)
      if muonJdx.size == 0:
        print("pc=", pc, " match=", match, " No muon in MC")
      # For all MC muon, find if dectected in reco 
      for j, idxCol in enumerate(muonJdx):
        # tfMuonRows = tfMuons[:,j]
        tfMuonRows = tfMatrix[:,idxCol]
        rowSum = np.sum(tfMuonRows)
        if rowSum == 0:
          # Muon not detected
          muonNotDetectedPerEv += 1
          tId = tracks[idxCol]
          tIdx = tracksIdx[idxCol]
          xMuon = mcObj.trackX[ev][tId][tIdx]
          yMuon = mcObj.trackY[ev][tId][tIdx]
          """
          # Reco debug
          xRecoHits = recoObj.rClusterX[ev][pc]
          yRecoHits = recoObj.rClusterY[ev][pc]
          x2 = xRecoHits - xMuon
          x2 = x2 * x2
          y2 = yRecoHits - yMuon
          y2 = y2 * y2
          dMin2 = np.min( x2 + y2)
          # fakeMuonsDMin.append( np.sqrt( dMin2) )
          """
          fakeMuonsDMin.append( np.sqrt( 0.0) )
          print("pc=", pc, " match=", match," UNDETECTED muon, smallest dist", 0.0 )
        elif rowSum == 1:
          # Muon detected
          muonDetectedPerEv += 1
          idxRow = np.where( tfMuonRows == 1 )[0]
          dxyMinIdx = np.where( dxyMinTotfMatrixRow == idxRow )[0]  
          if dxyMinIdx.size != 1:
            input("Problem !")
            exit()
          tId = tracks[idxCol]
          tIdx = tracksIdx[idxCol]
          #
          # For debuging with reco (recoObj)
          """
          verifX = recoObj.rClusterX[ev][pc][idxRow] - mcObj.trackX[ev][tId][tIdx] 
          verifY = recoObj.rClusterY[ev][pc][idxRow] - mcObj.trackY[ev][tId][tIdx] 
          # print( "verif", recoObj.rClusterX[ev][pc][idxRow] )
          # print( "verif", trackIdx[idxCol], isMuonIdx[j] )
          # print( "verif", mcObj.trackX[ev][trackIdx[idxCol]][idxCol] )
          # print( "verif", recoObj.rClusterX[ev][pc][idxRow] - mcObj.trackX[ev][trackIdx[idxCol]][idxCol] )
          """
          print( "pc=", pc, " match=", match,"DETECTED muon, dist", dxMin[dxyMinIdx], dyMin[dxyMinIdx])
          #
          # Debug with recoObj
          # if (dxMin[dxyMinIdx] != verifX) or (dyMin[dxyMinIdx] !=  verifY ):
          #  input("Not equal values dxMin, dyMin")
          allDxMin[chId].append( dxMin[dxyMinIdx] )
          allDyMin[chId].append( dyMin[dxyMinIdx] )
        else:
          input("Impossible !")
          exit()
    print( "ev=", ev, "detected", muonDetectedPerEv, muonNotDetectedPerEv)
    muonDetected += muonDetectedPerEv
    muonNotDetected += muonNotDetectedPerEv
    # input("next")
  # ev loop
  print( "Total dectected muons", muonDetected)
  print( "Total unDectected muons", muonNotDetected)
  input("next")


  U = np.empty( shape=(0) )
  V = np.empty( shape=(0) )
  # X
  fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(13, 7) ) 
  for ch in range(1,11,2):
    print( "ch=", ch, "All matched hits TP, FP, FN", allTP[ch], allFP[ch], allFN[ch] )
    allDMin[ch] = np.concatenate( allDMin[ch] ).ravel()
    dxMin = np.concatenate(allDxMin[ch]).ravel()
    dxMin = np.hstack( [dxMin, np.concatenate(allDxMin[ch+1]).ravel()])
    dyMin = np.concatenate(allDyMin[ch]).ravel()
    dyMin = np.hstack( [dyMin, np.concatenate(allDyMin[ch+1]).ravel()])
    # dyMin = np.concatenate(allDyMin[ch]).ravel()
    nValues = dxMin[ch].size
    totalnValues += nValues 
    U = np.hstack( [ U, dxMin ])
    V = np.hstack( [ V, dyMin ])
    # aver = float( np.sum( allDMin[ch] ) ) / nValues
    # print("Distances", aver, np.max( allDMin[ch] ) )
    jp = (ch - 1)  // 2
    ax[0,jp].grid(True)
    ax[1,jp].grid(True)

    n, bins, patches  = ax[0, jp].hist(dxMin, bins=100, range= (-0.5, 0.5))
    n, bins, patches  = ax[1, jp].hist(dyMin, bins=100, range= (-0.1, 0.1))
    # X Std dev
    xMean = np.mean( dxMin)
    xStd = np.std( dxMin )
    t = r'$\sigma=%.3f$' % xStd
    plu.setText( ax[0,jp], (0.6, 0.9), t, ha='left', fontsize=10)
    t = r'St %1d' % (jp+1)
    plu.setText( ax[0,jp], (0.1, 0.9), t, ha='left', fontsize=11)
    # Y Std dev
    yMean = np.mean( dyMin)
    yStd = np.std( dyMin )
    t = r'$\sigma=%.3f$' % yStd
    plu.setText( ax[1,jp], (0.6, 0.9), t, ha='left', fontsize=10)
    print( "mean / std", xMean, xStd)
    t = r'St %1d' % (jp+1)
    plu.setText( ax[1,jp], (0.1, 0.9), t, ha='left', fontsize=11)
    
  ax[0,0].set_ylabel( "Hist. of X residues" )
  ax[1,0].set_ylabel( "Hist. of Y residues" )
  label = "Residues (cm)"
  ax[1,0].set_xlabel( label )
  ax[1,1].set_xlabel( label )
  ax[1,2].set_xlabel( label )
  ax[1,3].set_xlabel( label )
  ax[1,4].set_xlabel( label )

  plt.suptitle('Histogram of the residues between Reco & MC clusters')
  plt.show()
  # All chambers      plt.show()

  print("???", totalnValues)
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13, 7) ) 
  n, bins, patches  = ax[0,0].hist(U, bins=100, range=(-0.5, 0.5))
  n, bins, patches  = ax[0,1].hist(V, bins=100, range=(-0.1, 0.1))
  xMean = np.mean( U )
  xStd = np.std( U )
  t = r'$\sigma=%.3f$' % xStd
  plu.setText( ax[0,0], (0.8, 0.9), t, ha='left', fontsize=10)
  xMean = np.mean( V )
  xStd = np.std( V )
  t = r'$\sigma=%.3f$' % xStd
  plu.setText( ax[0,1], (0.8, 0.9), t, ha='left', fontsize=10)
  ax[0,0].set_title( "Histogram of X residues" )
  ax[0,1].set_title( "Histogram of Y residues" )
  ax[0,0].set_xlabel( "X residues (cm)" )
  ax[0,1].set_xlabel( "Y residues (cm)" )
  plt.show()
  return

def worstCases( measure, mcObj, recoObj, emObj, evRange ):
  maxFP = 0
  maxFN = 0
  allTP = []
  allFP = []
  allFN = []
  allPC = []
  allEv = []
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, allDMin, assign = measure[ev]
    for m in assign :
      (pc, match, nbrOfHits, TP, FP,FN, dMin ) = m
      allTP.append( TP )
      allFP.append( FP )
      allFN.append( FN )
      allEv.append( ev )
      allPC.append( pc )
  allTP = np.array( allTP, dtype=int )
  allFP = np.array( allFP, dtype=int )
  allFN = np.array( allFN, dtype=int)
  allPC = np.array( allPC, dtype=int)
  allEv = np.array( allEv, dtype=int)
  FPIdx = np.argsort( allFP )
  FNIdx = np.argsort( allFN  )
  FPFNIdx = np.argsort( allFP + allFN )
  #
  nFigRow = 1
  nFigCol = 1
  #
  # FP
  #
  n = len(allFP)
  for k in range(n-5, n):
    ev = allEv[FPIdx[k]]
    pc = allPC[FPIdx[k]]
    #
    # ChIds = np.unique( recoObj.rClusterChId[ev][pc] )
    ChIds = np.unique( recoObj.padChId[ev][pc] )
    if ChIds.shape[0] != 1:
        print("Must have only 1 chamber")
        print("reco Ch", ChIds)
        exit()
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(10, 7) )
    #
    # PreCluster
    frame = drawPreClusterPads( ax, recoObj, ev, pc )
    #
    # Reco
    drawRecoClusterHits( ax, recoObj, ev, pc, emObj)
    #
    # MC
    #DEIds = np.unique( recoObj.rClusterDEId[ev][pc] )
    DEIds = np.unique( recoObj.padDEId[ev][pc] )
    drawMCHitsInFrame ( ax, frame, mcObj, ev, DEIds )
    
    #
    print( "ev = ", allEv[FPIdx[k]], "pc = ", allPC[FPIdx[k]], "TP, FP, FN = ", allTP[FPIdx[k]],allFP[FPIdx[k]], allFN[FPIdx[k]] )

    fig.suptitle('Worst FP clusters' + 'Ev='+str(ev)+", pc="+str(pc) , fontsize=16)  
    plt.show()
  #
  # FN
  #
  n = len(allFN)
  for k in range(n-5, n):
    ev = allEv[FNIdx[k]]
    pc = allPC[FNIdx[k]]
    ChIds = np.unique( recoObj.padChId[ev][pc] )
    if ChIds.shape[0] != 1:
        print("Must have only 1 chamber")
        print("reco Ch", ChIds)
        exit()
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
    #
    # PreCluster
    frame = drawPreClusterPads( ax, recoObj, ev, pc )
    #
    # Reco
    drawRecoClusterHits( ax, recoObj, ev, pc, emObj)
    #
    # MC
    DEIds = np.unique( recoObj.padDEId[ev][pc] )
    drawMCHitsInFrame ( ax, frame, mcObj, ev, DEIds )
    
    print( "ev = ", allEv[FNIdx[k]], "pc = ", allPC[FNIdx[k]], "TP, FP, FN = ", allTP[FNIdx[k]],allFP[FNIdx[k]], allFN[FNIdx[k]] )
    #
    fig.suptitle('Worst FN clusters' + 'Ev='+str(ev)+", pc="+str(pc) , fontsize=16)  
    plt.show()
  #
  # FP+FN
  #
  n = len(allFN)
  for k in range(n-5, n):
    ev = allEv[FPFNIdx[k]]
    pc = allPC[FPFNIdx[k]]
    ChIds = np.unique( recoObj.padChId[ev][pc] )
    if ChIds.shape[0] != 1:
        print("Must have only 1 chamber")
        print("reco Ch", ChIds)
        continue
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(10, 7) )
    #
    # PreCluster
    frame = drawPreClusterPads( ax, recoObj, ev, pc )
    #
    # Reco
    drawRecoClusterHits( ax, recoObj, ev, pc, emObj)
    #
    # MC
    DEIds = np.unique( recoObj.padDEId[ev][pc] )
    drawMCHitsInFrame ( ax, frame, mcObj, ev, DEIds )
    
    print( "ev = ", allEv[FPFNIdx[k]], "pc = ", allPC[FPFNIdx[k]], "TP, FP, FN = ", allTP[FPFNIdx[k]],allFP[FPFNIdx[k]], allFN[FPFNIdx[k]] )
    #
    fig.suptitle('Worst FP+FN clusters' + 'Ev='+str(ev)+", pc="+str(pc) , fontsize=16)  
    plt.show()    
  return

def worstCasesByChamber( measure, mcObj, recoObj, emObj, evRange, inChId ):
  maxFP = 0
  maxFN = 0
  allTP = []
  allFP = []
  allFN = []
  allPC = []
  allEv = []
  allDMinMatrix = []
  for ev in evRange:
    evNbrOfHits, evTP, evFP, evFN, allDMin, assign = measure[ev]
    for m in assign :
      (pc, match, nbrOfHits, TP, FP,FN, dMin ) = m
      ChIds = np.unique( recoObj.padChId[ev][pc] )
      if ChIds.shape[0] != 1:
        prin("Not a unique Chamber", ChIds)
        exit()
      if ChIds[0] == inChId:
        allTP.append( TP )
        allFP.append( FP )
        allFN.append( FN )
        allEv.append( ev )
        allPC.append( pc )
        allDMinMatrix.append( dMin )
  allTP = np.array( allTP, dtype=int )
  allFP = np.array( allFP, dtype=int )
  allFN = np.array( allFN, dtype=int)
  allPC = np.array( allPC, dtype=int)
  allEv = np.array( allEv, dtype=int)
  FPIdx = np.argsort( allFP )
  FNIdx = np.argsort( allFN  )
  FPFNIdx = np.argsort( allFP + allFN )
  #
  nFigRow = 1
  nFigCol = 1
  #
  # FP
  #
  n = len(allFP)
  for k in range(n-5, n):
    ev = allEv[FPIdx[k]]
    pc = allPC[FPIdx[k]]
    #
    # ChIds = np.unique( recoObj.rClusterChId[ev][pc] )
    ChIds = np.unique( recoObj.padChId[ev][pc] )
    if ChIds.shape[0] != 1:
        print("Must have only 1 chamber")
        print("reco Ch", ChIds)
        exit()
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(10, 7) )
    #
    # PreCluster
    frame = drawPreClusterPads( ax, recoObj, ev, pc )
    #
    # Reco
    drawRecoClusterHits( ax, recoObj, ev, pc, emObj)
    #
    # MC
    #DEIds = np.unique( recoObj.rClusterDEId[ev][pc] )
    DEIds = np.unique( recoObj.padDEId[ev][pc] )
    drawMCHitsInFrame ( ax, frame, mcObj, ev, DEIds )
    
    #
    print( "ev = ", allEv[FPIdx[k]], "pc = ", allPC[FPIdx[k]], "TP, FP, FN = ", allTP[FPIdx[k]],allFP[FPIdx[k]], allFN[FPIdx[k]] )
    print( "allDMinMatrix", allDMinMatrix[FPIdx[k]])
    
    fig.suptitle('Worst FP clusters' + 'Ev='+str(ev)+", pc="+str(pc) , fontsize=16)  
    plt.show()
  #
  # FN
  #
  n = len(allFN)
  for k in range(n-5, n):
    ev = allEv[FNIdx[k]]
    pc = allPC[FNIdx[k]]
    ChIds = np.unique( recoObj.padChId[ev][pc] )
    if ChIds.shape[0] != 1:
        print("Must have only 1 chamber")
        print("reco Ch", ChIds)
        exit()
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(10, 7) )
    #
    # PreCluster
    frame = drawPreClusterPads( ax, recoObj, ev, pc )
    #
    # Reco
    drawRecoClusterHits( ax, recoObj, ev, pc, emObj)
    #
    # MC
    DEIds = np.unique( recoObj.padDEId[ev][pc] )
    drawMCHitsInFrame ( ax, frame, mcObj, ev, DEIds )
    
    print( "ev = ", allEv[FNIdx[k]], "pc = ", allPC[FNIdx[k]], "TP, FP, FN = ", allTP[FNIdx[k]],allFP[FNIdx[k]], allFN[FNIdx[k]] )
    #
    fig.suptitle('Worst FN clusters' + 'Ev='+str(ev)+", pc="+str(pc) , fontsize=16)  
    plt.show()
  #
  # FP+FN
  #
  n = len(allFN)
  for k in range(n-5, n):
    ev = allEv[FPFNIdx[k]]
    pc = allPC[FPFNIdx[k]]
    ChIds = np.unique( recoObj.padChId[ev][pc] )
    if ChIds.shape[0] != 1:
        print("Must have only 1 chamber")
        print("reco Ch", ChIds)
        exit()
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(10, 7) )
    #
    # PreCluster
    frame = drawPreClusterPads( ax, recoObj, ev, pc )
    #
    # Reco
    drawRecoClusterHits( ax, recoObj, ev, pc, emObj)
    #
    # MC
    DEIds = np.unique( recoObj.padDEId[ev][pc] )
    drawMCHitsInFrame ( ax, frame, mcObj, ev, DEIds )
    
    print( "ev = ", allEv[FPFNIdx[k]], "pc = ", allPC[FPFNIdx[k]], "TP, FP, FN = ", allTP[FPFNIdx[k]],allFP[FPFNIdx[k]], allFN[FPFNIdx[k]] )
    #
    fig.suptitle('Worst FP+FN clusters' + 'Ev='+str(ev)+", pc="+str(pc) , fontsize=16)  
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
        print( "key, number of identical Clusters", key,  l)
        tRef = sameClusters[0][0]
        cRef = sameClusters[0][1]
        print( "tRef, cRef =", tRef, cRef)
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
        wf, muf, varf = simpleProcessEMCluster( cl, wi, mui, vari, cstVar=False )
        wff, muff, varff = filterModel( wf, muf, varf )
        wfff, mufff, varfff = simpleProcessEMCluster( cl, wff, muff, varff )
        
        """
        print( "wi", wi
        print( 'mui', mui
        print( 'vari', vari
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
                print( "NOT THE SAME TRACK/CLUSTER t=", t,"/", tRef, ", c=",c ,"/", cRef)
                print( "ev=", ev, ", mcLabel=", mcLabel, ", code=", partCode, ", clID=", iCl, ", nPads=", nPads )
            # Bounds of the Reco Cluster
            xMin = np.min( tracksObj.tracks[t][c].x - tracksObj.tracks[t][c].dx) 
            xMax = np.max( tracksObj.tracks[t][c].x + tracksObj.tracks[t][c].dx) 
            yMin = np.min( tracksObj.tracks[t][c].y - tracksObj.tracks[t][c].dy) 
            yMax = np.max( tracksObj.tracks[t][c].y + tracksObj.tracks[t][c].dy) 
            if xMin != xInf or xMax != xSup or yMin != yInf or yMax != ySup :
                print( "WARNING: xyInfSupRef", xInf, xSup, yInf, ySup )
                print( "WARNING: xyMinMax", xMin, xMax, yMin, yMax )
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
          print( "WARNING: ", nbMCOutside, "/", len(xyMCs), "MC particle outside of the cluster")
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
    print( " TrackRef keys number", len(keys) )
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
            print( key1 )
            print( key2 )
          else:
            nbrKeyEqual +=1
    print( nbrKeyEqual )
    
    """
    # Loop on the "same" cluster
    for key, l in orderedTrackKeys:
        sameClusters = tracksObj.clusterHash[key]
        print( "key, number of identical Clusters", key,  l
        tRef = sameClusters[0][0]
        cRef = sameClusters[0][1]
        print( "tRef, cRef =", tRef, cRef
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
        # print( "nbRecoClusters=",nbRecoClusters
        if (nbRecoClusters == 1):
           minCharge.append( np.min( tracksObj.tracks[tStore][cStore].charge ))
           maxCharge.append( np.max( tracksObj.tracks[tStore][cStore].charge ))
           meanCharge.append( np.mean( tracksObj.tracks[tStore][cStore].charge ))
           
           clusterCharge.append( np.sum( tracksObj.tracks[tStore][cStore].charge ))
           clusterNbrOfPads.append( tracksObj.tracks[tStore][cStore].x.shape[0] )
        if (nbRecoClusters == 0):
            print( "not recognized", sameClusters )

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
    print( cl.x )
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
 
    print( "neigh", neigh )
    # Get the max
    locMax = []
    for cath in range(len(x)):
      locMaxCath = []
      mask = np.ones( ch[cath].shape )
      while np.sum(mask) != 0 :
        maxIdx = np.argmax( ch[cath] * mask  )
        print( "sum mask", np.sum(mask), ",cath=", cath )
        print( "  max ch", ch[cath][maxIdx] )
        print( "  neigh idx", neigh[cath][maxIdx] )
        print( "  neigh values", ch[cath][neigh[cath][maxIdx]] )
        
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

def filterModelWithMag( w0, mu0, var0 ):
    
    
    K = mu0.shape[0]
    cutOff = 1.0 / K * 0.01
    w0 = np.where( w0 < cutOff, 0, w0 )
    idx = np.argsort( - w0 )
    w = np.copy(w0[idx])
    mu = np.copy(mu0[idx])
    var = np.copy(var0[idx])
    sig = np.sqrt( var )

    for k in range( K ):
      far = True
      if w[k] != 0:
        for l in range(K):
          if w[l] != 0:
            dmux = np.abs(mu[k,0] - mu[l, 0])
            dmuy = np.abs(mu[k,1] - mu[l, 1])
            r = w[l] / w[k]
            r = 1.0
            xDelta = max( sig[k][0], sig[l][0] ) * 0.5 * r
            yDelta = max( sig[k][1], sig[l][1] ) * 0.5 * r 
            # Test if positions are close
            close = (k != l) and ( dmux < xDelta) and (dmuy < yDelta)
            if close:
              print(" l point close to k ", l, k )
              print(" x axe: ", np.abs(mu[k,0] - mu[l, 0]),  "<",  xDelta)
              print(" y axe: ", np.abs(mu[k,1] - mu[l, 1]),  "<",  yDelta)
              # weight += w[l] 

            far = far and (not close)
        if not far:
          print("remove k pont", k)
          w[k] = 0.0
    #
    idx = np.where( w > 0.0)
    return w[idx], mu[idx], var[idx]

def computeBarycenter( xi, yi, dxi, dyi, cathi, chi ):
    x = []; dx = []
    y = []; dy = []
    ch = []
    c0Idx = np.where( cathi == 0)
    c1Idx = np.where( cathi == 1)
    x.append(xi[c0Idx])
    x.append(xi[c1Idx])
    dx.append(dxi[c0Idx])
    dx.append(dxi[c1Idx])    
    y.append(yi[c0Idx])
    y.append(yi[c1Idx])
    dy.append(dyi[c0Idx])
    dy.append(dyi[c1Idx])
    ch.append( np.copy( chi[c0Idx] ))
    ch.append( np.copy( chi[c1Idx] ))  
    # Extract neighbours
    eps = 1.0e-10
    neigh = []
    for cath in range(len(x)):
      neighCath = []
      for i in range( x[cath].shape[0]):
        xMask = np.abs( x[cath][i] - x[cath] ) <= ( (1.0 + eps) * (dx[cath][i] + dx[cath]) )
        yMask = np.abs( y[cath][i] - y[cath] ) <= ( (1.0 + eps) * (dy[cath][i] + dy[cath]) )
        neighCath.append( np.where( np.bitwise_and(xMask, yMask) ) ) 
      neigh.append( neighCath )

    # Get the max
    maxVal = [0., 0.]
        
    # Find the max
    # Rq: Max location Can be refined if 
    # there is an intersection between the 2 pads
    if ch[0].shape[0] == 0:
        maxVal[0] = -1
        print("???", ch)
    else:
      maxVal[0] = np.max( ch[0][:] )
    if ch[1].shape[0] == 0:
        maxVal[1] = -1
        print("???", ch )
    else:
      maxVal[1] = np.max( ch[1][:]  )
    print( "??? maxVal", maxVal[0], maxVal[1] )
    if( maxVal[1] > maxVal[0]):
      maxCath = 1
    else:
      maxCath = 0
    maxIdx =np.argmax( ch[maxCath][:] )
    print("??? maxVal", maxVal, maxIdx, )
    
    # Refine mu
    # Compute mu (barycenter of the charge)
    idx = neigh[maxCath][maxIdx]
    u = ch[maxCath][idx]*x[maxCath][idx]
    mux = np.sum(u) / np.sum( ch[maxCath][idx] )
    u = ch[maxCath][idx]*y[maxCath][idx]
    muy = np.sum(u) / np.sum( ch[maxCath][idx] )
        
    print( "cath, mu, x, y ", maxCath, mux, muy, x[maxCath][maxIdx], y[maxCath][maxIdx] )
    print( "xmin, xmax, ymin, ymax ", np.min( x[maxCath] ), np.max( x[maxCath] ), \
                np.min( y[maxCath] ), np.max( y[maxCath] ) )
    mu = np.zeros(2)    
    mu[0] = mux
    mu[1] = muy
        
    # Computing w
    """
    muTmp = np.array( [ mu ] ).T
    # dxyTmp = np.array([ [dx[maxCath][maxIdx], dy[maxCath][maxIdx] ]]).T
    xyMax = np.array( [ [ x[maxCath][maxIdx], y[maxCath][maxIdx] ] ] ).T 
    dxyMax = np.array( [ [dx[maxCath][maxIdx], dy[maxCath][maxIdx]] ] ).T
    integral = EM.computeDiscretizedGaussian2D( xyMax, dxyMax, mu, var )
    w = ch[maxCath][maxIdx] / integral        
    """
    # return w, mu, maxCath, np.argmax( chi ) 
    return 0, mu, maxCath, np.argmax( chi ) 

def computeResidu( x, y, dx, dy, ch0, w, mu, sig ):
  #
  # Remove the gaussian
  #
  xy = np.vstack([ x, y ])
  dxy = np.vstack([ dx, dy ])
  ch = np.copy( ch0 )
  if ( xy.size != 0 ):
    print( "  gauss : ",  w*EM.computeDiscretizedGaussian2D( xy, dxy, mu, var) )
    ch[:] = ch[:] - w * EM.computeDiscretizedGaussian2D(xy, dxy, mu, var )
  else:
    print("No chage on this cathode")
  #
  # compute residu
  n = ch.size[0]
  res = np.sum( ch ) / n
  res1 = np.sum( np.abs( ch )) / n
  res2 = np.sqrt( np.sum( ch * ch ) ) / n
  res3 = np.max( np.abs( ch ))
  return res, res1, res2, res3

def findLocalMaxWithSubstractionV2( xi, yi, dxi, dyi, cathi, chi_, var0, ax=None, maxIter=200 ):
    x = []; dx = []
    y = []; dy = []
    ch = []
    c0Idx = np.where( cathi == 0)
    c1Idx = np.where( cathi == 1)
    x.append(xi[c0Idx])
    x.append(xi[c1Idx])
    dx.append(dxi[c0Idx])
    dx.append(dxi[c1Idx])    
    y.append(yi[c0Idx])
    y.append(yi[c1Idx])
    dy.append(dyi[c0Idx])
    dy.append(dyi[c1Idx])
    # ch.append( np.copy( chi[c0Idx] ))
    # ch.append( np.copy( chi[c1Idx] ))  
    chi = np.copy( chi_ )
    # sig0 = np.array( [0.35, 0.35] )
    """
    sig0 = np.array( [0.4, 0.4] )
    var0 = sig0 * sig0
    """
    mu = np.zeros(2)
    # Extract neighbours
    eps = 1.0e-10
    """
    neigh = []
    for cath in range(len(x)):
      neighCath = []
      for i in range( x[cath].shape[0]):
        xMask = np.abs( x[cath][i] - x[cath] ) <= ( (1.0 + eps) * (dx[cath][i] + dx[cath]) )
        yMask = np.abs( y[cath][i] - y[cath] ) <= ( (1.0 + eps) * (dy[cath][i] + dy[cath]) )
        neighCath.append( np.where( np.bitwise_and(xMask, yMask) ) ) 
      neigh.append( neighCath )
    """

    # Get the max
    locMax = []
    mask = np.ones( chi.shape )
    maxVal = [0., 0.]
    wa = []
    mua = []
    vara = []
    nIter = 0
    sig0 = np.sqrt( var0)
    while ( np.sum( mask[:]) + np.sum(mask[:] ) > 0.01 ) and nIter < maxIter :

        print("ITER ", nIter)
        print( "  Mask sum", np.sum( mask[:] ), np.sum( mask[:] ) )
        
        # Find the max
        # Rq: Max location Can be refined if 
        # there is an intersection between the 2 pads
        if chi.shape[0] == 0:
            maxVal = -1
            print("  ???", ch, mask)
        else:
          maxVal = np.max( chi[:] * mask[:]  )
        print( "  maxVal cath 0, 1", maxVal, maxVal )
        maxIdx =np.argmax( chi[:]*mask[:] )
        
        # Refine mu
        # Compute mu (barycenter of the charge)
        
        print("???", sig0)
        imax  = maxIdx
        idx = np.where( np.bitwise_and( (np.abs( xi[imax] - xi[:] ) < 2.1 * dxi[imax]), (np.abs( yi[imax] - yi[:]) < 2.1 * dxi[imax] ) ) ) 
        u = chi[idx]*xi[idx]*mask[idx]
        mux = np.sum(u) / np.sum( chi[idx]*mask[idx] )
        u = chi[idx]*yi[idx]*mask[idx]
        muy = np.sum(u) / np.sum( chi[idx]*mask[idx] )
        

        
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
        print( "y, cst", ch[maxCath][maxIdx], EM.TwoPi * sig0[0] * sig0[1]
        """
        # Pad Adatative
        #
        # var = np.array([ 1.0 * dx[maxCath][maxIdx], 1.0 *dy[maxCath][maxIdx]])
        # var = var * var
        # or from fixed var
        var = var0
        #
        vara.append( var )
        #
        # w = ch[maxCath][maxIdx] * EM.TwoPi * dx[maxCath][maxIdx] * dy[maxCath][maxIdx]
        muTmp = np.array( [ mu ] ).T
        dxyTmp = np.array([ [dxi[imax], dyi[imax ]] ]).T

        xyMax = np.array( [ [ xi[imax], yi[imax] ] ] ).T 
        dxyMax = np.array( [ [dxi[imax], dyi[imax]] ] ).T
        integral = EM.computeDiscretizedGaussian2D( xyMax, dxyMax, mu, var )
        w = chi[imax] / integral
        print( "  ch. max, integral at max, w", chi[imax], integral, w )
        print( "  mu=", mu ) 
        wa.append( w[0] )
        mua.append( np.copy(mu) )

        xyi = [ xi, yi ]
        dxy = [ dxi, dyi ]
        print( "    max ch, w, gauss(max):", chi[imax], w, EM.computeDiscretizedGaussian2D( muTmp, dxyTmp, mu, var ) )
        xyMax = np.array( [ [ xi[imax], yi[imax] ] ] ).T 
        dxyMax = np.array( [ [dxi[imax], dyi[imax]] ] ).T
#       # print( " value at index max :", w*EM.computeGaussian2D(xyMax, mu, var)
        # print( " value at index max :", w*EM.computeDiscretizedGaussian2D(xyMax, dxyMax, mu, var) )
#       # print( "  gauss : ",  w*EM.computeGaussian2D(xy, mu, var)
        xyTmp = np.array( xyi )
        if ( xyTmp.size != 0 ):
            dxyTmp = np.array( dxy)
            print( "    gauss : ",  w*EM.computeDiscretizedGaussian2D( xyTmp, dxyTmp, mu, var) )

            # print( "    charges :", ch[cath][:] 
#           ch[cath][:] = ch[cath][:] - mask[cath][:]* w * EM.computeGaussian2D(xy, mu, var )
            chi[:] = chi[:] - mask[:]* w * EM.computeDiscretizedGaussian2D(xyTmp, dxyTmp, mu, var )
            # print( "  charges :", ch[cath][:] 
            # print( "sum mask", np.sum( ch[0][:]*mask[0][:] >= 5 ), np.sum( ch[1][:]*mask[1][:] >= 5 )
            #  ( "dx ", dx[cath]
            # print( "dy ", dy[ca th]
        else:
            print("No chage on this cathode")
        #
        wi = np.array(wa)
        s = np.sum( wi)
        wi = wi / s
        mui = np.array( mua )
        vari = np.array(vara)
        # wf, muf, varf = simpleProcessEMCluster( cl, wi, mui, vari, cstVar=False )
        
        mask[:] = ( chi[:]*mask[:] >= 5 )
        # print( "mask 0", mask[0]
        # print( "mask 1", mask[1]
        
        if ax is not None and( nIter==0 or nIter==1):
          ch = []
          ch.append( chi[c0Idx] )
          ch.append( chi[c1Idx] ) 
          plu.drawPads(ax[0+2*nIter],  x[0], y[0], dx[0], dy[0], np.where( ch[0] > 0, ch[0], 0),
                title="", alpha=1.0, noYTicksLabels=True, doLimits=False )   
          plu.drawPads(ax[1+2*nIter],  x[1], y[1], dx[1], dy[1], np.where( ch[1] > 0, ch[1], 0),
                title="", alpha= 1.0, noYTicksLabels=True, doLimits=False )
        nIter += 1
    print( "wi ", wi )
    print( "mui ", mui )
    print( "vari", vari )

    return np.array( wi ) , np.array( mui ), np.array( vari )

def findLocalMaxWithLaplacian( xi, yi, dxi, dyi, chi, var0, graphTest=False, verbose = False  ):
    w = []
    mu = []
    var = []
    x = xi
    dx = dxi
    y = yi
    dy = dyi
    ch =  np.copy( chi )
    #
    # Extract neighbours
    #
    neigh = plu.getFirstNeighbours( x, y, dx, dy)
    #
    lapl = np.zeros( (x.shape[0]) )
    other = np.zeros( (x.shape[0]) )
    # print("LocalMax ???", x)
    # print("LocalMax ???", neigh)
    for i in range(x.shape[0]):
        centralIdx = np.where( x[ neigh[i]]  == i )
        # print("neigh[i]", neigh[i] )
        nNeigh = neigh[i].size
        """
        # Criteriom 1
        inf =  ( ch[i] >= ch[ neigh[i] ] ) 
        sup =  ( ch[i] < ch[ neigh[i] ] ) 
        lapl[i] = np.sum( inf ) - np.sum( sup )
        other[i] = np.sum( sup )
        """
        # Criteriom 2
        inf =  ( ch[i] >= ch[ neigh[i] ] )
        if ( nNeigh > 1):
          # When only two pads lapl=0 not a good thing 
          if nNeigh == 2 :
            lapl[i] = 0 
          else:
            lapl[i] = np.sum( inf ) / (nNeigh)
        else:
          lapl[i] = 0 
          """
          print("??? i, neigh[i]", i, neigh[i])
          print("??? nNeigh", nNeigh)
          print("??? x", x)
          print("??? y", y)
          print("??? ch", ch)
          input("Warning: no neigbour")
          """
        # print("inf / sup ",  lapl[i], np.sum( inf ), np.sum( sup ), nNeigh )
        # if lapl[i] == 1 :
        if lapl[i] >= 0.99 and ch[i] > 5.0 :
            w.append( ch[i] )
            mu.append( [ x[i], y[i] ] )
            var.append( var0 )
    #
    if verbose :
      print("[findLocalMax...] lapl", lapl)
      print("[findLocalMax...] w", w)
      print("[findLocalMax...] mu ", mu )
      print("[findLocalMax...] var", var)
    # if ax is not None:
    if graphTest:
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 7) )

        zMin = np.min(lapl)
        zMax = np.max(lapl)
        if (zMin == zMax):
          zMax = zMin + 1 
        plu.setLUTScale( 0.0, np.max( ch ) )
        plu.drawPads(ax[0,0],  x, y, dx, dy, ch,
                title="", alpha=1.0, noYTicksLabels=False, doLimits=True )  
        plu.setLUTScale( zMin, zMax )
        plu.drawPads(ax[0,1],  x, y, dx, dy, lapl,
                title="", alpha=1.0, noYTicksLabels=False, doLimits=True )  
        plt.show()
    if len(w) == 0:
      lMax = np.max(lapl)
      idx = np.where( lapl == lMax)[0]
      for k in idx:
        # if ( ch[k] > 5.0 ):
          w.append( ch[k] )
          mu.append( [ x[k], y[k] ] )
          var.append( var0 )
      #
      print( "not clear max w", w)
      print( "not clear max mu", mu)
      if InputWarning:
        input("Not clear max")

    return np.array( w ), np.array( mu ), np.array( var ), lapl

def findLocalMaxWithLaplacianOldWithCath( xi, yi, dxi, dyi, cathi, chi, var0, graphTest=False, verbose = False  ):
    w = []
    mu = []
    var = []
    c0Idx = np.where( cathi == 0)
    x = xi[c0Idx]
    dx = dxi[c0Idx]
    y = yi[c0Idx]
    dy = dyi[c0Idx]
    ch =  np.copy( chi[c0Idx] )
    #
    # Extract neighbours
    #
    neigh = plu.getFirstNeighbours( x, y, dx, dy)
    #
    lapl = np.zeros( (x.shape[0]) )
    other = np.zeros( (x.shape[0]) )
    # print("LocalMax ???", x)
    # print("LocalMax ???", neigh)
    for i in range(x.shape[0]):
        centralIdx = np.where( x[ neigh[i]]  == i )
        # print("neigh[i]", neigh[i] )
        nNeigh = neigh[i].size
        """
        # Criteriom 1
        inf =  ( ch[i] >= ch[ neigh[i] ] ) 
        sup =  ( ch[i] < ch[ neigh[i] ] ) 
        lapl[i] = np.sum( inf ) - np.sum( sup )
        other[i] = np.sum( sup )
        """
        # Criteriom 2
        inf =  ( ch[i] >= ch[ neigh[i] ] )
        if ( nNeigh > 1):
          # When only two pads lapl=0 not a good thing 
          if nNeigh == 2 :
            lapl[i] = 0 
          else:
            lapl[i] = np.sum( inf ) / (nNeigh)
        else:
          lapl[i] = 0 
          """
          print("??? i, neigh[i]", i, neigh[i])
          print("??? nNeigh", nNeigh)
          print("??? x", x)
          print("??? y", y)
          print("??? ch", ch)
          input("Warning: no neigbour")
          """
        # print("inf / sup ",  lapl[i], np.sum( inf ), np.sum( sup ), nNeigh )
        # if lapl[i] == 1 :
        if lapl[i] >= 0.99 and ch[i] > 5.0 :
            w.append( ch[i] )
            mu.append( [ x[i], y[i] ] )
            var.append( var0 )
    #
    if verbose :
      print("[findLocalMax...] lapl", lapl)
      print("[findLocalMax...] w", w)
      print("[findLocalMax...] mu ", mu )
      print("[findLocalMax...] var", var)
    # if ax is not None:
    if graphTest:
        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 7) )

        zMin = np.min(lapl)
        zMax = np.max(lapl)
        if (zMin == zMax):
          zMax = zMin + 1 
        plu.setLUTScale( 0.0, np.max( ch ) )
        plu.drawPads(ax[0,0],  x, y, dx, dy, ch,
                title="", alpha=1.0, noYTicksLabels=False, doLimits=True )  
        plu.setLUTScale( zMin, zMax )
        plu.drawPads(ax[0,1],  x, y, dx, dy, lapl,
                title="", alpha=1.0, noYTicksLabels=False, doLimits=True )  
        plt.show()
    if len(w) == 0:
      lMax = np.max(lapl)
      idx = np.where( lapl == lMax)[0]
      for k in idx:
        # if ( ch[k] > 5.0 ):
          w.append( ch[k] )
          mu.append( [ x[k], y[k] ] )
          var.append( var0 )
      #
      print( "not clear max w", w)
      print( "not clear max mu", mu)
      if InputWarning:
        input("Not clear max")

    return np.array( w ), np.array( mu ), np.array( var ), lapl

def findLocalMaxWithSubstraction( xi, yi, dxi, dyi, cathi, chi, var0, mcObj, preClusters, ev, pc, DEIds,fitParameter=5.0, ax=None, maxIter=200, chId=0 ):

    graphTest = chId > 4
    # graphTest = True
    x = []; dx = []
    y = []; dy = []
    ch = []
    c0Idx = np.where( cathi == 0)
    c1Idx = np.where( cathi == 1)
    x.append(xi[c0Idx])
    x.append(xi[c1Idx])
    dx.append(dxi[c0Idx])
    dx.append(dxi[c1Idx])    
    y.append(yi[c0Idx])
    y.append(yi[c1Idx])
    dy.append(dyi[c0Idx])
    dy.append(dyi[c1Idx])
    ch.append( np.copy( chi[c0Idx] ))
    ch.append( np.copy( chi[c1Idx] ))  
    
    print("??? x", x[0])
    # if wFinal.size > 0 or wFinal.size == 0 : 
    frame = []
    if graphTest : 
      fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 7) )
      plu.setLUTScale( 0, np.max(ch[0]) )
      print("??? ch[0]", ch[0])
      title = "max %.1f" % np.max( ch[0])
      plu.drawPads(ax[0,0],  x[0], y[0], dx[0], dy[0], ch[0],
                title=title, alpha=1.0, noYTicksLabels=False, doLimits=True )  
      # ACtual Reco
      pMu = np.array( [ preClusters.rClusterX[ev][pc], preClusters.rClusterY[ev][pc] ] ).T
      pW = np.ones( preClusters.rClusterX[ev][pc].shape )
      pVar = np.array( [ np.ones( pW.shape ), np.ones( pW.shape ) ] ).T
      plu.drawModelComponents( ax[0, 0], pW, pMu, pVar, color='green', pattern="o" ) 
      frame = plu.getPadBox( x[0], y[0], dx[0], dy[0] )
      drawMCHitsInFrame ( ax[0,0], frame, mcObj, ev, DEIds )
      title = "Ev=" +str(ev)+ ", pc=" + str(pc) + ", DEId="+ str(DEIds)
      fig.suptitle( title )
      findLocalMaxWithLaplacian( xi, yi, dxi, dyi, cathi, chi, var0, ax=ax[1,:] )

    # sig0 = np.array( [0.35, 0.35] )
    """
    sig0 = np.array( [0.4, 0.4] )
    var0 = sig0 * sig0
    """
    mu = np.zeros(2)
    #
    # Extract neighbours
    #
    eps = 1.0e-7
    neigh = []
    for cath in range(len(x)):
      neighCath = []
      for i in range( x[cath].shape[0]):
        xMask = np.abs( x[cath][i] - x[cath] ) <= ( (1.0 + eps) * (dx[cath][i] + dx[cath]) )
        yMask = np.abs( y[cath][i] - y[cath] ) <= ( (1.0 + eps) * (dy[cath][i] + dy[cath]) )
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

        print("ITER ", nIter)
        print( "  Mask sum", np.sum( mask[0][:] ), np.sum( mask[1][:] ) )
        
        # Find the max
        # Rq: Max location Can be refined if 
        # there is an intersection between the 2 pads
        if ch[0].shape[0] == 0:
            maxVal[0] = -1
            print("  ???", ch, mask)
        else:
          maxVal[0] = np.max( ch[0][:] * mask[0][:]  )
        if ch[1].shape[0] == 0:
            maxVal[1] = -1
            print("  ???", ch, mask)
        else:
          maxVal[1] = np.max( ch[1][:] * mask[1][:]  )
        print( "  maxVal on cath 0 &1", maxVal[0], maxVal[1] )
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
        mu[0] = mux
        mu[1] = muy
        print( "x[maxCath][idx]   ",x[maxCath][idx])
        print( "y[maxCath][idx]   ",x[maxCath][idx])
        print( "ch[maxCath][idx]  ",ch[maxCath][idx])
        print( "mask[maxCath][idx]",mask[maxCath][idx])
        print( "  maxCath, mux, muy, xMax, yMax ", maxCath, mux, muy, x[maxCath][maxIdx], y[maxCath][maxIdx] )
        print( "  xmin, xmax, ymin, ymax ", np.min( x[maxCath] ), np.max( x[maxCath] ), \
                    np.min( y[maxCath] ), np.max( y[maxCath] ) )
        
        # mu[0] = x[maxCath][maxIdx]
        # mu[1] = y[maxCath][maxIdx]

        #
        # Remove the gaussian
        #
        """ 
        # sig = cst
        w = ch[maxCath][maxIdx] * EM.TwoPi * sig0[0] * sig0[1]
        print( "y, cst", ch[maxCath][maxIdx], EM.TwoPi * sig0[0] * sig0[1]
        """
        #
        # Choice of the var method
        #
        # Pad Adatative
        #
        # var = np.array([ 1.0 * dx[maxCath][maxIdx], 1.0 *dy[maxCath][maxIdx]])
        # var = var * var
        #
        # or from fixed var
        #
        var = var0
        #
        vara.append( var )
        #
        # w = ch[maxCath][maxIdx] * EM.TwoPi * dx[maxCath][maxIdx] * dy[maxCath][maxIdx]
        muTmp = np.array( [ mu ] ).T
        dxyTmp = np.array([ [dx[maxCath][maxIdx], dy[maxCath][maxIdx] ]]).T

        xyMax = np.array( [ [ x[maxCath][maxIdx], y[maxCath][maxIdx] ] ] ).T 
        dxyMax = np.array( [ [dx[maxCath][maxIdx], dy[maxCath][maxIdx]] ] ).T
        # Enlarge gaussian
        varX = 1.2 * 1.2 * var
        integral = EM.computeDiscretizedGaussian2D( xyMax, dxyMax, mu, varX )
        w = ch[maxCath][maxIdx] / integral
        print( "  ch. max, integral at max, w", ch[maxCath][maxIdx], integral, w )
        print( "  mu=", mu ) 
        wa.append( w[0] )
        mua.append( np.copy(mu) )

        for cath in range(2):
            print( "    [Sub] cath=", cath )
            xy = [ x[cath], y[cath] ]
            dxy = [ dx[cath], dy[cath] ]
            print( "    [Sub] maxIdx", maxIdx )
            # print( "  max ch, w, gauss(max):", ch[maxCath][maxIdx], w,  EM.computeGaussian2D( mu, mu, var )
            print( "    [Sub] max ch, w, gauss(max):", ch[maxCath][maxIdx], w, EM.computeDiscretizedGaussian2D( muTmp, dxyTmp, mu, varX ) )
            # xyMax = np.array( [ [ x[maxCath][maxIdx], y[maxCath][maxIdx] ] ] ).T 
            # dxyMax = np.array( [ [dx[maxCath][maxIdx], dy[maxCath][maxIdx]] ] ).T
#           # print( " value at index max :", w*EM.computeGaussian2D(xyMax, mu, var)
            # print( " value at index max :", w*EM.computeDiscretizedGaussian2D(xyMax, dxyMax, mu, var) )
#            print( "  gauss : ",  w*EM.computeGaussian2D(xy, mu, var)
            xyTmp = np.array( xy )
            if ( xyTmp.size != 0 ):
              dxyTmp = np.array( dxy)
              print( "    [Sub] gauss : ",  w*EM.computeDiscretizedGaussian2D( xyTmp, dxyTmp, mu, varX) ) 

              # print( "    charges :", ch[cath][:] 
#             ch[cath][:] = ch[cath][:] - mask[cath][:]* w * EM.computeGaussian2D(xy, mu, var )
              # Enlarge gaussian
              varX = 1.2 * 1.2 * var
              ch[cath][:] = ch[cath][:] - mask[cath][:]* w * EM.computeDiscretizedGaussian2D(xyTmp, dxyTmp, mu, varX )
              # print( "  charges :", ch[cath][:] 
              # print( "sum mask", np.sum( ch[0][:]*mask[0][:] >= 5 ), np.sum( ch[1][:]*mask[1][:] >= 5 )
              #  ( "dx ", dx[cath]
              # print( "dy ", dy[ca th]
            else:
              print("    [Sub] No charge on this cathode")
            #
        wi = np.array(wa)
        s = np.sum( wi)
        wi = wi / s
        mui = np.array( mua )
        vari = np.array(vara)
        # wf, muf, varf = simpleProcessEMCluster( cl, wi, mui, vari, cstVar=False )
        
        mask[0][:] = ( ch[0][:]*mask[0][:] >= fitParameter )
        mask[1][:] = ( ch[1][:]*mask[1][:] >= fitParameter )
        # print( "mask 0", mask[0]
        # print( "mask 1", mask[1]
        maxCh = max( np.max(ch[0]), np.max(ch[1]) )
        print("Min/Max ch0", np.min(ch[0]), np.max(ch[0]))
        print("Min/Max ch1", np.min(ch[1]), np.max(ch[1]))
        print("surface", 4*np.dot(dx[0], dy[0]) ) 
        print("ratio surface/# max", 4*np.dot(dx[0], dy[0]) / wi.size)
        print("remaning surface", 4*np.dot(dx[1], dy[1]*mask[1][:]) ) 
        print("sum positive", np.sum( np.where(ch[0] > 0, ch[0], 0.0 )) )
        #
        if ax is not None:
          if nIter < 6:
              
              ix = (nIter+1) // 4
              jx = (nIter+1) % 4
              noLabel= False
          else:
              jx=2; ix = 1
              noLabel= True
          zz = np.where( ch[0] >= 0.0, ch[0], 0.0 )
          # print("zz", zz)
          # plu.setLUTScale( min( np.min(ch[0]), np.min(ch[1]) ), max( np.max(ch[0]), np.max(ch[1]) ) )
          if( np.max(zz) == 0.0):
            plu.setLUTScale( 0, 1.0 )
          else:
            plu.setLUTScale( 0, np.max(zz) )
          title = "max %.1f" % np.max( ch[0])
          plu.drawPads(ax[ix,jx],  x[0], y[0], dx[0], dy[0], zz,
                title=title, alpha=1.0, noYTicksLabels=True, doLimits=False )  
          plu.drawModelComponents( ax[ix,jx], wi, mui, vari, color='red', pattern="cross" )
          drawMCHitsInFrame ( ax[ix,jx], frame, mcObj, ev, DEIds )
          # plu.drawPads(ax[ix+1],  x[1], y[1], dx[1], dy[1], ch[1],
          #      title="", alpha= 1.0, noYTicksLabels=True, doLimits=False )
        nIter += 1
    # Tag geometric clusters
    # One cathode
    """
      """
    """
    if ax is not None:
        plu.setLUTScale( 0, np.max( grp ) )
        plu.drawPads(ax[3],  x[0], y[0], dx[0], dy[0], grp,
                title="", alpha=1.0, noYTicksLabels=True, doLimits=False )  
    """
    print( "wi ", wi )
    print( "mui ", mui )
    print( "vari", vari )
    if ax is not None:
      plt.show()
    return np.array( wi ) , np.array( mui ), np.array( vari )

def findLocalMaxWithSubstractionV0( xi, yi, dxi, dyi, cathi, chi, var0, fitParameter=5.0, ax=None, maxIter=200 ):
    x = []; dx = []
    y = []; dy = []
    ch = []
    c0Idx = np.where( cathi == 0)
    c1Idx = np.where( cathi == 1)
    x.append(xi[c0Idx])
    x.append(xi[c1Idx])
    dx.append(dxi[c0Idx])
    dx.append(dxi[c1Idx])    
    y.append(yi[c0Idx])
    y.append(yi[c1Idx])
    dy.append(dyi[c0Idx])
    dy.append(dyi[c1Idx])
    ch.append( np.copy( chi[c0Idx] ))
    ch.append( np.copy( chi[c1Idx] ))  

    # sig0 = np.array( [0.35, 0.35] )
    """
    sig0 = np.array( [0.4, 0.4] )
    var0 = sig0 * sig0
    """
    mu = np.zeros(2)
    # Extract neighbours
    eps = 1.0e-10
    neigh = []
    for cath in range(len(x)):
      neighCath = []
      for i in range( x[cath].shape[0]):
        xMask = np.abs( x[cath][i] - x[cath] ) <= ( (1.0 + eps) * (dx[cath][i] + dx[cath]) )
        yMask = np.abs( y[cath][i] - y[cath] ) <= ( (1.0 + eps) * (dy[cath][i] + dy[cath]) )
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

        print("ITER ", nIter)
        print( "  Mask sum", np.sum( mask[0][:] ), np.sum( mask[1][:] ) )
        
        # Find the max
        # Rq: Max location Can be refined if 
        # there is an intersection between the 2 pads
        if ch[0].shape[0] == 0:
            maxVal[0] = -1
            print("  ???", ch, mask)
        else:
          maxVal[0] = np.max( ch[0][:] * mask[0][:]  )
        if ch[1].shape[0] == 0:
            maxVal[1] = -1
            print("  ???", ch, mask)
        else:
          maxVal[1] = np.max( ch[1][:] * mask[1][:]  )
        print( "  maxVal cath 0, 1", maxVal[0], maxVal[1] )
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
        
        print( "  cath, mu, x, y ", maxCath, mux, muy, x[maxCath][maxIdx], y[maxCath][maxIdx] )
        print( "  xmin, xmax, ymin, ymax ", np.min( x[maxCath] ), np.max( x[maxCath] ), \
                    np.min( y[maxCath] ), np.max( y[maxCath] ) )
        
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
        print( "y, cst", ch[maxCath][maxIdx], EM.TwoPi * sig0[0] * sig0[1]
        """
        # Pad Adatative
        #
        # var = np.array([ 1.0 * dx[maxCath][maxIdx], 1.0 *dy[maxCath][maxIdx]])
        # var = var * var
        # or from fixed var
        var = var0
        #
        vara.append( var )
        #
        # w = ch[maxCath][maxIdx] * EM.TwoPi * dx[maxCath][maxIdx] * dy[maxCath][maxIdx]
        muTmp = np.array( [ mu ] ).T
        dxyTmp = np.array([ [dx[maxCath][maxIdx], dy[maxCath][maxIdx] ]]).T

        xyMax = np.array( [ [ x[maxCath][maxIdx], y[maxCath][maxIdx] ] ] ).T 
        dxyMax = np.array( [ [dx[maxCath][maxIdx], dy[maxCath][maxIdx]] ] ).T
        integral = EM.computeDiscretizedGaussian2D( xyMax, dxyMax, mu, var )
        w = ch[maxCath][maxIdx] / integral
        print( "  ch. max, integral at max, w", ch[maxCath][maxIdx], integral, w )
        print( "  mu=", mu ) 
        wa.append( w[0] )
        mua.append( np.copy(mu) )

        for cath in range(2):
            print( "  cath=", cath )
            xy = [ x[cath], y[cath] ]
            dxy = [ dx[cath], dy[cath] ]
            print( "    maxIdx", maxIdx )
            # print( "  max ch, w, gauss(max):", ch[maxCath][maxIdx], w,  EM.computeGaussian2D( mu, mu, var )
            print( "    max ch, w, gauss(max):", ch[maxCath][maxIdx], w, EM.computeDiscretizedGaussian2D( muTmp, dxyTmp, mu, var ) )
            xyMax = np.array( [ [ x[maxCath][maxIdx], y[maxCath][maxIdx] ] ] ).T 
            dxyMax = np.array( [ [dx[maxCath][maxIdx], dy[maxCath][maxIdx]] ] ).T
#           # print( " value at index max :", w*EM.computeGaussian2D(xyMax, mu, var)
            # print( " value at index max :", w*EM.computeDiscretizedGaussian2D(xyMax, dxyMax, mu, var) )
#            print( "  gauss : ",  w*EM.computeGaussian2D(xy, mu, var)
            xyTmp = np.array( xy )
            if ( xyTmp.size != 0 ):
              dxyTmp = np.array( dxy)
              print( "    gauss : ",  w*EM.computeDiscretizedGaussian2D( xyTmp, dxyTmp, mu, var) )

              # print( "    charges :", ch[cath][:] 
#             ch[cath][:] = ch[cath][:] - mask[cath][:]* w * EM.computeGaussian2D(xy, mu, var )
              varX = 1.2 * 1.2 * var
              ch[cath][:] = ch[cath][:] - mask[cath][:]* w * EM.computeDiscretizedGaussian2D(xyTmp, dxyTmp, mu, varX )
              # print( "  charges :", ch[cath][:] 
              # print( "sum mask", np.sum( ch[0][:]*mask[0][:] >= 5 ), np.sum( ch[1][:]*mask[1][:] >= 5 )
              #  ( "dx ", dx[cath]
              # print( "dy ", dy[ca th]
            else:
              print("No chage on this cathode")
            #
        wi = np.array(wa)
        s = np.sum( wi)
        wi = wi / s
        mui = np.array( mua )
        vari = np.array(vara)
        # wf, muf, varf = simpleProcessEMCluster( cl, wi, mui, vari, cstVar=False )
        
        mask[0][:] = ( ch[0][:]*mask[0][:] >= fitParameter )
        mask[1][:] = ( ch[1][:]*mask[1][:] >= fitParameter )
        # print( "mask 0", mask[0]
        # print( "mask 1", mask[1]
        
        if ax is not None and( nIter==0 or nIter==1):
          plu.drawPads(ax[0+2*nIter],  x[0], y[0], dx[0], dy[0], np.where( ch[0] > 0, ch[0], 0),
                title="", alpha=1.0, noYTicksLabels=True, doLimits=False )   
          plu.drawPads(ax[1+2*nIter],  x[1], y[1], dx[1], dy[1], np.where( ch[1] > 0, ch[1], 0),
                title="", alpha= 1.0, noYTicksLabels=True, doLimits=False )
        nIter += 1
    print( "wi ", wi )
    print( "mui ", mui )
    print( "vari", vari )

    return np.array( wi ) , np.array( mui ), np.array( vari )

def printC( clusters, pClusters, pixels ):
    print( "shape cluster, pre-cluster, pixels", len(clusters.x), len(pClusters.x) )
    if ( len(clusters.x)  !=  len(pClusters.x) ):
        print( "displayClusters: Clusters and pre-Clusters have diff. size")
        # exit()
    # Assign event ID to clusters 
    n = len( clusters.id["DetectElemID"] )
    print( n )
    evID = 0 
    old = clusters.id["DetectElemID"][0]/100
    clusters.id["Event"][0] = evID
    for i in range(1, n):
     new = clusters.id["DetectElemID"][i] /100
     if (new < old):
         print( "i found", i )
         evID += 1
     clusters.id["Event"][i] = evID
     old = new
    print( "nev=", evID )
    nEv = evID+1
    
    print( "Cl Events:", clusters.id["Event"][0:22] )
    print( "Cl ServerID:", clusters.id["cServerID"][0:22] )
    print( "Cl DE Id:", clusters.id["DetectElemID"][0:22] )
    print( "Cl Ch Id:", clusters.id["ChamberID"][0:22] )
    x = clusters.cFeatures["X"]
    dx = clusters.cFeatures["ErrorX"]
    y = clusters.cFeatures["Y"]
    dy = clusters.cFeatures["ErrorY"]
    print( "preCl Events:", pClusters.id["Event"][0:22] )
    print( "preCl ServerID:", pClusters.id["cServerID"][0:22] )
    print( "preCl preClusterID :", pClusters.id["preClusterID"][0:22] )
    print( "preCl DE Id:", pClusters.id["DetectElemID"][0:22] )
    print( "preCl Ch Id:", pClusters.id["ChamberID"][0:22] )
    """
    for i in range(n):
      if (pClusters.id["Event"][i] != clusters.id["Event"][i]):
        print( "problem i=", i, pClusters.id["Event"][i], clusters.id["Event"][i]
        print( "Cl DE Id:", clusters.id["DetectElemID"][i-1:i+20]
        print( "preCl DE Id:", pClusters.id["DetectElemID"][i-1:i+20]
    """
    pcEv = np.array(pClusters.id["Event"])
    csEv = np.array(clusters.id["Event"])
    pcDEId = np.array(pClusters.id["DetectElemID"])
    csDEId = np.array(clusters.id["DetectElemID"])
    
    for ev in range(nEv):
        pcidx = np.where( pcEv == ev)
        csidx = np.where( csEv == ev)
        if np.all( pcDEId[pcidx] != csDEId[csidx] ):
            print( "ev=", ev, ", pCluster DEId=", pcDEId[pcidx] )
            print( "ev=", ev, ", ServCluster DEId=", csDEId[csidx] )
            pc = 0; cs = 0
            npc = pcidx[0].shape[0]
            ncs = csidx[0].shape[0]
            done = ((cs == ncs) and (pc == npc))

            while ( not done ):
                if ( pcDEId[pcidx[0][pc]] < csDEId[csidx[0][cs]] ):
                    print( "pre-cl < cluster serv.:", pcDEId[pcidx[0][pc]], csDEId[csidx[0][cs]] )
                    pc += 1
                elif ( pcDEId[pcidx[0][pc]] > csDEId[csidx[0][cs]] ):
                    print( "pre-cl >cluster serv.:", pcDEId[pcidx[0][pc]], csDEId[csidx[0][cs]] )            
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
    print( n )
    evID = 0 
    old = clusters.id["DetectElemID"][0]/100
    clusters.id["Event"][0] = evID
    for i in range(1, n):
     new = clusters.id["DetectElemID"][i] /100
     if (new < old):
         print( "i found", i )
         evID += 1
     clusters.id["Event"][i] = evID
     old = new
    print( "nev=", evID )
    nEv = evID+1
    
    print( "Cl Events:", len( clusters.id["Event"]), clusters.id["Event"][0:22] )
    print( "Tr Events:", len( tracks.id["Event"]), tracks.id["Event"][0:22] )
    clEv = np.array( clusters.id["Event"] )
    trEv = np.array( tracks.id["Event"] )
    nEv = max( np.max(clEv), np.max(trEv))
    
    for ev in range(nEv):
        clIdx = np.where( clEv == ev )
        trIdx = np.where( trEv == ev )
        print( "Cl ev=", ev, "[", )
        for i in clIdx[0]:
            print( clusters.id["DetectElemID"][i], )
            # print( "ev=", ev, clusters.id["DetectElemID"][i for i in clIdx[0]]
        print( "]" )
        #print( "ev=", ev, tracks.id["DetectElemID"][trIdx]
        print( "Tr ev=", ev, "[" )
        for i in trIdx[0]:
            print( tracks.id["DetectElemID"][i] )
            # print( "ev=", ev, clusters.id["DetectElemID"][i for i in clIdx[0]]
        print( "]")
        
        for icl in clIdx[0]:
            DEId = clusters.id["DetectElemID"][icl]
            print( "trIdx", trIdx )
            for itr in trIdx[0]:
                print( "itr", itr, tracks.id["DetectElemID"][itr] )
                npDEId = np.array( tracks.id["DetectElemID"][itr])
                sameDEId = np.where( npDEId == DEId )
                print( "ev=", ev, "cluster DEid=", DEId, "tr same DEId", sameDEId )
                for id in sameDEId[0].tolist():
                    print( "id=", id )
                    print( "Cluster X,Y from clsrv", clusters.cFeatures["X"][icl], clusters.cFeatures["Y"][icl] )
                    print( "Cluster X,Y from tr", tracks.cFeatures["X"][id], tracks.cFeatures["Y"][id] )
                    
            # print( "ev=", ev, clusters.id["DetectElemID"][i for i in clIdx[0]]
        print( "]")
    
    """
    print( "Cl ServerID:", clusters.id["cServerID"][0:22]
    print( "Cl DE Id:", clusters.id["DetectElemID"][0:22]
    print( "Cl Ch Id:", clusters.id["ChamberID"][0:22]
    x = clusters.cFeatures["X"]
    dx = clusters.cFeatures["ErrorX"]
    y = clusters.cFeatures["Y"]
    dy = clusters.cFeatures["ErrorY"]
    print( "preCl Events:", pClusters.id["Event"][0:22]
    print( "preCl ServerID:", pClusters.id["cServerID"][0:22]
    print( "preCl preClusterID :", pClusters.id["preClusterID"][0:22]
    print( "preCl DE Id:", pClusters.id["DetectElemID"][0:22]
    print( "preCl Ch Id:", pClusters.id["ChamberID"][0:22]
    """
    """
    for i in range(n):
      if (pClusters.id["Event"][i] != clusters.id["Event"][i]):
        print( "problem i=", i, pClusters.id["Event"][i], clusters.id["Event"][i]
        print( "Cl DE Id:", clusters.id["DetectElemID"][i-1:i+20]
        print( "preCl DE Id:", pClusters.id["DetectElemID"][i-1:i+20]
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
            print( "ev=", ev, ", pCluster DEId=", pcDEId[pcidx]
            print( "ev=", ev, ", ServCluster DEId=", csDEId[csidx]
            pc = 0; cs = 0
            npc = pcidx[0].shape[0]
            ncs = csidx[0].shape[0]
            done = ((cs == ncs) and (pc == npc))

            while ( not done ):
                if ( pcDEId[pcidx[0][pc]] < csDEId[csidx[0][cs]] ):
                    print( "pre-cl < cluster serv.:", pcDEId[pcidx[0][pc]], csDEId[csidx[0][cs]]
                    pc += 1
                elif ( pcDEId[pcidx[0][pc]] > csDEId[csidx[0][cs]] ):
                    print( "pre-cl >cluster serv.:", pcDEId[pcidx[0][pc]], csDEId[csidx[0][cs]]                    
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
          print( "s, ds", s, ds)
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
        
def getTracks( mcObj, ev, DEId ):
    DEIdIdx = self.trackDEId[ev]
    return
""" ??? Inv
def addContributingMCTracks( mcObj, ev ):
  nbrOfTracks = len( mcObj.trackDEId[ev] )  
  for tidx in range( nbrOfTracks):
    DEIds = np.unique( mcObj.trackDEId[ev][tidx] )
    for deid in DEIds:
      idx = np.where( mcObj.trackDEId[ev][tidx] == deid)
      trackIds = mcObj.trackId[ev][tidx][idx]
      DEIdx = np.where( mcObj.padDEId[ev] == deid )
      pads = mcObj.padDEId[ev][DEIdx]
      for tid in trackIds:
          
  contrib = []
  contribCount = []
  for chId in md.ChId:
    trackList = []
    idx = (mcObj.padChId[ev] == chId )
    nb = mcObj.padNbrTracks[ev][idx]
    start = mcObj.padTrackIdx[ev][idx]
    # u = mcObj.padTrackId[ev][start]
    # print ("u", u)
    for s,k in zip(start, nb) :
        trackList.append( mcObj.padTrackId[ev][s:s+k] )
    # print("trackList", trackList)
    c = np.unique( np.concatenate( trackList ).reshape(-1) ) 
    contrib.append(c)
    contribCount.append( c.shape[0] )
    # chId loop
  print ("contrib",contribCount)
  return contrib, contribCount
"""

def getContributingMCTracks( mcObj, ev ):
  contrib = []
  contribCount = []
  for chId in md.ChId:
    trackList = []
    idx = (mcObj.padChId[ev] == chId )
    nb = mcObj.padNbrTracks[ev][idx]
    start = mcObj.padTrackIdx[ev][idx]
    # u = mcObj.padTrackId[ev][start]
    # print ("u", u)
    for s,k in zip(start, nb) :
        trackList.append( mcObj.padTrackId[ev][s:s+k] )
    # print("trackList", trackList)
    c = np.unique( np.concatenate( trackList ).reshape(-1) ) 
    contrib.append(c)
    contribCount.append( c.shape[0] )
    # chId loop
  print ("contrib",contribCount)
  return contrib, contribCount

def getTotalMCTrackChargeInChId(mcObj, ev, trackId, chId, x0, y0 ):
    """
    """
    
    #
    # Get the tracks with the same ID=trackId in pads 
    # with the same DEId
    #
    chargeOfTrackIdInChId = 0
    padList = []
    # Pad indexes of the same chId
    chIdIdx = np.where( mcObj.padChId[ev] == chId )[0]
    startIdx = mcObj.padTrackIdx[ev][chIdIdx]
    count = mcObj.padNbrTracks[ev][chIdIdx]
    print("??? count", np.sum( count) )
    # mask0 state if a pad has a trackId charge contibution
    # mask0 scope on the pads list in the same ChId  
    mask0 = (mcObj.padTrackId[ev][startIdx] == trackId)
    print( "??? mask0", np.sum(mask0) )
    chargeOfTrackIdInChId = np.sum( mcObj.padTrackCharges[ev][startIdx]*mask0 )
    # print("cObj.padTrackId[ev][startIdx]", np.array( mcObj.padTrackId[ev][startIdx]))
    # print("mask0", mask0.dtype, mask0)
    count = count-1
    startIdx += 1
    while ( np.sum(count > 0) > 0):
      # Indexes with have another track contrib
      maskIdx = np.where( count > 0)
      iii = startIdx[ (count > 0) ]
      # print("count (> 0)", count )
      # print ("iii", iii)
      mask = (mcObj.padTrackId[ev][iii] == trackId)
      chargeOfTrackIdInChId += np.sum( mcObj.padTrackCharges[ev][iii]*mask )
      if (mask.shape[0] != maskIdx[0].shape[0]):
          print( "mask.shape[0] != maskIdx.shape[0]")
          exit()
      mask0[maskIdx] = np.bitwise_or( mask0[maskIdx], mask )
      # print("mask0", mask0)
      # print("mask", mask)
      count = count-1 
      startIdx += 1
    #
    # Verify is the pad set contributing to the trackId charge 
    # in the chamber is not to large ... to avoid 2 different 
    # clusters i.e. a track with 2 clusters in the same chamber  
    #
    # Extract pads which have a trackId conribution
    padIdx = chIdIdx [ np.where( mask0)[0]]
    if (padIdx.shape[0] == 0) and (padIdx.shape[0] > 20):
      print("Cluster too large", padIdx.shape[0], "pads")
      input("WARNING")
    #
    nbrOfPads = padIdx.shape[0]
    return chargeOfTrackIdInChId, nbrOfPads

def hasTrackAChargeDeposit(mcObj, ev, trackId, d, x0, y0 ):
    """
    Return boolean meaning if "MChit" has deposited a charge  in pads
    """
    
    #
    # Get the tracks with the same ID=trackId in pads 
    # with the same DEId
    #
    padList = []
    # Pad indexes of the same DEId
    sDEidx = np.where( mcObj.padDEId[ev] == d )[0]
    startIdx = mcObj.padTrackIdx[ev][sDEidx]
    count = mcObj.padNbrTracks[ev][sDEidx]
    mask0 = (mcObj.padTrackId[ev][startIdx] == trackId)
    # print("cObj.padTrackId[ev][startIdx]", np.array( mcObj.padTrackId[ev][startIdx]))
    # print("mask0", mask0.dtype, mask0)
    count = count-1
    startIdx += 1
    while ( np.sum(count > 0) > 0):
      # Indexes with have another track contrib
      maskIdx = np.where( count > 0)
      iii = startIdx[ (count > 0) ]
      # print("count (> 0)", count )
      # print ("iii", iii)
      mask = (mcObj.padTrackId[ev][iii] == trackId)
      if (mask.shape[0] != maskIdx[0].shape[0]):
          print( "mask.shape[0] != maskIdx.shape[0]")
          exit()
      mask0[maskIdx] = np.bitwise_or( mask0[maskIdx], mask )
      # print("mask0", mask0)
      # print("mask", mask)
      count = count-1 
      startIdx += 1
    #
    #  Find if x0, y0 a MCHit is in the pad list with a trackId contrib
    # print( "np.where( mask0)", np.where( mask0) )
    padIdx = sDEidx [ np.where( mask0)[0]]
    # print("padIdx", padIdx)
    x = mcObj.padX[ev][padIdx]
    y = mcObj.padY[ev][padIdx]
    dx = mcObj.padDX[ev][padIdx]
    dy = mcObj.padDY[ev][padIdx]
    chargeContribution =np.zeros( x0.shape, dtype=np.int )
    for k in range( x0.shape[0] ):
      matchx = np.bitwise_and( x0[k] >= (x[:] - dx[:]), x0[k] < (x[:] + dx) )
      matchy = np.bitwise_and( y0[k] >= (y[:] - dy[:]), y0[k] < (y[:] + dy) )
      match = np.bitwise_and( matchx, matchy )
      # print( "nbr of matched pads", np.sum(match) )
      chargeContribution[k] = np.any( match )

    print("chargeContribution", chargeContribution)
    if (chargeContribution.shape[0] != 1):
      print("chargeContribution.shape[0] != 1")
      exit()
    return chargeContribution

def getRecoPositions( recoObj, ev):
  recoHitsFromChId = []
  for i in range(11):  
    recoHitsFromChId.append( {"x":[], "y":[], "DEId":[], "ClusterId":[] } ) 
  for chid in range(1,11):
    for preCl in range( len(recoObj.rClusterId[ev]) ):
      idx = np.where ( recoObj.rClusterChId[ev][preCl] == chid)
      if idx[0].shape[0] != 0: 
        recoHitsFromChId[chid]["x"].append( recoObj.rClusterX[ev][preCl][idx] )
        recoHitsFromChId[chid]["y"].append( recoObj.rClusterY[ev][preCl][idx] )
        recoHitsFromChId[chid]["DEId"].append( recoObj.rClusterDEId[ev][preCl][idx] )
        recoHitsFromChId[chid]["ClusterId"].append( recoObj.rClusterId[ev][preCl][idx] )
  #
  # Linerize
  print("before", recoHitsFromChId[3])
  for chid in range(1,11):
    recoHitsFromChId[chid]["x"] = np.concatenate( recoHitsFromChId[chid]["x"] ).ravel()
    recoHitsFromChId[chid]["y"] = np.concatenate( recoHitsFromChId[chid]["y"] ).ravel()
    recoHitsFromChId[chid]["DEId"] = np.concatenate( recoHitsFromChId[chid]["DEId"] ).ravel()
    recoHitsFromChId[chid]["ClusterId"] = np.concatenate( recoHitsFromChId[chid]["ClusterId"] ).ravel()    
  #
  # ??? print("after", recoHitsFromChId[3])
  # a = wait("pause")
  return recoHitsFromChId

def getMCPositions( mcObj, ev ):
  """
  Return a list per ev of GT MCHits : [chId] { x[], y[], DEId[], trackId, trackIdx] 
  """
  mcHitsFromChId = []
  for i in range(11):  
    mcHitsFromChId.append( {"x":[], "y":[], "DEId":[], "trackId":[], "trackIdx":[] } )
  print("trackId", mcObj.trackId[ev] )
  for tidx, tid in enumerate( mcObj.trackId[ev]):    
    deids = mcObj.trackDEId[ev][tidx]
    DEIds = np.unique(deids)
    print("trackDEId", tid, DEIds )
    for d in DEIds:
      deIdx = np.where( mcObj.trackDEId[ev][tidx] == d)
      x = mcObj.trackX[ev][tidx][deIdx]
      y = mcObj.trackY[ev][tidx][deIdx]      
      flag = hasTrackAChargeDeposit(mcObj, ev, tid, d, x, y )
      idx = np.where( flag )
      chId = d // 100
      if ( np.sum(flag) > 0):
        mcHitsFromChId[chId]["x"].append(x[idx])
        mcHitsFromChId[chId]["y"].append(y[idx])
        mcHitsFromChId[chId]["DEId"].append(d)
        mcHitsFromChId[chId]["trackId"].append(tid)
        mcHitsFromChId[chId]["trackIdx"].append(tidx)
    
      # print("mcHitsFromChId ", mcHitsFronChId)  
      print("getMCPositions, tid=", tid, ", DEid=", d, "GT/NbrMCHit=" , np.sum(flag), "/", x.shape[0])
  
  for chId in range(1, 11):
    mcHitsFromChId[chId]["x"] = np.concatenate( mcHitsFromChId[chId]["x"] ).ravel() 
    mcHitsFromChId[chId]["y"] = np.concatenate( mcHitsFromChId[chId]["y"] ).ravel() 
    mcHitsFromChId[chId]["DEId"] = np.array( mcHitsFromChId[chId]["DEId"] ) 
    mcHitsFromChId[chId]["trackId"] = np.array( mcHitsFromChId[chId]["trackId"] ) 
    mcHitsFromChId[chId]["trackIdx"] = np.array( mcHitsFromChId[chId]["trackIdx"] )
    
  # print( "???", mcHitsFromChId[3] )
  # wait = input("Press Enter to continue.")
  return mcHitsFromChId

def getMCPositionsV2( mcObj, ev, contribTrackId ):
    print("trackId", mcObj.trackId[ev] )
    print("particleId", mcObj.trackParticleId[ev] )
    for c in range( len(contribTrackId) ):
      print("Contrib", c, contribTrackId[c])
      for tid in contribTrackId[c]:
        idx = np.where( np.array(mcObj.trackId[ev]) == tid )
        trIdx = idx[0][0]
        if len(idx[0]) != 1 :
          print("Several trackId in mcObj.trackId[ev] list ev, chIdx, tid", ev, c, tid)
          exit()
        print("ch, tid, id", c, tid, mcObj.trackId[ev][trIdx] )
        print("ch, tid, chId", c, tid, mcObj.trackChId[ev][trIdx] )
        chIdx = np.where( mcObj.trackChId[ev][trIdx] == (c+1) )
        if len(chIdx[0]) != 1 :
          print("----------WARNING--------------")
          print("Several hits of the same track in the same chamber (ev, chIdx, tid, chIdx)", ev, c, tid, chIdx)
          print("trackId", mcObj.trackId[ev])
          print("Contrib", c, contribTrackId[c]) 
          print("hits trackId, partId", mcObj.trackId[ev][trIdx], mcObj.trackParticleId[ev][trIdx] )
          print("hits chId ", mcObj.trackChId[ev][trIdx][ chIdx ] )
          print("hits DEId ", mcObj.trackDEId[ev][trIdx][ chIdx ] )
          print("hits X ", mcObj.trackX[ev][trIdx][ chIdx ] )
          print("hits Y ", mcObj.trackY[ev][trIdx][ chIdx ] )
          wait = input("Press Enter to continue.")
                    
    return

def getMCHits( mcObjByChId, preCluster, ev, pc, xInf, xSup, yInf, ySup  ):
    chId = preCluster.rClusterChId[ev][pc][0]
    xPad = preCluster.padX[ev][pc]
    dxPad = preCluster.padDX[ev][pc]
    yPad = preCluster.padY[ev][pc]
    dyPad = preCluster.padDY[ev][pc]
    x = mcObjByChId[ev][chId]['x'].copy()
    y = mcObjByChId[ev][chId]['y'].copy()
    tid = mcObjByChId[ev][chId]['trackId'].copy()
    xMin = xPad - dxPad
    xMax = xPad + dxPad
    yMin = yPad - dyPad
    yMax = yPad + dyPad
    flags =np.zeros( x.shape, dtype=int)
    for k in range( x.shape[0] ):
      flags[k] = np.any( (x[k] > xMin) * (x[k] <= xMax ) * (y[k] > yMin) * (y[k] <= yMax) )
    print("???", flags)
    idx = np.where( flags )
    return x[idx], y[idx],tid[idx] 

def displayMCTracks( mcObj, ev ):
    nChId = len(md.ChId)
    nFigCol = 4
    nFigRow = 3
    nFig = nFigCol * nFigRow
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
    print( "ax shape", ax.shape)
    fig.suptitle('MC Pads ev=' + str(ev) + " with " + str( len(mcObj.trackId[ev]) ) + " tracks", fontsize=16)
    chMin = 50
    chMax = -50
    for chId in md.ChId:
      x = []; y = []
      for t in range( len(mcObj.trackX[ev]) ):
        idx = (mcObj.trackChId[ev][t] == (chId) )
        x.append( mcObj.trackX[ev][t][idx] )
        y.append( mcObj.trackY[ev][t][idx] )
        chMin = min( chMin, np.min( mcObj.trackChId[ev][t] ) )
        chMax = max( chMax, np.max( mcObj.trackChId[ev][t] ) )

      # print(x)
      print( "chMin/max", chMin, ' ', chMax)
      xx = np.concatenate( x ).ravel()
      yy = np.concatenate( y ).ravel()
      if (xx.shape[0] != 0):
        #
        xSup = np.max( xx  )
        ySup = np.max( yy )
        xInf = np.min( xx )
        yInf = np.min( yy )
        #
        #plu.setLUTScale( 0, maxCh)
        #
        kx = (chId-1) // nFigCol
        ky = (chId-1) % nFigCol
        print( "graph pos", kx, ky)
        print(xx)
        print(yy)
        ax[kx, ky].plot( xx, yy, "o", color='blue', markersize=3 )
        ax[kx,ky].set_xlim( xInf, xSup )
        ax[kx,ky].set_ylim( yInf, ySup )
    plt.show()  
    
def displayPreClusters( preClusters, mcObj, ev, mcObjByChId, recoByChId ):
    # nChId = len(md.ChId)
    nFigCol = 4
    nFigRow = 3
    nFig = nFigCol * nFigRow

    minmaxCharges = []
    nbrOfPreClusters = len( preClusters.padId[ev] )
    print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
    pc = 0

    while pc < nbrOfPreClusters :
      iFig = 0
      fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
      fig.suptitle('PreCluster ev=' + str(ev) )
      while ( iFig < nFig and pc < nbrOfPreClusters ):
        x = preClusters.padX[ev][pc]
        dx = preClusters.padDX[ev][pc]
        y = preClusters.padY[ev][pc]
        dy = preClusters.padDY[ev][pc]
        charges = preClusters.padCharge[ev][pc]
        cath = preClusters.padCath[ev][pc]
        DEId = preClusters.padDEId[ev][pc]
        nPads = x.shape[0]
        flags = (DEId == DEId[0])
        if np.sum( flags ) != nPads:
          print("Pads with different DEIds")
          print("DEId ", DEId)
          m = wait("WARNING")
        if (charges.shape[0] != 0) and (x.shape[0] != 0):
          #
          xSup = np.max( x + dx )
          ySup = np.max( y + dy )
          xInf = np.min( x - dx )
          yInf = np.min( y - dy )
          #
          minCh =  np.min( charges )
          maxCh =  np.max( charges )
          minmaxCharges.append( [ minCh, maxCh] )
          # Set Lut scale
          plu.setLUTScale( 0, maxCh)
          #
          kx = iFig // nFigCol
          ky = iFig % nFigCol
          print("???", pc, iFig, kx, ky)
          c0Idx = np.where( cath == 0)
          c1Idx = np.where( cath == 1)
          plu.drawPads( ax[kx, ky],  x[c0Idx], y[c0Idx], dx[c0Idx], dy[c0Idx], charges[c0Idx],
                      title= "", alpha=0.5, doLimits=False)
          plu.drawPads( ax[kx, ky],  x[c1Idx], y[c1Idx], dx[c1Idx], dy[c1Idx], charges[c1Idx],
                       title= str(DEId[0]), alpha=0.5, doLimits=False) # noYTicksLabels=True, doLimits=False)
          ax[kx,ky].set_xlim( xInf, xSup )
          ax[kx,ky].set_ylim( yInf, ySup )

          #
          # MC Hits
          #
          # Get MCHits
          chId = DEId[0] // 100
          if (chId == 0):
              input("Warning: chId == 0")
          u, v, tids = getMCHits( mcObjByChId, preClusters, ev, pc, xInf, xSup, yInf, ySup  )
          print("??? u", u.shape, u)
          print("??? tids", tids)
          print("??? chId", chId)
          chargeTrack = np.zeros( u.shape[0] )
          tidNPads = np.zeros( u.shape[0], dtype=int )
          for k in range( u.shape[0] ):
            chargeTrack[k], tidNPads[k]= getTotalMCTrackChargeInChId( mcObj, ev, tids[k], chId, u[k:k+1], v[k:k+1] )
            print( "??? charge/npads", chargeTrack[k], tidNPads[k] )
          mu = np.array( [ u, v ] ).T
          # w = np.ones( u.shape )
          w = np.sqrt( chargeTrack ) / 50.0
          var = np.array( [ np.ones( w.shape ), np.ones( w.shape ) ] ).T
          plu.drawModelComponents( ax[kx, ky], w, mu, var, color='black', pattern='show w' )
          #
          # Reco Hits
          #
          # mu = np.array( [ recoByChId[ev][chId]['x'], recoByChId[ev][chId]['y'] ] ).T
          print ("???", preClusters.rClusterX[ev][pc], preClusters.rClusterY[ev][pc])
          mu = np.array( [ preClusters.rClusterX[ev][pc], preClusters.rClusterY[ev][pc] ] ).T
          w = np.ones( preClusters.rClusterX[ev][pc].shape )
          var = np.array( [ np.ones( w.shape ), np.ones( w.shape ) ] ).T
          plu.drawModelComponents( ax[kx, ky], w, mu, var, color='red', pattern="+" )          
          iFig += 1
        else:
          print("Empty preCluster")
          print(preClusters.padX[ev][pc])
          w =input("Next ?")
        pc += 1
      # while iFig
      plt.show()
    # while pc
    return

def displayAPrecluster( ax, xy, dxy, cath, z, w, mu, var, mcObj, preClusters, ev, pc, DEIds ):
    idx0 = np.where( cath==0 )
    idx1 = np.where( cath==1 )
    frame = plu.getPadBox(xy[0], xy[1], dxy[0], dxy[1] )
    xInf, xSup, yInf, ySup = frame
    zMax = np.max( z )
    zMin = np.min( z )
    plu.setLUTScale( zMin, zMax )    
    print( xInf, xSup, yInf, ySup )
    if( xy[0][idx0].size != 0 ):
      plu.drawPads( ax[0,0], xy[0][idx0],  xy[1][idx0], dxy[0][idx0], dxy[1][idx0], z[idx0],  alpha=0.5 )
      plu.drawPads( ax[0,1], xy[0][idx0],  xy[1][idx0], dxy[0][idx0], dxy[1][idx0], z[idx0],  alpha=0.5 )
      plu.drawPads( ax[0,2], xy[0][idx0],  xy[1][idx0], dxy[0][idx0], dxy[1][idx0], z[idx0],  alpha=0.5 )
      plu.drawPads( ax[0,3], xy[0][idx0],  xy[1][idx0], dxy[0][idx0], dxy[1][idx0], z[idx0],  alpha=0.5 )
      #plu.drawPads( ax[1,1], xy[0][idx0],  xy[1][idx0], dxy[0][idx0], dxy[1][idx0], z[idx0],  alpha=0.5, title="Integral")
    if( xy[0][idx1].size != 0 ):
      plu.drawPads( ax[0,0], xy[0][idx1],  xy[1][idx1], dxy[0][idx1], dxy[1][idx1], z[idx1],  alpha=0.5)
      plu.drawPads( ax[0,1], xy[0][idx1],  xy[1][idx1], dxy[0][idx1], dxy[1][idx1], z[idx1],  alpha=0.5)
      plu.drawPads( ax[0,2], xy[0][idx1],  xy[1][idx1], dxy[0][idx1], dxy[1][idx1], z[idx1],  alpha=0.5)
      plu.drawPads( ax[0,3], xy[0][idx1],  xy[1][idx1], dxy[0][idx1], dxy[1][idx1], z[idx1],  alpha=0.5)
      # plu.drawPads( ax[1,1], xy[0][idx1],  xy[1][idx1], dxy[0][idx1], dxy[1][idx1], z[idx1],  alpha=0.5, title="Integral")

    for i in range(2):
      for j in range(4):
        ax[i,j].set_xlim( xInf, xSup)
        ax[i,j].set_ylim( yInf, ySup)
    plu.drawModelComponents( ax[1, 0], w, mu, var, color='red', pattern="o" )
    plu.drawModelComponents( ax[1, 1], w, mu, var, color='red', pattern="o" )
    plu.drawModelComponents( ax[1, 2], w, mu, var, color='red', pattern="o" )
    plu.drawModelComponents( ax[1, 3], w, mu, var, color='red', pattern="o" )
    # ACtual Reco
    pMu = np.array( [ preClusters.rClusterX[ev][pc], preClusters.rClusterY[ev][pc] ] ).T
    pW = np.ones( preClusters.rClusterX[ev][pc].shape )
    pVar = np.array( [ np.ones( pW.shape ), np.ones( pW.shape ) ] ).T
    plu.drawModelComponents( ax[0, 0], pW, pMu, pVar, color='green', pattern="o" )      

    # drawMCHitsInFrame ( ax[1,1], frame, mcObj, ev, DEIds )
    # plu.displayLUT( ax[1, 1] )
    return 

def matchMCTrackHits(x0, y0, spanDEIds, boxOfRecoPads, rejectedDx, rejectedDy, mcObj, ev, verbose = False ):
  # Inv x0 = preClusters.rClusterX[ev][pClusterIdx]
  # Inv ??? y0 = preClusters.rClusterY[ev][pClusterIdx]
  x1 = []; y1 = []
  nPreClusters = x0.shape[0]
  if (nPreClusters == 0 ): 
    # return match, nPreClusters, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved 
    return ( 0, 0, 0, 0, 0, 
            np.array([],dtype=np.float), np.array([],dtype=np.float), np.array([],dtype=np.float), 
            np.empty( shape=(0, 0)), [] )
  # Inv ??? deIds = np.unique( preClusters.rClusterDEId[ev][pClusterIdx] )
  deIds = spanDEIds
  if verbose :
    print("matchMCTrackHits x0.shape=", x0.shape, "spanDEIds=", spanDEIds)
  # Pads half-size
  # Inv ??? dx0 = rejectPadFactor * np.max( preClusters.padDX[ev][pClusterIdx] )
  # Inv dy0 = rejectPadFactor * np.max( preClusters.padDX[ev][pClusterIdx] )
  dr0 = rejectedDx*rejectedDx + rejectedDy*rejectedDy
  nbrOfTracks = len( mcObj.trackDEId[ev] )
  dMatrix = np.empty(shape=[nPreClusters, 0])
  dMin = np.empty(shape=[0])
  x = np.empty(shape=[0])
  y = np.empty(shape=[0])
  # Keep the location of McHits involved
  mcHitsInvolved = []
  totalMCHits = 0
  for t in range( nbrOfTracks):
    xl = []; yl = []
    # Select track hits in DEIds
    for k, d in enumerate(deIds):
      # Must belong to the same DE and have a trackCharge in the DEId 
      # AND must be in the box delimited by the reco pads
      flag0 = (mcObj.trackDEId[ev][t] == d) 
      flag1 = (mcObj.trackCharge[ev][t] > 0)
      flag2 = plu.isInBox ( mcObj.trackX[ev][t], mcObj.trackY[ev][t], boxOfRecoPads )
      flags = np.bitwise_and( flag0, flag1)
      flags = np.bitwise_and( flags, flag2) 
      # flag1 = (mcObj.trackDEId[ev][t] == d) 
      idx = np.where( flags )[0]
      nbrOfHits = idx.shape[0]
      totalMCHits += nbrOfHits
      xl.append( mcObj.trackX[ev][t][idx] )
      yl.append( mcObj.trackY[ev][t][idx] )
      if k > 0:
        input("??? Take care")
    x = np.vstack(xl)
    y = np.vstack(yl)
    if ( x.shape[1] != 0 ):
      # Build distance matrix
      # [nbrOfHits][nbrX0]
      #
      # To debug
      x1.append(x)
      y1.append(y)
      
      dx = np.tile( x0, (nbrOfHits,1) ).T - x
      dy = np.tile( y0, (nbrOfHits,1) ).T - y
      # dxMin = dx[ np.argmin(np.abs(dx)) ]
      # dyMin = dx[ np.argmin(np.abs(dy)) ]
      dist = np.multiply( dx, dx) + np.multiply( dy, dy)
      # Not used
      selected = (dist < dr0 )
      """
      dist = np.where(  dist < dr0, dist, 0 )
      addHit = np.sum(selected)
      """
      # Select all tracks
      dMatrix = np.hstack( [dMatrix, dist])
      mcHitsInvolved.append( (t, idx) )

    # if x.shape[1] != 0
  # loop on track (t)
  #
  if verbose:
    print("matchMCTrackHits  dMatrix(distances) :")
    print(dMatrix)

  #
  # TP selection
  if dMatrix.size != 0 :
    # For debugging
    x1 = np.concatenate(x1).ravel()
    y1 = np.concatenate(y1).ravel()
    # dMin = np.min(dMatrix, axis=1) # Min per line
    # dMatrix = np.where( dMatrix == 0, 10000, dMatrix)
    dMinIdx = np.argmin( dMatrix, axis=0 )
    # Build TF Matrix
    tfMatrix = np.zeros( dMatrix.shape, dtype=np.int )
    for i in range(dMatrix.shape[1]):
      # Process on assigned hits
      if dMatrix[ dMinIdx[i], i ] < dr0 :
        # Search other value on the same row
        jIdx = np.where( tfMatrix[dMinIdx[i],:] == 1)
        # Check empy set
        if jIdx[0].shape[0] == 0:
          # No other occurence in the tfMatrix line
          # So can set it to 1
          tfMatrix[ dMinIdx[i], i ] = 1
        else:
          # One and only one occurence is possible
          j = jIdx[0][0]
          # Search the distance matrix minimun
          if dMatrix[ dMinIdx[i], j ] < dMatrix[ dMinIdx[i], i ]:
             # Other solution is less than the current index
             # Set to 0 the current element 
             tfMatrix[ dMinIdx[i], i ] = 0
          else:
             # the current element is the best solution (minimum)  
             tfMatrix[ dMinIdx[i], i ] = 1
             # Clean the other solution
             tfMatrix[ dMinIdx[i], j ] = 0
          #    
      else:
        tfMatrix[ dMinIdx[i], i ] = 0
    if verbose:
      print("matchMCTrackHits  tfMatrix :")
      print(tfMatrix)

    #
    # TP/FP
    #
    tpRow = np.sum( tfMatrix, axis=1)
    FP = np.sum( tpRow == 0)
    # Debug
    """
    iii = np.where( tpRow == 0 )
    for kk in list( iii[0] ):
      print( x0[ kk ], y0[ kk] )
    print("FP", FP, tpRow.shape)
    """
    TP = np.sum( tpRow > 0 )
    # Remove the TP and count otheroccurence as FP
    tpRow = tpRow - 1
    # print("tpRow", tpRow)
    FP += np.sum( np.where( tpRow > 0, tpRow, 0) )
    # Debug
    """
    iii = np.where( tpRow > 0 )
    for kk in list( iii[0] ):
      print( x0[ kk ], y0[ kk] )
    print("FP", FP)
    """
    #
    # FN and 
    #
    tpCol = np.sum( tfMatrix, axis=0)
    # print("tpCol", tpCol)
    FN = np.sum( tpCol == 0)
    # Debug
    """
    print("FN", FN)
    iii = np.where( tpCol == 0 )
    print(iii)
    print("??? x1",x1)
    for kk in list( iii[0] ):
      print( x1[ kk ], y1[ kk] )    
    """
    # The min has been performed by column
    # FP += np.sum( np.where( tpCol > 0, tpCol, 0) )
    if verbose : 
      print("matchMCTrackHits # Reco Hits, # MC Hits", tfMatrix.shape )  
      print("matchMCTrackHits TP, FP, FN", TP, FP, FN)
    # 
    # Minimal dsitances for assigned hits given by tfMatrix
    dMin = np.sqrt( dMatrix[ np.where(tfMatrix == 1) ].ravel() )
    """
    print("??? dMatrix", dMatrix )
    print("??? dr0", dr0 )
    print("??? tfMatrix", tfMatrix )
    print("???", np.where(tfMatrix == 1) )
    print("??? em x", x0 )
    print("??? mc x", x1 )
    print("??? em y", y0 )
    print("??? mc y", y1 )
    """
    idx0, idx1 = np.where(tfMatrix == 1)
    if idx0.size != idx1.size :
      input("Pb : one and only one TP per reco hit point")
    dxMin = x0[ idx0 ] - x1[ idx1 ]
    dyMin = y0[ idx0 ] - y1[ idx1 ]
    #
    # Verification
    if tfMatrix.size != 0:
      sumLines = np.sum(tfMatrix, axis=0)
      sumColumns = np.sum(tfMatrix, axis=1)
      if ( (np.where( sumLines  > 1)[0].size != 0) or (np.where( sumColumns > 1)[0].size != 0)  ):
        print("tfMatrix", tfMatrix)
        print("sumLines", sumLines)
        print("sumColumns", sumColumns)
        print("len(mcHitsInvolved)", len(mcHitsInvolved))
        input("Pb in tfMatrix")
  else:
    TP=0; FP=0; FN=0; dMin = np.array([],dtype=np.float)
    dxMin = np.array([],dtype=np.float); dyMin = np.array([],dtype=np.float)
    tfMatrix = np.empty( shape=(0, 0) )
  match = float(TP) / nPreClusters  
  # input("One Cluster")
 
  return match, nPreClusters, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved 

def assignMCToRecoHits( preClusters, mcObj, ev, rejectPadFactor=0.5):
  nbrOfPreClusters = len( preClusters.padId[ev] )
  print( "displayPreClusters: nbrOfPreClusters =", nbrOfPreClusters)
  assign = []
  allDMin = []
  evTP = 0; evFP = 0 ; evFN = 0
  evNbrOfHits = 0
  for pc in range(nbrOfPreClusters):
    nbrOfHits = preClusters.rClusterX[ev][pc].shape[0]
    x0 = preClusters.rClusterX[ev][pc]
    y0 = preClusters.rClusterY[ev][pc]
    spanDEIds = np.unique( preClusters.padDEId[ev][pc] )
    rejectedDx = rejectPadFactor * np.max( preClusters.padDX[ev][pc] )
    rejectedDy = rejectPadFactor * np.max( preClusters.padDY[ev][pc] )
    
    spanBox = plu.getPadBox( preClusters.padX[ev][pc], preClusters.padY[ev][pc],
                             preClusters.padDX[ev][pc], preClusters.padDY[ev][pc])
    # print("matchMCTrackHits x0, rejectedDx", x0, rejectedDx )
    # u = matchMCTrackHits(x0, y0, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
    # print("???",u)
    match, nbrOfHits, TP, FP, FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved = matchMCTrackHits(x0, y0, spanDEIds, spanBox, rejectedDx, rejectedDy, mcObj, ev )
    if (match > 0.33):
      assign.append( (pc, match, nbrOfHits, TP, FP,FN, dMin, dxMin, dyMin, tfMatrix, mcHitsInvolved ) )
      evTP += TP; evFP += FP; evFN += FN
      allDMin.append( dMin )
      evNbrOfHits += nbrOfHits
    #
    # Debug
    """
    if (pc == 74 ):
      input("pc =" + str(pc) + ", match=" + str(match) )
    """
  # end loop pc
  print( "ev=", ev, "# hits=", evNbrOfHits, "TP, FP, FN = (", evTP, evFP, evFN, "), assigned preClusters", len(assign),"/", nbrOfPreClusters  ) 
  return evNbrOfHits, evTP, evFP, evFN, allDMin, assign

def displayMCPads( mcObj, ev ):
    nChId = len(md.ChId)
    nFigCol = 4
    nFigRow = 3
    nFig = nFigCol * nFigRow
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
    fig.suptitle('MC Pads ev=' + str(ev) + " with " + str( len(mcObj.trackId[ev]) ) + " tracks", fontsize=16)
    minmaxCharges = []
    for chId in md.ChId:
      idx = (mcObj.padChId[ev] == chId)
      x = mcObj.padX[ev][idx]
      dx = mcObj.padDX[ev][idx]
      y = mcObj.padY[ev][idx]
      dy = mcObj.padDY[ev][idx]
      charges = mcObj.padCharge[ev][idx]
      cath = mcObj.padCath[ev][idx]
      print ("# of elements", np.sum(idx), "/", idx.shape[0])
      if (charges.shape[0] != 0) and (x.shape[0] != 0):
        # print("shapes", chId, dx.shape, dy.shape, charges.shape, cath.shape)
        # print("min/max cath", chId, np.min(cath), np.max(cath))
        #
        xSup = np.max( x + dx )
        ySup = np.max( y + dy )
        xInf = np.min( x - dx )
        yInf = np.min( y - dy )
        #
        minCh =  np.min( charges )
        maxCh =  np.max( charges )
        minmaxCharges.append( [ minCh, maxCh] )
        # Set Lut scale
        plu.setLUTScale( 0, maxCh)
        #
        kx = (chId-1) // nFigCol
        ky = (chId-1) % nFigCol
        c0Idx = np.where( cath == 0)
        c1Idx = np.where( cath == 1)
        plu.drawPads( ax[kx, ky],  x[c0Idx], y[c0Idx], dx[c0Idx], dy[c0Idx], charges[c0Idx],
                      title= "", alpha=0.5, doLimits=False)
        plu.drawPads( ax[kx, ky],  x[c1Idx], y[c1Idx], dx[c1Idx], dy[c1Idx], charges[c1Idx],
                       title= str(chId), alpha=0.5, doLimits=False) # noYTicksLabels=True, doLimits=False)
        ax[kx,ky].set_xlim( xInf, xSup )
        ax[kx,ky].set_ylim( yInf, ySup )
    print("min/max Charges", minmaxCharges)
    plt.show()   
    
def getDEDescription( preClusters):
  DE = {}
  for ev in range(len( preClusters.padDEId) ):
    for pc in range(len( preClusters.padDEId[ev] )): 
      deids = preClusters.padDEId[ev][pc]
      for i, id in enumerate(deids):
        x = preClusters.padDX[ev][pc][i]
        y = preClusters.padDY[ev][pc][i]
        if id in DE.keys():
          found = False
          for (a,b) in DE[id]:
             if a == x and b == y:
               found = True
               break
          if not found:
            DE[id].append( (x, y) )
        else:
          DE[id] = [ (preClusters.padDX[ev][pc][i], preClusters.padDY[ev][pc][i]) ]
      # id loop
    # pc loop
  # ev loop
  #
  for key in DE.keys():
    """
    unique = set()
    temp = [unique.add( (int(a*100), int(b*100)) ) for (a, b) in DE[key] 
              if ( int(a*100), int(b*100) ) not in unique]
    """
    print(key, ":", DE[key])
  """           
  for key in DE.keys():
    print(key, ":", DE[key])
  """
  return
def displayMCAndRecoHits( mcObj, ev, mcHits, recoHits, nFigCol=4, nFigRow=3 ):
    nChId = len(md.ChId)

    nFig = nFigCol * nFigRow
    fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(17, 7) )
    if nFigCol==1 and nFigRow==1:
      ax = np.array( [[ax]])

    fig.suptitle('MC Pads ev=' + str(ev) + " with " + str( len(mcObj.trackId[ev]) ) + " tracks", fontsize=16)
    minmaxCharges = []
    for chId in md.ChId:
      idx = (mcObj.padChId[ev] == chId)
      x = mcObj.padX[ev][idx]
      dx = mcObj.padDX[ev][idx]
      y = mcObj.padY[ev][idx]
      dy = mcObj.padDY[ev][idx]
      charges = mcObj.padCharge[ev][idx]
      cath = mcObj.padCath[ev][idx]
      print ("# of elements", np.sum(idx), "/", idx.shape[0])
      if (charges.shape[0] != 0) and (x.shape[0] != 0):
        # print("shapes", chId, dx.shape, dy.shape, charges.shape, cath.shape)
        # print("min/max cath", chId, np.min(cath), np.max(cath))
        #
        xSup = np.max( x + dx )
        ySup = np.max( y + dy )
        xInf = np.min( x - dx )
        yInf = np.min( y - dy )
        #
        minCh =  np.min( charges )
        maxCh =  np.max( charges )
        minmaxCharges.append( [ minCh, maxCh] )
        # Set Lut scale
        plu.setLUTScale( 0, maxCh)
        #
        kx = (chId-1) // nFigCol
        ky = (chId-1) % nFigCol
        c0Idx = np.where( cath == 0)
        c1Idx = np.where( cath == 1)
        plu.drawPads( ax[kx, ky],  x[c0Idx], y[c0Idx], dx[c0Idx], dy[c0Idx], charges[c0Idx],
                      title= "", alpha=0.5, doLimits=False)
        plu.drawPads( ax[kx, ky],  x[c1Idx], y[c1Idx], dx[c1Idx], dy[c1Idx], charges[c1Idx],
                       title= str(chId), alpha=0.5, doLimits=False) # noYTicksLabels=True, doLimits=False)
        ax[kx,ky].set_xlim( xInf, xSup )
        ax[kx,ky].set_ylim( yInf, ySup )
        #
        # MC Hits
        #
        mu = np.array( [ mcHits[ev][chId]['x'], mcHits[ev][chId]['y'] ] ).T
        w = np.ones( mcHits[ev][chId]['x'].shape )
        var = np.array( [ np.ones( w.shape ), np.ones( w.shape ) ] ).T
        plu.drawModelComponents( ax[kx, ky], w, mu, var, color='black', pattern="o" )
        #
        # Reco Hits
        #
        mu = np.array( [ recoHits[ev][chId]['x'], recoHits[ev][chId]['y'] ] ).T
        w = np.ones( recoHits[ev][chId]['x'].shape )
        var = np.array( [ np.ones( w.shape ), np.ones( w.shape ) ] ).T
        plu.drawModelComponents( ax[kx, ky], w, mu, var, color='red', pattern="+" )
        # plt.show() 
    print("min/max Charges", minmaxCharges)
    plt.show()    
                        
    """

    for t in range(nTracks):
        nClusters = len(tracks.tracks[t])
        clusters = tracks.tracks[t] 
        N = int( np.ceil( float(len(clusters)) / nFigRow ) )
        cID = 0
        for k in range(N):
    """     
if __name__ == "__main__":
    
    mcData = IO.MCData(fileName="MCDataDump.dat")
    mcData.read()
    #
    recoData = IO.PreCluster()
    recoData.read()
    print( "recoData.padChIdMinMax=", recoData.padChIdMinMax )
    print( "recoData.rClusterChIdMinMax=", recoData.rClusterChIdMinMax )
    # displayMCPads( mcData, 0 )
    # displayMCTracks( mcData, 0 )
    # displayMCPads( mcData, 0 )
    nEvents = len( recoData.padX )
    
    # ???
    """
    mcGTByChId = []
    recoByChId = []
    for ev in range( len(mcData.padX)):
      contrib, counts =getContributingMCTracks( mcData, ev)
      # print( contrib )
      # print( counts )
      mcGTByChId.append( getMCPositions( mcData, ev ) )
      recoByChId.append( getRecoPositions( recoData, ev) )
    
    #    displayMCPads( mcData, ev )
    displayMCAndRecoHits( mcData, 5, mcGTByChId, recoByChId, nFigCol=1, nFigRow=1)
    # displayPreClusters( recoData, mcData, 5, mcGTByChId, recoByChId   )
    """

    # Build Reco measure
    """
    recoMeasure = []
    for ev in range(nEvents): 
      recoMeasure.append( assignMCToRecoHits( recoData, mcData, ev ))
    #
    utl.writePickle( "recoMeasure.obj", recoMeasure )
    """
    recoMeasure = utl.readPickle("recoMeasure.obj")
    emptyEM = EM.EMResults() 
    # statOnEvents( recoMeasure, mcData, recoData, emptyEM, range(nEvents)  )
    # worstCases( mReco, mcData, recoData, emptyEM, range(nEvents)  )
    
    emMeasure = []
    emObj = EM.EMResults()

    #
    # Build EM measure 
    emMeasure = []
    emObj = EM.EMResults()
    
    # for ev in range(29, 30):
    # for ev in range(2, 3):
    
    for ev in range(nEvents):
      emMeasure.append( processEMOnPreClustersV5( recoData, mcData, ev, emObj) )
    """
    utl.writePickle( "emMeasure.obj", (emMeasure, emObj) )
    """
    #
    (emMeasure, emObj) = utl.readPickle("emMeasure.obj")
    #
    # Fit LocalMaxWithSub
    #
    """
    for i in range(3):
      clusterSize = []
      mcClusterSize = []
      recoClusterSize = []
      fp = 15.0 + i * 5
      for ev in range(nEvents):
        fitLocalMaxWithSubtraction( recoData, mcData, ev, clusterSize, mcClusterSize, recoClusterSize, fitParameter=fp)    
      cSize = np.array( clusterSize )
      mcCSize = np.array( mcClusterSize )
      r = cSize / mcCSize
      print( "# max ratio ", np.sum( r ) / mcCSize.size,"@ fp", fp)
      input("next")
    """
    
    #
    # Build Fit measure 
    fitMeasure = []
    """
    fitObj = EM.EMResults()
    for ev in range(nEvents):
      fitMeasure.append( processFitingThePreClusters( recoData, mcData, ev, fitObj) )
    utl.writePickle( "fitMeasure.obj", (fitMeasure, fitObj) )

    (fitMeasure, fitObj) = utl.readPickle("fitMeasure.obj")
    """
    
    #statOnEvents( emMeasure, mcData, recoData, emptyEM, range(nEvents)  )
    # Stat on var
    # statWithOneHitClusters( recoMeasure, mcData, recoData, range(nEvents), nGaussians=1 )
    # statWithOneHitClusters( emMeasure, mcData, recoData, range(2), nGaussians=1 )
    # getDEDescription( recoData )

    
    # testFindLocalMax( recoData, mcData, range(nEvents))

    
    # selectOneHitClusters( emMeasure, emObj, mcData, recoData, range(nEvents), nGaussians=1 )
    # refineOneHitClusters( emMeasure, mcData, recoData, range(nEvents), nGaussians=1 )
    
    #
    statOnDistancesMuons( recoMeasure, mcData, recoData, emptyEM, range(nEvents) )
    statOnDistancesMuons( emMeasure, mcData, recoData, emptyEM, range(nEvents) )

    # statOnChambers( recoMeasure, mcData, recoData, emptyEM, range(nEvents) )
    # statOnChambers( emMeasure, mcData, recoData, emptyEM, range(nEvents) )  

    recoCSizes, recoDMin = statOnClusterSize( recoMeasure, mcData, recoData, emptyEM, range(nEvents) )
    emCSizes, emDMin = statOnClusterSize( emMeasure, mcData, recoData, emptyEM, range(nEvents) )
    compareClusterSizes( recoCSizes, recoDMin, emCSizes, emDMin)
    
    contrib, counts =getContributingMCTracks( mcData, 5)
    """
    emObj5 = EM.EMResults()
    emMeasure5 = [[],[],[],[], []]
    recoMeasure5 = [[],[],[],[],[]]
    emMeasure5.append( processEMOnPreClusters( recoData, mcData, 5, emObj5) )
    recoMeasure5.append( assignMCToRecoHits( recoData, mcData, 5 ) )
    statOnChambers( recoMeasure5, mcData, recoData, emptyEM, range(5,6) )
    statOnChambers( emMeasure5, mcData, recoData, emptyEM, range(5,6) )
    """
    # print(emObj)
    # worstCasesByChamber( emMeasure, mcData, recoData, emObj, range(nEvents), 5  )
    input("Worst Cases")
    # worstCases( recoMeasure, mcData, recoData, emObj  )
    # worstCases( emMeasure, mcData, recoData, emObj, range(nEvents)  )
