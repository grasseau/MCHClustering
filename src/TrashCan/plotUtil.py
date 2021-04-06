#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jlul 30, 2020 2:46:25 PM$"

import sys

import numpy as np
import matplotlib as plt
import matplotlib.patches as patches

Colors = None
Scale = None
StopOnWarning = False

def getColorMap(N=256):
    global Colors
    cmap = np.linspace( 0.0, 0.87, N)
    # colors_ = [ plt.cm.coolwarm(x) for x in cmap ]
    colors_ = [ plt.cm.rainbow(x) for x in cmap ]
    # colors_ = [ plt.cm.plasma(x) for x in cmap ]
    #colors_[0] = [ 1.0, 1.0, 1.0, 1.0]
    Colors = np.array( colors_)
    return Colors

def setLUTScale( ymin, ymax ):
    global Scale
    Scale = (ymin, ymax)

def displayLUT( ax,  colors=Colors, scale=Scale):
    global Colors, Scale
    colors = Colors
    nColors = len(colors)
    yMin = Scale[0]
    yMax = Scale[1]
    """
    d = yMax - yMin
    dy = d / nColors
    """
    dy = 1.0/nColors
    x = 0.0
    dx = 0.2 
    for i in range(nColors):
      # y = i * dy + yMin
      y = i * dy
      rect = patches.Rectangle( ( x, y), dx, dy,  linewidth=1, edgecolor=colors[i], facecolor=colors[ i ], alpha=1.0)
      ax.add_patch(rect)
    #
    ax.set_xlim( 0,  1.0 )
    # ax.set_ylim( yMin,  yMax )
    ax.set_ylim( 0,  1.0 )
    ax.set_title( "Color map (max=%5.2f)" % yMax)
    ax.set_xticks([])
    return

def setText( ax, ratio, str, **kwargs):
   xLim = ax.get_xlim()
   yLim = ax.get_ylim()
   x = (xLim[1] - xLim[0])* ratio[0] + xLim[0] 
   y = (yLim[1] - yLim[0])* ratio[1] + yLim[0] 
   ax.text( x, y, str, kwargs)
   
def getPadBox( x, y, dx, dy ):
  xMin = np.min( x - dx)
  xMax = np.max( x + dx)
  yMin = np.min( y - dy)
  yMax = np.max( y + dy)
  return (xMin, xMax, yMin, yMax )

def inBox( x, y, box ):
    flags = isInBox( x, y, box)
    return np.where( flags )

def isInBox( x, y, box ):
    (xMin, xMax, yMin, yMax ) = box
    flag0 = np.bitwise_and( (x >= xMin), (x < xMax) )
    flag1 = np.bitwise_and( (y >= yMin), (y < yMax) )
    flags = np.bitwise_and( flag0, flag1)
    return flags

def drawModelComponents( ax, w, mu, var, color='black', pattern="o" ):
    K = w.shape[0]
    if K == 0 : return
    ( bot, top) = ax.get_ylim()
    wMax = np.max(w)
    wMin = np.min(w)
    cst = (top - bot) * 0.20 / wMax
    for k in range(K):
        if w[k] > 1.e-12 :
          x = mu[k,0]; y = mu[k,1]
          dx = np.sqrt( var[k,0]); dy = np.sqrt( var[k,1] )

          if pattern == "cross":
            hx = [ x - 0.5*dx,  x + 0.5*dx ]      
            hy = [ y,  y]
            vx = [ x,  x]
            vy = [ y - 0.5*dy,  y + 0.5*dy]
            ax.plot( hx, hy, "-", color=color)
            ax.plot( vx, vy, "-", color=color)
          elif pattern == "rect":
            sox = [ x - 0.5*dx, x + 0.5*dx ]
            soy = [ y - 0.5*dy, y - 0.5*dy]
            nox = [ x - 0.5*dx, x + 0.5*dx ]
            noy = [ y + 0.5*dy, y + 0.5*dy]
            eastx = [ x - 0.5*dx, x - 0.5*dx ]          
            easty = [ y - 0.5*dy, y + 0.5*dy]
            westx = [ x + 0.5*dx, x + 0.5*dx ]          
            westy = [ y - 0.5*dy, y + 0.5*dy]          
            ax.plot( sox, soy, "-", color=color)
            ax.plot( nox, noy, "-", color=color)
            ax.plot( eastx, easty, "-", color=color)
            ax.plot( westx, westy, "-", color=color)
          elif pattern == "diam":
            sex = [ x,          x + 0.5*dx ]
            sey = [ y - 0.5*dy, y ]
            swx = [ x,          x - 0.5*dx ]
            swy = [ y - 0.5*dy, y]
            nex = [ x + 0.5*dx, x  ]          
            ney = [ y, y + 0.5*dy ]
            nwx = [ x - 0.5*dx, x ]          
            nwy = [ y, y + 0.5*dy]          
            ax.plot( sex, sey, "-", color=color)
            ax.plot( swx, swy, "-", color=color)
            ax.plot( nex, ney, "-", color=color)
            ax.plot( nwx, nwy, "-", color=color)
          elif pattern == "show w":
            # for k in range(K):
            circle = patches.Circle(  (mu[k, 0], mu[k, 1]) , w[k]*cst, linewidth=1, edgecolor='black', facecolor=None, fill=False) 
                    # facecolor=c[r], 
            ax.add_patch(circle)
          else:
            ax.plot( x, y, pattern, color=color, markersize=3 )
            
    return

def getBarycenter( x0, x1, w0, w1):
    # Explain ???
    sw0 = np.sum( w0) 
    sw1 = np.sum( w1)
    w = sw0 + sw1
    # sw0 = sw0 / w
    # sw1 = sw1 / w
    cst = 1.0 / w
    # Mean
    x = cst * np.dot(x0, w0) + cst* np.dot(x1, w1)
    # x = x / w
    # variance, std-error
    du0 = x0 - x
    du1 = x1 - x
    du0 = du0 * du0
    du1 = du1 * du1
    # dx = sw0 * np.dot(dx0, w0) + sw1 * np.dot(dx1, w1)
    dx = cst * np.dot(du0, w0) + cst * np.dot(du1, w1) 
    #
    return ( x , np.sqrt( dx ) )
def computeOverlapingPads( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1, verbose = 0 ):
    maxFloat = sys.float_info.max
    x0Inf = x0 - dx0
    x0Sup = x0 + dx0
    y0Inf = y0 - dy0
    y0Sup = y0 + dy0
    x1Inf = x1 - dx1
    x1Sup = x1 + dx1
    y1Inf = y1 - dy1
    y1Sup = y1 + dy1
    x = []; dx = []
    y = []; dy = []
    ch = []
    for i in range(x0Inf.shape[0]):
      jXInf = -1; jYInf = -1
      jXSup = -1; jYSup = -1
      XInf = maxFloat; XSup = - maxFloat
      YInf = maxFloat; YSup = - maxFloat
      for j in range(x1Inf.shape[0]):
          xmin = max( x0Inf[i], x1Inf[j] )
          xmax = min( x0Sup[i], x1Sup[j] )
          xInter = ( xmin < xmax )
          ymin = max( y0Inf[i], y1Inf[j] )
          ymax = min( y0Sup[i], y1Sup[j] )
          yInter = ( ymin < ymax )
          # Have an intersection
          if xInter and yInter:
              # if xmin < XInf and xmin != x0Inf[i]:
              if xmin < XInf :
                 XInf = xmin
                 jXInf = j
              # if xmax > XSup and xmax != x0Sup[i]:
              if xmax > XSup :
                 XSup = xmax
                 jXSup = j
              # if ymin < YInf and ymin != y0Inf[i]:
              if ymin < YInf:
                 YInf = ymin
                 jYInf = j
              # if ymax > YSup and ymax != y0Sup[i]:
              if ymax > YSup :
                 YSup = ymax
                 jYSup = j
              """
              if np.abs(xmin - x0Inf[i]) > SimplePrecision :
                  x0[i] = 
              x0[i] = xInter
              x1[j] = xInter
              y0[i] = yInter
              y1[j] = yInter
              """
      if jXInf == -1 and jYInf == -1:
        # No intersection
        # The pad must be removed
        print( "no intersection i=", i,  " pad should be removed")
        x.append( x0[i])
        y.append( y0[i] )
        dx.append( dx0[i] )
        dy.append( dy0[i] )
        ch.append( ch0[i])
        continue
      else:
          #
          # X left
          #
          if jXInf != -1:
            # Intercepting pads 
            if ( XInf < x0Inf[i]):
              # Smaller than i
              # Keep i as is
              xl = x0Inf[i]
            else:
              if verbose > 0 : print( "  x left shortened", XInf,", i=", i )
              xl = XInf
          else:
            # No intercepting pads with a lower x value
            # Keep the pad as is
            xl = x0Inf[i]
          #
          # X right
          #
          if jXSup != -1:
            if ( XSup > x0Sup[i]):
              # Greater than i
              # Keep i as is
              xr = x0Sup[i]
            else:
              if verbose > 0 : print( "  x right shortened", XInf )
              xr = XSup
          else:
            # No intercepting pads with a lower x value
            # Keep the pad as is
            xr = x0Sup[i]
          #
          # Y left
          #            
          if jYInf != -1:
            # Intercepting pads 
            if ( YInf < y0Inf[i]):
              # Smaller than i
              # Keep i as is
              yl = y0Inf[i]
            else:
              yl = YInf
          else:
            # No intercepting pads with a lower y value
            # Keep the pad as is
            yl = y0Inf[i]
          #
          # Y right
          #
          if jYSup != -1:
            if ( YSup > y0Sup[i]):
              # Greater than i
              # Keep i as is
              yr = y0Sup[i]
            else:
              yr = YSup
          else:
            # No intercepting pads with a lower x value
            # Keep the pad as is
            yr = y0Sup[i]
                 
      xx = (xl + xr) *0.5
      yy = (yl + yr) *0.5
      dxx = (xr - xl) *0.5
      dyy = (yr - yl) *0.5
      x.append( xx )
      y.append( yy )
      dx.append( dxx )
      dy.append( dyy )
      ch.append( ch0[i])
      #
      
    return np.array(x), np.array(y) , np.array(dx), np.array(dy), np.array(ch)

def shorteningPadsV0( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1, verbose = 0 ):
    maxFloat = sys.float_info.max
    x0Inf = x0 - dx0
    x0Sup = x0 + dx0
    y0Inf = y0 - dy0
    y0Sup = y0 + dy0
    x1Inf = x1 - dx1
    x1Sup = x1 + dx1
    y1Inf = y1 - dy1
    y1Sup = y1 + dy1
    newX = []; newDX = []
    newY = []; newDY = []
    newCh = []
    for i in range(x0Inf.shape[0]):
      # jXInf = -1; jYInf = -1
      # jXSup = -1; jYSup = -1
      # XInf = maxFloat; XSup = - maxFloat
      # YInf = maxFloat; YSup = - maxFloat
      interI = []
      for j in range(x1Inf.shape[0]):
          xmin = max( x0Inf[i], x1Inf[j] )
          xmax = min( x0Sup[i], x1Sup[j] )
          xInter = ( xmin <= xmax )
          ymin = max( y0Inf[i], y1Inf[j] )
          ymax = min( y0Sup[i], y1Sup[j] )
          yInter = ( ymin <= ymax )
          # Have an intersection
          if xInter and yInter:
            interI.append(j)  
      if StopOnWarning and len( interI ) == 0:
        # No intersection, i remove
        input("No intersection,  next")
        continue
      sumCh1Inter =  np.sum( ch1[ interI ])
      cst =  ch0[i] / sumCh1Inter
      #  New pads
      for j in interI :
        l = max( x0Inf[i], x1Inf[j])
        r = min( x0Sup[i], x1Sup[j])
        b = max( y0Inf[i], y1Inf[j])      
        t = min( y0Sup[i], y1Sup[j])
        newX.append( (l+r)*0.5 )
        newY.append( (b+t)*0.5 )
        newDX.append( (r-l)*0.5 )
        newDY.append( (t-b)*0.5 )
        newCh.append( ch1[j] * cst )

    return np.array(newX), np.array(newY) , np.array(newDX), np.array(newDY), np.array(newCh)

def shorteningPads( x0, y0, dx0, dy0, ch0, x1, y1, dx1, dy1, ch1, verbose = 0 ):
    print("Use of SHORTENING PADS")
    epsilon = 10.0e-5
    maxFloat = sys.float_info.max
    x0Inf = x0 - dx0
    x0Sup = x0 + dx0
    y0Inf = y0 - dy0
    y0Sup = y0 + dy0
    x1Inf = x1 - dx1
    x1Sup = x1 + dx1
    y1Inf = y1 - dy1
    y1Sup = y1 + dy1
    newX = []; newDX = []
    newY = []; newDY = []
    newCh0 = []
    interIJ = [ [] for j in range(x0.shape[0])]
    interJI = [ [] for j in range(x1.shape[0])]
    mapIJToK = {}

    # Probably unsused
    mapKToIJ = []
    k = 0
    for i in range(x0Inf.shape[0]):
      # jXInf = -1; jYInf = -1
      # jXSup = -1; jYSup = -1
      # XInf = maxFloat; XSup = - maxFloat
      # YInf = maxFloat; YSup = - maxFloat
      for j in range(x1Inf.shape[0]):
          xmin = max( x0Inf[i], x1Inf[j] )
          xmax = min( x0Sup[i], x1Sup[j] )
          xInter = ( xmin <= (xmax - epsilon) )
          ymin = max( y0Inf[i], y1Inf[j] )
          ymax = min( y0Sup[i], y1Sup[j] )
          yInter = ( ymin <= (ymax - epsilon))
          # Have an intersection
          if xInter and yInter:
            interIJ[i].append(j)
            interJI[j].append(i)
      if StopOnWarning and len( interIJ[i] ) == 0:
        # No intersection, i remove ???
        # ??? To treat
        input("[shorteningPads] No intersection between cath0 and cath 1, next")
        continue
      sumCh1Inter =  np.sum( ch1[ interIJ[i] ])
      cst =  ch0[i] / sumCh1Inter
      #  New pads
      for j in interIJ[i] :
        l = max( x0Inf[i], x1Inf[j])
        r = min( x0Sup[i], x1Sup[j])
        b = max( y0Inf[i], y1Inf[j])      
        t = min( y0Sup[i], y1Sup[j])
        newX.append( (l+r)*0.5 )
        newY.append( (b+t)*0.5 )
        newDX.append( (r-l)*0.5 )
        newDY.append( (t-b)*0.5 )
        newCh0.append( ch1[j] * cst )
        mapIJToK[(i,j)] = k
        mapKToIJ.append( (i,j) )
        k += 1
    #
    newCh1 = [0.0 for i in range(len(newCh0))]
    for j in range( len(interJI) ):
      if len(interJI[j]) != 0 :
        sj = ch1[j] /  np.sum( ch0[ interJI[j] ])
        for l, i in enumerate( interJI[j] ):
          chji =  ch0[ interJI[j][l] ] * sj            
          newCh1[ mapIJToK[(i,j)] ] = chji
      elif StopOnWarning:
        # No intersection, j remove ???
        # ??? To treat
        input("[shorteningPads] No intersection between cath1 and cath 0, next")          
    if len(newDX) != 0:
      dXMin = np.min( np.array(newDX) )
      dYMin = np.min( np.array(newDY) )
      if min( dXMin, dYMin ) < 10.0e-3:
        print( " dMin", min( dXMin, dYMin ) )
        print( newDX )
        print( newDY )
        input("Too thin")
    return (np.array(newX), np.array(newY) , np.array(newDX), np.array(newDY), np.array(newCh0), np.array(newCh1),
           mapIJToK, mapKToIJ, interIJ, interJI)


def associateGrpPads( nbrGrp, groups, chA, chB, ch0, ch1, mapIJToK, mapKToIJ, IInterJ, JInterI ):
    """
    Associate cathode pad to grp and remove isolated pad (size of the grp = 1) 
    chA: charge on 1-plane pad
    chB: charge on 1-plane pad
    ch0: charge on cathode 0
    ch1: charge on cathode 0
    mapIJToK[(i,j)]: mapping cathode pad indexes (i,j) to 1-plane index (k) to i, j
    mapKToIJ : mapping 1-plane index (k) to i, j cathode pad indexes
    IInterJ[i] : list of cathode_1 pads intersecting cathode-0 pad i  
    JInterI[j] : list of cathode_1 pads intersecting cathode-1 pad j  
    """
    nbrOfGrpToRemove = 0
    # Check grpId start at 1
    for g in range(1,nbrGrp+1):
      idx = np.where( groups==g)
      if idx[0].size == 1:
        groups[idx] = 0
        nbrOfGrpToRemove += 1
        # remove this 1-Plane pad
        k = idx[0][0]
        (i0, j0) = mapKToIJ[k]
        # j's interceting i0
        interJ = IInterJ[i0]
        interJ.remove(j0)
        sumCh1Inter =  np.sum( ch1[ interJ ])
        cst =  ch0[i0] / sumCh1Inter
        for j in interJ:
          # (i0,j) to k
          chA[ mapIJToK[(i0,j)] ] = ch1[j] * cst
        # Update 
        IInterJ[i0] = interJ
        # 
        # the same for chB
        # i's interceting j0
        interI = JInterI[j0]
        interI.remove(i0)
        sumCh0Inter =  np.sum( ch0[ interI ])
        cst =  ch1[j0] / sumCh0Inter
        for i in interI:
          # (i0,j) to k
          chB[ mapIJToK[(i,j0)] ] = ch0[i] * cst
        # Update 
        JInterI[j0] = interI
    #     
    # Renumber Grps ???
    newG = 0
    if nbrOfGrpToRemove > 0:
      # nbrGrp = nbrGrp - nbrOfGrpToRemove
      for g in range(1,nbrGrp+1):
         idx = np.where( groups==g)
         if idx[0].size != 0:
           newG += 1
           groups[idx] = newG
      nbrGrp = newG
      print("[associateGrpPads] remove a 1-pad group")
    #
    # Associate a cathode-pad to a groups
    wellSplit = np.ones( (nbrGrp+1) )
    padGrpCath0 = [ [] for j in range(ch0.size)]
    padGrpCath1 = [ [] for j in range(ch1.size)]
    for k, g in enumerate(groups):
      # k is 1D-plane pad
      (i, j) = mapKToIJ[k]
      padGrpCath0[i].append(g)
      padGrpCath1[j].append(g)
    for p in range( len( padGrpCath0)):
      gTmp = np.unique( np.array(padGrpCath0[p]) )
      l = len( gTmp )
      if l == 0:
        print("[associateGrpPads] pad with no groups")
        print("pad cath0", p )
        if StopOnWarning: input("next")
      elif l != 1:
        print(gTmp)
        wellSplit[gTmp] = False
        print(l, "groups associated with a pad")
        gTmp = [0]
        # input("next")
      #
      padGrpCath0[p] = gTmp
    for p in range( len(padGrpCath1) ):
      gTmp = np.unique( np.array(padGrpCath1[p]) )
      l = len( gTmp )
      if l == 0:
        print("[associateGrpPads] pad with no groups")
        print("pad cath1", p )
        if StopOnWarning : input("next")
      elif l != 1:
        wellSplit[gTmp] = False
        print(l, "groups associated with a pad")
        gTmp = [0]
      #
      padGrpCath1[p] = gTmp

        # input("next")
    return nbrGrp, groups, wellSplit, np.hstack( padGrpCath0 ), np.hstack( padGrpCath1 )

def getConnexComponents( xi, yi, dxi, dyi ):    
    grp = np.zeros( xi.shape[0] , dtype=np.int)
    nextGrp = True
    grpId = 1
    # neighs = copy.deepcopy( neigh )
    neighs = getFirstNeighbours( xi, yi, dxi, dyi )
    # print("neighs", neighs)
    while nextGrp:
      print( "NEW Group", grpId, grp)
      k = np.argmax( grp == 0 )
      grp[k] = grpId
      nList = np.array( neighs[k] )
      done = []
      while nList.size != 0 :
        # print( "  Neigh nList", nList)
        l = nList[0]
        done.append(l)
        grp[l] = grpId
        # print( "l, nList, neighs[l]", l, nList, neighs[l] )
        nList = np.concatenate( [ nList, neighs[l] ] )
        for kk in done:
          ii = np.where( nList != kk )
          nList = nList[ii]
        # input("Next")
      grpId += 1
      nextGrp = np.any( grp == 0 )
    # No other groups
    # 
    # Remove groups with 1 element
    n = grpId-1
    """
    n = 0
    for g in range(1, nGrp+1):
      idx = np.where( grp == g )[0]
      nItems = idx.size
      if nItems == 1:
        print( "????", g, idx )
        input("Groop of size =1")
        grp[idx[0]] = 0
      else:
        n += 1
    """
    print("# of Groups ", n, grp)
    return n, grp 

def getFirstNeighbours( x, y, dx, dy ):
    eps = 1.0e-7
    neigh = []
    for i in range( x.shape[0]):
        # 9 neighbors
        # xMask = np.abs( x[i] - x ) <= ( (1.0 + eps) * (dx[i] + dx) )
        #yMask = np.abs( y[i] - y ) <= ( (1.0 + eps) * (dy[i] + dy) )
        # neigh.append( np.where( np.bitwise_and(xMask, yMask) ) ) 
        # 5 neighbors
        xMask0 = np.abs( x[i] - x ) <= ( (1.0 + eps) * (dx[i] + dx) ) 
        yMask0 = np.abs( y[i] - y ) <= (1.0 + eps) * dy[i]
        xMask1 = np.abs( x[i] - x ) <= (1.0 + eps) * dx[i]
        yMask1 = np.abs( y[i] - y ) <= ( (1.0 + eps) * (dy[i] + dy) )  
        neigh.append( np.where( np.bitwise_or(
            np.bitwise_and(xMask0, yMask0), 
            np.bitwise_and(xMask1, yMask1) ) )[0] ) 
    
    return neigh

def drawPads(ax, x0, y0, dx0, dy0, z, title="", yTitle="", alpha=1.0, doLimits=True, noYTicksLabels= False, hatchPattern=None):
    """
    Arguments:
      x, y : center of the cell
      dx, dy : scalar o r array[], half-size of the cell
    """
    global Scale, Colors
    # Shift to the  bottom/left point
    xbl = x0 - dx0
    ybl = y0 - dy0
    # Allocate & init dx if dx0/dy0 scalar
    if not isinstance( dx0, np.ndarray):
        dx = np.full( x0.shape, dx0)
    else:
        dx = dx0
    if not isinstance( dy0, np.ndarray):
        dy = np.full( y0.shape, dy0)
    else:
        dy = dy0
    # Size of the pads
    dx = 2*dx
    dy = 2*dy
    minZ = Scale[0]
    maxZ = Scale[1]
    if ( maxZ <= minZ ):
      maxZ = minZ + 1
    norm = (256.0 - 1.0) / (maxZ - minZ)
        
    idx = np.round( (z - minZ) * norm ).astype( int ) 
    Colors =getColorMap()
    c = Colors[ idx ]
    for r in range( xbl.shape[0] ):
          # print( "colorMap:", norm, clusters.charge[cID][r], clusters.charge[cID][r] * norm,  colors[ int( np.round( clusters.charge[cID][r]*norm ) ) ] )
          # rect = patches.Rectangle( (clusters.x[cID][r] - clusters.dx[cID][r], clusters.y[cID][r] - clusters.dy[cID][r]) ,2*clusters.dx[cID][r], 2*clusters.dy[cID][r],
          #  linewidth=1, edgecolor='b', facecolor=colors[ int( np.round( clusters.charge[cID][r] * norm ) ) ], alpha=0.5)
          # print( "c", r, c[r]
          rect = patches.Rectangle(  (xbl[r], ybl[r]) , dx[r], dy[r], linewidth=1, edgecolor='b', facecolor=c[r], 
            alpha= alpha, hatch=hatchPattern )
          ax.add_patch(rect)
    if doLimits:
      ax.set_xlim( np.min(xbl) ,  np.max(xbl+dx) )
      ax.set_ylim( np.min(ybl) ,  np.max(ybl+dy) )
    if noYTicksLabels:
      y = ax.get_yticks()
      ax.set_yticklabels([])
    if ( yTitle != ""):
      ax.set_ylabel(yTitle)

    ax.set_title(title)

if __name__ == "__main__":
    print ("Hello World");
