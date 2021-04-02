#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jlul 30, 2020 2:46:25 PM$"

import sys

import numpy as np

StopOnWarning = False
   
def getPadBox( x, y, dx, dy ):
  xMin = np.min( x - dx)
  xMax = np.max( x + dx)
  yMin = np.min( y - dy)
  yMax = np.max( y + dy)
  return (xMin, xMax, yMin, yMax )

# ??? oldname def inBox( x, y, box ):
def getIndexesInBox( x, y, box ):
    flags = isInBox( x, y, box)
    return np.where( flags )

def isInBox( x, y, box ):
    (xMin, xMax, yMin, yMax ) = box
    flag0 = np.bitwise_and( (x >= xMin), (x < xMax) )
    flag1 = np.bitwise_and( (y >= yMin), (y < yMax) )
    flags = np.bitwise_and( flag0, flag1)
    return flags

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

# ??? Old to remove
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
    interJI = [ [] for j in range(x1.shape[0])]
    mapIJToK = {}
    k = 0
    for i in range(x0Inf.shape[0]):
      # jXInf = -1; jYInf = -1
      # jXSup = -1; jYSup = -1
      # XInf = maxFloat; XSup = - maxFloat
      # YInf = maxFloat; YSup = - maxFloat
      interIJ = []
      for j in range(x1Inf.shape[0]):
          xmin = max( x0Inf[i], x1Inf[j] )
          xmax = min( x0Sup[i], x1Sup[j] )
          xInter = ( xmin <= (xmax - epsilon) )
          ymin = max( y0Inf[i], y1Inf[j] )
          ymax = min( y0Sup[i], y1Sup[j] )
          yInter = ( ymin <= (ymax - epsilon))
          # Have an intersection
          if xInter and yInter:
            interIJ.append(j)
            interJI[j].append(i)
      if StopOnWarning and len( interIJ ) == 0:
        # No intersection, i remove
        input("No intersection, next")
        continue
      sumCh1Inter =  np.sum( ch1[ interIJ ])
      cst =  ch0[i] / sumCh1Inter
      #  New pads
      for j in interIJ :
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
        k += 1
    #
    newCh1 = [0.0 for i in range(len(newCh0))]
    for j in range( len(interJI) ):
      if len(interJI[j]) != 0 :
        sj = ch1[j] /  np.sum( ch0[ interJI[j] ])
        for l, i in enumerate( interJI[j] ):
          chji =  ch0[ interJI[j][l] ] * sj            
          newCh1[ mapIJToK[(i,j)] ] = chji 
    if len(newDX) != 0:
      dXMin = np.min( np.array(newDX) )
      dYMin = np.min( np.array(newDY) )
      if min( dXMin, dYMin ) < 10.0e-3:
        print( " dMin", min( dXMin, dYMin ) )
        print( newDX )
        print( newDY )
        input("Too thin")
    return np.array(newX), np.array(newY) , np.array(newDX), np.array(newDY), np.array(newCh0), np.array(newCh0)
    
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

