#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jlul 30, 2020 2:46:25 PM$"

import sys

import numpy as np
import Util.dataTools as uData

StopOnWarning = False
epsilon = 1.0e-04

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

def getFirstNeighboursInOneDir( x, y, dx, dy):
    eps = 1.0e-7
    neighI = []
    neighJ = []
    for i in range( x.shape[0]):
      xMask0 = np.abs( x[i] - x ) <= ( (1.0 + eps) * (dx[i] + dx) ) 
      yMask0 = np.abs( y[i] - y ) <= (1.0 + eps) * dy[i]            
      xMask1 = np.abs( x[i] - x ) <= (1.0 + eps) * dx[i]
      yMask1 = np.abs( y[i] - y ) <= ( (1.0 + eps) * (dy[i] + dy) )
        
      neighI.append( np.where( np.bitwise_and(xMask0, yMask0) )[0])
      neighJ.append( np.where( np.bitwise_and(xMask1, yMask1) )[0])
      #
    return neighI, neighJ

def getFirstNeighbours( x, y, dx, dy ):
    eps = 1.0e-5
    neigh = []
    for i in range( x.shape[0]):
        # 9 neighbors
        # xMask = np.abs( x[i] - x ) <= ( (1.0 + eps) * (dx[i] + dx) )
        #yMask = np.abs( y[i] - y ) <= ( (1.0 + eps) * (dy[i] + dy) )
        # neigh.append( np.where( np.bitwise_and(xMask, yMask) ) ) 
        # 5 neighbors
        # print("np.abs( x[i] - x ), (dx[i] + dx + eps)", np.abs( x[i] - x ), (dx[i] + dx + eps))
        xMask0 = np.abs( x[i] - x ) <= (dx[i] + dx + eps) 
        yMask0 = np.abs( y[i] - y ) <= (dy[i]+eps)
        xMask1 = np.abs( x[i] - x ) <= (dx[i] + eps)
        yMask1 = np.abs( y[i] - y ) <= ( dy[i] + dy +eps ) 
        # print("mask0 X", i, xMask0)
        # print("mask0 Y", i, yMask0)
        # print("mask0 X&Y", i, np.bitwise_and(xMask0, yMask0))
        # print("mask1 X", i, xMask1)
        # print("mask1 Y", i, yMask1)
        # print("mask1 X&Y", i, np.bitwise_and(xMask1, yMask1))

        neigh.append( np.where( np.bitwise_or(
            np.bitwise_and(xMask0, yMask0), 
            np.bitwise_and(xMask1, yMask1) ) )[0] ) 
    
    return neigh

def getFirstCrossNeighbours( x, y, dx, dy ):
    eps = 1.0e-7
    neigh = []
    for i in range( x.shape[0]):
        # 9 neighbors
        # xMask = np.abs( x[i] - x ) <= ( (1.0 + eps) * (dx[i] + dx) )
        #yMask = np.abs( y[i] - y ) <= ( (1.0 + eps) * (dy[i] + dy) )
        # neigh.append( np.where( np.bitwise_and(xMask, yMask) ) ) 
        # 5 neighbors
        xMask0 = ( np.abs( x[i] - x ) - (dx[i] + dx) )  <= eps   
        yMask0 = ( np.abs( y[i] - y ) - (dy[i] + dy) )  <= eps   
        # xMask0 = np.abs( x[i] - x ) <= ( (1.0 + eps) * (dx[i] + dx) ) 
        # yMask0 = np.abs( y[i] - y ) <= ( (1.0 + eps) * (dy[i] + dy) )  
        xMask1 = np.abs( x[i] - x ) <= (1.0 + eps) * dx[i]
        yMask1 = np.abs( y[i] - y ) <= (1.0 + eps) * dy[i]
        neigh.append( np.where( np.bitwise_or(
            np.bitwise_and(xMask0, yMask0), 
            np.bitwise_and(xMask1, yMask1) ) )[0] ) 
    
    return neigh

def laplacian2D( xyDxy, z ):
    (u, v, du, dv) = uData.asXYdXdY( xyDxy )  
    
    eps = 1.0e-7
    noise = 4. * 0.22875
    cutoff = noise 
    atLeastOneMax = -1
    # noise = 0
    neigh = getFirstNeighbours(u, v, du, dv)
    #
    # Laplacian in u direction
    nPads = u.size
    lapl = np.ones( u.size ) * (-1)
    unSelected = np.ones( u.size )
    qLissed = np.zeros( u.size )
    boundery = np.zeros( u.size )
    for i in range (u.size):
      neighI = neigh[i]
      # print("neighI", neighI)
      lapl[i] =  np.sum( z[neighI] <= ((z[i] + noise) * unSelected[neighI])) / ( neighI.size)
      qLissed[i] =  np.sum( z[neighI] ) / neighI.size
      if ( z[i] < cutoff ):
        lapl[i] = 0;
      unSelected[i] = (lapl[i] != 1)
      print("i, neighI, z[i], lapl[i]", i, z[i], neighI, lapl[i])

    print("Laplacian lapl", lapl)
    print("Laplacian qLissed", qLissed)
    locMax = np.where( lapl >= 1.0 )[0]

        
    #
    # Limit the number of loc. max if small nbr of pads
    if nPads != 0 :
      uInf = np.min( u - du)
      uSup = np.max( u + du)
      nU = int((uSup - uInf) / du[0] + eps)
      vInf = np.min( v - dv)
      vSup = np.max( v + dv)
      nV = int((vSup - vInf) / dv[0]  + eps)
      aspectRatio = min(nU, nV) / max(nU, nV)
    else:
      aspectRatio = 0
    if nPads < 6 and aspectRatio > 0.6 and locMax.size > 1:
      print ("Too many solutionslocMax, keep only one", locMax )
      idx = np.argmax(qLissed[ locMax ] )
      locMax = locMax[idx:idx+1]
      print ("locMax", locMax )
    # At least one pad
    if locMax.size == 0 and nPads != 0:
       locMax = np.array( [0])
    # At most
    if locMax.size <= (nPads + 1) / 3:
      n = np.floor( (nPads + 1) / 3.0 )
      locMax = np.sort( locMax )
      locMax  = locMax[0: int(n)]
    qLissedLocMax =  qLissed[ locMax ]      
    print("Laplacian locMax",locMax )
    return locMax, qLissedLocMax, qLissed


def laplacian1D( xyDxy, z ):
    (x, y, dx, dy) = uData.asXYdXdY( xyDxy )  
    
    # direction to process
    dxMin = np.min( dx )
    dyMin = np.min( dy )
    if dxMin < dyMin:
      u = x
      v = y
      du = dx
      dv = dy
      print("Laplacian x-direction")
    else:
      u = y
      v = x
      du = dy
      dv = dx
      print("Laplacian y-direction")
    # Neighbours in the direction
    eps = 1.0e-7
    neighU, neighV = getFirstNeighboursInOneDir(u, v, du, dv)
    #
    # Laplacian in u direction
    lapl = np.ones( u.size ) * (-1)
    qLissed = np.zeros( u.size )
    boundery = np.zeros( u.size )
    for i in range (u.size):
      neighI = neighU[i]
      # print("neighI", neighI)
      lapl[i] =  np.sum( (z[neighI] <= z[i]) ) / ( neighI.size)
      qLissed[i] =  np.sum( z[neighI] ) / neighI.size
      print("i, neighI, z[i], lapl[i]", i, z[i], neighI, lapl[i])
      """
      if neighI.size == 2 :
        if np.sum( boundery[ neighI ] > 0):
          # A neihbour is a boundery i.e the 2 pixel are isolated
          # Keep only one
          lapl[i] = 0
        boundery[i] = 1
      """
      """
      if neighI.size == 1 :
        # Isolated pad
        lapl[i] = 3
        qLissed[i] = z[i]
      else:
        lapl[i] =  np.sum( (z[neighI] <= z[i]) )
        qLissed[i] =  np.sum( z[neighI] )
        if neighI.size == 2 :
          edge[i] = 1
      """  
    # print("z",z)
    print("Laplacian lapl", lapl)
    print("Laplacian qLissed", qLissed)
    locMax = np.where( lapl >= 1.0 )[0]
    # print("locMax",locMax )
    
    # Filter remove same position in x or y
    keep = np.ones( locMax.size );
    for lidx, l in enumerate(locMax):
      for midx, m in enumerate(locMax[lidx+1:]):
        # print( "lidx, midx", lidx, midx)
        if ( (np.abs( u[l] - u[m]) < epsilon) and (np.abs(v[l]-v[m]) < (1+epsilon)* (dv[l]+dv[m])) ):
            # print( "same x/y", l, m, lidx, midx)
            if z[l] < z[m]:
              keep[lidx] = 0
              print( "Laplacian suppress l", l, lidx)
            else:
              keep[midx+lidx+1] = 0
              print( "Laplacian suppress m", m, midx)
    # print("keep", keep )
    locMax = locMax[ keep == 1] 
    #
    # Cross filter
    """
    neighCross = getFirstCrossNeighbours(u, v, du, dv)
    keep = np.ones( locMax.size );
    laplX = np.zeros( locMax.size );
    # for i in range (u.size):
    for lidx, l in enumerate(locMax):
      neigh = neighCross[l]
      print("neigh X", neigh)
      # print("neigh (z[neigh] <= z[l])", (z[neigh] <= z[l]))
      laplX[lidx] =  np.sum( (z[neigh] <= z[l]) ) / ( neigh.size)
      if laplX[lidx] < 1:
        keep[lidx] = 0
      # qLissed[i] =  np.sum( z[neighI] ) / neigh.size
      # print("i, neighI, z[i], lapl[i]", i, z[i], neighI, lapl[i])
    print("lapl X", laplX)
    locMax = locMax[ keep == 1] 
    """
    
    #
    # Fuse 2 local neigh which are neighbors
    
    keep = np.ones( locMax.size );
    print("??? u", u)
    print("??? v", v)
    print("??? z", z)
    print("??? locMax", locMax)
    for lidx, l in enumerate(locMax):
      if keep[lidx] :
        neighUV = np.hstack( [ neighU[l], neighV[l] ] )
        print("??? l / neighxy[l]", l, neighUV)
        for neigh in neighUV:
          if ( neigh != l ):
            print("2 locmax neighbours", l, neigh, np.where(locMax == neigh))
            aaa = np.where(locMax == neigh)[0]
            if aaa.size != 0:
              m = aaa[0]
              # Suppres the neighbours
              print( "Laplacian 2 locmax neighbours suppress l", neigh, m)
              keep[m] = 0
              ### Not used
              #  if zu[m] != -1:
              #    # Fuse the 2 pads
              #    u[l] = 0.5 * ( u[m] + u[l] )
              #    du[l] = ( du[m] + du[l] )
              #    # y and dy are the same
              #    zu[l] =  0.5 * (zu[l] + zu[m])  
              #    zu[m] = -1
              #    # remove pad
              #    # removePads[m] = 1
    locMax = locMax[ keep == 1] 
    
    print("Laplacian locMax",locMax )
    return locMax, qLissed

def laplacian1DOld( xyDxy, z ):
    (x, y, dx, dy) = uData.asXYdXdY( xyDxy )  
    
    # direction to process
    dxMin = np.min( dx )
    dyMin = np.min( dy )
    if dxMin < dyMin:
      u = x
      v = y
      du = dx
      dv = dy
    else:
      u = y
      v = x
      du = dy
      dv = dx
    # Neighbours in the direction
    eps = 1.0e-7
    nextIdx = np.ones( u.size, dtype=np.int ) * -1
    prevIdx = np.ones( u.size, dtype=np.int ) * -1
    lapl = np.ones( u.size ) * (-1)
    for i in range( u.size):
        # 9 neighbors
        # xMask = np.abs( x[i] - x ) <= ( (1.0 + eps) * (dx[i] + dx) )
        # yMask = np.abs( y[i] - y ) <= ( (1.0 + eps) * (dy[i] + dy) )
        # neigh.append( np.where( np.bitwise_and(xMask, yMask) ) ) 
        # 2 neighbors
        xMask0 = np.bitwise_and( (u[i] - u ) > (  eps ), (u[i] - u ) < ( (1.0 + eps) * (du[i] + du) ) )
        xMask1 = np.bitwise_and( (u - u[i] ) > ( eps ),  (u - u[i]) < ( (1.0 + eps) * (du[i] + du) ) )
        yMask0 = np.abs( v[i] - v ) < eps
        idx = np.where( np.bitwise_and(xMask1, yMask0))[0]
        # print("idx", idx)
        if idx.size == 1: 
          nextIdx[i] = idx[0]
        if idx.size == 0:
          nextIdx[i] = -1
        idx = np.where( np.bitwise_and(xMask0, yMask0))[0]
        # print("idx", idx)
        if idx.size == 1: 
          prevIdx[i] = idx[0]
        if idx.size == 0:
          prevIdx[i] = -1    
    # print( "nextIdx", nextIdx)
    # print( "prevIdx", prevIdx)
    for i in range (u.size):
      if prevIdx[i] != -1:
        lapl[i] += (z[ prevIdx[i] ] <= z[i])
      if nextIdx[i] != -1:
        lapl[i] += ( z[ nextIdx[i] ] <= z[i])
    # print("z",z)
    # print("lapl", lapl)
    locMax = np.where( lapl == 1 )[0]
    # print("locMax",locMax )
    
    # Filter remove same position in x or y
    keep = np.ones( locMax.size );
    for lidx, l in enumerate(locMax):
      for midx, m in enumerate(locMax[lidx+1:]):
        # print( "lidx, midx", lidx, midx)
        if np.abs( u[l] - u[m]) < epsilon:
            # print( "same x/y", l, m, lidx, midx)
            if z[l] < z[m]:
              keep[lidx] = 0
              print( "suppress l", l, lidx)
            else:
              keep[midx+lidx+1] = 0
              print( "suppress m", m, midx)
    # print("keep", keep )
    locMax = locMax[ keep == 1] 
            
    print("locMax",locMax )
    return locMax

def findLocalMax( xyDxy0, xyDxy1, q0, q1 ):
    eps = 1.0e-8
    localMaxIdx = []
    localXMax = []
    localYMax = []

    (x0, y0, dx0, dy0) = uData.asXYdXdY( xyDxy0 )  
    (x1, y1, dx1, dy1) = uData.asXYdXdY( xyDxy1 )
    (xProj, yProj , dxProj, dyProj, qProj0, qProj1, mapIJToK, mapKToIJ, interIJ, interJI) = \
      shorteningPads( x0, y0, dx0, dy0, q0, x1, y1, dx1, dy1, q1)

    maxCath0, q0LissedLocMax, q0Liss = laplacian2D( xyDxy0, q0)
    maxCath1, q1LissedLocMax, q1Liss = laplacian2D( xyDxy1, q1)
    
    print("findLocalMax q0", q0)
    print("findLocalMax q1", q1)
    print("findLocalMax maxCath0", maxCath0)
    print("findLocalMax maxCath1", maxCath1)

    # input("next")

    # Sort the local max
    locMaxVal0 = q0Liss[maxCath0]
    locMaxVal1 = q1Liss[maxCath1]
    idx0 = np.argsort( -locMaxVal0 ) 
    idx1 = np.argsort( -locMaxVal1 ) 
    maxCath0 = maxCath0[idx0]
    maxCath1 = maxCath1[idx1]
    q0LissedLocMax = q0LissedLocMax[idx0]
    q1LissedLocMax = q1LissedLocMax[idx1]
    print( "q0Liss", q0Liss)
    print( "q1Liss", q1Liss)
    print( locMaxVal0,  locMaxVal1)
    print( "maxCath0", maxCath0)    
    print( "maxCath1", maxCath1)    

    # Select the cathode 
    if maxCath0.size < maxCath1.size:
      maxCathu = maxCath1
      maxCathv = maxCath0
      xu = x1; dxu = dx1
      yu = y1; dyu = dy1
      xv = x0; dxv = dx0
      yv = y0; dyv = dy0
      qu = q1Liss
      qv = q0Liss
      qvLissedLocMax = q0LissedLocMax
      interUV = interJI
      # qvAvailable = np.ones( q0.size )
      order = "Cath1 / Cath0"
      cath0CathU = False
    elif maxCath1.size < maxCath0.size:
      maxCathu = maxCath0
      maxCathv = maxCath1
      xu = x0; dxu = dx0
      yu = y0; dyu = dy0
      xv = x1; dxv = dx1
      yv = y1; dyv = dy1
      # qu = q0
      qu = q0Liss
      qv = q1Liss
      qvLissedLocMax = q1LissedLocMax
      interUV = interIJ
      # qvAvailable = np.ones( q1.size )
      order = "Cath1 / Cath0"
      cath0CathU = True
    else:
      # Same numer of loc max on both cath
      # Choose the max of the last loc. max.
      n = maxCath0.size
      if (maxCath0[-1] < maxCath1[-1]):
        maxCathu = maxCath1
        maxCathv = maxCath0
        xu = x1; dxu = dx1
        yu = y1; dyu = dy1
        xv = x0; dxv = dx0
        yv = y0; dyv = dy0
        qu = q1Liss
        qv = q0Liss
        qvLissedLocMax = q0LissedLocMax
        interUV = interJI
        # qvAvailable = np.ones( q0.size )
        order = "Cath1 / Cath0, equal nbr"        
        cath0CathU = False
      else:
        maxCathu = maxCath0
        maxCathv = maxCath1
        xu = x0; dxu = dx0
        yu = y0; dyu = dy0
        xv = x1; dxv = dx1
        yv = y1; dyv = dy1
        qu = q0Liss
        qv = q1Liss
        qvLissedLocMax = q1LissedLocMax
        interUV = interIJ
        # qvAvailable = np.ones( q1.size )
        order = "Cath0 / Cath1, equal nbr "
        cath0CathU = True
    #
    qvAvailable = np.ones( maxCathv.size )

    print( "findLocalMax cath0/1.size", maxCath0.size, maxCath1.size)
    print( "findLocalMax order", order)      
    #
    for ii,i in enumerate(maxCathu):
      print( "MaxCatU ", ii+1, "/", maxCathu.size, ", index=", i)
      print("qu", qu)
      print("qAvail", qvAvailable)

      # print("??? interIJ[i]", interUV[i] )
      interU = set( interUV[i] )
      inter = interU & set( maxCathv )
      # print("interU", interU)
      # print("set( maxCathv )", set( maxCathv ))
      # print("inter", inter)
      #
      # Build the set of pad index wich are in  maxCathv
      maxValue = 0.0
      j = -1
      maxCathvIdx = -1
      print(" maxCathv", maxCathv)
      print(" qvLissedLocMax", qvLissedLocMax)
      print(" qv[ maxCathv]", qv[ maxCathv])
      for ii in inter:
        idx = np.where(maxCathv == ii)[0]
        if idx.size == 1: 
          iii =idx[0]
          val = qvLissedLocMax[iii] * qvAvailable[iii]
          if ( val > maxValue):
            maxValue = val
            maxCathvIdx = iii
            # pad index
            j =  maxCathv[iii]

        # idx.append( np.where(maxCathv == ii)[0])
        # idx_size = idx_size + np.where(maxCathv == ii)[0].size
      print("  maxCathvIdx, maxValue, maxCathv[maxCathvIdx]", maxCathvIdx, maxValue, j ) 
      print("  maxCathvIdx maxCathv", maxCathvIdx, maxCathv )
      # Take the maximum
      """
      if idx_size > 0:
        idx = np.hstack( idx )
        print("idx maxCathv", idx, maxCathv )
        # print( " ", interUV[j])
        print("  Max possible in maxCathv[idx]", maxCathv[idx] )
        maxIdx = np.argmax( qvLissedLocMax[idx] * qvAvailable[idx] )
        print( "  maxIdx", maxIdx)
        print("maxCathv[idx][maxIdx]", maxCathv[idx[maxIdx]])
        print("   max possible Valuex",  qvLissedLocMax[idx] * qvAvailable[idx] )
        maxValue = (qvLissedLocMax[idx] * qvAvailable[idx])[maxIdx]
        print("   max value", (qvLissedLocMax[idx] * qvAvailable[idx])[maxIdx])
        print("   max index", (maxCathv[idx][maxIdx]))
        print("   max index", (maxCathv[idx[maxIdx]]))
        j = maxCathv[idx[maxIdx]]
        verifIdx = np.where(  qvLissedLocMax == maxValue)[0]
        print("verifIdx", verifIdx)
        # maxIdx = idx[maxIdx]
        maxIdx = verifIdx[0]
        print("  found max in Cathv", maxIdx, maxCathv)
        print("  qv",  qv)
      """
      # if idx_size > 0 and qvAvailable[maxIdx] != 0:
      if j != -1:
        #print("  selected max in Cathv", maxIdx, maxCathv)
        # j = maxCathv [ maxIdx ]
        print("    i,j", i, j)
        # Mapping found between the two cathode
        if cath0CathU:
          k = mapIJToK[(i, j)]
        else:
          k = mapIJToK[(j, i)]
            
        localXMax.append( xProj[k] )
        localYMax.append( yProj[k] )
        qvAvailable[ maxCathvIdx ] = 0
      else:
        print("Approximate solution")
        idx = interUV[i]
        if xv.size != 0 and len(idx) !=0:
          if (dxu[i] < dyu[i]):
            print("??? xv yv", xv, yv, dyv, idx)
            vMin = np.min( yv[idx] - dyv[idx] )
            vMax = np.max( yv[idx] + dyv[idx] )
            localXMax.append( xu[i] )
            localYMax.append( 0.5*(vMin+vMax) )
            # localYMax.append( yu[i] )
          else:
            vMin = np.min( xv[idx] - dxv[idx] )
            vMax = np.max( xv[idx] + dxv[idx] )
            localXMax.append( 0.5*(vMin+vMax) )
            localYMax.append( yu[i] )
        else:
          # No loc. max in cathv
          localXMax.append( xu[i] )
          localYMax.append( yu[i] )

            # localXMax.append(  )
        """
        if np.sum( maskUiV ) != 0:
        # Have an intersection with other Loc Max
          for j, m in enumerate(maskUiV):
            # j = 0
            # m = maskUiV[0]
            if m :
              localMaxIdx.append( (i, maxCathv[j]) )
              x = 0.5 * (xSup[j] + xInf[j])
              y = 0.5 * (ySup[j] + yInf[j])
              localXMax.append( x )
              localYMax.append( y )
              # print("??? maxCathv, qvAvailable.size", maxCathv, i, qvAvailable.size)
              print( "new seed with loc. max", x, y)
              qvAvailable[ maxCathv[j] ] = 0 
              mm = np.ones( maxCathv.size )
              mm[j] = 0 
              maxCathv = maxCathv[ mm == 1]
              break
        else:
          # No intersection with loc Max, find an
          # other possible maximun:
          # - no already chosen as loc Max, and 
          # - intersecting i loc max
          #
          # print("InterIJ", interUV[i] )
          # print( "qv", qv)
          interuv = np.zeros( qv.size )
          # Set all intersection with i loc Max
          interuv[interUV[i]] = 1
          idx = np.argsort( - qv * qvAvailable * interuv)
          print( "qv filtered", qv * qvAvailable * interuv)
          for ii in idx[0:1]:
            if dxu[i] < dxv[ii]:
              x = xu[i]; y = yv[ii];
            else:
              x = xv[ii]; y = yu[i];
            localXMax.append( x )
            localYMax.append( y )
            print( "new seed with a max", x, y, dxv[ii], dyv[ii], qv[ii])
            qvAvailable[ ii ] = 0 
          
          # input("next")
        """
        #
    #
    # Process to unselected maxCathv
    for l, todo in enumerate(qvAvailable):
      if todo:
        localXMax.append( xv[ maxCathv[l] ] )
        localYMax.append( yv[ maxCathv[l] ] )        
        
    x = np.array( localXMax )
    y = np.array( localYMax )
    return ( x, y)
    
def findLocalMax0( xyDxy0, xyDxy1, q0, q1 ):
    eps = 1.0e-8
    localMaxIdx = []
    localXMax = []
    localYMax = []

    (x0, y0, dx0, dy0) = uData.asXYdXdY( xyDxy0 )  
    (x1, y1, dx1, dy1) = uData.asXYdXdY( xyDxy1 )
    (xProj, yProj , dxProj, dyProj, qProj0, qProj1, mapIJToK, mapKToIJ, interIJ, interJI) = \
      shorteningPads( x0, y0, dx0, dy0, q0, x1, y1, dx1, dy1, q1)

    maxCath0, q0Liss = laplacian1D( xyDxy0, q0)
    maxCath1, q1Liss = laplacian1D( xyDxy1, q1)
    
    print("findLocalMax maxCath0", maxCath0)
    print("findLocalMax maxCath1", maxCath1)
    print("findLocalMax q0", q0)
    print("findLocalMax q1", q1)
    # input("next")

    # Sort the local max
    locMaxVal0 = q0Liss[maxCath0]
    locMaxVal1 = q1Liss[maxCath1]
    idx0 = np.argsort( -locMaxVal0 ) 
    idx1 = np.argsort( -locMaxVal1 ) 
    maxCath0 = maxCath0[idx0]
    maxCath1 = maxCath1[idx1]
    # print( locMaxVal0,  locMaxVal1)

    if maxCath0.size < maxCath1.size:
      maxCathu = maxCath1
      maxCathv = maxCath0
      xu = x1; dxu = dx1
      yu = y1; dyu = dy1
      xv = x0; dxv = dx0
      yv = y0; dyv = dy0
      qu = q1
      qv = q0
      interUV = interJI
      qvAvailable = np.ones( q0.size )
    elif maxCath1.size < maxCath0.size:
      maxCathu = maxCath0
      maxCathv = maxCath1
      xu = x0; dxu = dx0
      yu = y0; dyu = dy0
      xv = x1; dxv = dx1
      yv = y1; dyv = dy1
      # qu = q0
      qu = q0Liss
      qv = q1Liss
      interUV = interIJ
      qvAvailable = np.ones( q1.size )
    else:
      # Same numer of loc max on both cath
      n = maxCath0.size
      if (maxCath0[-1] < maxCath1[-1]):
        maxCathu = maxCath1
        maxCathv = maxCath0
        xu = x1; dxu = dx1
        yu = y1; dyu = dy1
        xv = x0; dxv = dx0
        yv = y0; dyv = dy0
        qu = q1Liss
        qv = q0Liss
        interUV = interJI
        qvAvailable = np.ones( q0.size )
      else:
        maxCathu = maxCath0
        maxCathv = maxCath1
        xu = x0; dxu = dx0
        yu = y0; dyu = dy0
        xv = x1; dxv = dx1
        yv = y1; dyv = dy1
        qu = q0Liss
        qv = q1Liss
        interUV = interIJ
        qvAvailable = np.ones( q1.size )
      
    for i in maxCathu:
      xInf0 = xu[i] - dxu[i]
      xSup0 = xu[i] + dxu[i]
      yInf0 = yu[i] - dyu[i]
      ySup0 = yu[i] + dyu[i]
      print( "cathu xInf/Sup, yInf/Sup", xInf0, xSup0, yInf0, ySup0)
      # Intersection with pad i and cath1
      xInf1 = xv[maxCathv] - dxv[maxCathv]
      xSup1 = xv[maxCathv] + dxv[maxCathv]
      yInf1 = yv[maxCathv] - dyv[maxCathv]
      ySup1 = yv[maxCathv] + dyv[maxCathv]
      """
      print("xInf0", np.array([xInf0]) )
      print("xInf1", xInf1 )
      xInf0bc = np.ones( xInf1.size) * xInf0
      xSup0bc = np.ones( xInf1.size) * xSup0
      xInf0bc = np.ones( xInf1.size) * xInf0
      xSup0bc = np.ones( xInf1.size) * xSup0
      print("xInf0", xInf0 )
      print("xInf1", xInf1)
      """
      xInf = np.maximum ( xInf0, xInf1) 
      xSup = np.minimum ( xSup0, xSup1) 
      yInf = np.maximum (yInf0, yInf1) 
      ySup = np.minimum (ySup0, ySup1)   
      
      """      
      print("----" )
      print("xInf0", xInf0 )
      print("xInf1", xInf1 )
      print("xSup0", xSup0 )
      print("xSup1", xSup1 )
      print("yInf0", yInf0 )
      print("yInf1", yInf1 )
      print("ySup0", ySup0 )
      print("ySup1", ySup1 )
      
      print("----" )
      print("xInf", xInf )
      print("xSup", xSup )
      print("yInf", yInf )
      print("ySup", ySup )
      
      print( "mask x",  (xInf)  < (xSup - eps) )
      print( "mask y", (yInf) < (ySup - eps) )
      print( "mask x",  np.abs( xSup - xInf ) < eps )
      print( "mask y", np.abs( ySup - yInf ) < eps )
      """
      # mask =  np.bitwise_and( np.abs( xSup - xInf ) < eps, np.abs( ySup - yInf ) < eps )
      mask =  np.bitwise_and( (xInf)  < (xSup - eps) ,  (yInf) < (ySup - eps) )
      print( "xInf", xInf)
      print( "xSup", xSup)
      print( "mask", mask)
      
      if np.sum( mask ) != 0:
      # Have an intersection with other Loc Max
        for j, m in enumerate(mask):
          # j = 0
          # m = mask[0]
          if m :
            localMaxIdx.append( (i, maxCathv[j]) )
            x = 0.5 * (xSup[j] + xInf[j])
            y = 0.5 * (ySup[j] + yInf[j])
            localXMax.append( x )
            localYMax.append( y )
            # print("??? maxCathv, qvAvailable.size", maxCathv, i, qvAvailable.size)
            print( "new seed with loc. max", x, y)
            qvAvailable[ maxCathv[j] ] = 0 
            mm = np.ones( maxCathv.size )
            mm[j] = 0 
            maxCathv = maxCathv[ mm == 1]
            break
      else:
        # No intersection with loc Max, find an
        # other possible maximun:
        # - no already chosen as loc Max, and 
        # - intersecting i loc max
        #
        # print("InterIJ", interUV[i] )
        # print( "qv", qv)
        interuv = np.zeros( qv.size )
        # Set all intersection with i loc Max
        interuv[interUV[i]] = 1
        idx = np.argsort( - qv * qvAvailable * interuv)
        print( "qv filtered", qv * qvAvailable * interuv)
        for ii in idx[0:1]:
          if dxu[i] < dxv[ii]:
            x = xu[i]; y = yv[ii];
          else:
            x = xv[ii]; y = yu[i];
          localXMax.append( x )
          localYMax.append( y )
          print( "new seed with a max", x, y, dxv[ii], dyv[ii], qv[ii])
          qvAvailable[ ii ] = 0 
        
        # input("next")
      #
    #
    x = np.array( localXMax )
    y = np.array( localYMax )
    return ( x, y)
    
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