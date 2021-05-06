#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jlul 30, 2020 2:46:25 PM$"

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as pltColors

import Util.dataTools as dUtl

Colors = None
Scale = None
StopOnWarning = False


def getColorMap(N=256):
    global Colors
    cmap = np.linspace( 0.0, 1.0 - 1.0/N, N)
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

def getMCHitsInFrame( frame, mcObj, ev, DEIds ):
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
  return (x, y)
    
def drawMCHitsInFrame ( ax, frame, mcObj, ev, DEIds ):
    (x, y) = getMCHitsInFrame( frame, mcObj, ev, DEIds )
    drawPoints( ax, x, y, color='black', pattern='x')
    #
    return

def drawModelComponents( ax, theta, color='black', pattern="o" ):
    (w, muX, muY, varX, varY) = dUtl.thetaAsWMuVar( theta )
    K = w.shape[0]
    if K == 0 : return
    ( bot, top) = ax.get_ylim()
    wMax = np.max(w)
    wMin = np.min(w)
    cst = (top - bot) * 0.20 / wMax
    for k in range(K):
        if w[k] > 1.e-12 :
          x = muX[k]; y = muY[k]
          dx = np.sqrt( varX[k]); dy = np.sqrt( varY[k] )

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
            circle = patches.Circle(  (muX[k], muY[k]) , w[k]*cst, linewidth=1, edgecolor='black', facecolor=None, fill=False) 
                    # facecolor=c[r], 
            ax.add_patch(circle)
          else:
            ax.plot( x, y, pattern, color=color, markersize=3 )
            
    return

# Old version ???
def drawModelComponentsV0( ax, w, mu, var, color='black', pattern="o" ):
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

def drawPoints( ax, x, y, color='black', pattern="o" ):
    K = x.shape[0]
    if K == 0 : return
    ( bot, top) = ax.get_ylim()
    """
    wMax = np.max(w)
    wMin = np.min(w)
    cst = (top - bot) * 0.20 / wMax
    """
    """
    for k in range(K):
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
          """
    ax.plot( x, y, pattern, color=color, markersize=3 )
            
    return

def drawPads( fig, ax, x0, y0, dx0, dy0, z, title="", yTitle="", alpha=1.0, 
                doLimits=True, displayLUT=True, noYTicksLabels= False, hatchPattern=None):
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
    # LUT
    minZ = Scale[0]
    maxZ = Scale[1]
    norm = (256.0 - 1.0) / (maxZ - minZ)
    idx = np.round( (z - minZ) * norm ).astype( int ) 
    Colors =getColorMap()
    c = Colors[ idx ]
    if (displayLUT):
      # display a dummy image to initialize LUT
      dummy = np.linspace(minZ, maxZ, num=256).reshape(1, 256)
      pcm = ax.pcolormesh(dummy, cmap="rainbow")
      fig.colorbar(pcm, ax=ax)
    #
    for r in range( xbl.shape[0] ):
          # print( "colorMap:", norm, clusters.charge[cID][r], clusters.charge[cID][r] * norm,  colors[ int( np.round( clusters.charge[cID][r]*norm ) ) ] )
          # rect = patches.Rectangle( (clusters.x[cID][r] - clusters.dx[cID][r], clusters.y[cID][r] - clusters.dy[cID][r]) ,2*clusters.dx[cID][r], 2*clusters.dy[cID][r],
          #  linewidth=1, edgecolor='b', facecolor=colors[ int( np.round( clusters.charge[cID][r] * norm ) ) ], alpha=0.5)
          # print( "c", r, c[r]
          rect = patches.Rectangle(  (xbl[r], ybl[r]) , dx[r], dy[r], linewidth=1, edgecolor='b', facecolor=c[r], 
            alpha= alpha, hatch=hatchPattern )
          patch = ax.add_patch(rect)
    if doLimits:
      ax.set_xlim( np.min(xbl) ,  np.max(xbl+dx) )
      ax.set_ylim( np.min(ybl) ,  np.max(ybl+dy) )
    if noYTicksLabels:
      y = ax.get_yticks()
      ax.set_yticklabels([])
    if ( yTitle != ""):
      ax.set_ylabel(yTitle)
    """    
    cMap = ListedColormap(['white', 'green', 'blue','red'])  
    ax.pcolor(data, cmap=cMap)
    #####################
    im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    ##################
    """
    
    ax.set_title(title)
    # ??? Color bar
    # cfig = plt.gcf()
    # cfig.colorbar( plt.cm.ScalarMappable(norm=[Scale[0], Scale[1]], cmap="rainbow"), ax=ax)
    # cfig.colorbar( plt.cm.ScalarMappable(cmap="rainbow"), ax=ax)
    # cNorm = pltColors.Normalize(0, 10)
    # cfig.colorbar( plt.cm.ScalarMappable(norm=cNorm, cmap="rainbow"), ax=ax)
    # ax.colorbar( )
    # ??? ig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    return

def drawPads0(ax, x0, y0, dx0, dy0, z, title="", yTitle="", alpha=1.0, doLimits=True, noYTicksLabels= False, hatchPattern=None):
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
          patch = ax.add_patch(rect)
    if doLimits:
      ax.set_xlim( np.min(xbl) ,  np.max(xbl+dx) )
      ax.set_ylim( np.min(ybl) ,  np.max(ybl+dy) )
    if noYTicksLabels:
      y = ax.get_yticks()
      ax.set_yticklabels([])
    if ( yTitle != ""):
      ax.set_ylabel(yTitle)
    
    ax.set_title(title)
    # ??? Color bar
    cfig = plt.gcf()
    # cfig.colorbar( plt.cm.ScalarMappable(norm=[Scale[0], Scale[1]], cmap="rainbow"), ax=ax)
    # cfig.colorbar( plt.cm.ScalarMappable(cmap="rainbow"), ax=ax)
    # cNorm = pltColors.Normalize(0, 10)
    # cfig.colorbar( plt.cm.ScalarMappable(norm=cNorm, cmap="rainbow"), ax=ax)
    # ax.colorbar( )
    # ??? ig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    return