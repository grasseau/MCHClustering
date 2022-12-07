#!/usr/bin/env python3
#encoding: UTF-8

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#
import Util.csvSpec as csv
# import Util.IOv5 as IO
# Analyses Tool Kit
# import analyseToolKit as aTK

class Timing(csv.CSV):
  def __init__(self, fName):
    data = csv.CSV(fName, ['i', 'i', 'i', 'i', 'i','i','f'])
    self.preClusterID = data.fields[0]
    self.bCrossing = data.fields[1]
    self.orbit = data.fields[2]
    self.nPads = data.fields[3]
    self.nSeeds = data.fields[4]
    self.DEIds = data.fields[5]
    self.dt = data.fields[6]
    self.minNPads = np.min(self.nPads)
    self.maxNPads = np.max(self.nPads)
    self.minNSeeds = np.min(self.nSeeds)
    self.maxNSeeds = np.max(self.nSeeds)
    self.minDt = np.min(self.dt)
    self.maxDt = np.max(self.dt)
    print("min/max nPads", self.minNPads, " ", self.maxNPads);
    print("min/max nSeeds", self.minNSeeds, " ", self.maxNSeeds);
    print("min/max dt", self.minDt, " ", self.maxDt);
    # self.chIds = np.rint( self.DEIds / 100, dtype=np.intc)
    self.chIds = (self.DEIds / 100).astype(np.int)
    print( np.min( self.chIds), " ", np.max( self.chIds))
    self.stationIds = ((self.DEIds/ 200 -1)).astype(np.int) + 1
    #, dtype=np.int8)
    print( np.min( self.stationIds), " ", np.max( self.stationIds))
    print("Total time :", np.sum(self.dt))

def plotSeeds( reco, em, padSeedMax= (100, 70), diffLimits=(-10, 10)):

    # Plot 
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 7) )
    # 
    print(ax.shape)
    print(ax[0,:].size)
    compareSeeds( fig, ax[0,:], reco, em, padSeedMax= (100, 70), diffLimits=(-15, 15) )
    compareSeeds( fig, ax[1,:], reco, em, padSeedMax= ( 50, 25), diffLimits=(-7, 7) )
    
    plt.show()
    
def compareSeeds( fig, ax, reco, em, padSeedMax= (100, 70), diffLimits=(-10, 10)):
    nMaxPads = np.max(reco.nPads)
    nMaxSeeds = max(np.max(reco.nSeeds), np.max(em.nSeeds))

    # Filter 
    nMaxPads = padSeedMax[0]
    nMaxSeeds =  padSeedMax[1]
    # maskSt = ( reco.stationIds == 3 )
    maskSt = np.ones( reco.stationIds.size, dtype=np.bool_ )
    mask = np.bitwise_and( reco.nPads < nMaxPads, (reco.nPads > 1))
    maskPads = np.bitwise_and(mask, maskSt) 
    maskRecoSeeds = (reco.nSeeds < nMaxSeeds)
    maskEMSeeds = (em.nSeeds < nMaxSeeds)
    # print("shape ", maskPads.shape, maskRecoSeeds.shape, maskEMSeeds.shape)
    maskReco = np.bitwise_and( maskPads, maskRecoSeeds)
    maskEM = np.bitwise_and( maskPads, maskEMSeeds)
    #
    seedsReco = reco.nSeeds[maskReco]
    seedsEM =  em.nSeeds[maskEM]
    #
    # Difference
    minDiff = diffLimits[0]
    maxDiff = diffLimits[1]
    diffSeeds0 = reco.nSeeds - em.nSeeds
    maskDiff0 = np.bitwise_and( (diffSeeds0 > minDiff), (diffSeeds0 < maxDiff))
    maskDiff = np.bitwise_and( maskPads, maskDiff0)
    diffPads = reco.nPads[ maskDiff ]
    diffSeeds = diffSeeds0[maskDiff]
    
    # print( "pads/diffSeeds", diffPads.shape, diffSeeds.shape)
    # print( "diff size", diffSeeds.size, "min/max/", np.min(diffSeeds), np.max(diffSeeds))

    # Plot 
    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 7) )
    #
    
    # Boxes
    xed = np.arange( nMaxPads+1)
    xed = xed -0.5
    # print( nMaxPads, xed.size, xed[0], xed[-1])
    yed = np.arange( nMaxSeeds+1)
    yed = yed -0.5
    yeddiff =  np.arange( maxDiff-minDiff-2+2)
    yeddiff = yeddiff -0.5 + minDiff +1
    print( maxDiff-minDiff +1, yeddiff.size, yeddiff[0], yeddiff[-1])

    ax[0].set_title("histogram(reco seeds)")
    pads = reco.nPads[ maskReco ]
    h, xe, ye, img = ax[0].hist2d(pads, seedsReco,  bins=(xed,yed), norm=mpl.colors.LogNorm(), cmap=plt.cm.jet)
    fig.colorbar(img, ax=ax[0])
    ax[1].set_title("histogram(EM seeds)")
    pads = reco.nPads[ maskEM ]
    # print("pads/seeds shape ", pads.shape, seedsEM.shape)
    h, xe, ye, img = ax[1].hist2d(pads, seedsEM,  bins=(xed, yed), norm=mpl.colors.LogNorm(), cmap=plt.cm.jet)
    fig.colorbar(img, ax=ax[1])
    ax[2].set_title("histogram([EM - Reco] seeds)")
    h, xe, ye, img = ax[2].hist2d(diffPads, diffSeeds, bins=(xed, yeddiff), norm=mpl.colors.LogNorm(), cmap=plt.cm.jet)
    fig.colorbar(img, ax=ax[2])
    
    
def plot( reco, em):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 7) )
    #
    ax[0, 0].set_title("EMdt=f(Recodt)")
    ax[0, 0].plot( [0,np.max(reco.dt)], [0, np.max( reco.dt)], color='red')
    ax[0,0].plot( reco.dt, em.dt, '+')
    #
    """
    ax[0, 1].set_title("distrib. st=1")
    st=1
    mask = np.where( reco.stationIds == st)
    dt = reco.dt[mask]
    ax[0,1].hist(dt )
    """
    ax[0,1].set_title("diff Seeds=f(nPads)")
    #ax[0,1].plot( reco.nPads, reco.nSeeds, '+', color='blue')    
    #ax[0,1].plot( em.nPads, em.nSeeds, '+', color='red')    
    ax[0,1].plot( reco.nPads, (em.nSeeds - reco.nSeeds), '+', color='red')    
    #
    ax[1,0].set_title("dt = f(nPads)")
    ax[1,0].plot( reco.nPads, reco.dt, '+', color='blue')
    ax[1,0].plot( em.nPads, em.dt, '+', color='red')
    #
    ax[1,1].set_title("dt = f(nSeeds)")
    ax[1,1].plot( reco.nSeeds, reco.dt, '+', color='blue')    
    ax[1,1].plot( em.nSeeds, em.dt, '+', color='red')    
    # 
    ax[0,2].set_title("nPads = f(nPads)")
    ax[0,2].plot( reco.nPads, reco.nPads, '+', color='blue')    
    # ax[1,1].plot( em.nSeeds, em.dt, '+', color='red')    
    plt.show()
    
def plotStations( reco, em):
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(14, 7) )
    #
    #
    for st in range(1,6):
      mask = np.where( reco.stationIds == st)
      dtReco = reco.dt[mask]
      nPadsReco = reco.nPads[mask]
      mask = np.where( em.stationIds == st)
      dtEM = em.dt[mask]
      nPadsEM = em.nPads[mask]
      ax[0,st-1].set_title("dt=fct(nPads) st={}".format(st))
      ax[0,st-1].plot( nPadsReco, dtReco, '+', color='blue')
      ax[0,st-1].plot( nPadsEM, dtEM, '+', color='red') 
      n0 = np.count_nonzero( dtReco )
      n = dtReco.size - n0
      # yMax = max( np.max( dtReco), np.max( dtEM))
      yMax = max( np.max( nPadsReco), np.max( nPadsEM))
      xMax = np.max( nPadsReco)
      # Verif
      """
      ax[1,st-1].text( 0.5 * xMax, 0.25 * yMax,  "n0={} (Reco)".format(n0), ha='left')
      ax[1,st-1].text( 0.5 * xMax, 0.20 * yMax, "n={}".format(n), ha='left')
      n0 = np.count_nonzero( dtEM )
      n = dtEM.size - n0 
      ax[1,st-1].text( 0.5 * xMax, 0.15 * yMax, "n0={} (EM)".format(n0), ha='left')
      ax[1,st-1].text( 0.5 * xMax, 0.1 * yMax, "n={}".format(n), ha='left')
      #
      ax[1,st-1].plot( nPadsReco, nPadsEM, '+', color='blue')
      """
      # padsHisto = np.log10( nPadsReco[ np.where(dtReco > 0) ] )
      padsHisto = nPadsReco[ np.where(dtReco > 0) ]
      ax[1,st-1].hist( padsHisto, log=True )
      #
      ax[2,st-1].set_title("dtEM=fct(dtReco) st={}".format(st))
      ax[2,st-1].plot( [0,np.max( dtReco)], [0, np.max( dtReco)], color='red')
      ax[2,st-1].plot( dtReco, dtEM, '+', color='blue')
      
    """
    ax[0,1].set_title("nSeeds=f(nPads)")
    ax[0,1].plot( reco.nPads, reco.nSeeds, '+', color='blue')    
    ax[0,1].plot( em.nPads, em.nSeeds, '+', color='red')    
    #
    ax[1,0].set_title("dt = f(nPads)")
    ax[1,0].plot( reco.nPads, reco.dt, '+', color='blue')
    ax[1,0].plot( em.nPads, em.dt, '+', color='red')
    #
    ax[1,1].set_title("dt = f(nSeeds)")
    ax[1,1].plot( reco.nSeeds, reco.dt, '+', color='blue')    
    ax[1,1].plot( em.nSeeds, em.dt, '+', color='red')
    """  
    plt.show() 
    
def plotCumuls( reco, em, limits=[]):
    fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(14, 7) )
    #
    #
    if len(limits) != 2:
      padLimits=[0, np.max(reco.nPads)]
    else:
      padLimits = limits
    maxPads = padLimits[1]
    # All station
    allRecoDt = np.zeros( maxPads )
    allEMDt = np.zeros( maxPads )
    allNPreClusters = np.zeros(maxPads )
    # Per station
    cumulRecoDt = np.zeros( (5, maxPads), dtype=np.double)
    cumulEMDt = np.zeros( (5, maxPads), dtype=np.double)
    cumulNPreClusters = np.zeros( (5, maxPads), dtype=np.double)   
    cumulSumRecoDt = np.zeros( (5, maxPads), dtype=np.double)  
    cumulSumEMDt = np.zeros( (5, maxPads), dtype=np.double)  
    #
    sumTimeReco = np.zeros(6)
    sumTimeEM = np.zeros(6)
    sumPreClusters = np.zeros(6)
    maxCumulDt = 0.0
    maxCumulNPreClusters = 0
    maxCumulSumDt = 0
    CumulNPreClusters = 0
    for st in range(1,6):
      maskSt = (reco.stationIds == st)
      masknPads = np.bitwise_and((reco.nPads >= padLimits[0]), (reco.nPads < padLimits[1]))
      mask = np.bitwise_and(maskSt, masknPads)
      idx=np.where(mask)
      print("idx.shape ", idx[0].size)
      for i in idx[0]:
        cumulRecoDt[st-1, reco.nPads[i]] += reco.dt[i]
        cumulEMDt[st-1, reco.nPads[i]] += em.dt[i]
        cumulNPreClusters[st-1, reco.nPads[i]] += 1.0
        allRecoDt[reco.nPads[i]] += reco.dt[i]
        allEMDt[reco.nPads[i]] += em.dt[i]
        allNPreClusters[reco.nPads[i]] += 1.0
      cumulSumRecoDt[st-1] = np.cumsum( cumulRecoDt[st-1]) 
      cumulSumEMDt[st-1] = np.cumsum( cumulEMDt[st-1])
      maxCumulDt = max([maxCumulDt , np.max( cumulRecoDt[st-1]), np.max( cumulEMDt[st-1])] )
      maxCumulSumDt = max([maxCumulSumDt , np.max( cumulSumRecoDt[st-1]), np.max( cumulSumEMDt[st-1])] )
      maxCumulNPreClusters = max([CumulNPreClusters, np.max(cumulNPreClusters[st-1])] )

    # Limits
    maxCumulDt = maxCumulDt * 1.05  
    maxCumulSumDt = maxCumulSumDt * 1.05
    maxCumulNPreClusters = maxCumulNPreClusters * 2.0
    yMax0 = max( np.max(allRecoDt[padLimits[0]:padLimits[1]]), np.max(allEMDt[padLimits[0]:padLimits[1]]) )
    yMax0 = yMax0 * 1.05
    # yMax0 = max( np.max(cumulSumAllRecoDt[padLimits[0]:padLimits[1]]), np.max(cumulSumAllEMDt[padLimits[0]:padLimits[1]]) )
    for st in range(1,6):
      # Limits
      # np.max( allRecoDt[padLimits[0]:padLimits[1]])
      # ax[0,5].fill_between( x, allEMDt[padLimits[0]:padLimits[1]], color='red', alpha=0.5)
      x= np.arange(padLimits[0], padLimits[1])
      ax[0,st-1].set_title("st={}".format(st))
      ax[0,st-1].fill_between( x, cumulRecoDt[st-1, padLimits[0]:padLimits[1]], color='blue', alpha=0.5)
      ax[0,st-1].fill_between( x, cumulEMDt[st-1, padLimits[0]:padLimits[1]], color='red', alpha=0.5)
      ax[0,st-1].set_ylim([0, maxCumulDt])
      if st >= 2 and st <= 5 : 
        ax[0,st-1].set_yticklabels([])
        ax[1,st-1].set_yticklabels([])
        ax[2,st-1].set_yticklabels([])
      ax[0,st-1].set_xticklabels([])
      ax[1,st-1].set_xticklabels([])
      #ax[2,st-1].set_yticklabels([])  
      # ax[1,st-1].set_title("sumDt=cumul(nPads) st={}".format(st))
      ax[1,st-1].fill_between( x, cumulSumRecoDt[st-1, padLimits[0]:padLimits[1]], color='blue', alpha=0.5)
      ax[1,st-1].fill_between( x, cumulSumEMDt[st-1, padLimits[0]:padLimits[1]], color='red', alpha=0.5)
      ax[1,st-1].set_ylim([0, maxCumulSumDt])
      # ax[2,st-1].set_title("nPreCl=d(nPads) st={}".format(st))
      ax[2,st-1].plot( x, cumulNPreClusters[st-1, padLimits[0]:padLimits[1]], color=plt.cm.RdYlBu(st/5.0), alpha=1.0)
      ax[2,5].plot( x, cumulNPreClusters[st-1, padLimits[0]:padLimits[1]], color=plt.cm.RdYlBu(st/5.0), alpha=1.0)
      ax[2,st-1].set_yscale('log')
      ax[2,st-1].set_ylim([1, maxCumulNPreClusters])
      ax[2,st-1].set_xlabel(r'$n_{pads}$')

      sumTimeReco[st] = np.sum(cumulRecoDt)
      sumTimeEM[st] = np.sum(cumulEMDt)
      sumPreClusters[st] = np.sum(cumulNPreClusters)
      x= np.arange(padLimits[0], padLimits[1])
    # Cumul on all st
    ax[0,5].set_title("all st")
    ax[0,5].fill_between( x, allRecoDt[padLimits[0]:padLimits[1]], color='blue', alpha=0.5)
    ax[0,5].fill_between( x, allEMDt[padLimits[0]:padLimits[1]], color='red', alpha=0.5)
    ax[0,5].yaxis.tick_right()
    # ax[0,5].set_ylim([0, yMax0])
    cumulSumAllRecoDt = np.cumsum( allRecoDt) 
    cumulSumAllEMDt = np.cumsum( allEMDt) 
    # ax[1,5].set_title("sumDt=d(nPads) all")
    ax[1,5].fill_between( x, cumulSumAllRecoDt[padLimits[0]:padLimits[1]], color='blue', alpha=0.5)
    ax[1,5].fill_between( x, cumulSumAllEMDt[padLimits[0]:padLimits[1]], color='red', alpha=0.5)  
    ax[1,5].yaxis.tick_right()
    # ax[1,5].set_ylim([0, yMax0])
    # 
    # ax[2,5].set_title("nPreCl=d(nPads) All")
    # ax[2,5].plot( x, allNPreClusters[padLimits[0]:padLimits[1]], color='blue', alpha=1.0)
    ax[2,5].set_yscale('log')
    ax[2,5].set_ylim([1, maxCumulNPreClusters])
    ax[2,5].yaxis.tick_right()
    ax[2,5].set_xlabel(r'$n_{pads}$')

    for st in range(1,6):
      print("st={}, ttimeReco={:+.0f} ms, timeEM={:+.0f} ms, nPreclusters={:+.0f}".format(st, sumTimeReco[st], sumTimeEM[st], sumPreClusters[st]) )
    sReco = np.sum(sumTimeReco)/100. 
    sEM = np.sum(sumTimeEM)/100.
    sPreCl = np.sum(sumPreClusters)/100.
    ax[0,0].set_ylabel(r'$\sum t(n_{pads})$')
    ax[1,0].set_ylabel(r'Cumul: $\int_{0}^{n_{pads}}\sum t(n_{pads}) $')
    ax[2,0].set_ylabel(r'$ Distribution(n_{pads}) $')
    for st in range(1,6):
        print("st={}, ttimeReco={:+.0f} %, timeEM={:+.0f} %, nPreclusters={:+.0f} %".format(st, sumTimeReco[st]/sReco , sumTimeEM[st]/sEM , sumPreClusters[st]/sPreCl) )        
    plt.show()
    
def locateEvent( algo, stId, nPads):
  maskSt = (algo.stationIds == stId)
  mask = np.bitwise_and( maskSt, algo.nPads == nPads)
  idx = np.where( mask )[0]
  for i in idx :
    print("locate events st={:d} nPads={:d} => dt={:f}, nSeeds={:d} --- Index={:d} preId={:d} orbit={:d} bc={:d}".format( algo.stationIds[i], algo.nPads[i], 
    algo.dt[i], algo.nSeeds[i], i, algo.preClusterID[i], algo.orbit[i], algo.bCrossing[i]) )
  

if __name__ == "__main__":
  # runReco = csv.CSV('statistics.reco.csv', ['i', 'i', 'i', 'i', 'i','i','f'])
  # reco = Timing('statistics3.reco.csv')
  # Test avec reco-wf em = Timing('statistics3.reco-wf.csv')
  # em = Timing('statistics3.em.csv')
  #reco = Timing('statistics3.em.csv')
  #em = Timing('statistics.st45.csv')
  
  # 40 TF
  reco = Timing('statistics.orig.100tf.csv')
  em = Timing('statistics.em.100tf.csv')
  # plot( reco, em)
  # plotStations( reco, em)
  st = 3; npads = 89  
  locateEvent( em, st, npads)
  locateEvent( reco, st, npads)
  st = 4; npads = 82  
  locateEvent( em, st, npads)
  st = 5; npads = 62  
  locateEvent( em, st, npads)
  st = 5; npads = 64
  locateEvent( em, st, npads)
  plotCumuls( reco, em)
  plotCumuls( reco, em, limits=[0, 200])
  plotCumuls( reco, em, limits=[0, 100])
  #plotCumuls( reco, em, limits=[0, 20])
  plotCumuls( reco, em, limits=[0, 10])
  plotSeeds( reco, em)


  