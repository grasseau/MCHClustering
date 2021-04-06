#! /usr/bin/python

__author__ = "grasseau"
__date__ = "$Jul 12, 2020 9:56:07 AM$"

import sys, traceback
import struct
import numpy as np

def readInt4(file, n):
    
    if ( n == 0 ): return np.zeros( (0), dtype=np.int32 )
    #
    # Read Nbr of items (8 bytes)
    raw = file.read(2 * 4)
    nData = struct.unpack('q', raw)[0]
    # print str(raw[0]).encode("hex") ,'-', str(raw[1]).encode("hex")
    if nData != n: 
        print ("Expected/Read number of values are different ", n, "/", nData); 
        # traceback.print_stack(); exit()
        getEOF(file)
        # return 
    
    # Read N Int32
    raw = file.read(nData * 4)
    array = struct.unpack(str(nData) + 'i', raw)
    if nData != n: 
      print("Expected/Read number of values are different ", n, "/", nData); 
      print( "raw", raw )
      print( "array", array )
      traceback.print_stack(); 
      exit()
    #
    return np.array(array, dtype=np.int32)

def readDouble(file, n):
    if ( n == 0 ): return np.zeros( (0), dtype=np.float64 )
    #
    # Read Nbr of items (8 bytes)
    raw = file.read(2 * 4)
    nData = struct.unpack('q', raw)[0]
    # print str(raw[0]).encode("hex") ,'-', str(raw[1]).encode("hex")

    # if nData != n: print("Expected/Read number of values are different ", n, "/", nData); traceback.print_stack(); exit()

    # Read N Double
    raw = file.read(nData * 8)
    array = struct.unpack(str(nData) + 'd', raw)
    if nData != n: 
      print("Expected/Read number of values are different ", n, "/", nData); 
      print( "raw", raw )
      print( "array", array )
      traceback.print_stack(); 
      exit()
    #
    return np.array(array, dtype=np.float64)

def getEOF(file, errorStr="", mode="quiet"):
    k = len(file.read(8))
    EOF = (k != 8)
    if (mode == "verbose" and EOF):
        print( "Warning: EOF reached ", errorStr, k, "bytes read", )
    file.seek(-k, 1)
    return EOF

class MCData:
  """
  loop on events
        //
        // Tracks
        //
        Int_t trackListHeader[] = { -1, event, -1, -1, 0, nTracks };        
        Loop on nTracks
          Int_t trackHeader[] = { -1, trackIdx, mcLabel, partCode, 0, nbrOfMCClusters };
          Double_t partInfo[] = {particle->Xv(), particle->Yv(), particle->Zv(), particle->Px(), particle->Py(), particle->Pz()};
          Double_t xc[nbrOfMCClusters], yc[nbrOfMCClusters]
          Int_t chamberId[nbrOfMCClusters]
          Int_t DEId[nbrOfMCClusters]
        //
        // Pads
        //
        Loop on Pads
          Int_t padHeader = { -1, event, preClusterIdx, -1, 0, nPads };
          Double_t xPad[ nPads ], yPad[ nPads ]
          Double_t dxPad[ nPads ], dyPad[ nPads ]: pad size
          Int_t padId[ nPads ], 
          Int_t DEId[ nPads ], 
          Int_t cath[ nPads ];
          Int_t padADC[ nPads ];
          Int_t hit[ nPads ];
          Int_t isSaturated[ nPads ], 
          Int_t isCalibrated[ nPads ];
          Int_t nTracks[ nPads ];
          Double_t trackCharges[ nTrack[ipad] ];
          Int_t    trackId[ nTrack[ipad] ];
  """
  def __init__(self, fileName="MCDataDump.dat"):
    #
    self.fileName = fileName
    self.file = open(fileName, 'rb')
    
    self.evId = []
    #
    # Tracks or MC Clusters
    #
    # Data dimension [ev][trIdx][nbrOfMCClusters]
    # Int_t trackHeader[] = { -1, event, mcLabel, partCode, 0, nbrOfMCClusters };
    self.trackKinetics = [] # {particle->Xv(), particle->Yv(), particle->Zv(), particle->Px(), particle->Py(), particle->Pz()};
    self.trackId = [] # [nbrOfMCClusters]
    self.trackParticleId = [] # [nbrOfMCClusters]
    self.trackX = [] # [nbrOfMCClusters]
    self.trackY = [] # [nbrOfMCClusters]
    self.trackChId = [] # [nbrOfMCClusters]
    self.trackDEId = [] # [nbrOfMCClusters]
    self.trackCharge = []
    #
    # MC Pads
    #
    # Data dimension [ev][nbrOfPads]
    self.padX = []
    self.padY = []
    self.padDX = []
    self.padDY = []
    self.padId = [] 
    self.padDEId = [] 
    self.padChId = [] 
    self.padCath = []
    self.padADC = []
    self.padHit = []
    self.padSaturated = [] 
    self.padCalibrated = []
    self.padNbrTracks = []
    self.padTrackCharges = []
    self.padTrackId = []
    self.padCharge = []
    self.padTrackIdx = []
    self.padChargeMax = []
    #
    return

  def readMCTrackData(self, ev):
    # New event
    # Read header
    # Int_t trackInfo[] = { -1, event, -1, -1, 0, nbrOfTracks };
    headerSize = 6
    trackListHeader = readInt4(self.file, headerSize)
    if (trackListHeader[0] != -1):
        print("readMCTrackData: bad trackListHeader", trackListHeader[0])
        exit()
    nbrOfTracks = trackListHeader[5]
    evId = trackListHeader[1]
    
    # New event
    self.evId.append(evId)
    self.trackKinetics.append([])
    self.trackId.append([])
    self.trackParticleId.append([])
    self.trackKinetics.append([])
    self.trackX.append([])
    self.trackY.append([])
    self.trackChId.append([])
    self.trackDEId.append([])
    self.trackCharge.append([])
    self.padX.append([])
    self.padY.append([])
    self.padDX.append([])
    self.padDY.append([])
    self.padId.append([]) 
    self.padDEId.append([]) 
    self.padChId.append([]) 
    self.padCath.append([])
    self.padADC.append([])
    self.padHit.append([])
    self.padSaturated.append([]) 
    self.padCalibrated.append([])
    self.padNbrTracks.append([])
    self.padTrackCharges.append([])
    self.padTrackId.append([])
    self.padCharge.append([])
    self.padTrackIdx.append([])
    self.padChargeMax.append(0.0)

    # Track part
    for t in range(nbrOfTracks):
        # Int_t trackInfo[] = { -1, trackIdx, mcLabel, partCode, 0, nbrOfClusters };
        trackHeader = readInt4(self.file, headerSize)
        # print( "??? t, trackheader", t , trackHeader)
        if (trackHeader[0] != -1):
            print("readMCTrackData: bad trackHeader", t, trackHeader[0])
            exit()
        mcLabel = trackHeader[2]
        partCode = trackHeader[3] 
        nbrOfClusters = trackHeader[5]
        self.trackId[ev].append(mcLabel)
        self.trackParticleId[ev].append(partCode)
        self.trackKinetics[ev].append(readDouble(self.file, headerSize))
        self.trackX[ev].append (readDouble(self.file, nbrOfClusters))
        self.trackY[ev].append (readDouble(self.file, nbrOfClusters))
        self.trackChId[ev].append (readInt4(self.file, nbrOfClusters))
        # Incoherent ChId !!! correcting it
        self.trackChId[ev][-1] = self.trackChId[ev][-1] + 1
        # print ("min/max trackChId", np.min(self.trackChId[ev][-1]), np.max(self.trackChId[ev][-1]))
        self.trackDEId[ev].append (readInt4(self.file, nbrOfClusters))
        self.trackCharge[ev].append( np.zeros( (nbrOfClusters), dtype = np.float ) )
    #
    # Read MC pads
    #
    # print("??? pads")
    padHeader=readInt4(self.file, headerSize)
    if (padHeader[0] != -1):
      print("readMCTrackData: bad padHeader", padHeader[0])
      exit()
    nbrOfPads=padHeader[5]
    # print("pads ???", nbrOfPads)
    self.padX[ev]=readDouble(self.file, nbrOfPads)
    self.padY[ev]=readDouble(self.file, nbrOfPads)
    self.padDX[ev]=readDouble(self.file, nbrOfPads)
    self.padDY[ev]=readDouble(self.file, nbrOfPads)
    self.padId[ev]=readInt4(self.file, nbrOfPads)
    self.padDEId[ev]=readInt4(self.file, nbrOfPads)
    self.padChId[ev] = self.padDEId[ev] // 100 
    # print ("min/max padChId", np.min(self.padChId[ev]), np.max(self.padChId[ev]))
    self.padCath[ev]=readInt4(self.file, nbrOfPads)
    self.padADC[ev]=readInt4(self.file, nbrOfPads)
    self.padHit[ev]=readInt4(self.file, nbrOfPads)
    self.padSaturated[ev]=readInt4(self.file, nbrOfPads)
    self.padCalibrated[ev]=readInt4(self.file, nbrOfPads)
    self.padNbrTracks[ev]=readInt4(self.file, nbrOfPads)
    n=np.sum(self.padNbrTracks[ev])
    print("  nTracks", n )
    self.padTrackCharges[ev]=readDouble(self.file, n)
    self.padTrackId[ev]=readInt4(self.file, n)
    # Compute Pad Charge
    padCharge = []
    idx = []
    nPads = self.padNbrTracks[ev].shape[0]
    i = 0   # Index for padNbrTracks 
    k = 0   # Index for padTrackCharges, padTrackId
    for i in range(nPads):
        inc = self.padNbrTracks[ev][i]
        c = 0.0
        for l in range(k, k+inc):
            q = self.padTrackCharges[ev][l]
            c += q
            # ??? self.track[ev][chId]
        idx.append(k)
        k += inc
        padCharge.append( c )
    self.padCharge[ev] = np.array( padCharge )
    #
    # Compute track Charge (deposit) in a DE 
    idxToTid = np.array( self.trackId[ev] ) 
    k=0
    for i in range(nPads):
        inc = self.padNbrTracks[ev][i]
        deid = self.padDEId[ev][i]
        for l in range(k, k+inc):
            # Pad infos
            tid = self.padTrackId[ev][l]
            c   = self.padTrackCharges[ev][l]
            # Track Hit impacted (tid & deid)
            idxOfTrackId = np.where( idxToTid == tid )
            for tidx in list( idxOfTrackId[0] ) :
              deidx = (self.trackDEId[ev][tidx] == deid)
              """
              both = np.bitwise_and( flag0, flag1 )
              idx = np.where( both )
              np.sum( both )
              """
              self.trackCharge[ev][tidx][deidx] += c
            # ??? self.track[ev][chId]
        k += inc
    #
    self.padChargeMax[ev] = np.max( self.padCharge[ev] )
    self.padTrackIdx[ev] = np.array( idx, dtype = np.int )

    print( "track nbr", k, n)
    
    return

  def read(self):
    # 
    EOF=False
    self.readBytes=0
    # Event Loop
    ev=0
    while not EOF:
      # Read new event Track
      self.readMCTrackData(ev)
      EOF=getEOF(self.file, "reading event", ev)
      ev += 1; 
    return ev
    
    return

class PreCluster:
  """
  LOOP Event
    Int_t prClusterListHeader[] = { -1, event, -1, -1, 0, nPreClusters };
    LOOP PreCluster 
      //
      // Pads/Digits
      //
      Int_t padHeader[] = { -1, event, iPreCluster, -1, 0, nPads };
      //
      dumpFloat64( dumpFiles, 0, nPads, xPad);
      dumpFloat64( dumpFiles, 0, nPads, yPad);
      dumpFloat64( dumpFiles, 0, nPads, dxPad);
      dumpFloat64( dumpFiles, 0, nPads, dyPad);
      dumpFloat64( dumpFiles, 0, nPads, charge);
      dumpInt32( dumpFiles, 0, nPads, padId);
      dumpInt32( dumpFiles, 0, nPads, DEId);            
      dumpInt32( dumpFiles, 0, nPads, cath);
      dumpInt32( dumpFiles, 0, nPads, padADC);
      dumpInt32( dumpFiles, 0, nPads, hit);
      dumpInt32( dumpFiles, 0, nPads, isSaturated);
      dumpInt32( dumpFiles, 0, nPads, isCalibrated);
      dumpInt32( dumpFiles, 0, nPads, nTracks);
      //
      // RecoCluster (Mathieson center found)
      //
      Int_t clusterHeader[] = { -1, event, iPreCluster, -1, 0, nbrOfRecoClusters };
      //
      dumpFloat64( dumpFiles, 0, nbrOfRecoClusters, xx);
      dumpFloat64( dumpFiles, 0, nbrOfRecoClusters, yy);
      dumpInt32( dumpFiles, 0, nbrOfRecoClusters, rClusterId); 
      dumpInt32( dumpFiles, 0, nbrOfRecoClusters, chamberId);
      dumpInt32( dumpFiles, 0, nbrOfRecoClusters, detElemId); 
  """
  def __init__(self, fileName="RecoDataDump.dat"):
    self.fileName = fileName
    self.file = 0
    self.file = open(fileName, 'rb')
    #
    # Data members
    self.evId = []
    self.padX = []
    self.padY = []
    self.padDX = []
    self.padDY = []
    self.padCharge = []
    self.padId = [] 
    self.padDEId = [] 
    self.padChId = [] 
    self.padCath = []
    self.padADC = []
    self.padHit = []
    self.padSaturated = [] 
    self.padCalibrated = []
    self.padNbrTracks = []
    self.padChIdMinMax = (100, -1)
    #
    self.rClusterX = []
    self.rClusterY = []
    self.rClusterId = []
    self.rClusterChId = []
    self.rClusterDEId = []
    self.rClusterChIdMinMax = (100, -1)
    return

  def readPreClusterData(self, ev, verbose=False):
    # New event
    # Read header
    # Int_t trackInfo[] = { -1, event, -1, -1, 0, nbrOfTracks };
    if (verbose): print("readPreClusterData ev=", ev)
    headerSize = 6
    preClusterListHeader = readInt4(self.file, headerSize)
    if (preClusterListHeader[0] != -1):
        print("readPreClusterData: bad preClusterListHeader", preClusterListHeader[0])
        exit()
    nbrOfPreClusters = preClusterListHeader[5]
    evId = preClusterListHeader[1]
    if (verbose): print("readPreClusterData ev=", ev, ", nbrOfPreClusters", nbrOfPreClusters)
    
    # New event
    self.evId.append(evId)
    self.padX.append([])
    self.padY.append([])
    self.padDX.append([])
    self.padDY.append([])
    self.padCharge.append([])
    self.padId.append([]) 
    self.padDEId.append([]) 
    self.padChId.append([]) 
    self.padCath.append([])
    self.padADC.append([])
    
    self.padHit.append([])
    self.padSaturated.append([]) 
    self.padCalibrated.append([])
    self.padNbrTracks.append([])
    #
    self.rClusterX.append([])
    self.rClusterY.append([])
    self.rClusterId.append([])
    self.rClusterChId.append([])
    self.rClusterDEId.append([])
    #
    for t in range(nbrOfPreClusters):
        #
        #  Pad part
        #
        #  Int_t padHeader = { -1, event, -1, -1, 0, nPads };
        if (verbose): print("readPreClusterData ev=", ev, ", preCluster=", t)
        padHeader = readInt4(self.file, headerSize)
        if (padHeader[0] != -1):
            print("readMCTrackData: bad padHeader", t, padHeader[0])
            exit()
        preClusterIdx = padHeader[2]
        nbrOfPads = padHeader[5]
        # print("pads ???", nbrOfPads)
        if (verbose): print("  readPreClusterData ev=", ev, ", precluster=", t, ", nbrOfPads", nbrOfPads)
        self.padX[ev].append( readDouble(self.file, nbrOfPads) )
        self.padY[ev].append( readDouble(self.file, nbrOfPads) )
        self.padDX[ev].append( readDouble(self.file, nbrOfPads) )
        self.padDY[ev].append( readDouble(self.file, nbrOfPads) )
        self.padCharge[ev].append( readDouble(self.file, nbrOfPads) )
        self.padId[ev].append( readInt4(self.file, nbrOfPads) )
        self.padDEId[ev].append( readInt4(self.file, nbrOfPads) )
        self.padChId[ev].append(  self.padDEId[ev][-1] // 100 )
        if self.padChId[ev][-1].shape[0] != 0:
          (chIdMin, chIdMax) = self.padChIdMinMax
          chIdMin = min( chIdMin, np.min( self.padChId[ev][-1] ) )
          chIdMax = max( chIdMax, np.max( self.padChId[ev][-1] ) )
          self.padChIdMinMax = (chIdMin, chIdMax )
        #
        self.padCath[ev].append( readInt4(self.file, nbrOfPads) )
        self.padADC[ev].append( readInt4(self.file, nbrOfPads) )
        self.padHit[ev].append( readInt4(self.file, nbrOfPads) )
        self.padSaturated[ev].append( readInt4(self.file, nbrOfPads) )
        self.padCalibrated[ev].append( readInt4(self.file, nbrOfPads) )
        self.padNbrTracks[ev].append( readInt4(self.file, nbrOfPads) )
        n=np.sum(self.padNbrTracks[ev][-1])
        # n Tracks Always = 0
        # print("nTracks", n )        
        #
        # RecoCluster part
        #
        # Int_t clusterHeader[] = { -1, event, iPreCluster, -1, 0, nbrOfRecoClusters };
        recoClusterHeader = readInt4(self.file, headerSize)
        if (recoClusterHeader[0] != -1):
            print("readMCTrackData: bad recoClusterHeader", t, recoClusterHeader[0])
            exit()
        #    
        preClusterIdx = recoClusterHeader[2]
        nbrOfRecoClusters = recoClusterHeader[5]        
        #
        if (verbose): print("  readPreClusterData ev=", ev, ", precluster=", t, ", nbrOfRecoClusters", nbrOfRecoClusters)
        self.rClusterX[ev].append (readDouble(self.file, nbrOfRecoClusters))
        self.rClusterY[ev].append (readDouble(self.file, nbrOfRecoClusters))
        self.rClusterId[ev].append (readInt4(self.file, nbrOfRecoClusters))
        self.rClusterChId[ev].append (readInt4(self.file, nbrOfRecoClusters))
        self.rClusterDEId[ev].append (readInt4(self.file, nbrOfRecoClusters))
        if (self.rClusterChId[ev][-1].shape[0] != 0):
          self.rClusterChId[ev][-1] = self.rClusterChId[ev][-1] + 1
          (chIdMin, chIdMax) = self.rClusterChIdMinMax
          chIdMin = min( chIdMin, np.min( self.rClusterChId[ev][-1] ) )
          chIdMax = max( chIdMax, np.max( self.rClusterChId[ev][-1] ) )
          self.rClusterChIdMinMax = (chIdMin, chIdMax )
        #    
    # Precluster loop
    return

  def read(self):
    # 
    EOF=False
    self.readBytes=0
    # Event Loop
    ev=0
    while not EOF:
      # Read new event Track
      self.readPreClusterData(ev)
      EOF=getEOF(self.file, "reading event", ev)
      ev += 1; 
    return ev
    
if __name__ == "__main__":
    print("Hello")

