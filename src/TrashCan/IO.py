#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 12, 2020 9:56:07 AM$"

import sys, traceback
import struct
import numpy as np

def readInt4( file, n):
    # Read Nbr of items (8 bytes)
    raw = file.read( 2 * 4 )
    nData = struct.unpack( 'q' , raw)[0]
    # print str(raw[0]).encode("hex") ,'-', str(raw[1]).encode("hex")

    if nData != n : 
        print ("Expected/Read number of values are different ", n, "/", nData); 
        # traceback.print_stack(); exit()
        getEOF( file )
        return 
    
    # Read N Int32
    raw = file.read( nData * 4 )
    array = struct.unpack( str(nData)+'i' , raw)
    return np.array( array, dtype=np.int32 )

def readDouble( file, n):
    # Read Nbr of items (8 bytes)
    raw = file.read( 2 * 4 )
    nData = struct.unpack( 'q' , raw)[0]
    # print str(raw[0]).encode("hex") ,'-', str(raw[1]).encode("hex")

    if nData != n : print ("Expected/Read number of values are different ", n,"/", nData); traceback.print_stack(); exit()
    
    # Read N Double
    raw = file.read( nData * 8 )
    array = struct.unpack( str(nData)+'d' , raw)
    return np.array( array, dtype=np.float64 )

def getEOF( file, errorStr="", mode="quiet"):
    k = len( file.read( 8 ) )
    EOF = ( k != 8)
    if (mode=="verbose" and EOF) :
        print "Warning: EOF reached ", errorStr, k, "bytes read", 
    file.seek(-k, 1)
    return EOF

class Pads:
    def __init__(self, file, nData ):
        self.x      = readDouble( file, nData) 
        self.y      = readDouble( file, nData)
        self.dx     = readDouble( file, nData)
        self.dy     = readDouble( file, nData)
        self.charge = readDouble( file, nData)
        self.cathode    = readInt4( file, nData)
        self.padId      = readInt4( file, nData)
        self.ADC        = readInt4( file, nData)
        self.saturated  = readInt4( file, nData)
        self.calibrated = readInt4( file, nData)

class TrackMCFeatures:
    def __init__(self, file, nData ):
        self.x      = readDouble( file, nData) 
        getEOF( file, "TrackMCFeature x")
        self.y      = readDouble( file, nData)
        getEOF( file, "TrackMCFeature y")
        self.ChamberId = readInt4( file, nData)
        getEOF( file, "TrackMCFeature ch")
        self.DEId      = readInt4( file, nData)
        getEOF( file, "TrackMCFeature de")

        
class TrackRef:
  """
  Format:
  + dumpInt32( dumpFiles, 0, 6, trackHeader ); # New Track, Give nClusters in the Track
    for c in nClusters:
    + dumpInt32( dumpFiles, 0, 6, padHeader);  # NewCluster, Give nPads in the Cluster, it can be zero
        + pad list (x, y, dx, dy, charge, ...)
    + dumpInt32( dumpFiles, 0, 6, padHeader); # NewCluster, Give nPads in the Cluster, it can be zero
      ...
  + trackInfo MC Info list (x[nClusters],y[nClusters], chamberId[:], ...) : End of Track
  + dumpInt32( dumpFiles, 0, 6, trackHeader ); # New Track, Give nClusters in the Track
    ...
  """
  def __init__(self, fileName="MCTrackRefDump.dat"):
    self.fileName = fileName
    self.file = open(fileName, 'rb')
    self.trackHeaders = []  # list[t]
    self.clusterHeaders = [] # list [t][c]
    self.tracks = [] # tracks[t][c] gives the pad (arrays)arrays
    self.MCTracks = [] # MCtracks[t].x/y/DEId/ChId[c] gives the MC feature (arrays)
    self.readBytes = 0
    self.clusterHash = {}
    self.dXdYHash = {}

  def hashClusters(self):
    print "**** hashClusters"
    totalNbrOfClusters = 0
    code = [0]*10
    nTracks = len( self.tracks )  
    for t in range(nTracks):
      nClusters = len(self.tracks[t])
      totalNbrOfClusters += nClusters
      for c in range(nClusters):
        cl = self.tracks[t][c]
        # Cathode 0
        Idx = np.where( cl.cathode == 0)
        if ( cl.x[Idx].size !=0 ):
          x = np.mean( cl.x[Idx] )
          y = np.mean( cl.y[Idx] )
          varx = x - cl.x[Idx]
          varx = varx * varx
          sigx = np.sqrt( np.mean( varx ) )
          vary = y - cl.y[Idx]
          vary = vary * vary
          sigy = np.sqrt( np.mean( vary ) )
          ch = np.sum(cl.charge[Idx])
          code[0] = int(x * 100)
          code[1] = int(y * 100)
          code[2] = int( sigx * 100)
          code[3] = int( sigy * 100)
          code[4] = int( ch * 100)
        else:
          code[0] = 0; code[1] = 0; code[2] = 0; code[3] = 0; code[4] = 0; 
        # Cathode 1
        Idx = np.where( cl.cathode == 1)  
        if (cl.x[Idx]).size != 0:
          x = np.mean( cl.x[Idx] )
          y = np.mean( cl.y[Idx] )
          varx = x - cl.x[Idx]
          varx = varx * varx
          sigx = np.sqrt( np.mean( varx ) )
          vary = y - cl.y[Idx]
          vary = vary * vary
          sigy = np.sqrt( np.mean( vary ) )
          ch = np.sum(cl.charge[Idx])
          code[5] = int(x * 100)
          code[6] = int(y * 100)
          code[7] = int( sigx * 100)
          code[8] = int( sigy * 100)
          code[9] = int( ch * 100)
        else:
          code[5] = 0; code[6] = 0; code[7] = 0; code[8] = 0; code[9] = 0; 
        # Code
        key = str(code[0]) + ' ' + str(code[1]) + ' ' + str(code[2]) + ' ' + str(code[3]) + ' ' + str(code[4]) + ' ' \
            + str(code[5]) + ' ' + str(code[6]) + ' ' + str(code[7]) + ' ' + str(code[8]) + ' ' + str(code[9])
        # Debug
        # print key
        if self.clusterHash.has_key(key):
          self.clusterHash[key].append( (t,c) )
        else:
          self.clusterHash[key] = [(t,c)]
    lMax = 0
    keyMax = ""
    sumClusters = 0
    for key in self.clusterHash.keys():          
      l = len( self.clusterHash[ key ] )
      sumClusters += l
      if l > lMax:
          lMax = l
          keyMax = key
          
    print "clusterMax", self.clusterHash[ keyMax ]
    print "keyMax", keyMax
    print "totalNbrOfClusters=", totalNbrOfClusters, "sumClusters=", sumClusters
    print "nTracks=", nTracks
    return

  def extractdXdY(self):
    print "**** extractDxDy"
    totalNbrOfClusters = 0
    code = [0]*3
    nTracks = len( self.tracks )  
    for t in range(nTracks):
      nClusters = len(self.tracks[t])
      totalNbrOfClusters += nClusters
      for c in range(nClusters):
        cl = self.tracks[t][c]
        # Cathode 0
        Idx = np.where( cl.cathode == 0)
        dx = cl.dx[Idx] 
        dy = cl.dy[Idx] 
        cath = 0
        for k in range(len(dx)):
          code[0] = int(dx[k] * 100)
          code[1] = int(dy[k] * 100)
          code[2] = int(cath)
          key = str(code[0]) + ' ' + str(code[1]) + ' ' + str(code[2])
          if not self.dXdYHash.has_key(key):
            self.dXdYHash[key] = cath
        # Cathode 1
        Idx = np.where( cl.cathode == 1)
        dx = cl.dx[Idx] 
        dy = cl.dy[Idx] 
        cath = 1
        for k in range(len(dx)):
          code[0] = int(dx[k] * 100)
          code[1] = int(dy[k] * 100)
          code[2] = int(cath)
          key = str(code[0]) + ' ' + str(code[1]) + ' ' + str(code[2])
          if not self.dXdYHash.has_key(key):
            self.dXdYHash[key] = cath
            
    print "totalNbrOfClusters=", totalNbrOfClusters
    print "nTracks=", nTracks
    print self.dXdYHash.keys()
    return

  def getClusterInfos(self, trackIdx, clusterIdx):
    # Int_t padHeader[] = { -1, event, mcLabel, partCode, iCl, nPads };
     infos = self.clusterHeaders[trackIdx][clusterIdx]
     return infos[1:] 
      
      
  def readNewTrack( self, verbose=True ):
    #
    # Read cluster header v3
    #
    # V3
    # Int_t trackInfo[] = { -1, event, mcLabel, partCode, 0, nbrOfClusters };
    headerSize = 6
    trackInfo = readInt4( self.file, headerSize )
    # Test tag
    if (trackInfo[0] != -1):
        print("readNewTrack: bad header",  trackInfo[0])
        exit()
    if verbose == True :
        print "New Track ev=", trackInfo[1], ", Label=", trackInfo[2], "part. code=", trackInfo[3] 
    self.readBytes += 4*headerSize
    # New Track
    self.tracks.append([])
    self.MCTracks.append( None  )
    self.clusterHeaders.append([])
    self.trackHeaders.append( trackInfo )
    
    nClusters = trackInfo[headerSize-1]
    return nClusters

  def readNewCluster( self, t ):
    #
    # Read cluster header
    #
    # Int_t padHeader[] = { -1, event, mcLabel, partCode, iCl, nPads };
    headerSize = 6
    clusterInfo = readInt4( self.file, headerSize )
    # Test tag
    if (clusterInfo[0] != -1):
        print("readNewCluster: bad header",  clusterInfo[0])
        exit()

    self.readBytes += 4*headerSize
    self.clusterHeaders[t].append( clusterInfo )
    # Nbr of values    
    nClusters = clusterInfo[headerSize-1]
    return nClusters

  def readPads( self, t, nData ):
    self.tracks[t].append( Pads( self.file, nData ) )
    self.readBytes += (nData * (5*8 + 5*4))


  def readTrackMCFeatures(self, t, nData):
      self.MCTracks[t] = TrackMCFeatures( self.file, nData )
      self.readBytes += (nData * (2*8 + 2*4))
      
  def check(self, nTracks):
    # Tracks
    print
    print "**** CHECK ****"
    print "Track Stored =", len(self.tracks) , "/", nTracks
    print "Track header Stored =", len(self.trackHeaders) , "/", nTracks
    print "MCTrack Stored =", len(self.MCTracks) , "/", nTracks
    # Clusters
    for t in range(nTracks):
      ncInTracks = len( self.tracks[t] )
      ncInMCTracks = self.MCTracks[t].x.shape[0]
      if (ncInTracks != ncInMCTracks):
        print "Track=", t, ":", ncInMCTracks, "/", ncInTracks 
        exit()
    return

  def read( self ):
    """
    + dumpInt32( dumpFiles, 0, 6, trackHeader ); # New Track, Give nClusters in the Track
    for c in nClusters:
    + dumpInt32( dumpFiles, 0, 6, padHeader);  # NewCluster, Give nPads in the Cluster, it can be zero
        + pad list (x, y, dx, dy, charge, ...)
    + dumpInt32( dumpFiles, 0, 6, padHeader); # NewCluster, Give nPads in the Cluster, it can be zero
      ...
  + trackInfo MC Info list (x[nClusters],y[nClusters], chamberId[:], ...) : End of Track
  + dumpInt32( dumpFiles, 0, 6, trackHeader ); # New Track, Give nClusters in the Track
    ...
    """
    EOF = False
    iTrack = 0
    self.readBytes = 0
    while not EOF :
        # Read new Track
        nClusters = self.readNewTrack( )
        EOF = getEOF( self.file, "reading Track Header")
        # print "To read nClusters=", nClusters
        #
        # Read New Cluster associated with the track
        for c in range(nClusters):
          nPads = self.readNewCluster( iTrack )
          # print "To read nPads=", nPads, ", cluster=", c
          self.readPads( iTrack, nPads)
          EOF = getEOF( self.file, "reading Pads")
        #
        self.readTrackMCFeatures( iTrack, nClusters)
        EOF = getEOF( self.file, "try to read after MC Track Features", mode="verbose")
        #
        iTrack += 1; 
    # Debug
    # print "Track i=", iTrack
    # print "total bytes read:", self.readBytes
    self.check(iTrack)
    return iTrack

class TrackInfo:
    
  def __init__(self, fileName="test.dat"):
    self.fileName = fileName
    self.file = open(fileName, 'rb')
    self.clusterInfos = []
    self.clusterRecos = []
    self.x = []
    self.y = []
    self.dx = []
    self.dy = []
    self.charge = []
    self.cathode = []
    self.clusterID = []        
    self.status = []        
    self.saturated = []
    # Cluster identification
    self.id = {}
    self.id["Event"] = []
    self.id["cServerID"] = []
    self.id["preClusterID"] = []
    self.id["DetectElemID"] = []
    self.id["ChamberID"] = []
    self.id["MCLabel"] = []
    self.id["ParticleID"] = []
    # Cluster features (center, erro rbar, ..)
    self.cFeatures = {}
    self.cFeatures["X"] = []
    self.cFeatures["Y"] = []
    self.cFeatures["globalX"] = []
    self.cFeatures["globalY"] = []
    self.cFeatures["globalZ"] = []
    self.cFeatures["ErrorX"] = []
    self.cFeatures["ErrorY"] = []
    self.cFeatures["Chi2"] = []
    self.cFeatures["ChargeCath0"] = []
    self.cFeatures["ChargeCath1"] = []
    self.cFeatures["SaturatedCath0"] = []
    self.cFeatures["SaturatedCath1"] = []

    self.clusterHash = {}
    
  def readMCTrackHeader( self ):
    #
    # Read cluster header v3
    #
    # V3
    # Int_t trackInfo[] = { -1, event, mcLabel, partCode, 0, nbrOfClusters };
    headerSize = 6
    trackInfo = readInt4( self.file, headerSize )
    # Test tag
    if (trackInfo[0] != -1):
        print("readPreCluster: bad header",  clusterInfo[0])
        exit()
    # Nbr of values
    # v2 - nData = clusterInfo[3]
    nData = trackInfo[headerSize-1]
    self.clusterInfos.append( trackInfo )
    self.id["Event"].append( trackInfo[1] )
    self.id["MCLabel"].append( trackInfo[2] )
    self.id["ParticleID"].append( trackInfo[3] )
    # self.id["ChamberID"].append( clusterInfo[4] )
    return nData

  def readPreClusterHeader( self, version ):
    #
    # Read cluster header v3
    #
    # V3
    # clusterInfo[] = { -1, evID, clServId, detElemId, chamberId, np};
    headerSize = 6
    if (version=="v2"):
        headerSize = 4
    clusterInfo = readInt4( self.file, headerSize )
    # Test tag
    if (clusterInfo[0] != -1):
        print("readPreCluster: bad header",  clusterInfo[0])
        exit()
    # Nbr of values
    # v2 - nData = clusterInfo[3]
    nData = clusterInfo[headerSize-1]
    self.clusterInfos.append( clusterInfo )
    self.id["Event"].append( clusterInfo[1] )
    self.id["cServerID"].append( clusterInfo[3] )
    self.id["preClusterID"].append( clusterInfo[4] )
    self.id["DetectElemID"].append( clusterInfo[2] )
    # self.id["ChamberID"].append( clusterInfo[4] )
    return nData

  def readClusterServerHeader( self ):
    #
    # Read cluster header v3
    #
    # V3
    # clusterInfo[] = { -1, evID, clServId, detElemId, chamberId, np};
    clusterInfo = readInt4( self.file, 6 )
    # Test tag
    if (clusterInfo[0] != -1):
        print("readPreCluster: bad header",  clusterInfo[0])
        exit()
    # Nbr of values
    # v2 - nData = clusterInfo[3]
    nData = clusterInfo[5]
    self.clusterInfos.append( clusterInfo )
    self.id["Event"].append( clusterInfo[1] )
    self.id["cServerID"].append( clusterInfo[2] )
    # self.id["preClusterID"].append( clusterInfo[] )
    self.id["DetectElemID"].append( clusterInfo[3] )
    self.id["ChamberID"].append( clusterInfo[4] )
    reco = readDouble( self.file, 14)
    self.clusterRecos.append( reco )
    self.cFeatures["X"].append( reco[0] )
    self.cFeatures["Y"].append( reco[1] )
    self.cFeatures["globalX"].append( reco[2] )
    self.cFeatures["globalY"].append( reco[3] )
    self.cFeatures["globalZ"].append( reco[4] )
    self.cFeatures["ErrorX"].append( reco[5] )
    self.cFeatures["ErrorY"].append( reco[6] )
    self.cFeatures["Chi2"].append( reco[7] )
    self.cFeatures["ChargeCath0"].append( reco[8] )
    self.cFeatures["ChargeCath1"].append( reco[9] )
    self.cFeatures["SaturatedCath0"].append( reco[12] )
    self.cFeatures["SaturatedCath1"].append( reco[13] )
    return nData

  def readTrack( self, nData ):
    """
    C++ format
    dumpFloat64( dumpFiles, 0, nbrOfClusters, x);
    dumpFloat64( dumpFiles, 0, nbrOfClusters, y);
    dumpInt32( dumpFiles, 0, nbrOfClusters, chamberId);
    dumpInt32( dumpFiles, 0, nbrOfClusters, detElemId);
    """
    self.cFeatures["X"].append( readDouble( self.file, nData) )
    self.cFeatures["Y"].append( readDouble( self.file, nData) )
    self.id["ChamberID"].append( readInt4( self.file, nData) )
    self.id["DetectElemID"].append( readInt4( self.file, nData) )

    EOF = ( len( self.file.read( 8) )  != 8)
    self.file.seek(-8, 1)
    return EOF

  def readPads( self, nData ):
    self.x.append( readDouble( self.file, nData) )
    self.y.append( readDouble( self.file, nData) )
    self.dx.append( readDouble( self.file, nData) )
    self.dy.append( readDouble( self.file, nData) )
    self.charge.append( readDouble( self.file, nData) )
    self.cathode.append( readInt4( self.file, nData) )       # V2.0
    self.clusterID.append( readInt4( self.file, nData) )
    self.status.append( readInt4( self.file, nData) )
    self.saturated.append( readInt4( self.file, nData) )

    EOF = ( len( self.file.read( 8) )  != 8)
    self.file.seek(-8, 1)
    return EOF

  def read( self, mode="PreCluster", version="v3" ):
    EOF = False
    count = 0
    while not EOF :
        if ( mode == "PreCluster"):
          nPads = self.readPreClusterHeader( version )
          EOF = self.readPads( nPads )
        elif mode == "ClusterServer":
          nPads = self.readClusterServerHeader( )
          EOF = self.readPads( nPads )
        else:
          nClusters = self.readMCTrackHeader( )
          EOF = self.readTrack( nClusters )

        if not EOF: count += 1
    return count

  def hashClusters(self):
    print "**** hashClusters"
    totalNbrOfClusters = 0
    code = [0]*10
    nClusters = len( self.x )  
    for c in range(nClusters):
        # Cathode 0
        Idx = np.where( self.cathode[c] == 0)
        if ( self.x[c][Idx].size !=0 ):
          x = np.mean( self.x[c][Idx] )
          y = np.mean( self.y[c][Idx] )
          varx = x - self.x[c][Idx]
          varx = varx * varx
          sigx = np.sqrt( np.mean( varx ) )
          vary = y - self.y[c][Idx]
          vary = vary * vary
          sigy = np.sqrt( np.mean( vary ) )
          ch = np.sum( self.charge[c][Idx])
          code[0] = int(x * 100)
          code[1] = int(y * 100)
          code[2] = int( sigx * 100)
          code[3] = int( sigy * 100)
          code[4] = int( ch * 100)
        else:
          code[0] = 0; code[1] = 0; code[2] = 0; code[3] = 0; code[4] = 0; 
        # Cathode 1
        Idx = np.where( self.cathode[c] == 1)  
        if ( self.x[c][Idx]).size != 0:
          x = np.mean( self.x[c][Idx] )
          y = np.mean( self.y[c][Idx] )
          varx = x - self.x[c][Idx]
          varx = varx * varx
          sigx = np.sqrt( np.mean( varx ) )
          vary = y - self.y[c][Idx]
          vary = vary * vary
          sigy = np.sqrt( np.mean( vary ) )
          ch = np.sum(self.charge[c][Idx])
          code[5] = int(x * 100)
          code[6] = int(y * 100)
          code[7] = int( sigx * 100)
          code[8] = int( sigy * 100)
          code[9] = int( ch * 100)
        else:
          code[5] = 0; code[6] = 0; code[7] = 0; code[8] = 0; code[9] = 0; 
        # Code
        key = str(code[0]) + ' ' + str(code[1]) + ' ' + str(code[2]) + ' ' + str(code[3]) + ' ' + str(code[4]) + ' ' \
            + str(code[5]) + ' ' + str(code[6]) + ' ' + str(code[7]) + ' ' + str(code[8]) + ' ' + str(code[9])
        # Debug
        # print key
        if self.clusterHash.has_key(key):
          self.clusterHash[key].append( c )
        else:
          self.clusterHash[key] = [ c ]
    lMax = 0
    keyMax = ""
    sumClusters = 0
    for key in self.clusterHash.keys():          
      l = len( self.clusterHash[ key ] )
      sumClusters += l
      if l > lMax:
          lMax = l
          keyMax = key
          
    print "clusterMax", self.clusterHash[ keyMax ]
    print "keyMax", keyMax
    print "totalNbrOfClusters=", totalNbrOfClusters, "sumClusters=", sumClusters
    print "nClusters=", nClusters
    return

if __name__ == "__main__":
    tracks = TrackInfo()
    tracks.read()

