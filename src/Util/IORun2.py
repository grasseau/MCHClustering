#! /usr/bin/python

__author__ = "grasseau"
__date__ = "$Jul 12, 2020 9:56:07 AM$"

import sys, traceback
import struct
import numpy as np
import pickle 

def readInt2(file, n):
    
    if ( n == 0 ): return np.zeros( (0), dtype=np.int16 )
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
    
    # Read N Int16
    raw = file.read(nData * 2)
    array = struct.unpack(str(nData) + 'h', raw)
    if nData != n: 
      print("Expected/Read number of values are different ", n, "/", nData); 
      print( "raw", raw )
      print( "array", array )
      traceback.print_stack(); 
      exit()
    #
    return np.array(array, dtype=np.int16)

def readUInt4(file, n):
    
    if ( n == 0 ): return np.zeros( (0), dtype=np.uint32 )
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
    
    # Read N UInt32
    raw = file.read(nData * 4)
    array = struct.unpack(str(nData) + 'I', raw)
    if nData != n: 
      print("Expected/Read number of values are different ", n, "/", nData); 
      print( "raw", raw )
      print( "array", array )
      traceback.print_stack(); 
      exit()
    #
    return np.array(array, dtype=np.uint32)

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
    print("len(raw ", len(raw) )
    # print("fmt unpack ", str(nData) + 'd')
    array = struct.unpack(str(nData) + 'd', raw)
    if nData != n: 
      print("Expected/Read number of values are different ", n, "/", nData); 
      print( "raw", raw )
      print( "array", array )
      traceback.print_stack(); 
      exit()
    #
    return np.array(array, dtype=np.float64)

def getEOF(file, errorStr="", verbose=False):
    k = len(file.read(8))
    EOF = (k != 8)
    if ( verbose  and EOF):
        print( "Warning: EOF reached ", errorStr, k, "bytes read", )
    file.seek(-k, 1)
    return EOF

class Run2PreCluster:
  """
  loop on preClusters
    ################ STORE Pads #################################
    if ( !(samePreCluster) && (digits.size() != 0) ) {
      uint32_t header[6] = { (uint32_t) (bunchCrossing), orbit, iROF, (uint32_t) (0), N, (uint32_t) (DEId)};
      pClusterDump->dumpUInt32(0, 6, header );
      pClusterDump->dumpFloat64(0, N, xPad);    
      pClusterDump->dumpFloat64(0, N, yPad);
      pClusterDump->dumpFloat64(0, N, dxPad);
      pClusterDump->dumpFloat64(0, N, dyPad);
      pClusterDump->dumpFloat64(0, N, padCharge );
      pClusterDump->dumpInt16(0, N, saturated );
      pClusterDump->dumpInt16(0, N, cathode );
      pClusterDump->dumpUInt32(0, N, padADC );
  
    ############### STORE reco Hits #############################
  
    uint32_t N = mClusters.size() - nPreviousCluster;
    // Header
    uint32_t header[6] = { (uint32_t) (bunchCrossing),  (orbit), iROF, (0), N, 0 };    
    pClusterDump->dumpUInt32(0, 6, header );
    //
    if (N > 0 ) {
      pClusterDump->dumpFloat64(0, N, x);
      pClusterDump->dumpFloat64(0, N, y);
      pClusterDump->dumpFloat64(0, N, ex);
      pClusterDump->dumpFloat64(0, N, ey);
      pClusterDump->dumpUInt32(0, N, uid);
      pClusterDump->dumpUInt32(0, N, firstDigit);
      pClusterDump->dumpUInt32(0, N, nDigits);
    }
    With 
      // struct ClusterStruct {
      //  float x;             ///< cluster position along x
      //  float y;             ///< cluster position along y
      //  float z;             ///< cluster position along z
      //  float ex;            ///< cluster resolution along x
      //  float ey;            ///< cluster resolution along y
      //  uint32_t uid;        ///< cluster unique ID
      //  uint32_t firstDigit; ///< index of first associated digit in the ordered vector of digits
      //  uint32_t nDigits;    ///< number of digits attached to this cluster      
  """
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
    self.preClusterId = []   
    self.BC = []
    self.orbit = []
    self.iROF = []
    self.DEId = []
    #
    self.padX = []
    self.padY = []
    self.padDX = []
    self.padDY = []
    # 
    self.padCharge = []
    self.padSaturated = []
    self.padCathode = []
    self.padADC = []
    #
    self.hitX = []
    self.hitY = []
    self.hitErrX = []
    self.hitErrY = []
    self.hitUID  = []
    self.hitFirstPadIdx = []
    self.hitNbrPads = []
    #
    self.nbrOfReadPreclusters = 0

    return

  def readOnePreCluster(self, verbose=False):
    # New PreCluster
    # Read header
    # Int_t trackInfo[] = { -1, event, -1, -1, 0, nbrOfTracks };
    # uint32_t header[6] = { bunchCrossing, orbit, iROF, (0), n, DEId };    
    # ??? if (verbose): print("readPreClusterData preClusterId=", preClusterId)
    headerSize = 6
    preClusterListHeader = readUInt4(self.file, headerSize)
    if (preClusterListHeader[3] != 0):
        print("readPreClusterData: bad preClusterListHeader", preClusterListHeader[0])
        exit()
    
    bc = preClusterListHeader[0]
    orbit = preClusterListHeader[1]
    iROF = preClusterListHeader[2]
    nbrOfPads = preClusterListHeader[4]
    DEId = preClusterListHeader[5]

    if (verbose): 
      print("readPreClusterData bc=", bc, "orbit=", orbit, ", iROF=", iROF, ", nbrOfPads", nbrOfPads)
    
    id = ( bc, orbit, iROF, DEId, nbrOfPads)
    # pads
    x = readDouble(self.file, nbrOfPads)
    y = readDouble(self.file, nbrOfPads)
    dx = readDouble(self.file, nbrOfPads)
    dy = readDouble(self.file, nbrOfPads)
    # 
    charge = readDouble(self.file, nbrOfPads) 
    saturated = readInt2(self.file, nbrOfPads) 
    cathode = readInt2(self.file, nbrOfPads) 
    adc = readUInt4(self.file, nbrOfPads)

    pads = ( x, y, dx, dy, charge, saturated, cathode, adc )
    #
    # Int_t clusterHeader[] = { -1, event, iPreCluster, -1, 0, nbrOfRecoClusters };
    # uint32_t header[6] = { bunchCrossing, orbit, iROF, (0), nHits, 0 };    

    recoHitHeader = readUInt4(self.file, headerSize)
    if (recoHitHeader[3] != 0):
        print("readMCTrackData: bad recoClusterHeader", bc, orbit, iROF, recoHitHeader[0])
        exit()
    #
    nHits = recoHitHeader[4]
    if nHits != 0:
      xr = readDouble(self.file, nHits)
      yr = readDouble(self.file, nHits)
      errX = readDouble(self.file, nHits)
      errY = readDouble(self.file, nHits)
      uid = readUInt4(self.file, nHits)
      startPadIdx = readUInt4(self.file, nHits)
      nPadIdx = readUInt4(self.file, nHits)
    else :
      xr = np.zeros((0)) 
      yr = np.zeros((0)) 
      errX = np.zeros((0)) 
      errY = np.zeros((0)) 
      uid  = np.zeros((0), dtype=np.uint32) 
      startPadIdx = np.zeros((0), dtype=np.uint32) 
      nPadIdx     = np.zeros((0), dtype=np.uint32) 
    # Hits
    hits = (nHits, xr, yr, errX, errY, uid, startPadIdx, nPadIdx)
    # Precluster
    return id, pads, hits

  def __iter__(self):
    self.file.close()
    self.file = open( self.fileName, 'rb')
    self.nbrOfReadPreclusters = 0
    return self

  def __next__(self): 
    EOF=getEOF(self.file, "reading preCluster", verbose=True)
    if not EOF:
      data = self.readOnePreCluster()
      self.nbrOfReadPreclusters += 1
    else:
      data = ()
      raise StopIteration
    return ( data )

  def readPreCluster(self, bc, orbit, iROF):
    for pc in self:
      (id, pads, hits) = pc
      ( bc_, orbit_, irof_, deid_, nbrOfPads) = id
      if (bc_ == bc) and (orbit_ == orbit) and (irof_ == iROF) :
         return pc

  def readPreClusterData(self, preClusterId, verbose=False):
    # New PreCluster
    # Read header
    # Int_t trackInfo[] = { -1, event, -1, -1, 0, nbrOfTracks };
    # uint32_t header[6] = { bunchCrossing, orbit, iROF, (0), n, DEId };    
    if (verbose): print("readPreClusterData preClusterId=", preClusterId)
    headerSize = 6
    preClusterListHeader = readUInt4(self.file, headerSize)
    if (preClusterListHeader[3] != 0):
        print("readPreClusterData: bad preClusterListHeader", preClusterListHeader[0])
        exit()
    
    bc = preClusterListHeader[0]
    orbit = preClusterListHeader[1]
    iROF = preClusterListHeader[2]
    nbrOfPads = preClusterListHeader[4]
    DEId = preClusterListHeader[5]

    if (verbose): 
      print("readPreClusterData bc=", bc, "orbit=", orbit, ", iROF=", iROF, ", nbrOfPads", nbrOfPads)
    
    # New PreCluster
    self.preClusterId.append( preClusterId )    
    self.BC.append(bc)
    self.orbit.append(orbit)
    self.iROF.append(iROF)
    self.DEId.append(DEId)

    self.padX.append( readDouble(self.file, nbrOfPads) )
    self.padY.append( readDouble(self.file, nbrOfPads) )
    self.padDX.append( readDouble(self.file, nbrOfPads) )
    self.padDY.append( readDouble(self.file, nbrOfPads) )
    # 
    self.padCharge.append( readDouble(self.file, nbrOfPads) )
    self.padSaturated.append( readInt2(self.file, nbrOfPads) )
    self.padCathode.append( readInt2(self.file, nbrOfPads) )
    self.padADC.append( readUInt4(self.file, nbrOfPads) )
    
    #
    # Int_t clusterHeader[] = { -1, event, iPreCluster, -1, 0, nbrOfRecoClusters };
    # uint32_t header[6] = { bunchCrossing, orbit, iROF, (0), nHits, 0 };    

    recoHitHeader = readUInt4(self.file, headerSize)
    if (recoHitHeader[3] != 0):
        print("readMCTrackData: bad recoClusterHeader", bc, orbit, iROF, recoHitHeader[0])
        exit()
    #
    nHits = recoHitHeader[4]
    if nHits != 0:
      self.hitX.append (readDouble(self.file, nHits))
      self.hitY.append (readDouble(self.file, nHits))
      self.hitErrX.append (readDouble(self.file, nHits))
      self.hitErrY.append (readDouble(self.file, nHits))
      self.hitUID.append (readUInt4(self.file, nHits))
      self.hitFirstPadIdx.append (readUInt4(self.file, nHits))
      self.hitNbrPads.append (readUInt4(self.file, nHits))
    else :
      self.hitX.append ( np.zeros((0)) )
      self.hitY.append ([])
      self.hitErrX.append ([])
      self.hitErrY.append ([])
      self.hitUID.append ([])
      self.hitFirstPadIdx.append ([])
      self.hitNbrPads.append ([])        
    # Precluster
    return

  def read(self, verbose=False):
    # 
    EOF=False
    self.readBytes=0
    # Event Loop
    preClusterId=0
    while not EOF:
      # Read new precluster
      self.readPreClusterData(preClusterId, verbose=verbose)
      EOF=getEOF(self.file, "reading preCluster", verbose=verbose)
      preClusterId+= 1; 
    return preClusterId

def writePickle( fileName, obj ):
   file = open( fileName, "wb" )
   pickle.dump( obj, file )
   file.close()
   
def readPickle( fileName ):
   file = open( fileName, "rb" )
   obj = pickle.load( file )
   file.close()
   return obj

if __name__ == "__main__":
    print("Hello")

