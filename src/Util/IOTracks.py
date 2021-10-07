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
    # print("len(raw) ", len(raw) )
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

class Tracks:
  """
    Int_t trackListHeader[] = { -1, iEvent, -1, -1, 0, nTracks };
    for (auto& track : tracks) {
        Int_t trackInfo[] = { trackIdx, chi2x100, -1, -1, -1, nClusters};
        //
        Int_t DeIds[nClusters];
        Int_t UIDs[nClusters];
        Double_t X[nClusters];
        Double_t Y[nClusters];
        Double_t Z[nClusters];
        Double_t errX[nClusters];
        Double_t errY[nClusters];
  """
  # ???
  def __init__(self, fileName="TracksReco.dat"):
    self.fileName = fileName
    self.file = 0
    self.file = open(fileName, 'rb')
    #
    # Data members
    # tracks[ev][iTrack].nparray[nHits]
    self.tracks = []   
  
    return

  def readATrack(self, verbose=False):
    headerSize = 6
    header = readInt4(self.file, headerSize)
    if (header[4] != -1 ):
        print("readATrack: bad preClusterListHeader", header[4])
        exit()
    # { trackIdx, chi2x100, -1, -1, -1, nClusters};
    trackIdx = header[0]
    chi2 = header[1] / 100.0
    nHits = header[5]

    if (verbose): 
      print("readAtrack trackIdx=", trackIdx, "chi2=", chi2, ", nHits=", nHits )
    # Hits
    aTrack = ()
    if nHits != 0:
      DEIds = readInt4(self.file, nHits)
      UIDs  = readInt4(self.file, nHits)
      #
      x = readDouble(self.file, nHits)
      y = readDouble(self.file, nHits)
      z = readDouble(self.file, nHits)
      errX = readDouble(self.file, nHits)
      errY = readDouble(self.file, nHits)
      aTrack = ( trackIdx, chi2, nHits, DEIds, UIDs, x, y, z, errX, errY)
    else:
      empty = np.empty(0)
      aTrack = ( trackIdx, chi2, nHits, empty, empty, empty, empty, empty, empty, empty)
    #
    return aTrack

  def __iter__(self):
    self.file.close()
    self.file = open( self.fileName, 'rb')
    self.nbrOfReadPreclusters = 0
    return self

  def __next__(self): 
    EOF=getEOF(self.file, "reading preCluster", verbose=True)
    if not EOF:
      data = self.readATrack()
      self.nbrOfReadTracks += 1
    else:
      data = ()
      raise StopIteration
    return ( data )

  def read(self, verbose=False):
    # 
    EOF=False
    self.readBytes=0
    # Read header
    # ??? self.tracks = [None] * nEvents
    while not EOF:
      # Read the tracks
      headerSize = 6
      header = readInt4(self.file, headerSize)
      if (header[3] != -1 ):
        print("readATrack: bad preClusterListHeader", header[3])
        print(header)
        exit()
      # Int_t trackListHeader[] = { -1, iEvent, -1, -1, 0, nTracks };
      iEvent = header[1]  
      nTracks = header[5]  
      nEvents = len(self.tracks)
      if iEvent != ( nEvents - 1 ):
          for i in range(nEvents, iEvent+1):
            self.tracks.append([])
      for iTrack in range(nTracks):
        self.tracks[iEvent].append( self.readATrack(verbose=verbose))
      EOF=getEOF(self.file, "reading a new Track", verbose=verbose)
    return

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

