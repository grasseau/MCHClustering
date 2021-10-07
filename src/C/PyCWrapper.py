# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import numpy as np
import numpy.ctypeslib as np_ct
from ctypes import c_int
from ctypes import c_double
from ctypes import c_int
from ctypes import cdll
from ctypes.util import find_library

pcWrap = 0

def setupPyCWrapper():
  # Tests, unused here
  # print( find_library("c") )
  # print( find_library("g") )
  # 
  # load the library, using numpy mechanisms
  # libc = cdll.LoadLibrary("/usr/local/lib/libgslcblas.so")
  # print(libc)
  #
  # extCLib = np_ct.load_library("libgslcblas.so","/usr/local/lib")   
  # extCLib = np_ct.load_library("libgsl.so","/usr/local/lib") 
  extCLib = np_ct.load_library("../C/libExternalC.so", ".")

  # input type for the cos_doubles function
  # must be a double array, with single dimension that is contiguous
  array_1d_double = np_ct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
  array_1d_int    = np_ct.ndpointer(dtype=np.int32,  ndim=1, flags='CONTIGUOUS')
  array_1d_short  = np_ct.ndpointer(dtype=np.int16,  ndim=1, flags='CONTIGUOUS')
  #  
  # Setup Mathieson functions
  #
  extCLib.initMathieson.restype  = None
  extCLib.initMathieson.argtypes = None
  #
  extCLib.compute2DPadIntegrals.resType = None
  extCLib.compute2DPadIntegrals.argtypes = [ array_1d_double,
                                             c_int, c_int, array_1d_double]
  #
  extCLib.compute2DMathiesonMixturePadIntegrals.resType = 0
  extCLib.compute2DMathiesonMixturePadIntegrals.argtypes = [ 
                                                array_1d_double,     # xyInfSup 
                                                array_1d_double,     # theta,
                                                c_int, c_int, c_int, #  N, K, chamberId, 
                                                array_1d_double      # Integrals[]
                                                ]
  #
  extCLib.computeCij.resType = 0
  extCLib.computeCij.argtypes = [ 
                                                array_1d_double,     # xyInfSup 
                                                array_1d_double,     # theta,
                                                c_int, c_int, c_int, #  N, K, chamberId, 
                                                array_1d_double      # Integrals[]
                                                ]
  #
  # Setup Mathieson fitting functions
  #
  extCLib.fitMathieson.resType = None
  extCLib.fitMathieson.argtypes = [ array_1d_double,  # thetai
                                    array_1d_double,  # xyAndDxy
                                    array_1d_double,  # z (mesurred)
                                    array_1d_short,   # cath
                                    array_1d_double,  # zCathTotalCharge
                                    c_int, c_int,     # K, N 
                                    c_int, c_int,     # chamberId, jacobian
                                    array_1d_double,  # thetaf
                                    array_1d_double,  # Khi2
                                    array_1d_double   # pError
                                ]
  #
  # Setup Gaussian EM functions
  #

  extCLib.computeDiscretizedGaussian2D.resType = None
  extCLib.computeDiscretizedGaussian2D.argtypes = [ 
                                    array_1d_double,  # xyInfSup
                                    array_1d_double,  # theta
                                    c_int, c_int,     # K, N 
                                    c_int,            # k component 
                                    array_1d_double,  # z
                                ]
                                
  extCLib.generateMixedGaussians2D.resType = None
  extCLib.generateMixedGaussians2D.argtypes = [ 
                                    array_1d_double,  # xyInfSup
                                    array_1d_double,  # theta
                                    c_int, c_int,     # K, N 
                                    array_1d_double   # z
                                ]
                                
  extCLib.weightedEMLoop.resType = c_double
  extCLib.weightedEMLoop.argtypes = [ 
                                    array_1d_double,  # xyDxy
                                    array_1d_short,   # saturated
                                    array_1d_double,  # zObs
                                    array_1d_double,  # theta0
                                    array_1d_short,   # theta mask                                    
                                    c_int, c_int,     # K, N 
                                    c_int,            # mode
                                    c_double,         # LConvergence
                                    c_int,            # verbose
                                    array_1d_double   # theta
                                ]
  #
  extCLib.computeResidual.resType = None
  extCLib.computeResidual.argtypes = [ 
                                    array_1d_double,  # xyInfSup
                                    array_1d_double,  # z
                                    array_1d_double,  # theta
                                    c_int, c_int,     # K, N 
                                    array_1d_double   # residual
                                ]
  #
  #
  extCLib.computeMathiesonResidual.resType = None
  extCLib.computeMathiesonResidual.argtypes = [ 
                                    array_1d_double,  # xyInfSup
                                    array_1d_short,   # cath
                                    array_1d_double,  # z
                                    array_1d_double,  # theta
                                    c_int,            # chId
                                    c_int, c_int,     # K, N 
                                    array_1d_double   # residual
                                ]
  #
  # Setup Pad Processing functions
  #
  extCLib.projectChargeOnOnePlane.resType = None
  extCLib.projectChargeOnOnePlane.argtypes = [
                                                array_1d_double,  # xy0Dxy
                                                array_1d_double,  # ch0  
                                                array_1d_double,  # xy1Dxy  
                                                array_1d_double,  # ch1
                                                c_int, c_int,     # N0, N1
                                                c_int             # includeAlonePads
                                            ]
  extCLib.projectChargeOnOnePlaneWithTheta.resType = None
  extCLib.projectChargeOnOnePlaneWithTheta.argtypes = [
                                                array_1d_double,  # xy0Dxy
                                                array_1d_double,  # ch0  
                                                array_1d_double,  # xy1Dxy  
                                                array_1d_double,  # ch1
                                                array_1d_double,  # chTheta0
                                                array_1d_double,  # chTheta1
                                                c_int, c_int,     # N0, N1
                                                c_int             # includeAlonePads
                                            ]
  #
  extCLib.copyProjectedPads.resType = None
  extCLib.copyProjectedPads.argtypes = [
                                        array_1d_double,  # xyDxy
                                        array_1d_double,  # chA
                                        array_1d_double,  # chB
                                       ]
  #
  extCLib.collectProjectedMinMax.resType = None
  extCLib.collectProjectedMinMax.argtypes = [
                                        array_1d_double,  # Max Ch
                                        array_1d_double   # Max Ch
                                       ]
  # 
  extCLib.findLocalMaxWithLaplacian.resType = c_int
  extCLib.findLocalMaxWithLaplacian.argtypes = [
                                                array_1d_double,  # xyDxy
                                                array_1d_double,  # z
                                                c_int, c_int,     # N, NAllocated
                                                array_1d_double,  # Laplacian
                                                array_1d_double,  # Theta
                                                ]
  #
  extCLib.getConnectedComponentsOfProjPads.resType = c_int
  extCLib.getConnectedComponentsOfProjPads.argtypes = [ array_1d_short ]  # Pad Groups
  #
  extCLib.assignCathPadsToGroupFromProj.resType = None
  extCLib.assignCathPadsToGroupFromProj.argtypes = [
                                            array_1d_short,     # padGroup
                                            c_int, c_int,       # nPads, nGroup (# of pad Groups)
                                            c_int, c_int,       # nCath0, nCath1
                                            array_1d_short,     # well split groups
                                            ]
  #                                            
  extCLib.copyCathToGrpFromProj.resType = None
  extCLib.copyCathToGrpFromProj.argtypes =  [
                                    array_1d_short,     # cath0Grp
                                    array_1d_short,     # cath1Grp
                                    c_int, c_int,       # nCath0, nCath1
                                    ]
  #
  extCLib.clusterProcess.resType = c_int
  extCLib.clusterProcess.argtypes = [
                                    array_1d_double,    # xyDxy
                                    array_1d_short,     # cathi
                                    array_1d_short,     # saturated
                                    array_1d_double,    # zi
                                    c_int,              # chId
                                    c_int               # nPads
                                    ] 
  #
  extCLib.setMathiesonVarianceApprox.resType = None
  extCLib.setMathiesonVarianceApprox.argtypes = [
                                                c_int,              # chId
                                                array_1d_double,    # theta
                                                c_int               # K
                                                ]
  #
  extCLib.collectTheta.resType  = None
  extCLib.collectTheta.argtypes = [
                                    array_1d_double,    # theta
                                    array_1d_short,     # thetaToGrp
                                    c_int               # N
                                  ]
  #
  extCLib.getNbrOfPadsInGroups.resType = c_int
  extCLib.getNbrOfPadsInGroups.argtypes = None
  #
  extCLib.getNbrOfProjPads.resType = c_int
  extCLib.getNbrOfProjPads.argtypes = None
  #
  extCLib.getNbrOfThetaEMFinal.resType = c_int
  extCLib.getNbrOfThetaEMFinal.argtypes = None
  #
  extCLib.collectPadsAndCharges.resType  = None
  extCLib.collectPadsAndCharges.argtypes = [
                                    array_1d_double,    # xyDxy
                                    array_1d_double,    # z
                                    array_1d_short,     # padToGrp
                                    c_int               # nTot
                                    ] 
  #
  extCLib.getKThetaInit.resType = c_int
  extCLib.getKThetaInit.argtypes = None
  #
  extCLib.collectLaplacian.resType  = None
  extCLib.collectLaplacian.argtypes = [
                                    array_1d_double,    # Laplacian
                                    c_int               # nTot
                                    ] 
  #
  extCLib.collectPadToCathGroup.resType  = None
  extCLib.collectPadToCathGroup.argtypes = [
                                    array_1d_short,    # padToGrp
                                    c_int              # nPads
                                    ] 
  #
  extCLib.collectResidual.resType  = None
  extCLib.collectResidual.argtypes = [
                                    array_1d_double,    # Residual
                                    c_int               # nTot
                                    ] 
  #  
  extCLib.collectThetaInit.resType  = None
  extCLib.collectThetaInit.argtypes = [
                                    array_1d_double,    # theta init
                                    c_int               # K
                                    ]  
  #
  extCLib.collectThetaEMFinal.resType  = None
  extCLib.collectThetaEMFinal.argtypes = [
                                          array_1d_double,    # theta EM Final
                                          c_int               # K
                                         ]  
  #
  extCLib.freeMemoryPadProcessing.resType = c_int
  extCLib.freeMemoryPadProcessing.argtypes = None
  #
  global CLib
  CLib = extCLib
  #
  return extCLib

def compute2DPadIntegrals( xyInfSup, chId ):
  
  N = int( xyInfSup.size / 4)
  z = np.zeros( (N), dtype=np.float64 )
  CLib.compute2DPadIntegrals( xyInfSup, N, chId, z)
  return z

def compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, chamberId ):
  N = int( xyInfSup.size / 4)
  K = int( theta.size / 5)
  integrals = np.zeros( (N) )
  CLib.compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, N, K, chamberId, integrals )
  return integrals

def computeCij( xyInfSup, theta, chamberId ):
  N = int( xyInfSup.size / 4)
  K = int( theta.size / 5)
  Cij = np.zeros( (N*K) )
  CLib.computeCij( xyInfSup, theta, N, K, chamberId, Cij )
  return Cij.reshape((N,K), order="F") 


def fitMathieson( thetai, xyAndDxy, cath, z, chId, verbose=0, doJacobian=0, doKhi=0, doStdErr=0):
  # process : verbose(2bits) + doJacobian(1bit) + computeKhi2(1bit) + computeStdDev(1bit)
  N = int( xyAndDxy.size / 4)
  K = int( thetai.size / 5)
  doProcess = verbose + (doJacobian << 2) + ( doKhi << 3) + (doStdErr << 4)
  zCathTotalCharge = np.array([ np.sum(z[cath==0]), np.sum(z[cath==1]) ])
  # Returned values
  thetaf = np.zeros( (5*K) )
  khi2 = np.zeros( (1) )
  pError = np.zeros( (3*K * 3*K) )
  """
  extCLib.fitMathieson.argtypes = [ array_1d_double,  # thetai
                                    array_1d_double,  # xyAndDxy
                                    array_1d_double,  # z (mesurred)
                                    array_1d_short,   # cath
                                    array_1d_double,  # zCathTotalCharge
                                    c_int, c_int,     # K, N 
                                    c_int, c_int,     # chamberId, jacobian
                                    array_1d_double,  # thetaf
                                    array_1d_double,  # Khi2
                                    array_1d_double   # pErr
  """                               
  CLib.fitMathieson( thetai, xyAndDxy, z, cath, zCathTotalCharge, K, N,
                         chId, doProcess, 
                         thetaf, khi2, pError 
                      )
  return ( thetaf, khi2, pError)

def computeDiscretizedGaussian2D( xyInfSup, theta, K, N, k):
  N = int( xyInfSup.size / 4)
  K = int( theta.size / 5)
  z = np.zeros( (N) )
  Clib.computeDiscretizedGaussian2D( xyInfSup, theta, K, N, k, z)
  return z

def generateMixedGaussians2D( xyInfSup, theta): 
  N = int( xyInfSup.size / 4)
  K = int( theta.size / 5)
  z = np.zeros( (N) )
  CLib.generateMixedGaussians2D( xyInfSup, theta, K, N, z)
  return z

def weightedEMLoop( xyDxy, saturated, zObs, thetai, thetaMask, mode, LConvergence, verbose):
  N = int( xyDxy.size / 4)
  K = int( thetai.size / 5)
  thetaf = np.zeros( K*5 )
  sat = saturated.astype( np.int16 )
  logL = CLib.weightedEMLoop( xyDxy, sat, zObs, thetai, thetaMask, K, N, mode, LConvergence, verbose, thetaf )
  logL = thetaf[0]
  return ( thetaf, logL )

def copyProjectedPads( ):
  #
  nbrOfProjPads = CLib.getNbrProjectedPads()
  xyDxyProj = np.zeros( (4*nbrOfProjPads), dtype=np.float64 )
  chA = np.zeros( nbrOfProjPads, dtype=np.float64 )
  chB = np.zeros( nbrOfProjPads, dtype=np.float64 )
  #
  CLib.copyProjectedPads( xyDxyProj, chA, chB )
  npj = nbrOfProjPads
  xProj = np.zeros(npj)
  yProj = np.zeros(npj)
  dxProj = np.zeros(npj)
  dyProj = np.zeros(npj)
  xProj[:] = xyDxyProj[0*npj:1*npj]
  yProj[:] = xyDxyProj[1*npj:2*npj]
  dxProj[:] = xyDxyProj[2*npj:3*npj]
  dyProj[:] = xyDxyProj[3*npj:4*npj] 
  #
  return (xProj, dxProj, yProj, dyProj, chA, chB)

def collectProjectedMinMax(nProj):
  minCh = np.zeros( nProj)
  maxCh = np.zeros( nProj)
  CLib.collectProjectedMinMax( minCh, maxCh )
  return minCh, maxCh

def projectChargeOnOnePlane( x0, dx0, y0, dy0, x1, dx1, y1, dy1, ch0, ch1):
  N0 = x0.size
  xy0InfSup = np.zeros( (4*N0) )
  xy0InfSup[0*N0:1*N0] = x0 - dx0
  xy0InfSup[1*N0:2*N0] = y0 - dy0
  xy0InfSup[2*N0:3*N0] = x0 + dx0
  xy0InfSup[3*N0:4*N0] = y0 + dy0
  N1 = x1.size
  xy1InfSup = np.zeros( (4*N1) )
  xy1InfSup[0*N1:1*N1] = x1 - dx1
  xy1InfSup[1*N1:2*N1] = y1 - dy1
  xy1InfSup[2*N1:3*N1] = x1 + dx1
  xy1InfSup[3*N1:4*N1] = y1 + dy1
  includeAlonePads = 1
  CLib.projectChargeOnOnePlane( xy0InfSup, ch0, xy1InfSup, ch1, N0, N1, includeAlonePads)
  #
  # Get results
  #
  # proj = (xProj, dxProj, yProj, dyProj, chA, chB)
  proj = copyProjectedPads()
  #
  return proj

def projectChargeOnOnePlaneWithTheta( x0, dx0, y0, dy0, x1, dx1, y1, dy1, ch0, ch1, chTheta0, chTheta1):
  N0 = x0.size
  xy0InfSup = np.zeros( (4*N0) )
  xy0InfSup[0*N0:1*N0] = x0 - dx0
  xy0InfSup[1*N0:2*N0] = y0 - dy0
  xy0InfSup[2*N0:3*N0] = x0 + dx0
  xy0InfSup[3*N0:4*N0] = y0 + dy0
  N1 = x1.size
  xy1InfSup = np.zeros( (4*N1) )
  xy1InfSup[0*N1:1*N1] = x1 - dx1
  xy1InfSup[1*N1:2*N1] = y1 - dy1
  xy1InfSup[2*N1:3*N1] = x1 + dx1
  xy1InfSup[3*N1:4*N1] = y1 + dy1
  includeAlonePads = 1
  CLib.projectChargeOnOnePlaneWithTheta( xy0InfSup, ch0, xy1InfSup, ch1, chTheta0, chTheta1, N0, N1, includeAlonePads)
  #
  # Get results
  #
  # proj = (xProj, dxProj, yProj, dyProj, chA, chB)
  proj = copyProjectedPads()
  #
  return proj

def getConnectedComponentsOfProjPads():
  nbrOfProjPads = CLib.getNbrProjectedPads()
  padGrp = np.zeros( (nbrOfProjPads), dtype=np.int16 )
  nbrGroups = CLib.getConnectedComponentsOfProjPads( padGrp )
  return nbrGroups, padGrp

def findLocalMaxWithLaplacian( xyDxy, z, laplacian=None):
  N = int( xyDxy.size / 4)
  if laplacian is None:
    laplacian = np.zeros(N)
  theta0 = np.zeros(5*N)
  K = CLib.findLocalMaxWithLaplacian( xyDxy, z, N, N, laplacian, theta0)
  theta = np.zeros(5*K)
  theta[0*K:1*K] = theta0[0*N:0*N+K]
  theta[1*K:2*K] = theta0[1*N:1*N+K]
  theta[2*K:3*K] = theta0[2*N:2*N+K]
  theta[3*K:4*K] = theta0[3*N:3*N+K]
  theta[4*K:5*K] = theta0[4*N:4*N+K]
  return theta

def assignCathPadsToGroup( padGroup, nGroup, nCath0, nCath1 ):
  nPads = padGroup.size
  wellSplitGroup = np.zeros( nGroup, dtype=np.int16 )
  CLib.assignCathPadsToGroup( padGroup, nPads, nGroup, nCath0, nCath1, wellSplitGroup )
  return wellSplitGroup

def copyCathToGrp( nCath0, nCath1):
  cath0Grp = np.zeros( nCath0, dtype=np.int16)
  cath1Grp = np.zeros( nCath1, dtype=np.int16)
  CLib.copyCathToGrp( cath0Grp, cath1Grp, nCath0, nCath1)
  return (cath0Grp, cath1Grp)

def clusterProcess( xyDxyi, cathi, saturated, zi, chId ):
  N = int( xyDxyi.size / 4)
  sat = saturated.astype( np.int16 )
  nbrOfHits = CLib.clusterProcess( xyDxyi, cathi, sat, zi, chId, N)
  return nbrOfHits

def setMathiesonVarianceApprox( chId, theta):
  K = int( theta.size / 5)
  CLib.setMathiesonVarianceApprox(chId, theta, K)

def collectTheta( K ):
  if (K==0): 
    return (np.zeros( 0 ), np.zeros( 0 ))
  theta = np.zeros( K*5)
  thetaInGrp = np.zeros( K, dtype=np.int16)
  CLib.collectTheta( theta, thetaInGrp, K)
  return (theta, thetaInGrp)

def collectPadsAndCharges():
  N = CLib.getNbrOfPadsInGroups()
  print("[python] collectPadsAndCharges N=", N)
  xyDxy = np.zeros( N*4)
  z     = np.zeros( N)
  padToGrp = np.zeros( N, dtype=np.int16)
  CLib.collectPadsAndCharges( xyDxy, z, padToGrp, N)
  return ( xyDxy, z, padToGrp)

def collectLaplacian():
  N = CLib.getNbrOfProjPads()
  laplacian = np.zeros( N)
  CLib.collectLaplacian( laplacian, N)
  return laplacian

def collectPadToCathGroup( nPads ):
  padToMGrp = np.zeros( (nPads),  dtype = np.int16 )
  CLib.collectPadToCathGroup ( padToMGrp, nPads)
  return padToMGrp

def computeResidual( xyDxy, zObs, theta ):
  K = int( theta.size / 5)
  N = int( zObs.size )
  residual = np.zeros(N)
  CLib.computeResidual( xyDxy, zObs, theta, K, N, residual)
  return residual

def computeMathiesonResidual( xyDxy, cath, zObs, theta, chId ):
  K = theta.size // 5
  N = zObs.size
  residual = np.zeros(N)
  CLib.computeMathiesonResidual( xyDxy, cath, zObs, theta, chId, K, N, residual)
  return residual

def collectResidual():
  N = CLib.getNbrOfProjPads()
  residual = np.zeros( N)
  CLib.collectResidual( residual, N)
  return residual

def collectThetaInit():
  K = CLib.getKThetaInit()
  thetai = np.zeros( 5*K)
  CLib.collectThetaInit( thetai, K)
  return thetai

def collectThetaEMFinal():
  K = CLib.getNbrOfThetaEMFinal()
  thetaf = np.zeros( 5*K)
  CLib.collectThetaEMFinal( thetaf, K)
  return thetaf
  
def freeMemoryPadProcessing():
  CLib.freeMemoryPadProcessing()
  return
