# include <string.h>
# include <vector>
# include <algorithm> 
# include <stdexcept>

# include "dataStructure.h"
# include "mathUtil.h"
# include "gaussianEM.h"
# include "mathiesonFit.h"
# include "padProcessing.h"
# include "mathUtil.h"
# include "clusterProcessing.h"

bool orderFct(double i, double j) { return i<j; }

// To keep internal data
# define DoEM 1
# define INSPECTMODEL 1
# define VERBOSE 1
# define CHECK 1
# define DISABLE_EM_SATURATED 1

// Type of projection
// Here add alone pads
static int includeAlonePads = 1;
//
// EM parameters
static const double EMConvergence = 10.0e-6;
// EM mode : 1 constant variance
static const int EMmode = 1;
static const int EMverbose = 1;
// Mathieson approximation with one gaussian 
/*
static double cstSigXCh1ToCh2 = 0.1814;    
static double cstSigYCh1ToCh2 = 0.1835;
static double cstSigXCh3ToCh10 = 0.2128;
static double cstSigYCh3ToCh10 = 0.2145;
*/
static double cstSigXCh1ToCh2 = 0.1690;    
static double cstSigYCh1ToCh2 = 0.1738;
static double cstSigXCh3ToCh10 = 0.1978 ;
static double cstSigYCh3ToCh10 = 0.2024;
//
// Fit parameters
// doProcess = verbose + (doJacobian << 2) + ( doKhi << 3) + (doStdErr << 4)
static const int processFit = 1 + (0 << 2) + ( 1 << 3) + (1 << 4);
// Limit of pad number  to perform the fitting
static const int nbrOfPadsLimitForTheFitting = 50;
//
// Component (hit/seeds) selection
// w cutoff
static const double wCutoff = 5.e-2; 

// Private variables
//
// Pads with cathodes
static double *xy0Dxy = 0;
static double *xy1Dxy = 0;
static double *ch0=0, *ch1=0;
static Mask_t *satPads0 = 0;
static Mask_t *satPads1 = 0;
// Mapping from cathode-pad to original pads
static PadIdx_t *cath0ToPadIdx = 0; 
static PadIdx_t *cath1ToPadIdx = 0; 
static int nMergedGrp = 0;
static short *padToMergedGrp = 0;

//
// Projection
static int nProjPads = 0;
static double *xyDxyProj = 0;
static Mask_t *saturatedProj=0;
static double *chProj    = 0;
static short *wellSplitGroup = 0;
//
// Hits/seeds founds per sub-cluster
static std::vector< DataBlock_t > subClusterThetaList;

// Inspect data
typedef struct dummy_t {
    // Data on Projected Pads
    int nbrOfProjPads;
    double *laplacian;
    // Residual between projected charges and the EM model
    double *residualProj;
    // Theta init
    double *thetaInit;
    int kThetaInit;
    // Data about subGroups
    int totalNbrOfSubClusterPads; 
    int totalNbrOfSubClusterThetaEMFinal; 
    std::vector< DataBlock_t > subClusterPadList;
    std::vector< DataBlock_t > subClusterChargeList;
    std::vector< DataBlock_t > subClusterThetaEMFinal;
    // Cath groups
    int nCathGroups;
    short *padToCathGrp;
} InspectModel_t;
//
static InspectModel_t inspectModel={.nbrOfProjPads=0, .laplacian=0, .residualProj=0, .thetaInit=0, .kThetaInit=0, 
  .totalNbrOfSubClusterPads=0, .totalNbrOfSubClusterThetaEMFinal=0, .nCathGroups=0, .padToCathGrp=0};

// Total number of hits/seeds in the precluster;
static int nbrOfHits = 0;
//
void setMathiesonVarianceApprox( int chId, double *theta, int K ) {
  double *varX = getVarX( theta, K);
  double *varY = getVarY( theta, K);
  double cstVarX, cstVarY;
  if ( chId <= 2 ) { 
    cstVarX = cstSigXCh1ToCh2 * cstSigXCh1ToCh2; 
    cstVarY = cstSigYCh1ToCh2 * cstSigYCh1ToCh2; 
  } else {
    cstVarX = cstSigXCh3ToCh10 * cstSigXCh3ToCh10; 
    cstVarY = cstSigYCh3ToCh10 * cstSigYCh3ToCh10;
  }
  for (int k=0; k<K; k++) {
    varX[k] = cstVarX; 
    varY[k] = cstVarY; 
  }
}

void deleteDouble( double *ptr) {
  if( ptr != 0) {
    delete[] ptr;
    ptr = 0;
  }
}

// Remove hits/seeds according to w-magnitude 
// and hits/seeds proximity
// Return the new # of components (hits/seeds)
int filterEMModel( double *theta, int K, Mask_t *maskFilteredTheta) {
  // w cut-off
  double cutOff = 1.0 / K * wCutoff;
  //
  double *w    = getW   ( theta, K);
  double *muX  = getMuX ( theta, K);
  double *muY  = getMuY ( theta, K);
  double *varX = getVarX( theta, K);
  double *varY = getVarY( theta, K);
  //
  // Sort w (index sorting)
  // Indexes for sorting
  int index[K]; 
  for( int k=0; k<K; k++) index[k]=k;
  std::sort( index, &index[K], [=](int a, int b){ return w[a] > w[b]; });
  // Test ??? to supress
  if (VERBOSE) {
    vectorPrint("  sort w", w, K);
    vectorPrintInt("  sort w-indexes",index, K);
  }
  // Mode constant variance for all hits/seeds 
  double sigX = sqrt( varX[0]);
  double sigY = sqrt( varY[0]);
  //
  // Reorder theta in fct of the w sort
  // and suppress closed theta[k] (hit/seeds)
  // ( w is set to zero)
  for ( int i=0; i < K; i++) {
    double muXi = muX[index[i]];
    double muYi = muY[index[i]];
    // Suppress low w
    if ( w[index[i]] > cutOff) {
      for ( int j=i+1; j < K; j++) {
        double dMuX = fabs( muXi - muX[index[j]] );
        double dMuY = fabs( muYi - muY[index[j]] );
        // Suppress closed hits/seeds
        if (( dMuX < sigX ) && ( dMuY < sigY )) w[index[j]] = 0;
      }  
    } else { w[index[i]] = 0;} 
  }
  
  
  // Build the mask w[k] >= 0.0
  for ( int k=0; k < K; k++) maskFilteredTheta[k] = ( w[k] > 0.0 );
  int newK = vectorSumShort( maskFilteredTheta, K);
  return newK;
}

int filterFitModel( double *theta, int K, Mask_t *maskFilteredTheta) {
  // w cut-off
  double cutOff = 0.05 / K;
  //
  double *w    = getW   ( theta, K);
  int newK = 0;
  for( int k=0; k<K; k++) {
    maskFilteredTheta[k] = (w[k] > cutOff); 
    newK += (w[k] > cutOff);
  }
  return newK;
}

// Used when several sub-cluster occur in the precluster
// Append the new hits/clusters in the thetaList of the pre-cluster
void copyInGroupList( const double *values, int N, int item_size, std::vector< DataBlock_t > &groupList) {
  double *ptr = new double[N*item_size];
  // memcpy( (void *) ptr, (const void*) values, N*item_size*sizeof(double));
  vectorCopy( values, N*item_size, ptr);
  groupList.push_back( std::make_pair(N, ptr));
}

void appendInThetaList( const double *values, int N, std::vector< DataBlock_t > &groupList) {
  // double *ptr = new double[N];
  // memcpy( (void *) ptr, (const void*) theta, N*sizeof(double));
  groupList.push_back( std::make_pair(N, values));
}

void cleanThetaList( ) {
  for (int i = 0; i < subClusterThetaList.size(); i++) 
    delete[] subClusterThetaList[i].second;
  subClusterThetaList.clear();
}


void collectTheta( double *theta, Group_t *thetaToGroup, int K) {
  int sumK=0;
  for (int h=0; h < subClusterThetaList.size(); h++) {
    int k = subClusterThetaList[h].first;
    copyTheta( subClusterThetaList[h].second, k, &theta[sumK], K, k );
    if ( thetaToGroup ) {
      vectorSetShort( &thetaToGroup[sumK], h+1, k);
    }
    sumK += k;
    if (VERBOSE) {
        printf("collect theta grp=%d,  grpSize=%d, adress=%p\n", h, k, subClusterThetaList[h].second);
    }
    delete[] subClusterThetaList[h].second;
  }
  if ( sumK > K) {
    printf("Bad allocation for collectTheta sumK=%d greater than K=%d\n", sumK, K);
    throw std::overflow_error("Bad Allocation");
  }
  subClusterThetaList.clear();
}

void cleanInspectModel( ) {
  //
  for (int i = 0; i < inspectModel.subClusterPadList.size(); i++) 
    delete[] inspectModel.subClusterPadList[i].second;
  inspectModel.subClusterPadList.clear();
  //
  for (int i = 0; i < inspectModel.subClusterChargeList.size(); i++) 
    delete[] inspectModel.subClusterChargeList[i].second;
  inspectModel.subClusterChargeList.clear();
  //
  for (int i = 0; i < inspectModel.subClusterThetaEMFinal.size(); i++) 
    delete[] inspectModel.subClusterThetaEMFinal[i].second;
  inspectModel.subClusterThetaEMFinal.clear();
  //  
  if ( inspectModel.laplacian != 0) { delete[] inspectModel.laplacian; inspectModel.laplacian=0; }
  if ( inspectModel.residualProj != 0) { delete[] inspectModel.residualProj; inspectModel.residualProj=0; }
  if ( inspectModel.thetaInit != 0) { delete[] inspectModel.thetaInit; inspectModel.thetaInit = 0;}
  //
  inspectModel.totalNbrOfSubClusterPads = 0;
  inspectModel.totalNbrOfSubClusterThetaEMFinal = 0;
  // Cath group
  delete [] inspectModel.padToCathGrp;
  inspectModel.padToCathGrp = 0;
  inspectModel.nCathGroups = 0;  
}

void finalizeInspectModel() {
  int sumN=0;
  for (int h=0; h < inspectModel.subClusterPadList.size(); h++) {
    int n = inspectModel.subClusterPadList[h].first;
    sumN += n;
  }
  inspectModel.totalNbrOfSubClusterPads = sumN;
  //
  int sumK=0;
  for (int h=0; h < inspectModel.subClusterThetaEMFinal.size(); h++) {
    int k = inspectModel.subClusterThetaEMFinal[h].first;
    sumK += k;
  }
  inspectModel.totalNbrOfSubClusterThetaEMFinal = sumK;  
}

int getNbrOfProjPads() {
 return inspectModel.nbrOfProjPads;  
}

int getNbrOfPadsInGroups() {
  return inspectModel.totalNbrOfSubClusterPads;
}

int getNbrOfThetaEMFinal() {
  return inspectModel.totalNbrOfSubClusterThetaEMFinal;
}

// ???
// Optim collectXXX can be replaced by getConstPtrXXX

void collectPadToCathGroup( Mask_t *padToMGrp, int nPads ) {
  vectorCopyShort( inspectModel.padToCathGrp, nPads, padToMGrp);
}

/// ???
void collectPadsAndCharges( double *xyDxy, double *z, Group_t *padToGroup, int N) {
  int sumN=0;
  for (int h=0; h < inspectModel.subClusterPadList.size(); h++) {
    int n = inspectModel.subClusterPadList[h].first;
    copyXYdXY( inspectModel.subClusterPadList[h].second, n, &xyDxy[sumN], N, n );
    vectorCopy( inspectModel.subClusterChargeList[h].second, n, &z[sumN]);
    if ( padToGroup ) {
      vectorSetShort( &padToGroup[sumN], h+1, n);
    }
    sumN += n;
    delete[] inspectModel.subClusterPadList[h].second;
    delete[] inspectModel.subClusterChargeList[h].second;
  }
  if ( sumN > N) {
    printf("Bad allocation for collectTheta sumN=%d greater than N=%d\n", sumN, N);
    throw std::overflow_error("Bad Allocation");
  }
  inspectModel.subClusterPadList.clear();
  inspectModel.subClusterChargeList.clear();
  inspectModel.totalNbrOfSubClusterPads = 0;
}

void collectLaplacian( double *laplacian, int N) {
  vectorCopy( inspectModel.laplacian, N, laplacian );
}

void collectResidual( double *residual, int N) {
  vectorCopy( inspectModel.residualProj, N, residual );
}

int getKThetaInit() {
 return inspectModel.kThetaInit;  
}

void collectThetaInit( double *thetai, int N) {
  vectorCopy( inspectModel.thetaInit, 5*N, thetai );
}

void collectThetaEMFinal( double *thetaEM, int K) {
  int sumK=0;
  for (int h=0; h < inspectModel.subClusterThetaEMFinal.size(); h++) {
    int k = inspectModel.subClusterThetaEMFinal[h].first;
    copyTheta( inspectModel.subClusterThetaEMFinal[h].second, k, &thetaEM[sumK], K, k );
    sumK += k;
    delete[] inspectModel.subClusterThetaEMFinal[h].second;
  }
  if ( sumK > K) {
    printf("Bad allocation for collectThetaEMFinal sumN=%d greater than N=%d\n", sumK, K);
    throw std::overflow_error("Bad Allocation");
  }
  inspectModel.subClusterThetaEMFinal.clear();
  inspectModel.totalNbrOfSubClusterThetaEMFinal = 0;
}

void getIndexInPadProjGrp( const Mask_t *maskThetaGrp, const int *thetaPadProjIdx, const int *mapProjIdxToProjGrpIdx, 
                           int KProj, int *thetaPadProjGrpIdx ) {
// maskThetaGrp    : Mask of projPads belonging to the current group
// thetaPadProjIdx : index in projPad array of a seed theta[k]
//                   In other words map a seed theta[k] on a projPad
// mapProjIdxToProjGrpIdx : map projPads idx to projPadGrp (subset projPads of the group)
// KProj : nbr of projPads
// thetaPadProjGrpIdx : index in projPadGroup array of a seed theta[k]
//                      In other words map a seed theta[k] on a projPadGroup
  int ii =0;
  for (int kk=0; kk < KProj; kk++) {
    if (maskThetaGrp[kk]) {
        // A max (or seed) belonging to the current grp
        // Get the pad-index in the whole set of projPad
        int idxProj = thetaPadProjIdx[kk];
        // Get the location in padProjGrp set
        // get the order projPadGrp
        int uu = mapProjIdxToProjGrpIdx[idxProj];
        if (uu < 0) {
          printf("Index in projPad %d padIdxInGrp %d \n", idxProj, uu);
          printf("#### Bad index %d %d\n", idxProj, kk);
          throw std::overflow_error("Bad Allocation");
        }
        thetaPadProjGrpIdx[ii] = uu;
        ii++;
    }
  }
}
      
int kOptimizer( double *xyDxy, Mask_t *saturated, double *z, 
                double *theta, int K, int N,
                int    *mapThetaToPadIdx,
                double *thetaOpt, Mask_t *thetaOptMask ) {
  // ??? Do something for thetaOptMask or take w[k] = 0
  //  
  const double relEps = (1.0 + 1.0e-4);
  // theta
  double *w    = getW   ( theta, K);
  double *muX  = getMuX ( theta, K);
  double *muY  = getMuY ( theta, K);
  // dx, dy pads
  double *projX = getX( xyDxy, N);
  double *projY = getY( xyDxy, N);
  double *projDX = getDX( xyDxy, N);
  double *projDY = getDY( xyDxy, N);  
  // Local allocation
  Mask_t proximityMatrix[K*K];
  short sumRow[K];
  // short done[K];
  vectorSetZeroShort( proximityMatrix, K*K);
  
  if (VERBOSE) printf("kOptimizer K=%d\n", K); 
  // vectorPrintInt("  pad associated with a max", mapThetaToPadIdx, K);
  //
  // Build the proximityMatrix (parameter's neighbors) 
  //
  for( int k0=0; k0 < K; k0++ ) {
    double x0 = muX[ k0 ];
    double y0 = muY[ k0 ];
    double dx0 = projDX[ mapThetaToPadIdx[k0] ];
    double dy0 = projDY[ mapThetaToPadIdx[k0] ];
    int rowCumul = 0;
    for( int k1=k0+1; k1 < K; k1++ ) {
      double x1 = muX[ k1 ];
      double y1 = muY[ k1 ];
      double dx1 = projDX[ mapThetaToPadIdx[k1] ];
      double dy1 = projDY[ mapThetaToPadIdx[k1] ];
      // printf(" x ... %g %g %g %g\n", x0, x1, dx0, dx1 );
      // printf(" y ... %g %g %g %g\n", y0, y1, dy0, dy1 );
      // printf(" dx, dx0+dx1,  dy, dy0+dy1,... %g %g %g %g\n", fabs (x0 - x1), ( dx0 + dx1), fabs (y0 - y1),  ( dy0 + dy1));
      Mask_t maskX = fabs (x0 - x1) < relEps * ( dx0 + dx1);
      Mask_t maskY = fabs (y0 - y1) < relEps * ( dy0 + dy1);
      proximityMatrix[ k0*K + k1] = maskX && maskY;
      // Not used proximityMatrix[ k1*K + k0] = maskX && maskY;
      if( maskX && maskY ) rowCumul += 1;
    }
    sumRow[k0] = rowCumul;
  }
  if (VERBOSE) printMatrixShort( "  proximity Matrix", proximityMatrix, K, K);
  //
  // EM on all k's
  double thetaTest[5*K];
  vectorCopy( theta, 5*K, thetaTest);
  // Mask_t thetaMaskOpt[K];
  vectorSetShort( thetaOptMask, 1, K);
  double minBIC = weightedEMLoop( xyDxy, saturated, z, 
              theta, thetaOptMask, K, N, 
              EMmode, EMConvergence, EMverbose, thetaOpt);
  if (VERBOSE) { 
    printTheta( "Config with all theta", theta, K);
    printf("  BIC %8g.3\n", minBIC);
  }
  //
  // Try to fusion k's
  int betterConfig = 0;
  Mask_t thetaTestMask[K];
  double thetaTestResult[5*K];
  for( int k0=0; k0 < K; k0++ ) {
    // printf(" ??? sumRow[%d] %d\n", k0, sumRow[k0]);
    if ( sumRow[k0] > 0 ) {
      // Theta to test
      vectorCopy( thetaOpt, 5*K, thetaTest);
      double *wTest  = getW   ( thetaTest, K);
      double *xTest  = getMuX ( thetaTest, K);
      double *yTest  = getMuY ( thetaTest, K);
      vectorCopyShort( thetaOptMask, K, thetaTestMask);
      double wSum = w[k0];
      double xSum = muX[ k0 ];
      double ySum = muY[ k0 ];
      int n = 1;
      for( int k1=k0+1; k1 < K; k1++ ) {  
        if ( proximityMatrix[ k0*K + k1] ) {
          xSum += muX[ k1 ];
          ySum += muY[ k1 ];
          wSum += w[k1];
          wTest[k1] = 0.;
          thetaTestMask[k1] = 0; 
          n++;
        }
      }
      xTest[k0] = xSum / n;
      yTest[k0] = ySum / n;
      wTest[k0] = wSum ;
      if (VERBOSE) {
        printTheta("  Config theta", thetaTest, K);
        vectorPrintShort("  thetaTestMask", thetaTestMask, K);
      }
      double BIC = weightedEMLoop( xyDxy, saturated, z, 
              thetaTest, thetaTestMask, K, N, 
              EMmode, EMConvergence, EMverbose, thetaTestResult);
      if (VERBOSE) { 
        printf(" kOptimizer BIC %10.5g\n", BIC );
      }
      // if ( BIC < minBIC) {
      printf(" kOptimizer BIC Test %10.5g > %10.5g \n", fabs( BIC - minBIC),  0.04 * fabs( minBIC));  
      if ( (BIC > minBIC) && fabs( BIC - minBIC) > 0.04 * fabs( minBIC) ) {
        minBIC = BIC;
        // Copy Configuration
        vectorCopy( thetaTestResult, 5*K, thetaOpt );
        vectorCopyShort( thetaTestMask, K, thetaOptMask);
        betterConfig = 1;
        if (VERBOSE) {
          printTheta("New best theta config.", thetaOpt, K);
        }
      }
    }
  }
  //
  int newK = K;
  if ( betterConfig) {
    newK = vectorSumShort( thetaOptMask, K );
  }
  return newK;
}

void computeMathiesonResidual( const double *xyDxy, const Mask_t *cath, const double *zObs, const double *theta, int chId, int K, int N, double *residual) {

  // GG duplicated code with EM module ???
  // define x, y, dx, dy description
  const double *x  = getConstX( xyDxy, N);
  const double *y  = getConstY( xyDxy, N);
  const double *dX = getConstDX( xyDxy, N);
  const double *dY = getConstDY( xyDxy, N); 
  
  // Compute boundary of each pads
  double xyInfSup[4*N];
  vectorAddVector( x, -1.0, dX, N, getXInf(xyInfSup, N) );
  vectorAddVector( y, -1.0, dY, N, getYInf(xyInfSup, N) );
  vectorAddVector( x, +1.0, dX, N, getXSup(xyInfSup, N) );
  vectorAddVector( y, +1.0, dY, N, getYSup(xyInfSup, N) );
  compute2DMathiesonMixturePadIntegrals( xyInfSup, theta, N, K, chId, residual );
  // generateMixedGaussians2D( xyInfSup, theta, K, N, residual);
  Mask_t cath0[N];  
  vectorNotShort( cath, N, cath0);
  double sumCh0 = vectorMaskedSum( zObs, cath0, N);
  double sumCh1 = vectorMaskedSum( zObs, cath, N);
  vectorMaskedMultScalar( residual, cath0, sumCh0, sumCh1, N);
  vectorAddVector( zObs, -1.0, residual, N, residual);
 }

 void computeResidual( const double *xyDxy, const double *zObs, const double *theta, int K, int N, double *residual) {

  // GG duplicated code with EM module ???
  // define x, y, dx, dy description
  const double *x  = getConstX( xyDxy, N);
  const double *y  = getConstY( xyDxy, N);
  const double *dX = getConstDX( xyDxy, N);
  const double *dY = getConstDY( xyDxy, N); 
  
  // Compute boundary of each pads
  double xyInfSup[4*N];
  vectorAddVector( x, -1.0, dX, N, getXInf(xyInfSup, N) );
  vectorAddVector( y, -1.0, dY, N, getYInf(xyInfSup, N) );
  vectorAddVector( x, +1.0, dX, N, getXSup(xyInfSup, N) );
  vectorAddVector( y, +1.0, dY, N, getYSup(xyInfSup, N) );     
  generateMixedGaussians2D( xyInfSup, theta, K, N, residual);
  double sumCh = - vectorSum( zObs, N);
  vectorAddVector( zObs, sumCh, residual, N, residual);
   
 }
 
// Extract hits/seeds of a pre-cluster 
int clusterProcess( const double *xyDxyi_, const Mask_t *cathi_, const Mask_t *saturated_, const double *zi_, int chId, int nPads) {
// Remarks:

  nbrOfHits = 0;
  cleanThetaList();
  if (INSPECTMODEL) cleanInspectModel();
  
  const double *xyDxyi;
  const double *zi;
  const Mask_t *cathi;
  const Mask_t *saturated;

  // Large and Noisy  clusters
  double *xyDxyi__;
  double *zi__;
  Mask_t *cathi__;
  Mask_t *saturated__;  
  Mask_t noiseMask[nPads];
  int nNewPads=0;
  if (nPads > 800) {
    // Remove noisy pads
    printf("remove noisy pads <z>=%f, min/max z=%f,%f", vectorSum(zi_, nPads)/ nPads, vectorMin(zi_, nPads), vectorMax(zi_, nPads));  
    vectorBuildMaskGreater( zi_, 2.0, nPads, noiseMask );
    nNewPads = vectorSumShort( noiseMask, nPads);
    xyDxyi__ = new double[nNewPads*4];
    zi__ = new double[nNewPads];
    saturated__ = new Mask_t[nNewPads];
    cathi__ = new Mask_t[nNewPads];
    
    vectorGather( zi_, noiseMask, nPads, zi__);
    vectorGatherShort(saturated_, noiseMask, nPads, saturated__);
    vectorGatherShort( cathi_, noiseMask, nPads, cathi__);
    maskedCopyXYdXY( xyDxyi_,  nPads, noiseMask,  nPads, 
                      xyDxyi__, nNewPads);
    xyDxyi = xyDxyi__;
    zi = zi__;
    cathi = cathi__;
    saturated = saturated__;
    nPads = nNewPads;
  } else {
    xyDxyi = xyDxyi_;
    zi = zi_;
    cathi = cathi_;
    saturated = saturated_;
  }
  
  int isSaturated = vectorSumShort( saturated, nPads );  
  
  // TODO ???
  // if nPads <= 6 go directly to fit
  // 1 seed is assumed compute the barycenter
  
  // Create mask on the cathode values
  // Must be the opposite of cathote value
  const Mask_t *maskCath1 = cathi;
  Mask_t maskCath0[nPads];
  vectorNotShort( maskCath1, nPads, maskCath0);
  double sumZ0 = vectorMaskedSum( zi, maskCath0, nPads);
  double sumZ1 = vectorMaskedSum( zi, maskCath1, nPads);
  int nbrCath0 = vectorSumShort( maskCath0, nPads );
  int nbrCath1 = vectorSumShort( maskCath1, nPads );
  if ( nbrCath0 > 0) cath0ToPadIdx = new PadIdx_t[nbrCath0];
  if ( nbrCath1 > 0) cath1ToPadIdx = new PadIdx_t[nbrCath1];
  vectorGetIndexFromMask( maskCath0, nPads, cath0ToPadIdx );
  vectorGetIndexFromMask( maskCath1, nPads, cath1ToPadIdx );
  PadIdx_t mapPadToCathIdx[nPads];
  vectorSetInt(mapPadToCathIdx, -1, nPads);
  for( int c=0; c< nbrCath0; c++) {
    mapPadToCathIdx[ cath0ToPadIdx[c] ] = c;
  }
  for( int c=0; c< nbrCath1; c++) {
    mapPadToCathIdx[ cath1ToPadIdx[c] ] = c;
  }
  vectorPrintInt( "cath0ToPadIdx", cath0ToPadIdx, nbrCath0);
  vectorPrintInt( "cath1ToPadIdx", cath1ToPadIdx, nbrCath1);
  vectorPrintInt( "mapPadToCathIdx", mapPadToCathIdx, nPads);
  //
  // uniqueCath = id (0/1) of the unique cathode, -1 (2 cath.) if not
  short uniqueCath = -1;
  if (nbrCath0 > 0 ) uniqueCath = 0;
  if (nbrCath1 > 0 ) uniqueCath = 1;
  if ( (nbrCath0 > 0) && (nbrCath1 > 0) ) uniqueCath = -1;
  //
  if (1 || VERBOSE) {
    printf("-----------------------------\n");
    printf("Starting CLUSTER PROCESSING\n");
    printXYdXY("Pads input xyDxyi, zi", xyDxyi, nPads, nPads, zi, 0);
    printf("# cath0=%2d, cath1=%2d\n", nbrCath0, nbrCath1);
    printf("# sum Z0=%7.3g, sum Z1=%7.3g\n", sumZ0, sumZ1);
    printf("uniqueCath=%2d\n", uniqueCath);
  }

  // 
  //  Get one plane charge
  // 
  double xy0InfSup[nbrCath0*4];    
  double xy1InfSup[nbrCath1*4];
  if ( uniqueCath == -1) {
    //
    // 2 planes of cathodes
    // Get xyDxy for cath0 & cath1
    xy0Dxy = new double[nbrCath0*4];
    xy1Dxy = new double[nbrCath1*4];
    maskedCopyXYdXY( xyDxyi, nPads, maskCath0, nPads, xy0Dxy, nbrCath0 );
    maskedCopyXYdXY( xyDxyi, nPads, maskCath1, nPads, xy1Dxy, nbrCath1 );

    // Get xyInfSup for cath0 & cath1
    xyDxyToxyInfSup( xy0Dxy, nbrCath0, xy0InfSup);
    xyDxyToxyInfSup( xy1Dxy, nbrCath1, xy1InfSup);
    // ch0, ch1
    ch0 = new double[nbrCath0];
    ch1 = new double[nbrCath1];
    vectorGather( zi, maskCath0, nPads, ch0 );
    vectorGather( zi, maskCath1, nPads, ch1 );
    //
    // Perform the projection on one plane
    nProjPads = projectChargeOnOnePlane( xy0InfSup, ch0, xy1InfSup, ch1, 
        nbrCath0, nbrCath1, includeAlonePads);
    //
    if (VERBOSE) {
      printf("One plane projection\n");
      printXYdXY("  Pads xy0InfSup, ch0", xy0InfSup, nbrCath0, nbrCath0, ch0, 0);
      printXYdXY("  Pads xy1InfSup, ch1", xy1InfSup, nbrCath1, nbrCath1, ch1, 0);
      printf("  nProjPads=%2d\n", nProjPads);
    }    
    // Get the projected plane and the charges
    xyDxyProj = new double[nProjPads*4];
    double chA[nProjPads];
    double chB[nProjPads];
    copyProjectedPads( xyDxyProj, chA, chB);   
    double sumChA = vectorSum(chA, nProjPads);
    double sumChB = vectorSum(chB, nProjPads);
    printXYdXY( "projection xyDxyProj", xyDxyProj, nProjPads, nProjPads, 0, 0);

    //
    if (VERBOSE) {
      printf("  sumZ0 =%7.3g sumZ1 =%7.3g\n", sumZ0, sumZ1);
      printf("  sumChA=%7.3g sumChB=%7.3g\n", sumChA, sumChB);
    }
    if (CHECK) { 
      // test Charge Equality
      if ( fabs( sumZ0 - sumChA) > 1.0) printf("  Charge cath0 and projection differ : %7.3g %7.3g\n", sumZ0, sumChA);
      if ( fabs( sumZ1 - sumChB) > 1.0) printf("  Charge cath1 and projection differ : %7.3g %7.3g\n", sumZ1, sumChB);
    }
    chProj = new double[nProjPads];
    // Compute the means between chA, chB
    vectorAddVector( chA, 1.0, chB, nProjPads, chProj);
    vectorMultScalar( chProj, 0.5, nProjPads, chProj);
    //
    // Saturated Pads on Projection
    satPads0 = new Mask_t[nbrCath0];
    satPads1 = new Mask_t[nbrCath1];
    vectorGatherShort( saturated, maskCath0, nPads, satPads0 );
    vectorGatherShort( saturated, maskCath1, nPads, satPads1 );
    saturatedProj = new Mask_t[nProjPads];
    buildProjectedSaturatedPads( satPads0, satPads1, saturatedProj);
  } else {
    // One Cathode
    // Pad Projection
    xyDxyProj = new double[nPads*4];
    vectorCopy(xyDxyi, nPads*4, xyDxyProj);
    nProjPads = nPads;
    chProj = new double[nPads];    
    vectorCopy(zi, nPads, chProj);
    // Neighbors
    // Must set nProjPads in padProcessing 
    setNbrProjectedPads( nProjPads);
    computeAndStoreFirstNeighbors(xyDxyProj, nPads, nPads);
    if (INSPECTMODEL) storeProjectedPads ( xyDxyProj, zi, nPads);
    // Saturated pads
    saturatedProj = new Mask_t[nProjPads];
    vectorCopyShort( saturated, nPads, saturatedProj);
    // New Laplacian
    if (nbrCath0 != 0) { 
      xy0Dxy = new double[nbrCath0*4];
      vectorCopy( xyDxyi, nbrCath0*4, xy0Dxy );
      ch0 = new double[nbrCath0];
      vectorCopy( zi, nbrCath0, ch0 );
    } else if(nbrCath1 != 0) {
      xy1Dxy = new double[nbrCath1*4];
      vectorCopy( xyDxyi, nbrCath1*4, xy1Dxy );
      ch1 = new double[nbrCath1];
      vectorCopy( zi, nbrCath1, ch1 );
    }
  }
  
  if ( nProjPads == 0 ) throw std::overflow_error("No pads !!!");

  //
  // Having one cathode plane (cathO, cath1 or projected cathodes)
  // Extract the sub-clusters
  Group_t projPadToGrp[nProjPads];
  // Set to 1 because 1-cathode mode
  vectorSetShort( projPadToGrp, 1, nProjPads);
  //
  int nProjGroups = 0;
  // V1 ??
  short *grpToCathGrp = 0;
  int nCathGroups = 0;
  int nGroups = 0;
  Mask_t cath0ToGrp[nbrCath0], cath1ToGrp[nbrCath1];
  if (uniqueCath == -1) {
    // 2 cathodes & projected cathodes
      
    nProjGroups = getConnectedComponentsOfProjPadsWOIsolatedPads( projPadToGrp );
    // nProjGroups = getConnectedComponentsOfProjPads( projPadToGrp );
    vectorPrintShort("  projPadToGrp", projPadToGrp, nProjPads);
    wellSplitGroup = new short[nProjGroups+1];
    // short matGrpGrp[ (nProjGroups+1)*(nProjGroups+1)];
    // Get the matrix grpToGrp to identify ovelapping groups
    // assignCathPadsToGroupFromProj( projPadToGrp, nProjPads, nProjGroups, nbrCath0, nbrCath1, wellSplitGroup, matGrpGrp);
    grpToCathGrp = new short[nProjGroups+1];
    // nCathGroups = assignCathPadsToGroup( matGrpGrp, nProjGroups, nbrCath0, nbrCath1, grpToCathGrp );
    // V1, now projPad is associated with a cathGroup
    // vectorMapShort( projPadToGrp, grpToCathGrp, nProjPads);
    //
    // New Laplacian
    //
    // vectorPrintShort( "newLB projPadToGrp ???", projPadToGrp, nProjPads);
    
    // Rq: Groups can be fused
    nCathGroups = assignGroupToCathPads( projPadToGrp, nProjPads, nProjGroups, nbrCath0, nbrCath1, cath0ToGrp, cath1ToGrp);
    // vectorPrintShort( "newL projPadToGrp ???", projPadToGrp, nProjPads);
    // vectorPrintShort( "newL cath0ToGrp ???", cath0ToGrp, nbrCath0);
    // vectorPrintShort( "newL cath1ToGrp ???", cath1ToGrp, nbrCath1);
    printf("newLM-0 nGroups=%d nCathGrp=%d\n",  nGroups, nCathGroups);
    vectorPrintShort( "newLM-0 projPadToGrp", projPadToGrp, nProjPads);
    vectorPrintShort( "newLM-0 cath0ToGrp", cath0ToGrp, nbrCath0);
    vectorPrintShort( "newLM-0 cath1ToGrp", cath1ToGrp, nbrCath1); 
    nGroups = nCathGroups;
    
    // Merge Groups or cathode group
    // 
    padToMergedGrp = new short[nPads];
    // Over allocated array
    // ??? short projGrpToMergedGrp[nGroups+1];
    short projGrpToMergedGrp[nPads];
    // printf("??? nPads=%d, nProjPads =%d\n", nPads, nProjPads);
    // TODO ???
    // Some alone pads in one cathode plane form a group but are
    // not set as a group (0), ex: Run2 orbit 0, ROF=7, cluster=319
    nMergedGrp = assignPadsToGroupFromProj( projPadToGrp, nProjPads, cath0ToPadIdx, cath1ToPadIdx, nGroups, nPads, padToMergedGrp, projGrpToMergedGrp );
    /// Force ????
    // vectorSetShort( &projGrpToMergedGrp[1], 1, nGroups);
    /// nGroups = 1;
    
    // Update proj and cath pads
    vectorMapShort(projPadToGrp, projGrpToMergedGrp, nProjPads);
    for (int u=0; u<nbrCath0; u++) {
      cath0ToGrp[u] = padToMergedGrp[ cath0ToPadIdx[u]];
    }
    for (int u=0; u<nbrCath1; u++) {
      cath1ToGrp[u] = padToMergedGrp[ cath1ToPadIdx[u]];
    }
    printf("newLM-1 nGroups=%d nMergedGrp=%d\n",  nGroups, nMergedGrp);
    vectorPrintShort( "newLM-1 projPadToGrp", projPadToGrp, nProjPads);
    vectorPrintShort( "newLM-1 grpToGrp", projGrpToMergedGrp, nGroups);
    vectorPrintShort( "newLM-1 cath0ToGrp", cath0ToGrp, nbrCath0);
    vectorPrintShort( "newLM-1 cath1ToGrp", cath1ToGrp, nbrCath1); 
    // Update the maping group to group
    for( int g=0; g<=nMergedGrp; g++) {
      projGrpToMergedGrp[g] = g;
    }
    nGroups = nMergedGrp;
    
    
    // Take care the number of Gropu nGrp increase ???????????????????????????????
    // Add new grp from isolated pad and merge them if it occures
    int nNewGrpCath0 = addIsolatedPadInGroups( xy0Dxy, cath0ToGrp, nbrCath0, 0, projGrpToMergedGrp, nGroups );
    vectorPrintShort( "projGrpToMergedGrp", projGrpToMergedGrp, nNewGrpCath0+1);
    printf("addIsolatedPadInGroups nNewGroups =%d\n", nNewGrpCath0);

    // nGroups = nNewGroups;
    int nNewGrpCath1 = addIsolatedPadInGroups( xy1Dxy, cath1ToGrp, nbrCath1, 1, projGrpToMergedGrp, nNewGrpCath0);
    printf("nGroups=%d, nNewGrpCath0=%d, nNewGrpCath1=%d\n", nGroups, nNewGrpCath0, nNewGrpCath1);
    int nNewGroups = renumberGroups( projGrpToMergedGrp, nNewGrpCath1);
    printf("nGroups=%d, nNewGrpCath0=%d, nNewGrpCath1=%d, nNewGroups=%d \n", nGroups, nNewGrpCath0,nNewGrpCath1, nNewGroups);

    vectorPrintShort("  projGrpToMergedGrp renumbering", projGrpToMergedGrp, nNewGrpCath1+1);
    // update Cathode 0 & 1 Dothe group merge
    vectorMapShort(cath0ToGrp,  projGrpToMergedGrp, nbrCath0);
    vectorMapShort(cath1ToGrp,  projGrpToMergedGrp, nbrCath1);
    
    printf("addIsolatedPadInGroups nNewGroups =%d\n", nNewGrpCath1);
    vectorPrintShort( "projGrpToMergedGrp", projGrpToMergedGrp, nGroups+1);
    printf("after Merge/fusion Grp nGrp=%d\n", nGroups);
    //
    vectorPrintShort("cath0ToGrp",cath0ToGrp, nbrCath0);
    vectorPrintShort("cath1ToGrp",cath1ToGrp, nbrCath1);
    
    // nGroups = vectorMaxShort( projGrpToMergedGrp, nGroups+1);
    nGroups = nNewGroups;
    updateProjectionGroups ( projPadToGrp, nProjPads, cath0ToGrp, cath1ToGrp, 
                             mapPadToCathIdx, projGrpToMergedGrp);
    // ??? vectorMapShort(projPadToGrp, projGrpToMergedGrp, nProjPads);
    /*
    for (int u=0; u<nbrCath0; u++) {
      cath0ToGrp[u] = padToMergedGrp[ cath0ToPadIdx[u]];
    }
    for (int u=0; u<nbrCath1; u++) {
      cath1ToGrp[u] = padToMergedGrp[ cath1ToPadIdx[u]];
    }
    */
    // nGroups = nNewGroups;
    vectorPrintShort( "newLM projPadToGrp ???", projPadToGrp, nProjPads);
    vectorPrintShort( "newLM cath0ToGrp ???", cath0ToGrp, nbrCath0);
    vectorPrintShort( "newLM cath1ToGrp ???", cath1ToGrp, nbrCath1);     
    
  } else {
    // One cathode
    // nProjGroups = getConnectedComponentsOfProjPads( projPadToGrp );
    nProjGroups = getConnectedComponentsOfProjPadsWOIsolatedPads( projPadToGrp );

    wellSplitGroup = new short[nProjGroups+1];
    vectorSetShort( wellSplitGroup, 1, nProjGroups+1);
    assignOneCathPadsToGroup( projPadToGrp, nProjPads, nProjGroups, nbrCath0, nbrCath1, wellSplitGroup);
    grpToCathGrp = new short[nProjGroups+1];
    for(int g=0; g < (nProjGroups+1); g++) grpToCathGrp[g] = g;
    // vectorPrintShort( "????", projPadToGrp, nProjPads );
    // printf("???? nProjGroups %d\n", nProjGroups);
    // printf("???? nCathGroups %d\n", nCathGroups);
    nCathGroups = nProjGroups;
    nGroups = nCathGroups;
    // Merged groups
    nMergedGrp = nGroups;
    padToMergedGrp = new short[nPads];    
    vectorCopyShort( projPadToGrp, nPads, padToMergedGrp);
    // New Laplacian
    nCathGroups = assignGroupToCathPads( projPadToGrp, nProjPads, nProjGroups, nbrCath0, nbrCath1, cath0ToGrp, cath1ToGrp);
    nGroups = nCathGroups;
    int nNewGroups;
    if (uniqueCath == 0) {
      nNewGroups = addIsolatedPadInGroups( xy0Dxy, cath0ToGrp, nbrCath0, 0, grpToCathGrp, nGroups );
    } else {
      nNewGroups = addIsolatedPadInGroups( xy1Dxy, cath1ToGrp, nbrCath1, 1, grpToCathGrp, nGroups ); 
    }
    nGroups = nNewGroups;
    //
    vectorPrintShort("cath0ToGrp",cath0ToGrp, nbrCath0);
    vectorPrintShort("cath1ToGrp",cath1ToGrp, nbrCath1);
  }
  
  if (INSPECTMODEL) {
    inspectModel.padToCathGrp = new Group_t[nPads];
    vectorCopyShort( padToMergedGrp, nPads, inspectModel.padToCathGrp );
    inspectModel.nCathGroups = 0;
  }

  //
  // Sub-Cluster loop
  //
  int nbrOfPadsInTheProjGroup = 0;
  // Group allocations
  double *xyDxyGrp;
  double *chGrp;
  Mask_t *saturatedGrp;
  PadIdx_t *thetaPadProjGrpIdx=0;
  // Fitting allocations
  double *xyDxyFit;
  double *zFit;
  double *thetaFit;
  // EM allocations
  double *thetaEMFinal = 0;
  
  //
  // Find local maxima (seeds)
  //
  
  // Array Overallocated
  /* New laplacian
  double thetaL[nProjPads*5];
  double laplacian[nProjPads];
  Group_t thetaLToGrp[nProjPads];
  PadIdx_t thetaPadProjIdx[nProjPads];
  vectorPrint( "xyDxyProj", xyDxyProj, 4*nProjPads);
  int KProj = findLocalMaxWithLaplacian( xyDxyProj, chProj, projPadToGrp, nGroups, nProjPads, nProjPads, 
                                     laplacian, thetaL, thetaPadProjIdx, thetaLToGrp);
  // printf("??? KProj %d\n", KProj);
  // printTheta("  ??? ThetaL", thetaL, nProjPads);
  // vectorPrintInt( " ??? thetaPadProjIdx", thetaPadProjIdx, KProj);
  
  if (CHECK) {
    double *muXc  = getMuX ( thetaL, nProjPads);
    double *muYc  = getMuY ( thetaL, nProjPads);
    double *projXc = getX( xyDxyProj, nProjPads);
    double *projYc = getY( xyDxyProj, nProjPads);    
    for (int k=0; k < KProj;  k++) {
      double  dx = fabs( muXc[k] - projXc[ thetaPadProjIdx[k]] ); 
      double  dy = fabs( muYc[k] - projYc[ thetaPadProjIdx[k]] ); 
      if ( (dx > 0.01) || (dy > 0.01)) {
        printf("##### pb with k=%d idsInPadGrp=%d,  max dx=%g, dy=%g \n", k, thetaPadProjIdx[k], dx, dy);
        throw std::overflow_error("Check thetaPadProjIdx");
      }
    } 
  }    

  if (INSPECTMODEL) { 
    inspectModel.laplacian =  new double[nProjPads];
    vectorCopy( laplacian, nProjPads, inspectModel.laplacian ); 
    inspectModel.nbrOfProjPads = nProjPads;
    inspectModel.thetaInit =  new double[5*KProj];
    inspectModel.kThetaInit = KProj;
    copyTheta( thetaL, nProjPads, inspectModel.thetaInit, KProj, KProj ); 
    inspectModel.residualProj = new double[nProjPads];
  }
  */
  
  // New laplacian ??? Mask_t maskThetaGrp[KProj];

  // Copy the K maxima   
  for( int g=1; g <= nGroups; g++ ) {
    //
    //  Exctract the current group
    //
    if (VERBOSE) printf("  group %d/%d \n", g, nGroups);
    //
    int K;
    Mask_t maskGrp[nProjPads];
    if (nGroups != 1) {
      // Extract data (xydxyGrp, chGrp, ...)
      // associated with the group g
      //
      // Build the group-mask for pad
      nbrOfPadsInTheProjGroup = vectorBuildMaskEqualShort( projPadToGrp, g, nProjPads, maskGrp);
      // Build the index mapping proj indexes to group indexes
      PadIdx_t mapProjIdxToProjGrpIdx[nProjPads];
      if (CHECK)
        // ??? Should be vectorSetShort
        vectorSetInt( mapProjIdxToProjGrpIdx, -1, nProjPads );
      int nVerif = vectorGetIndexFromMaskInt( maskGrp, nProjPads, mapProjIdxToProjGrpIdx);
      // printf("??? nVerif %d nbrOfPadsInTheProjGroup %d\n", nVerif, nbrOfPadsInTheProjGroup);
      // vectorPrintInt( " ??? mapProjIdxToProjGrpIdx", mapProjIdxToProjGrpIdx, nVerif);
      //
      xyDxyGrp = new double[nbrOfPadsInTheProjGroup*4];
      maskedCopyXYdXY( xyDxyProj, nProjPads, maskGrp,  nProjPads, xyDxyGrp, nbrOfPadsInTheProjGroup);
      chGrp = new double[nbrOfPadsInTheProjGroup];
      vectorGather( chProj, maskGrp, nProjPads, chGrp);
      /*
      vectorPrint( )
      printXYdXY( "xyDxyGrp projection grp", xyDxyGrp, nbrOfPadsInTheProjGroup);
      */
      // Saturated
      saturatedGrp = new Mask_t[nbrOfPadsInTheProjGroup];
      vectorGatherShort( saturatedProj, maskGrp, nProjPads, saturatedGrp);
      // Map of grp-Index to proj-index  
      // grpIdxToProjIdx = new PadIdx_t[nbrOfPadsInTheProjGroup]; 
      // vectorGetIndexFromMask( maskGrp,  nProjPads, grpIdxToProjIdx);
      // Theta Grp's
      // New Laplacian ??? K = vectorBuildMaskEqualShort( thetaLToGrp, g, KProj, maskThetaGrp);
      // Get the mapping:
      //   k (a seed) -> to the pad location in the projPads of 
      //   the current group (i.e. xyDxyGrp)
      /* New Laplacian ???
      thetaPadProjGrpIdx = new PadIdx_t[K];
      getIndexInPadProjGrp( maskThetaGrp, thetaPadProjIdx, mapProjIdxToProjGrpIdx, KProj, thetaPadProjGrpIdx );
      // vectorPrintShort( " ??? maskGrp", maskGrp, nProjPads);       
      // vectorPrintInt( " ??? thetaPadProjGrpIdx", thetaPadProjGrpIdx, K); 
      
      if (CHECK) {
        double *muXc  = getMuX ( thetaL, nProjPads);        
        double *muYc  = getMuY ( thetaL, nProjPads);
        double *projXc = getX( xyDxyGrp, nbrOfPadsInTheProjGroup);
        double *projYc = getY( xyDxyGrp, nbrOfPadsInTheProjGroup);
        int kForTheta = 0;
        for (int k=0; k < KProj;  k++) {
          if (maskThetaGrp[k]) { 
            double  dx = fabs( muXc[k] - projXc[ thetaPadProjGrpIdx[kForTheta]] ); 
            double  dy = fabs( muYc[k] - projYc[ thetaPadProjGrpIdx[kForTheta]] ); 
            if ( (dx > 0.01) || (dy > 0.01)) {
              printf("### kForTheta=%d muX, muY = %g %g\n", kForTheta, muXc[kForTheta], muYc[kForTheta]);
              printf("### thetaPadProjGrpIdx[kForTheta]=%d padX, padY = %g %g\n", thetaPadProjGrpIdx[kForTheta],
                      projXc[ thetaPadProjGrpIdx[kForTheta]], projYc[ thetaPadProjGrpIdx[kForTheta]]);
              printf("#### pb after thetaPadProjGrpIdx with kForTheta=%d idxInPadGrpIdx=%d,  max dx=%g, dy=%g \n", kForTheta, thetaPadProjGrpIdx[kForTheta], dx, dy);   
              throw std::overflow_error("Bad Allocation");
            }
            kForTheta++;
          }
        }
      }
      */
    } else {
      // nGroup == 1,
      // avoid performing masked copy
      // New Laplacian ??? K = KProj;
      nbrOfPadsInTheProjGroup = nProjPads;
      /// ??? Not good
      xyDxyGrp = xyDxyProj; 
      chGrp = chProj;
      vectorSetShort( maskGrp, 1, nProjPads);
      // Double delete ???
      saturatedGrp = saturatedProj;
      // New Laplacian ??? thetaPadProjGrpIdx = thetaPadProjIdx;
    }
    if (VERBOSE) {
      printf("Start processing group g=%2d/%2d \n", g, nGroups);
      printf("  # of pad in proj group g %3d\n", nbrOfPadsInTheProjGroup);
      printf("  Total charge in group (1-plane) %8.3g\n", vectorSum(chGrp, nbrOfPadsInTheProjGroup) );
    }
    //
    // Find local maxima (seeds)
    //
    // Array Overallocated
    // /* New Laplacian
    // ??? double thetaL[nbrOfPadsInTheProjGroup*5];
    // ??? double laplacian[nbrOfPadsInTheProjGroup];
    // nbrOfPadsInTheProjGroup = nPads;

    
    // New Laplacian ???int K = findLocalMaxWithLaplacian( xyDxyGrp, chGrp, grpIdxToProjIdx, nbrOfPadsInTheProjGroup, nbrOfPadsInTheProjGroup, 
    //                                   laplacian, thetaL);
    Mask_t maskGrpCath0[nbrCath0];
    vectorPrintShort("cath0ToGrp", cath0ToGrp, nbrCath0);
    vectorPrintShort("cath1ToGrp", cath1ToGrp, nbrCath1);
    vectorBuildMaskEqualShort( cath0ToGrp, g, nbrCath0, maskGrpCath0);
    int nbrOfPadsInTheGroupCath0 = vectorSumShort( maskGrpCath0,  nbrCath0);
    vectorPrintShort("maskGrpCath0", maskGrpCath0, nbrCath0);
    double xyDxyGrp0[nbrOfPadsInTheGroupCath0*4];
    maskedCopyXYdXY( xy0Dxy, nbrCath0, maskGrpCath0, nbrCath0, xyDxyGrp0, nbrOfPadsInTheGroupCath0);
    double qGrp0[nbrOfPadsInTheGroupCath0];
    vectorGather( ch0, maskGrpCath0, nbrCath0, qGrp0);
    // To use with glbal mapIJToK
    PadIdx_t mapGrpIdxToI[nbrOfPadsInTheGroupCath0];
    vectorGetIndexFromMask( maskGrpCath0, nbrCath0, mapGrpIdxToI);
    //
    Mask_t maskGrpCath1[nbrCath1];
    vectorBuildMaskEqualShort( cath1ToGrp, g, nbrCath1, maskGrpCath1);
    int nbrOfPadsInTheGroupCath1 = vectorSumShort( maskGrpCath1,  nbrCath1);
    vectorPrintShort("maskGrpCath1", maskGrpCath1, nbrCath1);

    double xyDxyGrp1[nbrOfPadsInTheGroupCath1*4];
    double qGrp1[nbrOfPadsInTheGroupCath1];
    vectorSet( xyDxyGrp1, -1.0e-06, nbrOfPadsInTheGroupCath1*4);
    vectorSet( qGrp1, -1, nbrOfPadsInTheGroupCath1);
    // To use with global mapIJToK
    PadIdx_t mapGrpIdxToJ[nbrOfPadsInTheGroupCath1];
    if (xy1Dxy != nullptr) {
      maskedCopyXYdXY( xy1Dxy, nbrCath1, maskGrpCath1, nbrCath1, xyDxyGrp1, nbrOfPadsInTheGroupCath1); 
      vectorGather( ch1, maskGrpCath1, nbrCath1, qGrp1); 
      printXYdXY("Pads input xyDxy1, qGrp1", xyDxyGrp1, nbrOfPadsInTheGroupCath1, nbrOfPadsInTheGroupCath1, qGrp1, 0);
      vectorGetIndexFromMask( maskGrpCath1, nbrCath1, mapGrpIdxToJ);
      vectorPrintShort("maskGrpCath1", maskGrpCath1, nbrCath1 );
    }
    // vectorPrintInt("mapGrpIdxToI",mapGrpIdxToI, nbrOfPadsInTheGroupCath0);
    // vectorPrintInt("mapGrpIdxToJ",mapGrpIdxToJ, nbrOfPadsInTheGroupCath1);

    // 
    printXYdXY("Pads input xyDxy0, qGrp0", xyDxyGrp0, nbrOfPadsInTheGroupCath0, nbrOfPadsInTheGroupCath0, qGrp0, 0);

    if (VERBOSE) {
    for ( int u=0; u< nbrOfPadsInTheGroupCath0; u++) {
        printf("i=%d, xg=%6.3f, qg=%6.3f,qg=%6.3f, map grp->global cath0=%d\n", u, xyDxyGrp0[u], xyDxyGrp0[u+nbrOfPadsInTheGroupCath0], qGrp1[u], mapGrpIdxToI[u]);
    }
    for ( int v=0; v< nbrOfPadsInTheGroupCath1; v++) {
        printf("j=%d, xg=%6.3f, qg=%6.3f,qg=%6.3f, map grp->global cath1=%d\n", v, xyDxyGrp1[v], xyDxyGrp1[v+nbrOfPadsInTheGroupCath1], qGrp1[v], mapGrpIdxToJ[v]);
    }
    }
    int nbrOfPadsInTheGroupCath = nbrOfPadsInTheGroupCath0 + nbrOfPadsInTheGroupCath1;
    double thetaL[nbrOfPadsInTheGroupCath*5];
    double laplacian[nbrOfPadsInTheGroupCath];
    
    K = findLocalMaxWithBothCathodes( xyDxyGrp0, qGrp0, nbrOfPadsInTheGroupCath0, 
                                      xyDxyGrp1, qGrp1, nbrOfPadsInTheGroupCath1, 
                                      xyDxyProj, nProjPads, chId, mapGrpIdxToI, mapGrpIdxToJ, nbrCath0, nbrCath1, thetaL, nbrOfPadsInTheGroupCath);
    if (K != 0) {

    // New Laplacian ??? if ( grpIdxToProjIdx !=0 ) { delete[] grpIdxToProjIdx; grpIdxToProjIdx=0; }
    // Copy the K maxima
    double theta0[K*5];
    copyTheta( thetaL, nbrOfPadsInTheGroupCath, theta0, K, K);
    // */ New Laplacian
    // */ New Laplacian ??? double theta0[K*5];
    /*
    if (nGroups != 1) {
      maskedCopyTheta( thetaL, nProjPads, maskThetaGrp, KProj, theta0, K); 
    } else {
      copyTheta( thetaL, nProjPads, theta0, K, K);
    }
    */
    //    
    // Set varX, varY in theta0
    setMathiesonVarianceApprox( chId, theta0, K);
    if (VERBOSE > 0) {
      printf("Find %2d local maxima : \n", K);
      // printXYdXY("  xyDxyGrp", xyDxyGrp, nbrOfPadsInTheProjGroup, nbrOfPadsInTheProjGroup, chGrp, 0);
      printTheta("  Theta0", theta0, K);
    }

    // New Projection
    if (uniqueCath == -1) {
      // vectorPrint("chGrp before", chGrp,  nProjPads);
      double qProjTheta0[nbrCath0];
      compute2DMathiesonMixturePadIntegrals( xy0InfSup, theta0, nbrCath0, K, chId, qProjTheta0);
      double sumCh0 = vectorSum( ch0, nbrCath0);
      vectorMultScalar(qProjTheta0, sumCh0, nbrCath0, qProjTheta0);
      printf("\n");
      printf("sum qProjTheta0 %f %f\n", sumCh0, vectorSum(qProjTheta0, nbrCath0));
      // vectorPrint("qProjTheta0", qProjTheta0, nbrCath0);
      double qProjTheta1[nbrCath1];
      compute2DMathiesonMixturePadIntegrals( xy1InfSup, theta0, nbrCath1, K, chId, qProjTheta1);
      double sumCh1 = vectorSum( ch1, nbrCath1);
      vectorMultScalar(qProjTheta1, sumCh1, nbrCath1, qProjTheta1);
      printf("sum qProjTheta1 %f %f \n", sumCh1, vectorSum(qProjTheta1, nbrCath1));
      // vectorPrint("qProjTheta0", qProjTheta0, nbrCath0);
      // Do not perform the charge projection
      //double *qProjTheta = chProj;
      // Do it
      
      double qProjTheta[nProjPads];
      printf("BEFORE Projection\n");

      int idxMax;
      idxMax = vectorArgMax( ch0, nbrCath0);
      printf("max ch0 x,y = %f, %f, idx=%d, value=%f\n",  xy0Dxy[idxMax], xy0Dxy[idxMax+nbrCath0], idxMax, ch0[idxMax]);
      idxMax = vectorArgMax( ch1, nbrCath1);
      printf("max ch1 x,y = %f, %f, idx=%d, value=%f\n",  xy1Dxy[idxMax], xy1Dxy[idxMax+nbrCath1], idxMax, ch1[idxMax]);
      
      idxMax = vectorArgMax( qProjTheta0, nbrCath0);
      printf("max theta ch0 x,y = %f, %f, idx=%d, value=%f\n",  xy0Dxy[idxMax], xy0Dxy[idxMax+nbrCath0], idxMax, qProjTheta0[idxMax]);
      idxMax = vectorArgMax( qProjTheta1, nbrCath1);
      printf("max theta ch1 x,y = %f, %f, idx=%d, value=%f\n",  xy1Dxy[idxMax], xy1Dxy[idxMax+nbrCath1], idxMax, qProjTheta1[idxMax]);
      
      // projectChargeOnOnePlaneWithTheta( xy0InfSup, ch0, xy1InfSup, ch1, 
      //     qProjTheta0, qProjTheta1, nbrCath0, nbrCath1, includeAlonePads, qProjTheta);
      // projectChargeOnOnePlaneWithTheta( xy0InfSup, qProjTheta0, xy0InfSup, qProjTheta0, 
      //    qProjTheta0, qProjTheta0, nbrCath0, nbrCath0, includeAlonePads, qProjTheta);
      // projectChargeOnOnePlaneWithTheta( xy0InfSup, qProjTheta0, xy1InfSup, qProjTheta1, 
      //    qProjTheta0, qProjTheta1, nbrCath0, nbrCath1, includeAlonePads, qProjTheta);
      double one0[nbrCath0];
      double one1[nbrCath1];
      vectorSet(one0, 1.0, nbrCath0);
      vectorSet(one1, 1.0, nbrCath1);
      projectChargeOnOnePlaneWithTheta( xy0InfSup, ch0, xy1InfSup, ch1, 
          one0, one1, nbrCath0, nbrCath1, includeAlonePads, qProjTheta);
      vectorGather( qProjTheta, maskGrp, nProjPads, chGrp);
      // vectorPrint("chGrp after", chGrp,  nProjPads);
    }
    //
    // EM
    //
    double *projXc = getX( xyDxyGrp, nbrOfPadsInTheProjGroup);
    double *projYc = getY( xyDxyGrp, nbrOfPadsInTheProjGroup);
    printf("projPads in the group=%d, Xmin/max = %f %f, min/max = %f %f\n", g, vectorMin( projXc, nbrOfPadsInTheProjGroup),vectorMax( projXc, nbrOfPadsInTheProjGroup),
              vectorMin( projYc, nbrOfPadsInTheProjGroup), vectorMax( projYc, nbrOfPadsInTheProjGroup)); 
    double thetaEM[K*5];
    int filteredK;
    if (DoEM==1) {
    double thetaOpt[K*5];
    Mask_t thetaOptMask[K];
    int kOpt = K;
    if ( 0 && (K > 1) && K < 15) {
      if (CHECK) {
        double *muXc  = getMuX ( theta0, K);        
        double *muYc  = getMuY ( theta0, K);
        double *projXc = getX( xyDxyGrp, nbrOfPadsInTheProjGroup);
        double *projYc = getY( xyDxyGrp, nbrOfPadsInTheProjGroup);
        for (int k=0; k < K;  k++) {
          double  dx = fabs( muXc[k] - projXc[ thetaPadProjGrpIdx[k]] ); 
          double  dy = fabs( muYc[k] - projYc[ thetaPadProjGrpIdx[k]] ); 
          if ( (dx > 0.01) || (dy > 0.01)) {
            printf("#### pb before kOptimizer with k=%d idsInPadGrp=%d,  max dx=%g, dy=%g \n", k, thetaPadProjGrpIdx[k], dx, dy);   
            throw std::overflow_error("Bad max/theta indexing before kOptimiser");          }
        }
      }
      // printf(" ??? before kOptim #saturatedGrp=%d\n", vectorSumShort(saturatedGrp, nbrOfPadsInTheProjGroup));
      kOpt = kOptimizer( xyDxyGrp, saturatedGrp, chGrp, 
                  theta0, K, nbrOfPadsInTheProjGroup,
                  thetaPadProjGrpIdx,
                  thetaOpt, thetaOptMask );
    }
    if (DISABLE_EM_SATURATED){
      vectorSetZeroShort( saturatedGrp, nbrOfPadsInTheProjGroup);
    }
    if ( 1 && (kOpt != K) ) {
      weightedEMLoop( xyDxyGrp, saturatedGrp, chGrp, thetaOpt, thetaOptMask, K, nbrOfPadsInTheProjGroup, EMmode, EMConvergence, EMverbose, thetaEM);
    } else {
      weightedEMLoop( xyDxyGrp, saturatedGrp, chGrp, theta0, 0, K, nbrOfPadsInTheProjGroup, EMmode, EMConvergence, EMverbose, thetaEM);
    }
    if (VERBOSE >0) printTheta("EM result Theta", thetaEM, K);
    Mask_t maskFilteredTheta[K*5];
    //
    // Filter the EM components
    filteredK = filterEMModel( thetaEM, K, maskFilteredTheta);
    thetaEMFinal = new double[5*filteredK];
    if ( filteredK != K) {
      double thetaFiltered[filteredK*5];
      maskedCopyTheta( thetaEM, K, maskFilteredTheta, K, thetaFiltered, filteredK);
      if (VERBOSE > 0) printTheta("Filtered Theta", thetaFiltered, filteredK);
      //
      // Final EM
      weightedEMLoop( xyDxyGrp, saturatedGrp, chGrp, thetaFiltered, 0, filteredK, nbrOfPadsInTheProjGroup, EMmode, EMConvergence, EMverbose, thetaEMFinal);
    } else {
      filteredK = K;
      vectorCopy( thetaEM, filteredK*5, thetaEMFinal);
    }
    } else {
      // Don't make EM  
      filteredK = K;
      thetaEMFinal= new double[K*5];
      vectorCopy( theta0, filteredK*5, thetaEMFinal);
    }
    /// ??????
    if (INSPECTMODEL ) {
      copyInGroupList( thetaEMFinal, filteredK, 5, inspectModel.subClusterThetaEMFinal ); 
      // Loc Max ??? inv double residual[nbrOfPadsInTheProjGroup]; 
      // Loc Max ??? inv computeResidual( xyDxyGrp, chGrp, thetaEMFinal, filteredK, nbrOfPadsInTheProjGroup, residual);
      // Loc Max ??? inv vectorScatter( residual, maskGrp, nProjPads, inspectModel.residualProj );
    }
    //
    //
    // To use to avoid fitting 
    // when filteredK > aValue and ratioPadPerSeed > 10 ????
    double ratioPadPerSeed =  nbrOfPadsInTheProjGroup / filteredK;
    //
    // Perform the fitting if the sub-cluster g
    // is well separated at the 2 planes level (cath0, cath1)
    // If not the EM result is kept
    //
    // Build the mask to handle pads with the g-group
    Mask_t maskFit0[nbrCath0];
    Mask_t maskFit1[nbrCath1];
    // printf(" ???? nbrCath0=%d, nbrCath1=%d\n", nbrCath0, nbrCath1);
    // Ne laplacian ??? getMaskCathToGrpFromProj( g, maskFit0, maskFit1, nbrCath0, nbrCath1);
    vectorBuildMaskEqualShort( cath0ToGrp, g, nbrCath0, maskFit0);
    vectorBuildMaskEqualShort( cath1ToGrp, g, nbrCath1, maskFit1);
    // vectorPrintShort("maskFit0", maskFit0, nbrCath0);
    // vectorPrintShort("maskFit1", maskFit1, nbrCath1);
    int n1 = vectorSumShort( maskFit1, nbrCath1);    
    int n0 = vectorSumShort( maskFit0, nbrCath0);
    int nFit = n0+n1;
    // if ( (nFit < nbrOfPadsLimitForTheFitting) && wellSplitGroup[g] ) {
    if ( (nFit < nbrOfPadsLimitForTheFitting) ) {
      //
      // Preparing the fitting
      //
      xyDxyFit = new double[nFit*4];
      zFit = new double[nFit];
      Mask_t cath[nFit];
      Mask_t notSaturatedFit[nFit];
      double zCathTotalCharge[2];
      //
      if (uniqueCath == -1) {
        // Build xyDxyFit  in group g
        //
        // Extract from cath0 the pads which belong to the group g
        maskedCopyXYdXY( xy0Dxy, nbrCath0, maskFit0, nbrCath0, xyDxyFit, nFit );
        // Extract from cath1 the pads which belong to the group g
        maskedCopyXYdXY( xy1Dxy, nbrCath1, maskFit1, nbrCath1, &xyDxyFit[n0], nFit );
        // Saturated pads
        // vectorPrintShort(" notSaturatedFit 0 ??? ", notSaturatedFit, nFit);
        vectorGatherShort( satPads0, maskFit0, nbrCath0, &notSaturatedFit[0]);
        vectorGatherShort( satPads1, maskFit1, nbrCath1, &notSaturatedFit[n0]);
        // vectorPrintShort(" notSaturatedFit 0 ??? ", notSaturatedFit, nFit);
        vectorNotShort( notSaturatedFit, nFit, notSaturatedFit);
        // vectorPrintShort(" MaskFit0 ??? ", maskFit0, nbrCath0);
        // vectorPrintShort(" MaskFit1 ??? ", maskFit1, nbrCath1);
        // vectorPrintShort(" saturatedFit ??? ", saturated, nFit);
        // vectorPrintShort(" notSaturatedFit ??? ", notSaturatedFit, nFit);
        // Chargei in group g 
        vectorGather( ch0, maskFit0, nbrCath0, zFit);
        vectorGather( ch1, maskFit1, nbrCath1, &zFit[n0]);
        // saturated pads are ignored
        // ??? Don't Set to zero the sat. pads 
        // vectorMaskedMult( zFit, notSaturatedFit, nFit, zFit);
        // Total Charge on both cathodes 
        zCathTotalCharge[0] = vectorMaskedSum( zFit,      &notSaturatedFit[0], n0);
        zCathTotalCharge[1] = vectorMaskedSum( &zFit[n0], &notSaturatedFit[n0], n1);
        // Merge the 2 Cathodes
        vectorSetShort( cath,      0, nFit);
        vectorSetShort( &cath[n0], 1, n1);
      } else {
        // In that case: there are only one cathode
        // It is assumed that there is no subcluster
        //
        // Extract from  all pads which belong to the group g
        printf("??? nPads, nFit, n0, n1 %d %d %d %d\n", nPads, nFit, n0, n1);
        vectorPrintShort( "maskFit0", maskFit0, nbrCath0);
        vectorPrintShort( "maskFit1", maskFit1, nbrCath1);
        /* Inv pb with the groups (not expected)
        copyXYdXY( xyDxyi, nPads, xyDxyFit, nFit, nFit );
        vectorCopy( zi, nPads, zFit);
        vectorCopyShort( saturated, nPads, notSaturatedFit );
        vectorNotShort( notSaturatedFit, nPads, notSaturatedFit);
        */
        //
        // GG ??? Maybe to shrink with the 2 cathodes processing
        // Total Charge on cathodes & cathode mask 
        if (nbrCath0 != 0) {
          maskedCopyXYdXY( xyDxyi, nbrCath0, maskFit0, nbrCath0, xyDxyFit, nFit );
          vectorGatherShort( saturated, maskFit0, nbrCath0, &notSaturatedFit[0]);
          vectorNotShort( notSaturatedFit, nFit, notSaturatedFit);
          vectorGather( zi, maskFit0, nbrCath0, zFit);
          // ??? Don't Set to zero the sat. pads 
          // vectorMaskedMult( zFit, notSaturatedFit, nFit, zFit);

          zCathTotalCharge[0] = vectorMaskedSum( zFit, notSaturatedFit, nFit);
          zCathTotalCharge[1] = 0;
          vectorSetShort(cath, 0, nFit);
        } else {
          maskedCopyXYdXY( xyDxyi, nbrCath1, maskFit1, nbrCath1, xyDxyFit, nFit );
          vectorGatherShort( saturated, maskFit1, nbrCath1, &notSaturatedFit[0]);
          vectorNotShort( notSaturatedFit, nFit, notSaturatedFit);
          vectorGather( zi, maskFit1, nbrCath1, zFit);
          // ??? Don't Set to zero the sat. pads 
          // vectorMaskedMult( zFit, notSaturatedFit, nFit, zFit);
          
          zCathTotalCharge[0] = 0;
          zCathTotalCharge[1] = vectorMaskedSum( zFit, notSaturatedFit, nFit);
          vectorSetShort(cath, 1, nFit);
        }
        // Don't take into account saturated pads
        vectorMaskedMult(  zFit, notSaturatedFit, nFit, zFit);
      }
      // ThetaFit (output)
      double thetaFit[filteredK*5];
      // khi2 (output)
      double khi2[1];
      // pError (output)
      double pError[3*filteredK*3*filteredK];
      if (VERBOSE) {
        printf( "Starting the fitting\n");
        printf( "- # cath0, cath1 for fitting: %2d %2d\n", n0, n1);
        printTheta("- thetaEMFinal", thetaEMFinal, filteredK);
      }
      // Fit
      if ( (filteredK*3 - 1) <= nFit) {
        fitMathieson( thetaEMFinal, xyDxyFit, zFit, cath, notSaturatedFit, zCathTotalCharge, filteredK, nFit,
                         chId, processFit, 
                         thetaFit, khi2, pError 
                  );
      } else {
        printf("WARNING: Fitting parameters to big - nbr parameters=%d > nbr of pads=%d\n", filteredK * 3 -1, nFit);
        vectorCopy( thetaEMFinal, filteredK*5, thetaFit);
      }
      if (VERBOSE) { 
        printTheta("- thetaFit", thetaFit, filteredK);
      }
      // Filter Fitting solution
      Mask_t maskFilterFit[filteredK];
      int finalK = filterFitModel( thetaFit, filteredK, maskFilterFit); 
      double *thetaFitFinal = new double[5*filteredK];
      if ( (finalK != filteredK) && (nFit >= finalK ) ) {
        if (VERBOSE) { 
          printf("Filtering the fitting %d >= %d\n", nFit, finalK);
          printTheta("- thetaFitFinal", thetaFitFinal, finalK);
        }
        maskedCopyTheta( thetaFit, filteredK, maskFilterFit, filteredK, thetaFitFinal, finalK);
        fitMathieson( thetaFitFinal, xyDxyFit, zFit, cath, notSaturatedFit, 
                      zCathTotalCharge, finalK, nFit,
                      chId, processFit, 
                      thetaFitFinal, khi2, pError 
                    );
      } else {
        vectorCopy( thetaFit, filteredK*5, thetaFitFinal);  
        finalK = filteredK;
      }
      nbrOfHits += finalK;

      // Store result (hits/seeds)
      appendInThetaList( thetaFitFinal, finalK, subClusterThetaList);
      deleteDouble( thetaEMFinal );
      if (INSPECTMODEL ) {
        copyInGroupList( xyDxyFit, nFit, 4, inspectModel.subClusterPadList);  
        copyInGroupList( zFit,     nFit, 1, inspectModel.subClusterChargeList);  
       }
    } else {
      nbrOfHits += filteredK;
      // Save the result of EM
      appendInThetaList( thetaEMFinal, filteredK, subClusterThetaList);      
      if (INSPECTMODEL ) {
        copyInGroupList( xyDxyGrp, nbrOfPadsInTheProjGroup, 4, inspectModel.subClusterPadList);  
        copyInGroupList( chGrp,    nbrOfPadsInTheProjGroup, 1, inspectModel.subClusterChargeList);  
      }
      
    }
    }
    // Release pointer for group
    // if nGroups =1 the deallocation is done with xyDxyProj 
    if (nGroups != 1) {
      if (chGrp != 0) { delete[] chGrp; chGrp = 0; }
      if (xyDxyGrp != 0) { delete[] xyDxyGrp; xyDxyGrp = 0; }
      if (saturatedGrp != 0) { delete[] saturatedGrp; saturatedGrp = 0; }
      if (thetaPadProjGrpIdx != 0) { delete[] thetaPadProjGrpIdx;  thetaPadProjGrpIdx = 0; }
    }
  } // next group
  
  // is the place ???
  delete [] grpToCathGrp;

  // Finalise inspectModel
  if ( INSPECTMODEL ) finalizeInspectModel();
  
  // Release memory need for preCluster
  cleanClusterProcessVariables( uniqueCath);

  if( nNewPads ) {
    delete [] xyDxyi__;
    delete [] cathi__;
    delete [] saturated__;
    delete [] zi__;
  }
  return nbrOfHits;
}

void cleanClusterProcessVariables( int uniqueCath) {
    // To verify the alloc/dealloc ????
    // if (uniqueCath == -1) {
      // Two cathodes case   
      if ( xy0Dxy != 0 ) { delete[] xy0Dxy; xy0Dxy = 0; }
      if ( xy1Dxy != 0 ) { delete[] xy1Dxy; xy1Dxy = 0; }
      if ( ch0 != 0) { delete[] ch0; ch0 = 0; }
      if ( ch1 != 0) { delete[] ch1; ch1 = 0; }
      if ( satPads0 != 0) { delete[] satPads0; satPads0 = 0; }
      if ( satPads1 != 0) { delete[] satPads1; satPads1 = 0; }
      if ( cath0ToPadIdx != 0) { delete[] cath0ToPadIdx; cath0ToPadIdx = 0; }
      if ( cath1ToPadIdx != 0) { delete[] cath1ToPadIdx; cath1ToPadIdx = 0; }
    // }
    if ( xyDxyProj != 0 ) { delete[] xyDxyProj; xyDxyProj =0; }
    if ( chProj != 0 ) { delete[] chProj; chProj = 0; }
    if ( saturatedProj != 0 ) { delete[] saturatedProj; saturatedProj = 0; }
    if ( wellSplitGroup != 0) { delete[] wellSplitGroup; wellSplitGroup=0; };
    if ( padToMergedGrp != 0) { delete[] padToMergedGrp; padToMergedGrp=0; };
    nProjPads = 0;
    nMergedGrp = 0;
  }