# include <string.h>
# include <vector>
# include <algorithm>
# include <stdexcept>

#include "MCHClustering/dataStructure.h"
#include "padProcessing.h"
#include "mathUtil.h"
#include "MCHClustering/clusterProcessing.h"
#include "MCHClustering/mathieson.h"
#include "gaussianEM.h"

#include "InspectModel.h"

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


//
// Component (hit/seeds) selection
// w cutoff
static const double wCutoff = 5.e-2;

// Private variables
//
// Pads with cathodes
static double *xy0Dxy = nullptr;
static double *xy1Dxy = nullptr;
static double *ch0=nullptr, *ch1=nullptr;
static Mask_t *satPads0 = nullptr;
static Mask_t *satPads1 = nullptr;
// Mapping from cathode-pad to original pads
static PadIdx_t *cath0ToPadIdx = nullptr;
static PadIdx_t *cath1ToPadIdx = nullptr;
static int nMergedGrp = 0;
static short *padToMergedGrp = 0;

//
// Projection
// static double *xyDxyProj = 0;
// ??? To remove
static Mask_t *saturatedProj=0;
static double *chProj    = 0;
static short *wellSplitGroup = 0;
//
using namespace o2::mch;


// Invalid
// static InspectModel_t inspectModel={.nbrOfProjPads=0, .laplacian=0, .residualProj=0, .thetaInit=0, .kThetaInit=0,
//  .totalNbrOfSubClusterPads=0, .totalNbrOfSubClusterThetaEMFinal=0, .nCathGroups=0, .padToCathGrp=0};

// Total number of hits/seeds in the precluster;
static int nbrOfHits = 0;
// Storage of the seeds found
std::vector< DataBlock_t > seedList;

// Unused
/*
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
*/


// Remove hits/seeds according to w-magnitude
// and hits/seeds proximity
// Return the new # of components (hits/seeds)
/*
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
*/

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
/* Unused
 void computeGaussianResidual( const double *xyDxy, const double *zObs, const double *theta, int K, int N, double *residual) {

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
*/
void cleanThetaList( ) {
  for (int i = 0; i < seedList.size(); i++)
    delete[] seedList[i].second;
  seedList.clear();
}

// Extract hits/seeds of a pre-cluster
int clusterProcess( const double *xyDxyi_, const Mask_t *cathi_, const Mask_t *saturated_, const double *zi_, int chId, int nPads) {
// Remarks:

  nbrOfHits = 0;
  cleanThetaList();
  if (INSPECTMODEL) {
    cleanInspectModel();
    // Inv ??? cleanInspectPadProcess();
  }

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

  // Pad filter when too large number of pads
  if (nPads > 800) {
    // Remove noisy event
    printf("WARNING: remove noisy pads <z>=%f, min/max z=%f,%f", vectorSum(zi_, nPads)/ nPads, vectorMin(zi_, nPads), vectorMax(zi_, nPads));
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

  // TODO ???
  // if nPads <= 6 go directly to fit
  // 1 seed is assumed compute the barycenter

  // Replace by constructor
  Cluster cluster( getConstX(xyDxyi, nPads), getConstY(xyDxyi, nPads), getConstDX(xyDxyi, nPads), getConstDY(xyDxyi, nPads),
          zi, cathi, saturated, chId, nPads);
  /* ??? To remove
  int isSaturated = vectorSumShort( saturated, nPads );

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
  // ??? Unused
  PadIdx_t mapPadToCathIdx[nPads];
  vectorSetInt(mapPadToCathIdx, -1, nPads);
  for( int c=0; c< nbrCath0; c++) {
    mapPadToCathIdx[ cath0ToPadIdx[c] ] = c;
  }
  for( int c=0; c< nbrCath1; c++) {
    mapPadToCathIdx[ cath1ToPadIdx[c] ] = c;
  }
  if (1 || VERBOSE > 3) {
    vectorPrintInt( "cath0ToPadIdx", cath0ToPadIdx, nbrCath0);
    vectorPrintInt( "cath1ToPadIdx", cath1ToPadIdx, nbrCath1);
    vectorPrintInt( "mapPadToCathIdx", mapPadToCathIdx, nPads);
  }
  //
  // uniqueCath = id (0/1) of the unique cathode, -1 (2 cath.) if not
  short uniqueCath = -1;
  if (nbrCath0 > 0 ) uniqueCath = 0;
  if (nbrCath1 > 0 ) uniqueCath = 1;
  if ( (nbrCath0 > 0) && (nbrCath1 > 0) ) uniqueCath = -1;
  //
  if (VERBOSE > 0 ) {
    printf("-----------------------------\n");
    printf("Starting CLUSTER PROCESSING\n");
    printf("# cath0=%2d, cath1=%2d\n", nbrCath0, nbrCath1);
    printf("# sum Z0=%7.3g, sum Z1=%7.3g\n", sumZ0, sumZ1);
    printf("# uniqueCath=%2d\n", uniqueCath);
  }
  //
  //  Get one plane charge
  //
  double xy0InfSup[nbrCath0*4];
  double xy1InfSup[nbrCath1*4];
  */

  // ???????????????????????????????????????
  int nProjPads = cluster.buildProjectedGeometry(includeAlonePads);
  //
  /* ??? To remove : build Projected geometry
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
    if (VERBOSE > 0) {
      printf("One plane projection\n");
      printf("  nProjPads=%2d\n", nProjPads);
    }
    if (VERBOSE > 1) {
      printXYdXY("  Pads xy0InfSup, ch0", xy0InfSup, nbrCath0, nbrCath0, ch0, 0);
      printXYdXY("  Pads xy1InfSup, ch1", xy1InfSup, nbrCath1, nbrCath1, ch1, 0);
    }
    // Get the projected plane and the charges
    xyDxyProj = new double[nProjPads*4];
    double chA[nProjPads];
    double chB[nProjPads];
    copyProjectedPads( xyDxyProj, chA, chB);
    double sumChA = vectorSum(chA, nProjPads);
    double sumChB = vectorSum(chB, nProjPads);
    //
    if (VERBOSE > 0) {
      printf("  sumZ0 =%7.3g sumZ1 =%7.3g\n", sumZ0, sumZ1);
      printf("  sumChA=%7.3g sumChB=%7.3g\n", sumChA, sumChB);
    }
    if (VERBOSE > 1) {
      printXYdXY( "  projection xyDxyProj", xyDxyProj, nProjPads, nProjPads, 0, 0);
    }
    //

    if (CHECK || VERBOSE > 1) {
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
    // Unused ???
    //saturatedProj = new Mask_t[nProjPads];
    //buildProjectedSaturatedPads( satPads0, satPads1, saturatedProj);
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
      satPads0 = new Mask_t[nbrCath0];
      vectorGatherShort( saturated, maskCath0, nPads, satPads0 );
    } else if(nbrCath1 != 0) {
      xy1Dxy = new double[nbrCath1*4];
      vectorCopy( xyDxyi, nbrCath1*4, xy1Dxy );
      ch1 = new double[nbrCath1];
      vectorCopy( zi, nbrCath1, ch1 );
      satPads1 = new Mask_t[nbrCath1];
      vectorGatherShort( saturated, maskCath1, nPads, satPads1 );
    }
  }
  */

  // ???????????????????????????????????????
  // END :: nProjPads = cluster.buildProjectedGeometry(includeAlonePads);

  if ( nProjPads == 0 ) throw std::range_error("No projected pads !!!");
  int nGroups = cluster.buildGroupOfPads();

  ///////////////////////////// TO SUPRESS (in Cluster Class)
  /*
  //
  // Having one cathode plane (cathO, cath1 or projected cathodes)
  // Extract the sub-clusters
  Group_t projPadToGrp[nProjPads];
  // Set to 1 because 1-cathode mode
  vectorSetShort( projPadToGrp, 0, nProjPads);
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
    if (VERBOSE > 0) {
      printf("Projected Groups %d\n", nProjGroups);
      vectorPrintShort("  projPadToGrp", projPadToGrp, nProjPads);
    }
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

    // Merge Groups or cathode group
    //
    padToMergedGrp = new short[nPads];
    // Over allocated array
    // ??? short projGrpToMergedGrp[nGroups+1];
    if (1) {
    // printf("??? nPads=%d, nProjPads =%d\n", nPads, nProjPads);
    // TODO ???
    // Some alone pads in one cathode plane form a group but are
    // not set as a group (0), ex: Run2 orbit 0, ROF=7, cluster=319
    // nMergedGrp = assignPadsToGroupFromProj( projPadToGrp, nProjPads, cath0ToPadIdx, cath1ToPadIdx, nGroups, nPads, padToMergedGrp, projGrpToMergedGrp );
    nMergedGrp = assignPadsToGroupFromProj( projPadToGrp, nProjPads, cath0ToPadIdx, cath1ToPadIdx, nProjGroups, nPads, padToMergedGrp );
    //
    // Update proj and cath pads
    for (int u=0; u<nbrCath0; u++) {
      cath0ToGrp[u] = padToMergedGrp[ cath0ToPadIdx[u]];
    }
    for (int u=0; u<nbrCath1; u++) {
      cath1ToGrp[u] = padToMergedGrp[ cath1ToPadIdx[u]];
    }
    nGroups = nMergedGrp;
    if (VERBOSE > 0) {
      printf("Groups after propagation to cathodes [assignPadsToGroupFromProj] nCathGroups=%d\n", nGroups);
      vectorPrintShort( "  projPadToGrp", projPadToGrp, nProjPads);
      vectorPrintShort( "  cath0ToGrp", cath0ToGrp, nbrCath0);
      vectorPrintShort( "  cath1ToGrp", cath1ToGrp, nbrCath1);
    }
    }

    // Take care the number of Gropu nGrp increase ???????????????????????????????
    // Add new grp from isolated pad and merge them if it occures
    //
    // Compute the max alloccation
    int nbrIsolatePads=0;
    for ( int p=0; p < nbrCath0; p++) {
      if( cath0ToGrp[p] == 0 ) nbrIsolatePads += 1;
    }
    for ( int p=0; p < nbrCath1; p++) {
      if( cath1ToGrp[p] == 0 ) nbrIsolatePads += 1;
    }
    int nbrMaxGroups = nGroups + nbrIsolatePads;
    Mask_t mapGrpToGrp[ nbrMaxGroups +1 ];
    for (int g=0; g < (nbrMaxGroups+1); g++) {
      mapGrpToGrp[g] = g;
    }
    //
    // int nNewGrpCath0 = addIsolatedPadInGroups( xy0Dxy, cath0ToGrp, nbrCath0, 0, projGrpToMergedGrp, nGroups );
    int nNewGrpCath0 = addIsolatedPadInGroups( xy0Dxy, cath0ToGrp, nbrCath0, 0, mapGrpToGrp, nGroups );
    nGroups += nNewGrpCath0;
    // Apply New Grp on cath1
    for ( int p=0; p < nbrCath1; p++) {
      cath1ToGrp[p] = mapGrpToGrp[cath1ToGrp[p]];
    }
    for ( int p=0; p < nProjPads; p++) {
      projPadToGrp[p] = mapGrpToGrp[ projPadToGrp[p]];
    }
    if( VERBOSE > 1) {
      printf("# addIsolatedPadInGroups cath-0 nNewGroups =%d\n", nGroups);
      vectorPrintShort( "  mapGrpToGrp", mapGrpToGrp, nGroups+1);
    }
    // nGroups = nNewGroups;
    // int nNewGrpCath1 = addIsolatedPadInGroups( xy1Dxy, cath1ToGrp, nbrCath1, 1, projGrpToMergedGrp, nNewGrpCath0);
    // Inv ??? int nNewGrpCath1 = addIsolatedPadInGroups( xy1Dxy, cath1ToGrp, nbrCath1, 1, mapGrpToGrp, nNewGrpCath0);
    int nNewGrpCath1 = addIsolatedPadInGroups( xy1Dxy, cath1ToGrp, nbrCath1, 1, mapGrpToGrp, nGroups);
    nGroups += nNewGrpCath1;
    // printf("nGroups=%d, nNewGrpCath0=%d, nNewGrpCath1=%d\n", nGroups, nNewGrpCath0, nNewGrpCath1);
    // Apply New Grp on cath0
    for ( int p=0; p < nbrCath0; p++) {
      cath0ToGrp[p] = mapGrpToGrp[cath0ToGrp[p]];
    }
    for ( int p=0; p < nProjPads; p++) {
      projPadToGrp[p] = mapGrpToGrp[ projPadToGrp[p]];
    }
    if( VERBOSE > 1) {
      printf("# addIsolatedPadInGroups cath-1 nNewGroups =%d\n", nGroups);
      vectorPrintShort( "  mapGrpToGrp", mapGrpToGrp, nGroups+1);
    }
    // int nNewGroups = renumberGroups( projGrpToMergedGrp, nNewGrpCath1);
    /// Inv ???? int nNewGroups = renumberGroupsV2( cath0ToGrp, nbrCath0, cath1ToGrp, nbrCath1, mapGrpToGrp, std::max( nNewGrpCath0, nNewGrpCath1));
    int nNewGroups = renumberGroupsV2( cath0ToGrp, nbrCath0, cath1ToGrp, nbrCath1, mapGrpToGrp, nGroups);
    // vectorPrintShort( "  mapGrpToGrp", mapGrpToGrp, nNewGroups);
    if (VERBOSE > 1) {
      printf("Groups after renumbering %d\n", nGroups);
      vectorPrintShort( "  projPadToGrp", projPadToGrp, nProjPads);
      printf("  nNewGrpCath0=%d, nNewGrpCath1=%d, nGroups=%d\n", nNewGrpCath0, nNewGrpCath1, nGroups);
      vectorPrintShort( "  cath0ToGrp  ", cath0ToGrp, nbrCath0);
      vectorPrintShort( "  cath1ToGrp  ", cath1ToGrp, nbrCath1);
      vectorPrintShort("   mapGrpToGrp ", mapGrpToGrp, nNewGroups);
    }
    for ( int p=0; p < nProjPads; p++) {
      projPadToGrp[p] = mapGrpToGrp[ projPadToGrp[p]];
    }
    // update Cathode 0 & 1 Dothe group merge
    // ????
    //vectorMapShort(cath0ToGrp,  projGrpToMergedGrp, nbrCath0);
    // vectorMapShort(cath1ToGrp,  projGrpToMergedGrp, nbrCath1);

    nGroups = nNewGroups;
    // Update proj-pads groups from cath grous
    updateProjectionGroups ( projPadToGrp, nProjPads, cath0ToGrp, cath1ToGrp );

    if (VERBOSE > 0) {
      printf("# Groups after adding isolate pads and renumbering %d\n", nGroups);
      vectorPrintShort( "  projPadToGrp", projPadToGrp, nProjPads);
      vectorPrintShort( "  cath0ToGrp  ", cath0ToGrp, nbrCath0);
      vectorPrintShort( "  cath1ToGrp  ", cath1ToGrp, nbrCath1);
    }
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
    nGroups += nNewGroups;
    //
  }

  if( VERBOSE > 0) {
    printf("# Final Groups %d\n", nGroups);
    vectorPrintShort("  cath0ToGrp",cath0ToGrp, nbrCath0);
    vectorPrintShort("  cath1ToGrp",cath1ToGrp, nbrCath1);
  }
  */
  if (INSPECTMODEL) {
    double *qProj = cluster.projectChargeOnProjGeometry(includeAlonePads);
    saveProjectedPads( cluster.getProjectedPads(), qProj);
    savePadToCathGroup( cluster.getCathGroup(0), cluster.getMapCathPadToPad(0), cluster.getNbrOfPads(0),
            cluster.getCathGroup(1), cluster.getMapCathPadToPad(1), cluster.getNbrOfPads(1) );
    // vectorCopyShort( padToMergedGrp, nPads, inspectModel.padToCathGrp );
    // Inv vectorCopyShort( cath0ToGrp, nbrCath0, inspectModel.padToCathGrp);
    // Inv vectorCopyShort( cath1ToGrp, nbrCath1, &inspectModel.padToCathGrp[nbrCath0]);
    // vectorScatterShort( cath0ToGrp, maskCath0, nPads, inspectModel.padToCathGrp);
    // vectorScatterShort( cath1ToGrp, maskCath1, nPads, inspectModel.padToCathGrp);

    // ??? inspectModel.nCathGroups = 0;
  }
// ##################" End of suppresss
  //
  // Sub-Cluster loop
  //
  int nbrOfProjPadsInTheGroup = 0;
  // Group allocations
  double *xyDxyGrp=nullptr;
  double *chGrp=nullptr;
  // ??? To remove
  Mask_t *saturatedGrp=nullptr;
  PadIdx_t *thetaPadProjGrpIdx=0;
  // Fitting allocations
  /* Inv ???
  double *xyDxyFit;
  double *zFit;
  double *thetaFit;
  */
  // EM allocations
  double *thetaEMFinal = 0;
  int finalK = 0;


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
    if (VERBOSE > 0) printf("Group %d/%d \n", g, nGroups);
    //
    int kEM;
    /*
    Mask_t maskGrp[nProjPads];
    if (nGroups != 1) {
      // Extract data (xydxyGrp, chGrp, ...)
      // associated with the group g
      //
      // Build the group-mask for pad
      nbrOfProjPadsInTheGroup = vectorBuildMaskEqualShort( cluster.getProjPadGroup(), g, nProjPads, maskGrp);
      // Build the index mapping proj indexes to group indexes
      // PadIdx_t mapProjIdxToProjGrpIdx[nProjPads];
      // if (CHECK)
        // ??? Should be vectorSetShort
        // vectorSetInt( mapProjIdxToProjGrpIdx, -1, nProjPads );
      // int nVerif = vectorGetIndexFromMaskInt( maskGrp, nProjPads, mapProjIdxToProjGrpIdx);
      // printf("??? nVerif %d nbrOfProjPadsInTheGroup %d\n", nVerif, nbrOfProjPadsInTheGroup);
      // vectorPrintInt( " ??? mapProjIdxToProjGrpIdx", mapProjIdxToProjGrpIdx, nVerif);
      //
      // ??? inv o2::mch::Pads tmp( xyDxyProj, &xyDxyProj[nProjPads], &xyDxyProj[2*nProjPads], &xyDxyProj[3*nProjPads], chProj, nProjPads );
      // ??? inspectSavePixels( 0, tmp);


      // Extract the projected-pads of the current group
      // xyDxyGrp = new double[nbrOfProjPadsInTheGroup*4];
      // maskedCopyXYdXY( xyDxyProj, nProjPads, maskGrp,  nProjPads, xyDxyGrp, nbrOfProjPadsInTheGroup);
      xyDxyGrp = cluster.getProjPadsAsXYdXY( g, maskGrp, nbrOfProjPadsInTheGroup);
      // Extract the charges of the current group
      // ??? Check if used in PEM

      // ??? the qProj is not calculated
      //chGrp = new double[nbrOfProjPadsInTheGroup];
      // vectorGather( chProj, maskGrp, nProjPads, chGrp);

      // Saturated
      // saturatedGrp = new Mask_t[nbrOfProjPadsInTheGroup];
      // vectorGatherShort( saturatedProj, maskGrp, nProjPads, saturatedGrp);
    } else {
      // nGroup == 1,
      // avoid performing masked copy
      // New Laplacian ??? K = KProj;
      nbrOfProjPadsInTheGroup = nProjPads;
      /// ??? Not good
      // xyDxyGrp = xyDxyProj;
      xyDxyGrp = cluster.getProjPadsAsXYdXY( g, maskGrp, nbrOfProjPadsInTheGroup);
      // ??? the qProj is not calculated
      // chGrp = chProj;
      vectorSetShort( maskGrp, 1, nProjPads);
      // Double delete ???
      saturatedGrp = saturatedProj;
      // New Laplacian ??? thetaPadProjGrpIdx = thetaPadProjIdx;
    }
    // ??? Dummy charge for this group
    chGrp = new double[nbrOfProjPadsInTheGroup];
    vectorSet( chGrp, 1, nbrOfProjPadsInTheGroup);

    if (VERBOSE > 0) {
      printf("------------------------------------\n");
      printf("Start processing group g=%2d/%2d \n", g, nGroups);
      printf("  # of pad in proj group g=%d : %3d\n", g, nbrOfProjPadsInTheGroup);
      printf("  Total charge in group (1-plane) %8.3g\n", vectorSum(chGrp, nbrOfProjPadsInTheGroup) );
    }
    //
    // Find local maxima (seeds)
    //
    // Array Overallocated
    // /* New Laplacian
    // ??? double thetaL[nbrOfProjPadsInTheGroup*5];
    // ??? double laplacian[nbrOfProjPadsInTheGroup];
    // nbrOfProjPadsInTheGroup = nPads;


    // New Laplacian ???int K = findLocalMaxWithLaplacian( xyDxyGrp, chGrp, grpIdxToProjIdx, nbrOfProjPadsInTheGroup, nbrOfProjPadsInTheGroup,
    //                                   laplacian, thetaL);
    int nbrCath0 = cluster.getNbrOfCathPad( 0 );
    Mask_t maskGrpCath0[nbrCath0];
    // vectorPrintShort("cath0ToGrp", cath0ToGrp, nbrCath0);
    // vectorPrintShort("cath1ToGrp", cath1ToGrp, nbrCath1);
    vectorBuildMaskEqualShort( cluster.getCathGroup(0), g, nbrCath0, maskGrpCath0);
    int nbrOfPadsInTheGroupCath0 = vectorSumShort( maskGrpCath0,  nbrCath0);
    // vectorPrintShort("maskGrpCath0", maskGrpCath0, nbrCath0);
    double xyDxyGrp0[nbrOfPadsInTheGroupCath0*4];
    maskedCopyXYdXY( xy0Dxy, nbrCath0, maskGrpCath0, nbrCath0, xyDxyGrp0, nbrOfPadsInTheGroupCath0);
    double qGrp0[nbrOfPadsInTheGroupCath0];
    vectorGather( ch0, maskGrpCath0, nbrCath0, qGrp0);
    Mask_t saturateGrp0[nbrOfPadsInTheGroupCath0];
    vectorGatherShort( satPads0, maskGrpCath0, nbrCath0, saturateGrp0);

    // To use with glbal mapIJToK
    PadIdx_t mapGrpIdxToI[nbrOfPadsInTheGroupCath0];
    vectorGetIndexFromMask( maskGrpCath0, nbrCath0, mapGrpIdxToI);
    //
    int nbrCath1 = cluster.getNbrOfCathPad( 1 );
    Mask_t maskGrpCath1[nbrCath1];
    vectorBuildMaskEqualShort( cluster.getCathGroup(1), g, nbrCath1, maskGrpCath1);
    int nbrOfPadsInTheGroupCath1 = vectorSumShort( maskGrpCath1,  nbrCath1);
    // vectorPrintShort("maskGrpCath1", maskGrpCath1, nbrCath1);

    double xyDxyGrp1[nbrOfPadsInTheGroupCath1*4];
    double qGrp1[nbrOfPadsInTheGroupCath1];
    vectorSet( xyDxyGrp1, -1.0e-06, nbrOfPadsInTheGroupCath1*4);
    vectorSet( qGrp1, -1, nbrOfPadsInTheGroupCath1);
    Mask_t saturateGrp1[nbrOfPadsInTheGroupCath1];
    vectorGatherShort( satPads1, maskGrpCath1, nbrCath1, saturateGrp1);
    // To use with global mapIJToK
    PadIdx_t mapGrpIdxToJ[nbrOfPadsInTheGroupCath1];
    if (xy1Dxy != nullptr) {
      maskedCopyXYdXY( xy1Dxy, nbrCath1, maskGrpCath1, nbrCath1, xyDxyGrp1, nbrOfPadsInTheGroupCath1);
      vectorGather( ch1, maskGrpCath1, nbrCath1, qGrp1);
      // printXYdXY("Pads input xyDxy1, qGrp1", xyDxyGrp1, nbrOfPadsInTheGroupCath1, nbrOfPadsInTheGroupCath1, qGrp1, 0);
      vectorGetIndexFromMask( maskGrpCath1, nbrCath1, mapGrpIdxToJ);
      // vectorPrintShort("maskGrpCath1", maskGrpCath1, nbrCath1 );
    }
    // vectorPrintInt("mapGrpIdxToI",mapGrpIdxToI, nbrOfPadsInTheGroupCath0);
    // vectorPrintInt("mapGrpIdxToJ",mapGrpIdxToJ, nbrOfPadsInTheGroupCath1);

    //
    // printXYdXY("Pads input xyDxy0, qGrp0", xyDxyGrp0, nbrOfPadsInTheGroupCath0, nbrOfPadsInTheGroupCath0, qGrp0, 0);

    if ( VERBOSE >1) {
      printf("cath0-Pads in group %d \n", g);
      for ( int u=0; u< nbrOfPadsInTheGroupCath0; u++) {
        printf("  i=%d, xg=%6.3f, yg=%6.3f,qg=%6.3f, map grp->global cath0=%d\n", u, xyDxyGrp0[u], xyDxyGrp0[u+nbrOfPadsInTheGroupCath0], qGrp0[u], mapGrpIdxToI[u]);
      }
      printf("cath1-Pads in group %d \n", g);
      for ( int v=0; v< nbrOfPadsInTheGroupCath1; v++) {
        printf("  j=%d, xg=%6.3f, yg=%6.3f,qg=%6.3f, map grp->global cath1=%d\n", v, xyDxyGrp1[v], xyDxyGrp1[v+nbrOfPadsInTheGroupCath1], qGrp1[v], mapGrpIdxToJ[v]);
      }
    }
    int nbrOfPadsInTheGroupCath = nbrOfPadsInTheGroupCath0 + nbrOfPadsInTheGroupCath1;
    double thetaL[nbrOfPadsInTheGroupCath*5];
    double laplacian[nbrOfPadsInTheGroupCath];
    */
    /*
    K = findLocalMaxWithBothCathodes( xyDxyGrp0, qGrp0, nbrOfPadsInTheGroupCath0,
                                      xyDxyGrp1, qGrp1, nbrOfPadsInTheGroupCath1,
                                      xyDxyProj, nProjPads, chId, mapGrpIdxToI, mapGrpIdxToJ, nbrCath0, nbrCath1, thetaL, nbrOfPadsInTheGroupCath);
    */
    Cluster *subCluster = nullptr;
    if ( nGroups == 1) {
      subCluster = &cluster;
    } else {
      subCluster = new Cluster( cluster, g);
    }
    if (VERBOSE > 0) {
      printf("Start findLocalMaxWithPET\n");
    }
    /* ??? Inv
    kEM = findLocalMaxWithPET( xyDxyGrp0, qGrp0, saturateGrp0, nbrOfPadsInTheGroupCath0,
                             xyDxyGrp1, qGrp1, saturateGrp1, nbrOfPadsInTheGroupCath1,
                             // xyDxyProj, nProjPads, chId, mapGrpIdxToI, mapGrpIdxToJ, nbrCath0, nbrCath1, thetaL, nbrOfPadsInTheGroupCath);
                             xyDxyGrp, chGrp, nbrOfProjPadsInTheGroup, chId,
                            // mapGrpIdxToI, mapGrpIdxToJ, nbrCath0, nbrCath1,
                            thetaL, nbrOfPadsInTheGroupCath);
    */
    int nbrOfPadsInTheGroup = subCluster->getNbrOfPads(0) + subCluster->getNbrOfPads(1);
    double thetaL[ nbrOfPadsInTheGroup*5];

    // Compute the local max with laplacian method
    // Used only to give insights of the cluster
    if (INSPECTMODEL) {
      int nProjPads = subCluster->buildProjectedGeometry(includeAlonePads);
      kEM = subCluster->findLocalMaxWithBothCathodes( thetaL, nbrOfPadsInTheGroup, 2 );
      double thetaExtra[kEM*5];
      copyTheta( thetaL, nbrOfPadsInTheGroup, thetaExtra, kEM, kEM);
      saveThetaExtraInGroupList( thetaExtra, kEM );
      printTheta("Theta findLocalMaxWithBothCathodes", thetaExtra, kEM);
    }
    kEM = subCluster->findLocalMaxWithPET( thetaL, nbrOfPadsInTheGroup );
    if (kEM != 0) {
    double thetaEM[kEM*5];
    copyTheta( thetaL, nbrOfPadsInTheGroup, thetaEM, kEM, kEM);
    // New Laplacian ??? if ( grpIdxToProjIdx !=0 ) { delete[] grpIdxToProjIdx; grpIdxToProjIdx=0; }
    // Copy the kEM maxima
    // ??? Inv double theta0[kEM*5];
    // ??? Inv copyTheta( thetaL, nbrOfPadsInTheGroupCath, theta0, kEM, kEM);
    // */ New Laplacian
    // */ New Laplacian ??? double theta0[kEM*5];
    /*
    if (nGroups != 1) {
      maskedCopyTheta( thetaL, nProjPads, maskThetaGrp, KProj, theta0, kEM);
    } else {
      copyTheta( thetaL, nProjPads, theta0, kEM, kEM);
    }
    */
    //
    // Set varX, varY in theta0
    // setMathiesonVarianceApprox( chId, theta0, kEM);
    if (VERBOSE > 0) {
      printf("Find %2d local maxima : \n", kEM);
      // printXYdXY("  xyDxyGrp", xyDxyGrp, nbrOfProjPadsInTheGroup, nbrOfProjPadsInTheGroup, chGrp, 0);
      printTheta("  ThetaEM", thetaEM, kEM);
    }

    //
    // EM
    //
    double *projXc = getX( xyDxyGrp, nbrOfProjPadsInTheGroup);
    double *projYc = getY( xyDxyGrp, nbrOfProjPadsInTheGroup);
    if (VERBOSE > 1) {
      printf("projPads in the group=%d, Xmin/max = %f %f, min/max = %f %f\n", g, vectorMin( projXc, nbrOfProjPadsInTheGroup),vectorMax( projXc, nbrOfProjPadsInTheGroup),
              vectorMin( projYc, nbrOfProjPadsInTheGroup), vectorMax( projYc, nbrOfProjPadsInTheGroup));
    }
    // Inv ??? double thetaEM[kEM*5];
    // ??? int filteredK;
    // ??? else {
    // ???  // Don't make EM
    // ???  filteredK = kEM;
    // ???   thetaEMFinal= new double[kEM*5];
    ///  vectorCopy( theta0, filteredK*5, thetaEMFinal);
    // }
    /// ??????
    if (INSPECTMODEL ) {
      // ??? Inv copyInGroupList( thetaEMFinal, filteredK, 5, inspectModel.subClusterThetaEMFinal );
      saveThetaEMInGroupList( thetaEM, kEM );
      // Loc Max ??? inv double residual[nbrOfProjPadsInTheGroup];
      // Loc Max ??? inv computeResidual( xyDxyGrp, chGrp, thetaEMFinal, filteredK, nbrOfProjPadsInTheGroup, residual);
      // Loc Max ??? inv vectorScatter( residual, maskGrp, nProjPads, inspectModel.residualProj );
    }
    //
    //
    //
    // Perform the fitting if the sub-cluster g
    // is well separated at the 2 planes level (cath0, cath1)
    // If not the EM result is kept
    //
    DataBlock_t newSeeds = subCluster->fit( thetaEM, kEM);
    finalK = newSeeds.first;
    // ??? (111) Invalid fiting
    /*
    // To use to avoid fitting
    // when filteredK > aValue and ratioPadPerSeed > 10 ????
    //
    double ratioPadPerSeed =  nbrOfProjPadsInTheGroup / filteredK;

    // ??? (111) Invalid fiting
    // Build the mask to handle pads with the g-group
    Mask_t maskFit0[nbrCath0];
    Mask_t maskFit1[nbrCath1];
    // printf(" ???? nbrCath0=%d, nbrCath1=%d\n", nbrCath0, nbrCath1);
    // Ne laplacian ??? getMaskCathToGrpFromProj( g, maskFit0, maskFit1, nbrCath0, nbrCath1);
    vectorBuildMaskEqualShort( cluster.getCathGroup(0), g, nbrCath0, maskFit0);
    vectorBuildMaskEqualShort( cluster.getCathGroup(1), g, nbrCath1, maskFit1);
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
        // printf("??? nPads, nFit, n0, n1 %d %d %d %d\n", nPads, nFit, n0, n1);
        // vectorPrintShort( "maskFit0", maskFit0, nbrCath0);
        // vectorPrintShort( "maskFit1", maskFit1, nbrCath1);

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
      if (VERBOSE > 0) {
        printf( "Starting the fitting\n");
        printf( "- # cath0, cath1 for fitting: %2d %2d\n", n0, n1);
        printTheta("- thetaEMFinal", thetaEMFinal, filteredK);
      }
      // Fit
      if ( (filteredK*3 - 1) <= nFit) {
        fitMathieson( thetaEMFinal, xyDxyFit, zFit, cath, notSaturatedFit, zCathTotalCharge, filteredK, nFit,
                         chId, processFitVerbose,
                         thetaFit, khi2, pError
                  );
      } else {
        printf("---> Fitting parameters to large : k=%d, 3k-1=%d, nFit=%d\n", filteredK, filteredK * 3 -1, nFit);
        printf("     keep the EM solution\n");
        vectorCopy( thetaEMFinal, filteredK*5, thetaFit);
      }
      if (VERBOSE) {
        printTheta("- thetaFit", thetaFit, filteredK);
      }
      // Filter Fitting solution
      Mask_t maskFilterFit[filteredK];
      int finalK = filterFitModelOnClusterRegion( xyDxyFit, nFit, thetaFit, filteredK, maskFilterFit);
      // int finalK = filterFitModelOnSpaceVariations( thetaEMFinal, filteredK, thetaFit, filteredK, maskFilterFit);
      double *thetaFitFinal = new double[5*finalK];
      if ( (finalK != filteredK) && (nFit >= finalK ) ) {
        if (VERBOSE) {
          printf("Filtering the fitting K=%d >= filteredK=%d\n", nFit, finalK);
          printTheta("- thetaFitFinal", thetaFitFinal, finalK);
        }
        if( finalK > 0) {
          maskedCopyTheta( thetaFit, filteredK, maskFilterFit, filteredK, thetaFitFinal, finalK);
          fitMathieson( thetaFitFinal, xyDxyFit, zFit, cath, notSaturatedFit,
                      zCathTotalCharge, finalK, nFit,
                      chId, processFit,
                      thetaFitFinal, khi2, pError
                    );
        } else {
           // No hit with the fitting
           vectorCopy( thetaEMFinal, filteredK*5, thetaFit);
           finalK = filteredK;
        }
      } else {
        vectorCopy( thetaFit, filteredK*5, thetaFitFinal);
        finalK = filteredK;
      }
      */
      nbrOfHits += finalK;
      // ??? (111) Invalid fiting

      // Store result (hits/seeds)
      // ??? Inv appendInThetaList( thetaFitFinal, finalK, subClusterThetaList);
      seedList.push_back( newSeeds );
      if (INSPECTMODEL) {
        saveThetaFitInGroupList( newSeeds.second, newSeeds.first);
      }
      // ??? deleteDouble( thetaEM );
      /* Inv ???
      if (INSPECTMODEL ) {
        // ??? InvcopyInGroupList( xyDxyFit, nFit, 4, inspectModel.subClusterPadList);
        // ??? copyInGroupList( zFit,     nFit, 1, inspectModel.subClusterChargeList);
        savePadsOfSubCluster( xyDxyFit, zFit, nFit);
       }
       */
    } else {
      // No EM seeds
      finalK = kEM;
      nbrOfHits += finalK;
      // Save the result of EM
      // ??? Inv appendInThetaList( thetaEMFinal, filteredK, subClusterThetaList);
      DataBlock_t newSeeds = std::make_pair(finalK, nullptr);
      seedList.push_back( newSeeds );
      /* Inv ???
      if (INSPECTMODEL ) {
        // Inv ??? copyInGroupList( xyDxyGrp, nbrOfProjPadsInTheGroup, 4, inspectModel.subClusterPadList);
        // ??? copyInGroupList( chGrp,    nbrOfProjPadsInTheGroup, 1, inspectModel.subClusterChargeList);
        savePadsOfSubCluster( xyDxyGrp, chGrp, nbrOfProjPadsInTheGroup);
      }
      */

    }
    if (INSPECTMODEL ) {
        // Inv ??? copyInGroupList( xyDxyGrp, nbrOfProjPadsInTheGroup, 4, inspectModel.subClusterPadList);
        // ??? copyInGroupList( chGrp,    nbrOfProjPadsInTheGroup, 1, inspectModel.subClusterChargeList);
        savePadsOfSubCluster( xyDxyGrp, chGrp, nbrOfProjPadsInTheGroup);
    }
    // Release pointer for group
    // if nGroups =1 the deallocation is done with xyDxyProj
    /*
    if (nGroups != 1) {
      if (chGrp != 0) { delete[] chGrp; chGrp = 0; }
      if (xyDxyGrp != 0) { delete[] xyDxyGrp; xyDxyGrp = 0; }
      if (saturatedGrp != 0) { delete[] saturatedGrp; saturatedGrp = 0; }
      if (thetaPadProjGrpIdx != 0) { delete[] thetaPadProjGrpIdx;  thetaPadProjGrpIdx = 0; }
    }
    */
    deleteDouble( xyDxyGrp );
    deleteDouble( chGrp );
    if ( nGroups > 1) {
      delete subCluster;
    }
  } // next group

  // is the place ???
  // Inv ??? delete [] grpToCathGrp;

  // Finalise inspectModel
  if ( INSPECTMODEL ) finalizeInspectModel();

  // Release memory need for preCluster
  // ??? To remove
  cleanClusterProcessVariables( );

  if( nNewPads ) {
    delete [] xyDxyi__;
    delete [] cathi__;
    delete [] saturated__;
    delete [] zi__;
  }
  return nbrOfHits;
}

/* Not Used ???
void finalizeSolution( &seedList ) {
  if ( seedList.size() == 1) return;

  double sumCharge = 0;
  for (int g=0;  < seedList.size(); g++) {
    int k = seedList[g].first;
    double theta = getW( seedList[g].second, k);
    double *w = getW(theta, k )
    for( int l=0; l<k; l++) {
      // w[k] contains the charge
      sumCharge += w[l];
    }
  }
  // Filter on relative weight (Charge)
  for (int g=0;  < seedList.size(); g++) {
    int k = seedList[g].first;
    double theta = getW( seedList[g].second, k);
    double *w = getW(theta, k )
    for( int l=0; l<k; l++) {
      // w[k] contains the charge
      w[l] = w[l]/sumCharge;
    }
    // Buid Mask
    vectorBuidMaskGreater( w, 0)
  }
  maskedCopyTheta(const double* theta, int K, const Mask_t* mask, int nMask, double* maskedTheta, int maskedK)
}
*/

void cleanClusterProcessVariables( ) {
    // To verify the alloc/dealloc ????
    // if (uniqueCath == -1) {
      // Two cathodes case
      // if ( xy0Dxy != 0 ) { delete[] xy0Dxy; xy0Dxy = 0; }
      // if ( xy1Dxy != 0 ) { delete[] xy1Dxy; xy1Dxy = 0; }
      // if ( ch0 != 0) { delete[] ch0; ch0 = 0; }
      // if ( ch1 != 0) { delete[] ch1; ch1 = 0; }
      // if ( satPads0 != 0) { delete[] satPads0; satPads0 = 0; }
      // if ( satPads1 != 0) { delete[] satPads1; satPads1 = 0; }
      // if ( cath0ToPadIdx != 0) { delete[] cath0ToPadIdx; cath0ToPadIdx = 0; }
      // if ( cath1ToPadIdx != 0) { delete[] cath1ToPadIdx; cath1ToPadIdx = 0; }
    // }
    // if ( xyDxyProj != 0 ) { delete[] xyDxyProj; xyDxyProj =0; }
    // if ( chProj != 0 ) { delete[] chProj; chProj = 0; }
    // if ( saturatedProj != 0 ) { delete[] saturatedProj; saturatedProj = 0; }
    // if ( wellSplitGroup != 0) { delete[] wellSplitGroup; wellSplitGroup=0; };
    // if ( padToMergedGrp != 0) { delete[] padToMergedGrp; padToMergedGrp=0; };
    //deleteDouble( xyDxyGrp );
    // deleteDouble( chGrp );
    nMergedGrp = 0;
  }