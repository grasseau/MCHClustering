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

// To keep internal data
# define SAVEInternVariables 1
# define VERBOSE 0
# define CHECK 0

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
static double *xy0Dxy = 0;
static double *xy1Dxy = 0;
static double *ch0=0, *ch1=0;
static Saturated_t *satPads0=0;
static Saturated_t *satPads1=0;
// Projection
static int nProjPads = 0;
static double *xyDxyProj = 0;
static Saturated_t *saturatedProj=0;
static double *chProj    = 0;
static short *wellSplitGroup = 0;
// hits/seeds founds per sub-cluster

static std::vector< DataBlock_t > subClusterThetaList;
static std::vector< DataBlock_t > subClusterPadList;
static std::vector< DataBlock_t > subClusterChargeList;
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
int filterModel( double *theta, int K, Mask_t *maskFilteredTheta) {
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

// Used when several sub-cluster occur in the precluster
// Append the new hits/clusters in the thetaList of the pre-cluster
void appendInGroupList( const double *values, int N, std::vector< DataBlock_t > &groupList) {
  // double *ptr = new double[N*];
  // memcpy( (void *) ptr, (const void*) theta, N*5*sizeof(double));
  groupList.push_back( std::make_pair(N, values));
}

void cleanGroupLists( ) {
  for (int i = 0; i < subClusterThetaList.size(); i++) 
    delete[] subClusterThetaList[i].second;
  subClusterThetaList.clear();
  //
  for (int i = 0; i < subClusterPadList.size(); i++) 
    delete[] subClusterPadList[i].second;
  subClusterPadList.clear();
  //
  for (int i = 0; i < subClusterChargeList.size(); i++) 
    delete[] subClusterChargeList[i].second;
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

void collectPadsAndCharges( double *xyDxy, double *z, Group_t *padToGroup, int N) {
  int sumN=0;
  for (int h=0; h < subClusterPadList.size(); h++) {
    int n = subClusterPadList[h].first;
    copyXYdXY( subClusterPadList[h].second, n, &xyDxy[sumN], N, n );
    vectorCopy( subClusterChargeList[h].second, n, &z[sumN]);
    if ( padToGroup ) {
      vectorSetShort( &padToGroup[sumN], h+1, n);
    }
    sumN += n;
    delete[] subClusterPadList[h].second;
    delete[] subClusterChargeList[h].second;
  }
  if ( sumN > N) {
    printf("Bad allocation for collectTheta sumN=%d greater than N=%d\n", sumN, N);
    throw std::overflow_error("Bad Allocation");
  }
  subClusterPadList.clear();
  subClusterChargeList.clear();
}

int getNbrOfPadsInGroups() {
  int N=0;
  for (int h=0; h < subClusterPadList.size(); h++) {
    N += subClusterPadList[h].first;
  }
  return N;
}

// Extract hits/seeds of a pre-cluster
int clusterProcess( const double *xyDxyi, const Mask_t *cathi, const Saturated_t *saturated, const double *zi, int chId, int nPads) {
// Remarks:

  nbrOfHits = 0;

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
  //
  // uniqueCath = id (0/1) of the unique cathode, -1 (2 cath.) if not
  short uniqueCath = -1;
  if (nbrCath0 > 0 ) uniqueCath = 0;
  if (nbrCath1 > 0 ) uniqueCath = 1;
  if ( (nbrCath0 > 0) && (nbrCath1 > 0) ) uniqueCath = -1;
  //
  if (VERBOSE) {
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
  if ( uniqueCath == -1) {
    //
    // 2 planes of cathodes
    // Get xyDxy for cath0 & cath1
    xy0Dxy = new double[nbrCath0*4];
    xy1Dxy = new double[nbrCath1*4];
    maskedCopyXYdXY( xyDxyi, nPads, maskCath0, nPads, xy0Dxy, nbrCath0 );
    maskedCopyXYdXY( xyDxyi, nPads, maskCath1, nPads, xy1Dxy, nbrCath1 );

    // Get xyInfSup for cath0 & cath1
    double xy0InfSup[nbrCath0*4];    
    double xy1InfSup[nbrCath1*4];
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
      // ??? printf("-----------------------------\n");
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
    //
    if (VERBOSE) {
      printf("  sumZ0 =%7.3g sumZ1 =%7.3g\n", sumZ0, sumZ1);
      printf("  sumChA=%7.3g sumChB=%7.3g\n", sumChA, sumChB);
    }
    if (CHECK) { 
      // test Charge Equality
      if ( fabs( sumZ0 - sumChA) > 1.0) printf("  Charge cath0 and projection differ : %7.3g %7.3g", sumZ0, sumChA);
      if ( fabs( sumZ1 - sumChB) > 1.0) printf("  Charge cath1 and projection differ : %7.3g %7.3g", sumZ1, sumChB);
    }
    chProj = new double[nProjPads];
    // Compute the means between chA, chB
    vectorAddVector( chA, 1.0, chB, nProjPads, chProj);
    vectorMultScalar( chProj, 0.5, nProjPads, chProj);
    // Saturated Pads
    satPads0 = new Saturated_t[nbrCath0];
    satPads1 = new Saturated_t[nbrCath1];
    vectorGatherShort( saturated, maskCath0, nPads, satPads0 );
    vectorGatherShort( saturated, maskCath1, nPads, satPads1 );
    saturatedProj = new Saturated_t[nProjPads];
    buildProjectedSaturatedPads( satPads0, satPads1, saturatedProj);
  } else {
    // One Cathode
    xyDxyProj = new double[nPads*4];
    vectorCopy(xyDxyi, nPads*4, xyDxyProj);
    nProjPads = nPads;
    chProj = new double[nPads];    
    vectorCopy(zi, nPads, chProj);
    // Neighbors
    computeAndStoreFirstNeighbors( xyDxyProj, nPads, nPads);
    // Group
    
    // Saturated pads
    saturatedProj = new Saturated_t[nProjPads];
    vectorCopyShort( saturated, nPads, saturatedProj);
  }
  
  if ( nProjPads == 0 ) throw std::overflow_error("No pads !!!");

  //
  // Having one cathode plane (cathO, cath1 or projected cathodes)
  // Extract the sub-clusters
  Group_t padToGrp[nProjPads];
  // Set to 1 because 1-cathode mode
  vectorSetShort( padToGrp, 1, nProjPads);
  //
  int nGroups = 0;
  if (uniqueCath == -1) {
    // 2 cathodes & projected cathodes
    nGroups = getConnectedComponentsOfProjPads( padToGrp );
    wellSplitGroup = new short[nGroups+1];
    assignCathPadsToGroup( padToGrp, nProjPads, nGroups, nbrCath0, nbrCath1, wellSplitGroup);
  } else {
    // One cathode 
    // Must have one group with 1 cathode
    nGroups = 1;
    wellSplitGroup = new short[1+1];
    wellSplitGroup[1] = 1;
  }
  //
  // Sub-Cluster loop
  //
  int nbrOfPadsInTheGroup = 0;
  // Group allocations
  double *xyDxyGrp;
  double *chGrp;
  Saturated_t *saturatedGrp;
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
  double thetaL[nProjPads*5];
  double laplacian[nProjPads];
  Group_t thetaLToGrp[nProjPads];
  int KProj = findLocalMaxWithLaplacian( xyDxyProj, chProj, padToGrp, nGroups, nProjPads, nProjPads, 
                                     laplacian, thetaL, thetaLToGrp);
  Mask_t maskThetaGrp[KProj];

  // Copy the K maxima   
  for( int g=1; g <= nGroups; g++ ) {
    //
    //  Exctract the current group
    //
    int K;
    if (nGroups != 1) {
      // Extract data (xydxyGrp, chGrp, ...)
      // associated with the group g
      Mask_t maskGrp[nProjPads];
      nbrOfPadsInTheGroup = vectorBuildMaskEqualShort( padToGrp, g, nProjPads, maskGrp);
      xyDxyGrp = new double[nbrOfPadsInTheGroup*4];
      maskedCopyXYdXY( xyDxyProj, nProjPads, maskGrp,  nProjPads, xyDxyGrp, nbrOfPadsInTheGroup);
      chGrp = new double[nbrOfPadsInTheGroup];
      vectorGather( chProj, maskGrp, nProjPads, chGrp);
      // Saturated
      saturatedGrp = new Saturated_t[nbrOfPadsInTheGroup];
      vectorGatherShort( saturatedProj, maskGrp, nProjPads, saturatedGrp);
      // Map of grp-Index to proj-index  
      // grpIdxToProjIdx = new PadIdx_t[nbrOfPadsInTheGroup]; 
      // vectorGetIndexFromMask( maskGrp,  nProjPads, grpIdxToProjIdx);
      // Theta Grp's
      K = vectorBuildMaskEqualShort( thetaLToGrp, g, KProj, maskThetaGrp);
    } else {
      // nGroup == 1,
      // avoid performing masked copy
      K = KProj;
      nbrOfPadsInTheGroup = nProjPads;
      xyDxyGrp = xyDxyProj; 
      chGrp = chProj;
      // Double delete ???
      saturatedGrp = saturatedProj;
    }
    if (VERBOSE) {
      printf("Start processing group g=%2d/%2d \n", g, nGroups);
      printf("  # of pad in group g %3d\n", nbrOfPadsInTheGroup);
      printf("  Total charge in group (1-plane) %8.3g\n", vectorSum(chGrp, nbrOfPadsInTheGroup) );
    }
    //
    // Find local maxima (seeds)
    //
    // Array Overallocated
    /*
    double thetaL[nbrOfPadsInTheGroup*5];
    double laplacian[nbrOfPadsInTheGroup];

    int K = findLocalMaxWithLaplacian( xyDxyGrp, chGrp, grpIdxToProjIdx, nbrOfPadsInTheGroup, nbrOfPadsInTheGroup, 
                                       laplacian, thetaL);
    if ( grpIdxToProjIdx !=0 ) { delete[] grpIdxToProjIdx; grpIdxToProjIdx=0; }
    // Copy the K maxima
    double theta0[K*5];
    copyTheta( thetaL, nbrOfPadsInTheGroup, theta0, K, K);
    */
    double theta0[K*5];
    if (nGroups != 1) {
      maskedCopyTheta( thetaL, nProjPads, maskThetaGrp, KProj, theta0, K); 
    } else {
      copyTheta( thetaL, nProjPads, theta0, K, K);
    }
    //    
    // Set varX, varY in theta0
    setMathiesonVarianceApprox( chId, theta0, K);
    if (VERBOSE > 0) {
      printf("Find %2d local maxima : \n", K);
      printXYdXY("  xyDxyGrp", xyDxyGrp, nbrOfPadsInTheGroup, nbrOfPadsInTheGroup, 0, 0);
      printTheta("  Theta0", theta0, K);
    }
    //
    // EM
    double thetaEM[K*5];
    //
    weightedEMLoop( xyDxyGrp, saturatedGrp, chGrp, theta0, K, nbrOfPadsInTheGroup, EMmode, EMConvergence, EMverbose, thetaEM);
    if (VERBOSE >0) printTheta("EM result Theta", thetaEM, K);
    Mask_t maskFilteredTheta[K*5];
    //
    // Filter the EM components
    int filteredK = filterModel( thetaEM, K, maskFilteredTheta);
    thetaEMFinal = new double[5*filteredK];
    if ( filteredK != K) {
      double thetaFiltered[filteredK*5];
      maskedCopyTheta( thetaEM, K, maskFilteredTheta, K, thetaFiltered, filteredK);
      if (VERBOSE > 0) printTheta("Filtered Theta", thetaFiltered, filteredK);
      //
      // Final EM
      weightedEMLoop( xyDxyGrp, saturatedGrp, chGrp, thetaFiltered, filteredK, nbrOfPadsInTheGroup, EMmode, EMConvergence, EMverbose, thetaEMFinal);
    } else {
      filteredK = K;
      vectorCopy( thetaEM, filteredK*5, thetaEMFinal);
    }
    nbrOfHits += filteredK;
    //
    //
    // To use to avoid fitting 
    // when filteredK > aValue and ratioPadPerSeed > 10 ????
    double ratioPadPerSeed =  nbrOfPadsInTheGroup / filteredK;
    //
    // Perform the fitting if the sub-cluster g
    // is well separated at the 2 planes level (cath0, cath1)
    // If not the EM result is kept
    //
    // Build the mask to handle pads with the g-group
    Mask_t maskFit0[nbrCath0], maskFit1[nbrCath1];
    getMaskCathToGrp( g, maskFit0, maskFit1, nbrCath0, nbrCath1);
    // vectorPrintShort("maskFit0", maskFit0, nbrCath0);
    // vectorPrintShort("maskFit1", maskFit1, nbrCath1);
    int n1 = vectorSumShort( maskFit1, nbrCath1);    
    int n0 = vectorSumShort( maskFit0, nbrCath0);
    int nFit = n0+n1;
    if ( (nFit < nbrOfPadsLimitForTheFitting) && wellSplitGroup[g] ) {
      //
      // Preparing the fitting
      //
      // Build xyDxyFit  in group g
      xyDxyFit = new double[nFit*4];
      //
      // Extract from cath0 the pads which belong to the group g
      maskedCopyXYdXY( xy0Dxy, nbrCath0, maskFit0, nbrCath0, xyDxyFit, nFit );
      // Extract from cath1 the pads which belong to the group g
      maskedCopyXYdXY( xy1Dxy, nbrCath1, maskFit1, nbrCath1, &xyDxyFit[n0], nFit );
      // Chargei in group g 
      zFit = new double[nFit];
      vectorGather( ch0, maskFit0, nbrCath0, zFit);
      vectorGather( ch1, maskFit1, nbrCath1, &zFit[n0]);
      // Total Charge on both cathodes 
      double zCathTotalCharge[2];
      zCathTotalCharge[0] = vectorSum( zFit, n0);
      zCathTotalCharge[1] = vectorSum( &zFit[n0], n1);
      // Merge the 2 Cathodes
      Mask_t cath[nFit];
      vectorSetShort(cath, 0, nFit);
      vectorSetShort(&cath[n0], 1, n1);
      // ThetaFit (output)
      double *thetaFit = new double[filteredK*5];
      // khi2 (output)
      double khi2[1];
      // pError (output)
      double pError[3*filteredK*3*filteredK];
      if (VERBOSE) {
        printf( "Starting the fitting\n");
        printf( "- # cath0, cath1 for fitting: %2d %2d\n", n0, n1);
        printXYdXY("- Pads input xyDxyFit, zi", xyDxyFit, nFit, nFit, zFit, 0);          
        printTheta("- thetaEMFinal", thetaEMFinal, filteredK);
      }
      // Fit
      fitMathieson( thetaEMFinal, xyDxyFit, zFit, cath, zCathTotalCharge, filteredK, nFit,
                         chId, processFit, 
                         thetaFit, khi2, pError 
                  );
      // Store result (hits/seeds)
      appendInGroupList( thetaFit, filteredK, subClusterThetaList);
      deleteDouble( thetaEMFinal );
      if (SAVEInternVariables ) {
        appendInGroupList( xyDxyFit, nFit, subClusterPadList);  
        appendInGroupList( zFit, nFit, subClusterChargeList);  
       }
    } else {
      // Save the result of EM
      appendInGroupList( thetaEMFinal, filteredK, subClusterThetaList);
      if (SAVEInternVariables ) {
        appendInGroupList( xyDxyGrp, filteredK, subClusterPadList);  
        appendInGroupList( chGrp,    filteredK, subClusterChargeList);  
       }
    }
    // Release pointer for group
    if ( SAVEInternVariables  == 0) {
      if (chGrp != 0) { delete[] chGrp; chGrp = 0; }
      if (xyDxyGrp != 0) { delete[] xyDxyGrp; xyDxyGrp = 0; }
      if (saturatedGrp != 0) { delete[] saturatedGrp; saturatedGrp = 0; }
    }
  } // next group
  //
  if ( SAVEInternVariables == 0) {
    cleanClusterProcessVariables( );
  }
  return nbrOfHits;
}

void cleanClusterProcessVariables() {
    if ( xy0Dxy != 0 ) { delete[] xy0Dxy; xy0Dxy = 0; }
    if ( xy1Dxy != 0 ) { delete[] xy1Dxy; xy1Dxy = 0; }
    if ( ch0 != 0) { delete[] ch0; ch0 = 0; }
    if ( ch1 != 0) { delete[] ch1; ch1 = 0; }
    if ( satPads0 != 0) { delete[] satPads0; satPads0 = 0; }
    if ( satPads1 != 0) { delete[] satPads1; satPads1 = 0; }
    if ( xyDxyProj != 0 ) { delete[] xyDxyProj; xyDxyProj =0; }
    if ( chProj != 0 ) { delete[] chProj; chProj = 0; }
    if ( saturatedProj != 0 ) { delete[] saturatedProj; saturatedProj = 0; }
    if (wellSplitGroup != 0) { delete[] wellSplitGroup; wellSplitGroup=0; };
    nProjPads = 0;
    // clean subCluster Data
    cleanGroupLists( );
  }