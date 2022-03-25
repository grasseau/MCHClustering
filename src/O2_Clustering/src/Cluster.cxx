// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterGeometry.h
/// \brief Definition of a class to reconstruct clusters with the original MLEM algorithm
///
/// \author Gilles Grasseau, Subatech

#include <cstdio>
#include <stdexcept>

#include "MCHClustering/PadsPEM.h"
#include "MCHClustering/Cluster.h"
#include "padProcessing.h"
#include "poissonEM.h"
#include "mathiesonFit.h"
#include "mathUtil.h"
#include "InspectModel.h"

#define VERBOSE 2
#define CHECK 1
// ??? maybe better in const header
#define INSPECTMODEL 1

namespace o2
{
namespace mch
{
// Fit parameters
// doProcess = verbose + (doJacobian << 2) + ( doKhi << 3) + (doStdErr << 4)
static const int processFitVerbose = 1 + (0 << 2) + ( 1 << 3) + (1 << 4);
static const int processFit = 0 + (0 << 2) + ( 1 << 3) + (1 << 4);

// Limit of pad number  to perform the fitting
static const int nbrOfPadsLimitForTheFitting = 50;

double epsilonGeometry = 1.0e-4;

Cluster::Cluster() {
}

Cluster::Cluster( Pads *pads0, Pads *pads1) {
  int nPlanes = 0;
  singleCathPlaneID = -1;
  if ( pads0 != nullptr ) {
    pads[nPlanes] = pads0;
    nPlanes++;
  } else {
    singleCathPlaneID = 1;
  }
  if ( pads1 != nullptr ) {
    pads[nPlanes] = pads1;
    nPlanes++;
  } else {
    singleCathPlaneID = 0;
  }
  nbrOfCathodePlanes = nPlanes;
}


Cluster::Cluster( const double *x, const double *y, const double *dx, const double *dy,
        const double *q, const short *cathodes, const short *saturated, int chId, int nPads) {

  chamberId = chId;
  nbrSaturated = vectorSumShort( saturated, nPads );

  int nbrCath1 = vectorSumShort( cathodes, nPads );
  int nbrCath0 = nPads - nbrCath1;

  // Build the pads for each cathode
  int nextCath=0;
  if (nbrCath0 != 0) {
    mapCathPadIdxToPadIdx[0] = new PadIdx_t[nbrCath0];
    pads[0] = new Pads( x, y, dx, dy, q, cathodes, saturated, 0, chId, mapCathPadIdxToPadIdx[0], nPads);
    singleCathPlaneID = 0;
    nextCath = 1;
  }
  if (nbrCath1 != 0) {
    mapCathPadIdxToPadIdx[nextCath] = new PadIdx_t[nbrCath1];
    pads[nextCath] = new Pads( x, y, dx, dy, q, cathodes, saturated, 1, chId, mapCathPadIdxToPadIdx[nextCath], nPads);
    singleCathPlaneID = 1;
  }
  // Number of cathodes & alone cathode
  nbrOfCathodePlanes = nextCath+1;
  if ( nbrOfCathodePlanes == 2) {
    singleCathPlaneID = -1;
  }
  // ??? To remove if default Constructor
  // Projection
  projectedPads=nullptr;
  projNeighbors=nullptr;
  projPadToGrp=nullptr;
  nbrOfProjGroups = 0;
  // Groups
  cathGroup[0] = nullptr;
  cathGroup[1] = nullptr;
  nbrOfCathGroups=0;
  // Geometry
  IInterJ=nullptr;
  JInterI=nullptr;
  mapKToIJ=nullptr;
  mapIJToK=nullptr;
  aloneIPads=nullptr;
  aloneJPads=nullptr;
  aloneKPads=nullptr;

  if (1 || VERBOSE > 3) {
    vectorPrintInt( "cath0ToPadIdx", mapCathPadIdxToPadIdx[0], nbrCath0);
    vectorPrintInt( "cath1ToPadIdx", mapCathPadIdxToPadIdx[1], nbrCath1);
  }
  //
  if (VERBOSE > 0 ) {
    printf("-----------------------------\n");
    printf("Starting CLUSTER PROCESSING\n");
    printf("# cath0=%2d, cath1=%2d\n", nbrCath0, nbrCath1);
    printf("# sum Q0=%7.3g, sum Q1=%7.3g\n", pads[0]->totalCharge, pads[1]->totalCharge);
    printf("# singleCathPlaneID=%2d\n", singleCathPlaneID);
  }
}

Cluster::Cluster( Cluster &cluster, Groups_t g) {
    chamberId = cluster.chamberId;
    //
    // Extract the pads of group g
    //
    Mask_t maskGrpCath0[cluster.pads[0]->nPads];
    Mask_t maskGrpCath1[cluster.pads[1]->nPads];
    Mask_t *maskGrpCath[2] = {maskGrpCath0, maskGrpCath1};
    int nbrCathPlanes_ = 0;
    int singleCathPlaneID_ = -1;
    for ( int c=0; c<2; c++) {
      // Build the mask mapping the group g
      // ??? Inv int nbr =
      int nbrPads = vectorBuildMaskEqualShort( cluster.cathGroup[c], g, cluster.pads[c]->nPads, maskGrpCath[c]);
      if (nbrPads != 0 ) {
        // Create the pads of the group g
        pads[c] = new Pads( *cluster.pads[c], maskGrpCath[c]);
        nbrCathPlanes_++;
        singleCathPlaneID_ = c;
      }
    }
    nbrOfCathodePlanes = nbrCathPlanes_;
    if (nbrCathPlanes_ != 2 ) {
      singleCathPlaneID = singleCathPlaneID_;
    }

    //
    // Extract the projected pads belonging to the group g
    //

    // Build the group-mask for proj pads
    Mask_t maskProjGrp[cluster.projectedPads->nPads];
    int nbrOfProjPadsInTheGroup = vectorBuildMaskEqualShort( cluster.projPadToGrp, g, cluster.projectedPads->nPads, maskProjGrp);
    projectedPads = new Pads( *cluster.projectedPads, maskProjGrp);
}

// ??? To remove
/*
void Cluster::membersInit() {
  chamberId=-1;
  singleCathPlaneID=-1;
  nbrOfCathodePlanes=0;
  // pads = {nullptr, nullptr};
  nbrSaturated=0;
  mapCathPadIdxToPadIdx={nullptr, nullptr};
  projectedPads=nullptr;
  projNeighbors=nullptr;
  projPadToGrp=nullptr;
  nbrOfProjGroups=0;
  cathGroup={nullptr, nullptr};
  nbrOfCathGroups=0;
  IInterJ=nullptr;
  JInterI=nullptr;
  mapKToIJ=nullptr;
  aloneKPads=nullptr;
  mapIJToK=nullptr;
}
*/

Cluster::~Cluster() {
  for (int c=0; c < 2; c++) {
    if (pads[c] == nullptr) {
      delete pads[c];
      pads[c] == nullptr;
    }
    deleteInt( mapCathPadIdxToPadIdx[c] );
    deleteShort(cathGroup[c]);
  }
  if (projectedPads == nullptr) {
    delete projectedPads;
    projectedPads = nullptr;
  }
  deleteInt(projNeighbors);
  deleteShort(projPadToGrp);
  deleteInt(IInterJ);
  deleteInt(JInterI);
  if ( mapKToIJ != nullptr ) {
    delete [] mapKToIJ;
    mapKToIJ == nullptr;
  }
  deleteInt(mapIJToK);
  deleteInt(aloneIPads);
  deleteInt(aloneJPads);
  deleteInt(aloneKPads);
}

int Cluster::getIndexByRow( const char *matrix, PadIdx_t N, PadIdx_t M, PadIdx_t *IIdx) {
  int k = 0;
  // printf("N=%d, M=%d\n", N, M);
  for( PadIdx_t i=0; i < N; i++) {
    for( PadIdx_t j=0; j < M; j++) {
      if (matrix[i*M+j] == 1) {
        IIdx[k]= j; k++;
        // printf("k=%d,", k);
      }
    }
    // end of row/columns
    IIdx[k]= -1; k++;
  }
  // printf("\n final k=%d \n", k);
  return k;
}

int Cluster::getIndexByColumns( const char *matrix, PadIdx_t N, PadIdx_t M, PadIdx_t *JIdx) {
  int k = 0;
  for( PadIdx_t j=0; j < M; j++) {
    for( PadIdx_t i=0; i < N; i++) {
      if (matrix[i*M+j] == 1) {
        JIdx[k]= i; k++;
      }
    }
    // end of row/columns
    JIdx[k]= -1; k++;
  }
  return k;
}

int Cluster::checkConsistencyMapKToIJ( const char *intersectionMatrix, const MapKToIJ_t *mapKToIJ, const PadIdx_t *mapIJToK, const PadIdx_t *aloneIPads, const PadIdx_t *aloneJPads, int N0, int N1, int nbrOfProjPads) {
  MapKToIJ_t ij;
  int n=0;
  int rc=0;
  // Consistency with intersectionMatrix
  // and aloneI/JPads
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    ij = mapKToIJ[k];
    if ((ij.i >= 0) && (ij.j >= 0)) {
      if ( intersectionMatrix[ij.i*N1 + ij.j] != 1) {
        printf("ERROR: no intersection %d %d %d\n", ij.i , ij.j, intersectionMatrix[ij.i*N1 + ij.j]);
        throw std::overflow_error("Divide by zero exception");
        rc = -1;
      } else {
          n++;
      }
    } else if (ij.i < 0) {
      if ( aloneJPads[ij.j] != k) {
        printf("ERROR: j-pad should be alone %d %d %d %d\n", ij.i , ij.j, aloneIPads[ij.j], k);
        throw std::overflow_error("Divide by zero exception");
        rc = -1;
      }
    } else if (ij.j < 0) {
        if ( aloneIPads[ij.i] != k) {
          printf("ERROR: i-pad should be alone %d %d %d %d\n", ij.i , ij.j, aloneJPads[ij.j], k);
          throw std::overflow_error("Divide by zero exception");
          rc = -1;
        }
    }
  }
  // TODO : Make a test with alone pads ???
  int sum = vectorSumChar( intersectionMatrix, N0*N1 );
  if( sum != n ) {
    printf("ERROR: nbr of intersection differs %d %d \n", n, sum);
    throw std::overflow_error("Divide by zero exception");
    rc = -1;
  }

  for ( int i=0; i<N0; i++ ) {
    for ( int j=0; j<N1; j++ ) {
      int k = mapIJToK[i*N1+ j];
      if (k >= 0) {
        ij = mapKToIJ[k];
        if ( ( ij.i != i) || ( ij.j != j) ) {
            throw std::overflow_error("checkConsistencyMapKToIJ: MapIJToK/MapKToIJ");
            printf( "ij.i=%d, ij.j=%d, i=%d, j=%d \n", ij.i, ij.j, i, j);

        }
      }
    }
  }
  // Check mapKToIJ / mapIJToK
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    ij = mapKToIJ[k];
    if ( ij.i < 0) {
      if (aloneJPads[ij.j] != k ) {
        printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
        throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK aloneJPads");
      }
    } else if( ij.j < 0 ) {
      if (aloneIPads[ij.i] != k ) {
        printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
        throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK aloneIPads");
      }
    } else if( mapIJToK[ij.i*N1+ ij.j] != k ) {
      printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
      throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK");
    }
  }

  return rc;
}

void Cluster::computeProjectedPads(
            const Pads &pad0InfSup, const Pads &pad1InfSup,
            PadIdx_t *aloneIPads, PadIdx_t *aloneJPads, PadIdx_t *aloneKPads, int includeAlonePads) {
  // Use positive values of the intersectionMatrix
  // negative ones are isolated pads
  // Compute the new location of the projected pads (projected_xyDxy)
  // and the mapping mapKToIJ which maps k (projected pads)
  // to i, j (cathode pads)
  const double *x0Inf = pad0InfSup.xInf;
  const double *y0Inf = pad0InfSup.yInf;
  const double *x0Sup = pad0InfSup.xSup;
  const double *y0Sup = pad0InfSup.ySup;
  const double *x1Inf = pad1InfSup.xInf;
  const double *y1Inf = pad1InfSup.yInf;
  const double *x1Sup = pad1InfSup.xSup;
  const double *y1Sup = pad1InfSup.ySup;
  int N0 = pad0InfSup.nPads;
  int N1 = pad1InfSup.nPads;
  //
  double *projX = projectedPads->x;
  double *projY = projectedPads->y;
  double *projDX = projectedPads->dx;
  double *projDY = projectedPads->dy;
  double l, r, b, t;
  int k=0; PadIdx_t *ij_ptr=IInterJ;
  double countIInterJ, countJInterI;
  PadIdx_t i, j;
  for( i=0; i < N0; i++) {
    // Nbr of j-pads intersepting  i-pad
    for( countIInterJ = 0; *ij_ptr != -1; countIInterJ++, ij_ptr++) {
      j = *ij_ptr;
      // Debug
      // printf("X[0/1]inf/sup %d %d %9.3g %9.3g %9.3g %9.3g\n", i, j,  x0Inf[i], x0Sup[i], x1Inf[j], x1Sup[j]);
      l = std::fmax( x0Inf[i], x1Inf[j]);
      r = std::fmin( x0Sup[i], x1Sup[j]);
      b = std::fmax( y0Inf[i], y1Inf[j]);
      t = std::fmin( y0Sup[i], y1Sup[j]);
      projX[k] = (l+r)*0.5;
      projY[k] = (b+t)*0.5;
      projDX[k] = (r-l)*0.5;
      projDY[k] = (t-b)*0.5;
      mapKToIJ[k].i = i;
      mapKToIJ[k].j = j;
      mapIJToK[i*N1+ j] = k;
      // Debug
      printf("newpad %d %d %d %9.3g %9.3g %9.3g %9.3g\n", i, j, k, projX[k], projY[k], projDX[k], projDY[k]);
      k++;
    }
    // Test if there is no intercepting pads with i-pad
    if( (countIInterJ == 0) && includeAlonePads ) {
      l = x0Inf[i];
      r = x0Sup[i];
      b = y0Inf[i];
      t = y0Sup[i];
      projX[k] = (l+r)*0.5;
      projY[k] = (b+t)*0.5;
      projDX[k] = (r-l)*0.5;
      projDY[k] = (t-b)*0.5;
      // printf("newpad alone cath0 %d %d %9.3g %9.3g %9.3g %9.3g\n", i, k, projX[k], projY[k], projDX[k], projDY[k]);
      // Not used ???
      mapKToIJ[k].i =i ; mapKToIJ[k].j=-1;
      aloneIPads[i] = k;
      aloneKPads[k] = i;
      k++;
    }
    // Row change
    ij_ptr++;
  }
  // Just add alone j-pads of cathode 1
  if ( includeAlonePads ) {
    ij_ptr = JInterI;
    for( PadIdx_t j=0; j < N1; j++) {
      for( countJInterI = 0; *ij_ptr != -1; countJInterI++, ij_ptr++);
      if( countJInterI == 0) {
        l = x1Inf[j];
        r = x1Sup[j];
        b = y1Inf[j];
        t = y1Sup[j];
        projX[k] = (l+r)*0.5;
        projY[k] = (b+t)*0.5;
        projDX[k] = (r-l)*0.5;
        projDY[k] = (t-b)*0.5;
        // Debug
        // printf("newpad alone cath1 %d %d %9.3g %9.3g %9.3g %9.3g\n", j, k, projX[k], projY[k], projDX[k], projDY[k]);
        // newCh0[k] = ch0[i];
        // Not used ???
        mapKToIJ[k].i=-1; mapKToIJ[k].j=j;
        aloneJPads[j] = k;
        aloneKPads[k] = j;
        k++;
      }
      // Skip -1, row/ col change
      ij_ptr++;
    }
  }
  if (VERBOSE > 2) {
    printf("builProjectPads mapIJToK=%p, N0=%d N1=%d\\n", mapIJToK, N0, N1);
    for (int i=0; i <N0; i++) {
      for (int j=0; j <N1; j++) {
        if ( (mapIJToK[i*N1+j] != -1))
          printf(" %d inter %d\n", i, j);
      }
    }
    vectorPrintInt("builProjectPads", aloneKPads, k);
  }
  projectedPads->nPads = k;
}

int Cluster::buildProjectedGeometry( int includeSingleCathodePads) {

  // Single cathode
  if (nbrOfCathodePlanes==1) {
    // One Cathode case
    // Pad Projection is the cluster itself
    /* Inv ???
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
    */
    projectedPads = new Pads( *pads[0], Pads::xydxdyMode);
    projNeighbors = projectedPads->buildFirstNeighbors();
    return projectedPads->nPads;
  }

  int N0 = pads[0]->nPads;
  int N1 = pads[1]->nPads;
  char intersectionMatrix[N0*N1];
  vectorSetZeroChar( intersectionMatrix, N0*N1);

  // Get the pad limits
  Pads padInfSup0( *pads[0], Pads::xyInfSupMode);
  Pads padInfSup1( *pads[1], Pads::xyInfSupMode);
  mapIJToK = new PadIdx_t[N0*N1];
  vectorSetInt( mapIJToK, -1, N0*N1);

  //
  // Build the intersection matrix
  // Looking for j pads, intercepting pad i
  //
  double xmin, xmax, ymin, ymax;
  PadIdx_t xInter, yInter;
  for( PadIdx_t i=0; i < N0; i++) {
    for( PadIdx_t j=0; j < N1; j++) {
      xmin = std::fmax( padInfSup0.xInf[i], padInfSup1.xInf[j] );
      xmax = std::fmin( padInfSup0.xSup[i], padInfSup1.xSup[j] );
      xInter = ( xmin <= (xmax - epsilonGeometry) );
      if( xInter ) {
        ymin = std::fmax( padInfSup0.yInf[i], padInfSup1.yInf[j] );
        ymax = std::fmin( padInfSup0.ySup[i], padInfSup1.ySup[j] );
        yInter = ( ymin <= (ymax - epsilonGeometry));
        intersectionMatrix[i*N1+j] =  yInter;
        // printf("inter i=%3d, j=%3d,  x0=%8.5f y0=%8.5f, x1=%8.5f y1=%8.5f\n", i, j, pads[0]->x[i], pads[0]->y[i], pads[1]->x[i], pads[1]->y[i]);
      }
    }
  }
  //
  if (VERBOSE) printMatrixChar( "  Intersection Matrix", intersectionMatrix, N0, N1);
  //
  // Compute the max number of projected pads to make
  // memory allocations
  //
  int maxNbrOfProjPads = vectorSumChar( intersectionMatrix, N0*N1 );
  int nbrOfSinglePads = 0;
  if (includeSingleCathodePads) {
    // Add alone cath0-pads
    for( PadIdx_t i=0; i < N0; i++) {
      if ( vectorSumRowChar( &intersectionMatrix[i*N1], N0, N1 ) == 0) nbrOfSinglePads++;
    }
    // Add alone cath1-pads
    for( PadIdx_t j=0; j < N1; j++) {
      if ( vectorSumColumnChar( &intersectionMatrix[j], N0, N1) == 0) nbrOfSinglePads++;
    }
  }
  // Add alone pas and row/column separators
  maxNbrOfProjPads += nbrOfSinglePads + fmax( N0, N1);
  if (VERBOSE) printf("  maxNbrOfProjPads %d\n", maxNbrOfProjPads);
  //
  //
  // Projected pad allocation
  // The limit maxNbrOfProjPads is alocated
  //
  projectedPads = new Pads(4*maxNbrOfProjPads, Pads::xydxdyMode);
  double *projX  = projectedPads->x;
  double *projY  = projectedPads->y;
  double *projDX  = projectedPads->dx;
  double *projDY  = projectedPads->dy;
  /* Inv ???
  projX = getX ( projected_xyDxy, maxNbrOfProjPads);
  projY  = getY ( projected_xyDxy, maxNbrOfProjPads);
  projDX = getDX( projected_xyDxy, maxNbrOfProjPads);
  projDY = getDY( projected_xyDxy, maxNbrOfProjPads);
  */
  //
  // Intersection Matrix Sparse representation
  //
  /// To Save ???
  IInterJ = new PadIdx_t[maxNbrOfProjPads];
  JInterI = new PadIdx_t[maxNbrOfProjPads];
  int checkr = getIndexByRow( intersectionMatrix, N0, N1, IInterJ);
  int checkc = getIndexByColumns( intersectionMatrix, N0, N1, JInterI);
  if (CHECK) {
    if (( checkr > maxNbrOfProjPads) || (checkc > maxNbrOfProjPads)) {
      printf("Allocation pb for  IInterJ or JInterI: allocated=%d, needed for row=%d, for col=%d \n",
          maxNbrOfProjPads, checkr, checkc);
      throw std::overflow_error(
         "Allocation pb for  IInterJ or JInterI" );
    }
  }
  if (VERBOSE) {
    printInterMap("  IInterJ", IInterJ, N0 );
    printInterMap("  JInterI", JInterI, N1 );
  }

  //
  // Remaining allocation
  //
  // ??? To allocate in computeProjectedPads
  // Stack allocation ????

  aloneIPads = new PadIdx_t[N0];
  aloneJPads = new PadIdx_t[N1];
  // PadIdx_t *aloneKPads = new PadIdx_t[N0*N1];
  mapKToIJ = new MapKToIJ_t[maxNbrOfProjPads];
  /*
  PadIdx_t aloneIPads[N0];
  PadIdx_t aloneJPads[N1];
  */
  // ??? Inv PadIdx_t aloneKPads[N0*N1];
  aloneKPads = new PadIdx_t[maxNbrOfProjPads];
  vectorSetInt( aloneIPads, -1, N0);
  vectorSetInt( aloneJPads, -1, N1);
  vectorSetInt( aloneKPads, -1, maxNbrOfProjPads);


  //
  // Build the new pads
  // ???
  computeProjectedPads( padInfSup0, padInfSup1,
                      aloneIPads, aloneJPads, aloneKPads, includeSingleCathodePads );

  if (CHECK == 1) checkConsistencyMapKToIJ( intersectionMatrix, mapKToIJ, mapIJToK, aloneIPads, aloneJPads, N0, N1, projectedPads->nPads);
  /* Inv ??? Done in clusterProcessing.cxx
  if (INSPECTMODEL) {
      saveProjectedPads( projectedPads );
  }
   */
  //
  // Get the isolated new pads
  // (they have no neighboring)
  //
  int thereAreIsolatedPads = 0;
  projNeighbors = projectedPads->buildFirstNeighbors();
  Pads::printPads("Projected Pads:", *projectedPads);
  Pads::printNeighbors( projNeighbors, projectedPads->nPads);

  int nbrOfProjPads = projectedPads->nPads;
  MapKToIJ_t ij;
  for( PadIdx_t k=0; k < nbrOfProjPads; k++) {
    if (getTheFirstNeighborOf(projNeighbors, k) == -1) {
      // pad k is isolated
      thereAreIsolatedPads = 1;
      ij = mapKToIJ[k];
      if( (ij.i >= 0) && (ij.j >= 0)) {
        if (VERBOSE) printf(" Isolated pad: nul intersection i,j = %d %d\n", ij.i, ij.j);
        intersectionMatrix[ij.i*N1 + ij.j] = 0;
      } else {
        throw std::overflow_error("I/j negative (alone pad)");
      }
    }
  }
  if (VERBOSE && thereAreIsolatedPads) printf("There are isolated pads %d\n", thereAreIsolatedPads);
  //
  if( thereAreIsolatedPads == 1) {
    // Recompute all
    // Why ???
    getIndexByRow( intersectionMatrix, N0, N1, IInterJ);
    getIndexByColumns( intersectionMatrix, N0, N1, JInterI);
    //
    // Build the new pads
    //
    delete projectedPads;
    computeProjectedPads( padInfSup0, padInfSup1,
                      aloneIPads, aloneJPads, aloneKPads, includeSingleCathodePads );
    delete [] projNeighbors;
    projNeighbors = projectedPads->buildFirstNeighbors( );
  }
  // Set null Charge
  vectorSetZero( projectedPads->q, projectedPads->nPads);
  return projectedPads->nPads;
}

double *Cluster::projectChargeOnProjGeometry(int includeAlonePads) {

  double *qProj;
  if( nbrOfCathodePlanes == 1 ) {
    Pads *sPads = pads[singleCathPlaneID];
    qProj = new double [sPads->nPads];
    vectorCopy( sPads->q, sPads->nPads, qProj);
    return qProj;
  }
  int nbrOfProjPads = projectedPads->nPads;
  double *ch0 = pads[0]->q;
  double *ch1 = pads[1]->q;
  //
  // Computing charges of the projected pads
  // Ch0 part
  //
  double minProj[nbrOfProjPads];
  double maxProj[nbrOfProjPads];
  double projCh0[nbrOfProjPads];
  double projCh1[nbrOfProjPads];
  int N0 = pads[0]->nPads;
  int N1 = pads[1]->nPads;
  PadIdx_t k=0;
  double sumCh1ByRow;
  PadIdx_t *ij_ptr = IInterJ;
  PadIdx_t *rowStart;
  for( PadIdx_t i=0; i < N0; i++) {
    // Save the starting index of the begining of the row
    rowStart = ij_ptr;
    // sum of charge with intercepting j-pad
    for( sumCh1ByRow = 0.0; *ij_ptr != -1; ij_ptr++) sumCh1ByRow += ch1[ *ij_ptr ];
    double ch0_i = ch0[i];
    if ( sumCh1ByRow != 0.0 ) {
      double cst = ch0[i] / sumCh1ByRow;
      for( ij_ptr = rowStart; *ij_ptr != -1; ij_ptr++) {
        projCh0[k] = ch1[ *ij_ptr ] * cst;
        minProj[k] = fmin( ch1[ *ij_ptr ], ch0_i );
        maxProj[k] = fmax( ch1[ *ij_ptr ], ch0_i );
        // Debug
        // printf(" i=%d, j=%d, k=%d, sumCh0ByCol = %g, projCh1[k]= %g \n", i, *ij_ptr, k, sumCh1ByRow, projCh0[k]);
        k++;
      }
    } else if (includeAlonePads){
      // Alone i-pad
      projCh0[k] = ch0[i];
      minProj[k] = ch0[i];
      maxProj[k] = ch0[i];
      k++;
    }
    // Move on a new row
    ij_ptr++;
  }
  // Just add alone pads of cathode 1
  if ( includeAlonePads ) {
    for( PadIdx_t j=0; j < N1; j++) {
      k = aloneJPads[j];
      if ( k >= 0 ){
        projCh0[k] = ch1[j];
        minProj[k] = ch1[j];
        maxProj[k] = ch1[j];
      }
    }
  }
  //
  // Computing charges of the projected pads
  // Ch1 part
  //
  k=0;
  double sumCh0ByCol;
  ij_ptr = JInterI;
  PadIdx_t *colStart;
  for( PadIdx_t j=0; j < N1; j++) {
    // Save the starting index of the beginnig of the column
    colStart = ij_ptr;
    // sum of charge intercepting i-pad
    for( sumCh0ByCol = 0.0; *ij_ptr != -1; ij_ptr++) sumCh0ByCol += ch0[ *ij_ptr ];
    if ( sumCh0ByCol != 0.0 ) {
      double cst = ch1[j] / sumCh0ByCol;
      for( ij_ptr = colStart; *ij_ptr != -1; ij_ptr++) {
        PadIdx_t i = *ij_ptr;
        k = mapIJToK[i * N1 + j];
        projCh1[k] = ch0[i] * cst;
        // Debug
        // printf(" j=%d, i=%d, k=%d, sumCh0ByCol = %g, projCh1[k]= %g \n", j, i, k, sumCh0ByCol, projCh1[k]);
      }
    } else if (includeAlonePads){
      // Alone j-pad
      k = aloneJPads[j];
      if ( CHECK && (k < 0) ) printf("ERROR: Alone j-pad with negative index j=%d\n", j);
      // printf("Alone i-pad  i=%d, k=%d\n", i, k);
      projCh1[k] = ch1[j];
    }
    ij_ptr++;
  }

  // Just add alone pads of cathode 0
  if ( includeAlonePads ) {
    ij_ptr = IInterJ;
    for( PadIdx_t i=0; i < N0; i++) {
      k = aloneIPads[i];
      if ( k >= 0) {
        // printf("Alone i-pad  i=%d, k=%d\n", i, k);
        projCh1[k] = ch0[i];
      }
    }
  }
  // Charge Result
  // Do the mean
  qProj = new double[nbrOfProjPads];
  vectorAddVector( projCh0, 1.0, projCh1, nbrOfProjPads, qProj);
  vectorMultScalar( qProj, 0.5, nbrOfProjPads, qProj );
  return qProj;
}

int Cluster::buildGroupOfPads() {

  // Having one cathode plane (cathO, cath1 or projected cathodes)
  // Extract the sub-clusters
  int nProjPads = projectedPads->nPads;
  projPadToGrp = new Groups_t[nProjPads];
  // Set to 1 because 1-cathode mode
  vectorSetShort( projPadToGrp, 0, nProjPads);

  //
  //  Part I - Extract groups from projection plane
  //

  //
  // V1 ??
  // Inv ??? short *grpToCathGrp = 0;
  int nCathGroups = 0;
  int nGroups = 0;
  int nbrCath0 = pads[0]->nPads;
  int nbrCath1 = pads[1]->nPads;
  //
  // Build the "proj-groups"
  // The pads which have no recovery with the other cathode plane
  // are not considered. They are named 'single-pads'
  nbrOfProjGroups = getConnectedComponentsOfProjPadsWOSinglePads();

  if(INSPECTMODEL) {
      saveProjPadToGroups( projPadToGrp, projectedPads->nPads );
  }

  // Single cathode case
  // The cath-group & proj-groups are the same
  Group_t grpToCathGrp[nbrOfProjGroups+1];
  if (nbrOfCathodePlanes == 1) {

    // Copy the projPadGroup to cathGroup[0/1]
    int nCathPads = pads[singleCathPlaneID]->nPads;
    cathGroup[singleCathPlaneID ] = new Group_t[ nCathPads];
    vectorCopyShort( projPadToGrp, nCathPads, cathGroup[singleCathPlaneID]);

    // Identity mapping
    for(int g=0; g < (nbrOfProjGroups+1); g++) grpToCathGrp[g] = g;

    nCathGroups = nbrOfProjGroups;
    nGroups = nCathGroups;
    // Merged groups
    // Unsused ??? int nMergedGrp = nGroups;
    short *padToMergedGrp = new short[nCathPads];
    vectorCopyShort( projPadToGrp, nCathPads, padToMergedGrp);
    // New Laplacian
    // nCathGroups = assignGroupToCathPads( projPadToGrp, nProjPads, nProjGroups, nbrCath0, nbrCath1, cath0ToGrp, cath1ToGrp);
    nCathGroups = assignGroupToCathPads();
    nGroups = nCathGroups;
    // Inv ??? int nNewGroups = addIsolatedPadInGroups( *pads[aloneCath], cathGroup[aloneCath ], grpToCathGrp, nGroups);
    // Add the pads (not present in the projected plane)
    int nNewGroups = pads[singleCathPlaneID]->addIsolatedPadInGroups( cathGroup[singleCathPlaneID ], grpToCathGrp, nGroups);
    // ??? Check if some time
    if (nNewGroups > 0) {
        throw std::overflow_error("New group with one cathode plane ?????");
    }
    nGroups += nNewGroups;
    //
  } else {
    // 2 cathodes & projected cathodes
    // Init. the new groups 'cath-groups'
    int nCath0 = pads[0]->nPads;
    int nCath1 = pads[1]->nPads;
    cathGroup[0] = new Group_t[ nCath0];
    cathGroup[1] = new Group_t[ nCath1];
    vectorSetZeroShort( cathGroup[0], nCath0 );
    vectorSetZeroShort( cathGroup[1], nCath1 );
    if (VERBOSE > 0) {
      printf("Projected Groups %d\n", nbrOfProjGroups);
      vectorPrintShort("  projPadToGrp", projPadToGrp, nProjPads);
    }
    // short matGrpGrp[ (nProjGroups+1)*(nProjGroups+1)];
    // Get the matrix grpToGrp to identify ovelapping groups
    // assignCathPadsToGroupFromProj( projPadToGrp, nProjPads, nProjGroups, nbrCath0, nbrCath1, wellSplitGroup, matGrpGrp);

    // ???? compil grpToCathGrp = new short[nbrOfProjGroups+1];
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
    // Inv ??? padToMergedGrp = new short[nPads];
    int nPads = pads[0]->nPads + pads[1]->nPads;
    short padToMergedGrp[nPads];
    // Over allocated array
    // ??? short projGrpToMergedGrp[nGroups+1];
    if (1) {
    // TODO ???
    // Some alone pads in one cathode plane form a group but are
    // not set as a group (0), ex: Run2 orbit 0, ROF=7, cluster=319

    // Propagate proj-groups on the cathode pads
    nGroups = assignPadsToGroupFromProj( nbrOfProjGroups );
    if (VERBOSE > 0) {
      printf("Groups after propagation to cathodes [assignPadsToGroupFromProj] nCathGroups=%d\n", nGroups);
      // vectorPrintShort( "  projPadToGrp", projPadToGrp, nProjPads);
      // vectorPrintShort( "  cath0ToGrp", cath0ToGrp, nbrCath0);
      // vectorPrintShort( "  cath1ToGrp", cath1ToGrp, nbrCath1);
    }
    }

    // Take care the number of Gropu nGrp increase ???????????????????????????????
    // Add new grp from isolated pad and merge them if it occures
    //
    // Compute the max allocation for the new cath-groups
    // Have to add single pads
    int nbrSinglePads=0;
    for (int c=0; c < 2; c++) {
      for ( int p=0; p < pads[c]->nPads; p++) {
        if( cathGroup[c][p] == 0 ) {
          nbrSinglePads += 1;
        }
      }
    }
    int nbrMaxGroups = nGroups + nbrSinglePads;
    // Use to map oldGroups to newGroups when
    // occurs the  groups change
    Mask_t mapGrpToGrp[ nbrMaxGroups +1 ];
    for (int g=0; g < (nbrMaxGroups+1); g++) {
      mapGrpToGrp[g] = g;
    }
    //
    // int nNewGrpCath0 = addIsolatedPadInGroups( xy0Dxy, cath0ToGrp, nbrCath0, 0, projGrpToMergedGrp, nGroups );
    // GG inv ??? int nNewGrpCath0 = addIsolatedPadInGroups( xy0Dxy, cath0ToGrp, nbrCath0, 0, mapGrpToGrp, nGroups );
    // ??? inv int nNewGrpCath0 = addIsolatedPadInGroups( *pads[0], cathGroup[0], mapGrpToGrp, nGroups);

    // Add single pads of cath-0 and modyfy the groups
    int nNewGrpCath0 = pads[0]->addIsolatedPadInGroups( cathGroup[0], mapGrpToGrp, nGroups);
// ??? May be to fuse with addIsolatedPads
    nGroups += nNewGrpCath0;
    // Apply the new Groups on cath1
    for ( int p=0; p < nbrCath1; p++) {
      cathGroup[1][p] = mapGrpToGrp[cathGroup[1][p]];
    }
    // and on proj-pads
    for ( int p=0; p < nProjPads; p++) {
      projPadToGrp[p] = mapGrpToGrp[ projPadToGrp[p]];
    }
    if( VERBOSE > 1) {
      printf("# addIsolatedPadInGroups cath-0 nNewGroups =%d\n", nGroups);
      vectorPrintShort( "  mapGrpToGrp", mapGrpToGrp, nGroups+1);
    }
    // nGroups = nNewGroups;
    // int nNewGrpCath1 = addIsolatedPadInGroups( xy1Dxy, cath1ToGrp, nbrCath1, 1, projGrpToMergedGrp, nNewGrpCath0);
    /// int nNewGrpCath1 = addIsolatedPadInGroups( xy1Dxy, cath1ToGrp, nbrCath1, 1, mapGrpToGrp, nGroups);
    // ??? Inv int nNewGrpCath1 = addIsolatedPadInGroups( *pads[1], cathGroup[1], mapGrpToGrp, nGroups);

    // Do the same on cath1
    // Add single pads of cath-0 and modyfy the groups
    int nNewGrpCath1 = pads[1]->addIsolatedPadInGroups( cathGroup[1], mapGrpToGrp, nGroups);
    nGroups += nNewGrpCath1;
    // Apply the new Groups on cath1
    for ( int p=0; p < nbrCath0; p++) {
      cathGroup[0][p] = mapGrpToGrp[cathGroup[0][p]];
    }
    // and on proj-pads
    for ( int p=0; p < nProjPads; p++) {
      projPadToGrp[p] = mapGrpToGrp[ projPadToGrp[p]];
    }
    // int nNewGroups = renumberGroups( projGrpToMergedGrp, nNewGrpCath1);
    /// Inv ???? int nNewGroups = renumberGroupsV2( cath0ToGrp, nbrCath0, cath1ToGrp, nbrCath1, mapGrpToGrp, std::max( nNewGrpCath0, nNewGrpCath1));

    // Some groups may be merged, others groups may diseappear
    // So the final groups must be renumbered
    int nNewGroups = renumberGroupsV2( cathGroup[0], pads[0]->nPads, cathGroup[1], pads[1]->nPads, mapGrpToGrp, nGroups);
    if (VERBOSE > 1) {
      printf("Groups after renumbering %d\n", nGroups);
      vectorPrintShort( "  projPadToGrp", projPadToGrp, nProjPads);
      printf("  nNewGrpCath0=%d, nNewGrpCath1=%d, nGroups=%d\n", nNewGrpCath0, nNewGrpCath1, nGroups);
      vectorPrintShort( "  cath0ToGrp  ", cathGroup[0], nbrCath0);
      vectorPrintShort( "  cath1ToGrp  ", cathGroup[1], nbrCath1);
      vectorPrintShort("   mapGrpToGrp ", mapGrpToGrp, nNewGroups);
    }
    // Apply this renumbering on projection-pads
    for ( int p=0; p < nProjPads; p++) {
      projPadToGrp[p] = mapGrpToGrp[ projPadToGrp[p]];
    }

    nGroups = nNewGroups;
    // Update proj-pads groups from cath grous
    updateProjectionGroups ();

    if (VERBOSE > 0) {
      printf("# Groups after adding isolate pads and renumbering %d\n", nGroups);
      vectorPrintShort( "  projPadToGrp", projPadToGrp, nProjPads);
      vectorPrintShort( "  cath0ToGrp  ", cathGroup[0], nbrCath0);
      vectorPrintShort( "  cath1ToGrp  ", cathGroup[1], nbrCath1);
    }
  }

  if( VERBOSE > 0) {
    printf("# Final Groups %d\n", nGroups);
    vectorPrintShort("  cath0ToGrp", cathGroup[0], nbrCath0);
    vectorPrintShort("  cath1ToGrp", cathGroup[1], nbrCath1);
  }
  /* Inv ???
  if (INSPECTMODEL) {
    inspectModel.padToCathGrp = new Group_t[nPads];
    // vectorCopyShort( padToMergedGrp, nPads, inspectModel.padToCathGrp );
    // Inv vectorCopyShort( cath0ToGrp, nbrCath0, inspectModel.padToCathGrp);
    // Inv vectorCopyShort( cath1ToGrp, nbrCath1, &inspectModel.padToCathGrp[nbrCath0]);
    vectorScatterShort( cath0ToGrp, maskCath0, nPads, inspectModel.padToCathGrp);
    vectorScatterShort( cath1ToGrp, maskCath1, nPads, inspectModel.padToCathGrp);

    inspectModel.nCathGroups = 0;
  }
  */
  return nGroups;
}

int Cluster::getConnectedComponentsOfProjPadsWOSinglePads(  ) {
  // Class from neighbors list of projected pads, the pads in groups (connected components)
  // projPadToGrp is set to the group Id of the pad.
  // If the group Id is zero, the the pad is unclassified
  // Return the number of groups
  int  N = projectedPads->nPads;
  projPadToGrp = new Groups_t[N];
  PadIdx_t *neigh = projNeighbors;
  PadIdx_t neighToDo[N];
  vectorSetZeroShort( projPadToGrp, N);
  // Nbr of pads alrready proccessed
  int nbrOfPadSetInGrp = 0;
  // Last projPadToGrp to process
  short *curPadGrp = projPadToGrp;
  short currentGrpId = 0;
  //
  int i, j, k;
  // printNeighbors();
  if (VERBOSE > 1) {
    printf("[getConnectedComponentsOfProjPadsWOIsolatedPads]\n");
  }
  while (nbrOfPadSetInGrp < N) {
    // Seeking the first unclassed pad (projPadToGrp[k]=0)
    for( ; (curPadGrp < &projPadToGrp[N]) && *curPadGrp != 0; curPadGrp++ );
    k = curPadGrp - projPadToGrp;
    if (VERBOSE > 1) {
      printf( "  k=%d, nbrOfPadSetInGrp g=%d: n=%d\n", k, currentGrpId, nbrOfPadSetInGrp);
    }
    //
    // New group for pad k - then search all neighbours of k
    // aloneKPads = 0 if only one cathode
    if (aloneKPads && (aloneKPads[k] != -1)) {
      // Alone Pad no group at the moment
      if (VERBOSE > 1) {
        printf("  isolated pad %d\n", k);
      }
      projPadToGrp[k] = -1;
      nbrOfPadSetInGrp++;
      continue;
    }
    currentGrpId++;
    if (VERBOSE > 1) {
      printf( "  NEW GRP, pad k=%d in new grp=%d\n", k, currentGrpId);
    }
    projPadToGrp[k] = currentGrpId;
    nbrOfPadSetInGrp++;
    PadIdx_t startIdx = 0, endIdx = 1;
    neighToDo[ startIdx ] = k;
    // Labels k neighbors
    // Propagation of the group in all neighbour list
    for( ; startIdx < endIdx; startIdx++) {
      i = neighToDo[startIdx];
      if (VERBOSE > 1) {
        printf("  propagate to neighbours of i=%d ", i );
      }
      //
      // Scan i neighbors
      for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh,i); *neigh_ptr != -1; neigh_ptr++) {
        j = *neigh_ptr;
        // printf("    neigh j %d\n, \n", j);
        if ((projPadToGrp[j] == 0)  ) {
          // Add the neighbors in the currentgroup
          //
          // aloneKPads = 0 if only one cathode
          if (aloneKPads && (aloneKPads[j] != -1)) {
            if (VERBOSE > 1) {
              printf("isolated pad %d, ", j);
            }
            projPadToGrp[j] = -1;
            nbrOfPadSetInGrp++;
            continue;
          }
          if (VERBOSE > 1) {
            printf("%d, ", j);
          }
          projPadToGrp[ j ] = currentGrpId;
          nbrOfPadSetInGrp++;
          // Append in the neighbor list to search
          neighToDo[ endIdx] = j;
          endIdx++;
        }
      }
      if (VERBOSE > 1) {
        printf("\n");
      }
    }
    // printf("make groups grpId=%d, nbrOfPadSetInGrp=%d\n", currentGrpId, nbrOfPadSetInGrp);
  }
  for ( int k=0; k < N; k++) {
    if ( projPadToGrp[k] == -1) {
      projPadToGrp[k] = 0;
    }
  }
  // return tne number of Grp
  return currentGrpId;
}

///??????????????????
/*
void Cluster::assignSingleCathPadsToGroup( short *padGroup, int nPads, int nGrp, int nCath0, int nCath1) {
  Group_t cath0ToGrpFromProj[nCath0];
  Group_t cath1ToGrpFromProj[nCath1];
  cath1ToGrpFromProj = 0;
  if ( nCath0 != 0) {
    cath0ToGrpFromProj = new short[nCath0];
    vectorCopyShort( padGroup, nCath0, cath0ToGrpFromProj);
  } else {
    cath1ToGrpFromProj = new short[nCath1];
    vectorCopyShort( padGroup, nCath1, cath1ToGrpFromProj);
  }
  vectorSetShort( wellSplitGroup, 1, nGrp+1);
}
*/

// Assign a group to the original pads
// Update the pad group and projected-pads group
int Cluster::assignPadsToGroupFromProj(
        // const PadIdx_t *cath0ToPadIdx, const PadIdx_t *cath1ToPadIdx,
        // int nGrp, int nPads, short *padMergedGrp ) {
        int nGrp ) {
// cath0ToPadIdx : pad indices of cath0 (cath0ToPadIdx[0..nCath0] -> i-pad
// outputs:
  short matGrpGrp[ (nGrp+1)*(nGrp+1)];
  //
  // vectorSetShort( wellSplitGroup, 1, nGrp+1);
  vectorSetZeroShort( matGrpGrp, (nGrp+1)*(nGrp+1) );
  //
  PadIdx_t i, j;
  short g, prevGroup;
  if (VERBOSE > 1) {
    printf( "[AssignPadsToGroupFromProj]\n");
  }
  // Expand the projected Groups
  // 'projPadToGrp' to the pad groups 'padToGrp'
  // If there are conflicts, fuse the groups
  // Build the Group-to-Group matrix matGrpGrp
  // which describe how to fuse Groups
  // with the projected Groups
  // projPadToGrp
  int nProjPads = projectedPads->nPads;
  for( int k=0; k < nProjPads; k++) {
    g = projPadToGrp[k];
    // give the indexes of overlapping pads
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    //
    // Cathode 0
    //
    // Inv ??? if ( (i >= 0) && (cath0ToPadIdx !=0) ) {
    if ( i >= 0 ) {
      // Remark: if i is an alone pad (j<0)
      // i is processed as well
      //
      // cath0ToPadIdx: map cathode-pad to the original pad
      /*
      PadIdx_t padIdx = cath0ToPadIdx[i];
      prevGroup = padToGrp[ padIdx ];
      */
      prevGroup = cathGroup[0][i];
      if ( (prevGroup == 0) || (prevGroup == g) ) {
        // Case: no group before or same group
        //
        // ???? padToGrp[ padIdx ] = g;
        cathGroup[0][i] = g;
        matGrpGrp[ g*(nGrp+1) +  g ] = 1;
      } else {
        // Already a grp (Conflict)
        // if ( prevGroup > 0) {
          // Invalid prev group
          // wellSplitGroup[ prevGroup ] = 0;
          // Store in the grp to grp matrix
          // Group to fuse
          cathGroup[0][i] = g;
          matGrpGrp[ g*(nGrp+1) +  prevGroup ] = 1;
          matGrpGrp[ prevGroup*(nGrp+1) +  g ] = 1;
        //}
        // padToGrp[padIdx] = -g;
      }
    }
    //
    // Cathode 1
    //
    // ??? if ( (j >= 0) && (cath1ToPadIdx != 0) ) {
    if ( (j >= 0) ) {
      // Remark: if j is an alone pad (j<0)
      // j is processed as well
      //
      // cath1ToPadIdx: map cathode-pad to the original pad
      // ??? PadIdx_t padIdx = cath1ToPadIdx[j];
      // ??? prevGroup = padToGrp[padIdx];
      prevGroup = cathGroup[1][j];

      if ( (prevGroup == 0) || (prevGroup == g) ){
        // No group before
        // padToGrp[padIdx] = g;
        cathGroup[1][j] = g;
        matGrpGrp[ g*(nGrp+1) +  g ] = 1;
      } else {
        // Already a Group (Conflict)
        // if ( prevGroup > 0) {
          cathGroup[1][j] = g;
          matGrpGrp[ g*(nGrp+1) +  prevGroup ] = 1;
          matGrpGrp[ prevGroup*(nGrp+1) +  g ] = 1;
        // }
        // padToGrp[padIdx] = -g;
      }
    }
  }
  if (VERBOSE > 0) {
    printMatrixShort("  Group/Group matrix", matGrpGrp, nGrp+1, nGrp+1);
    vectorPrintShort("  cathToGrp[0]", cathGroup[0], pads[0]->nPads);
    vectorPrintShort("  cathToGrp[1]", cathGroup[1], pads[1]->nPads);
  }
  //
  // Merge the groups (build the mapping grpToMergedGrp)
  //
  Group_t grpToMergedGrp[nGrp+1]; // Mapping old groups to new merged groups
  vectorSetZeroShort(grpToMergedGrp, nGrp+1);
  //
  int iGroup = 1; // Describe the current group
  int curGroup;   // Describe the mapping grpToMergedGrp[iGroup]
  while ( iGroup < (nGrp+1)) {
    // Define the new group to process
    if ( grpToMergedGrp[iGroup] == 0 ) {
        // newGroupID++;
        // grpToMergedGrp[iGroup] = newGroupID;
        grpToMergedGrp[iGroup] = iGroup;
    }
    curGroup = grpToMergedGrp[iGroup];
    // printf( "  current iGroup=%d -> grp=%d \n", iGroup, curGroup);
    //
      // Look for other groups in matGrpGrp
      int ishift = iGroup*(nGrp+1);
      // Check if there are an overlaping group
      for (int j=iGroup+1; j < (nGrp+1); j++) {
        if ( matGrpGrp[ishift+j] ) {
          // Merge the groups with the current one
          if ( grpToMergedGrp[j] == 0) {
            // printf( "    newg merge grp=%d -> grp=%d\n", j, curGroup);
            // No group assign before, merge the groups with the current one
            grpToMergedGrp[j] = curGroup;
          } else {
            // Fuse grpToMergedGrp[j] with
            // Merge curGroup and grpToMergedGrp[j]
            // printf( "    oldg merge grp=%d -> grp=%d\n", curGroup, grpToMergedGrp[j]);

            // A group is already assigned, the current grp takes the grp of ???
            // Remark : curGroup < j
            // Fuse and propagate
            grpToMergedGrp[ curGroup ] = grpToMergedGrp[j];
            for( int g=1; g < nGrp+1; g++) {
                if (grpToMergedGrp[g] == curGroup) {
                    grpToMergedGrp[g] = grpToMergedGrp[j];
                }
            }
          }
        }
      }
      iGroup++;
  }

  // Perform the mapping group -> mergedGroups
  if (VERBOSE >0 ) {
    vectorPrintShort( "  grpToMergedGrp", grpToMergedGrp, nGrp+1);
  }
  //
  // Renumber the fused groups
  //
  int newGroupID = 0;
  Mask_t map[nGrp+1];
  vectorSetZeroShort( map, (nGrp+1) );
  for (int g=1; g < (nGrp+1); g++) {
    int gm = grpToMergedGrp[g];
    if ( map[gm] == 0) {
      newGroupID++;
      map[gm] = newGroupID;
    }
  }
  // vectorPrintShort( "  map", map, nGrp+1);
  // Apply the renumbering
  for (int g=1; g < (nGrp+1); g++) {
    grpToMergedGrp[g] = map[ grpToMergedGrp[g] ];
  }

  // Perform the mapping grpToMergedGrp
  if ( VERBOSE >0 ) {
    vectorPrintShort( "  grpToMergedGrp", grpToMergedGrp, nGrp+1);
  }
  for ( int c=0; c<2; c++) {
    for ( int p=0; p< pads[c]->nPads; p++) {
      // ??? Why abs() ... explain
      cathGroup[c][p]= grpToMergedGrp[ std::abs(cathGroup[c][p]) ];
    }
  }

  if (CHECK) {
    for ( int c=0; c<2; c++) {
      for ( int p=0; p< pads[c]->nPads; p++) {
        // ??? Why abs() ... explain
        // ??? cathGroup[c][p]= grpToMergedGrp[ std::abs(cathGroup[c][p]) ];
        if (cathGroup[c][p] == 0) {
          printf("  Warning  assignPadsToGroupFromProj: pad %d with no group\n", p);
        }
      }
    }
  }

  // Update the group of the proj-pads
  vectorMapShort(projPadToGrp, grpToMergedGrp, nProjPads);

  //
  return newGroupID;
}

int Cluster::assignGroupToCathPads( ) {
// old arg short *projPadGroup, int nProjPads, int nGrp, int nCath0, int nCath1, short *cath0ToGrp, short *cath1ToGrp) {
  // From the cathode group found with the projection,
  //
  int nCath0 = pads[0]->nPads;
  int nCath1 = pads[1]->nPads;
  int nGrp = nbrOfProjGroups;
  // Group obtain with the projection
  Group_t cath0ToGrpFromProj[nCath0];
  Group_t cath1ToGrpFromProj[nCath1];
  vectorSetZeroShort( cath0ToGrpFromProj, nCath0);
  vectorSetZeroShort( cath1ToGrpFromProj, nCath1);
  vectorSetZeroShort( cathGroup[0], nCath0);
  vectorSetZeroShort( cathGroup[1], nCath1);
  Group_t projGrpToCathGrp[nGrp+1];
  vectorSetZeroShort( projGrpToCathGrp, nGrp+1);
  int nCathGrp = 0; // return value, ... avoid a sum ... ????
  //
  if (VERBOSE > 0) {
    printf("  assignGroupToCathPads\n");
  }
  //
  PadIdx_t i, j;
  short g, prevGroup0, prevGroup1;
  if (nbrOfCathodePlanes == 1) {
    vectorCopyShort( projPadToGrp, pads[singleCathPlaneID]->nPads, cathGroup[singleCathPlaneID]);
    return nGrp;
  }
  int nProjPads = projectedPads->nPads;
  for( int k=0; k < nProjPads; k++) {
    g = projPadToGrp[k];
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    if (VERBOSE > 1) {
      printf("map k=%d g=%d to i=%d/%d, j=%d/%d\n", k, g, i, nCath0, j, nCath1);
    }
    //
    // Cathode 0
    //
    if ( (i >= 0) && (nCath0 != 0) ) {
      prevGroup0 = cath0ToGrpFromProj[i];
      if ( (prevGroup0 == 0) ) {
        if( (projGrpToCathGrp[g] == 0 ) && (g !=0 )) {
          nCathGrp++;
          projGrpToCathGrp[g] = nCathGrp;
        }
        cath0ToGrpFromProj[i] = projGrpToCathGrp[g];
      } else if ( prevGroup0 != projGrpToCathGrp[g] ) {
         projGrpToCathGrp[g] = prevGroup0;
      }
    }
    //
    // Cathode 1
    //
    if ( (j >= 0) && (nCath1 != 0) ) {
      prevGroup1 = cath1ToGrpFromProj[j];
      if ( (prevGroup1 == 0) ) {
        if(( projGrpToCathGrp[g] == 0 ) && (g !=0 ) ){
          nCathGrp++;
          projGrpToCathGrp[g] = nCathGrp;
        }
        cath1ToGrpFromProj[j] = projGrpToCathGrp[g];
      } else if ( prevGroup1 != projGrpToCathGrp[g] ) {
         projGrpToCathGrp[g] = prevGroup1;
      }
    }
  }
  if (VERBOSE > 2) {
    printf("assignGroupToCathPads\n");
    vectorPrintShort( "  cath0ToGrpFromProj ??? ",cath0ToGrpFromProj,nCath0);
    vectorPrintShort( "  cath1ToGrpFromProj ??? ",cath1ToGrpFromProj,nCath1);
    vectorPrintShort( "  projGrpToCathGrp ??? ", projGrpToCathGrp, nGrp+1);
  }
  // Renumering cathodes groups
  // Inv ??? int nNewGrp = renumberGroups( projGrpToCathGrp, nGrp);
  int nNewGrp = renumberGroups( projGrpToCathGrp, nGrp);

  // vectorPrintShort("  projGrpToCathGrp renumbered", projGrpToCathGrp, nGrp+1);
  //
  vectorMapShort( cath0ToGrpFromProj, projGrpToCathGrp, nCath0);
  vectorCopyShort( cath0ToGrpFromProj, nCath0, cathGroup[0]);
  vectorMapShort( cath1ToGrpFromProj, projGrpToCathGrp, nCath1);
  vectorCopyShort( cath1ToGrpFromProj, nCath1, cathGroup[1]);

  for( i=0; i<nProjPads; i++) {
    projPadToGrp[i] = projGrpToCathGrp[ projPadToGrp[i] ];
  }
  if (VERBOSE > 1) {
    vectorPrintShort("  projPadToGrp", projPadToGrp, nProjPads);
    vectorPrintShort("  cath0ToGrp", cathGroup[0], nCath0);
    vectorPrintShort("  cath1ToGrp", cathGroup[1], nCath1);
  }
  return nNewGrp;
}

//
void Cluster::maskedCopyToXYdXY(const Pads &pads, const Mask_t* mask, int nMask,
                     double* xyDxyMasked, int nxyDxyMasked)
{
  /* ?? Inv
  const double* X = getConstX(xyDxy, nxyDxy);
  const double* Y = getConstY(xyDxy, nxyDxy);
  const double* DX = getConstDX(xyDxy, nxyDxy);
  const double* DY = getConstDY(xyDxy, nxyDxy);
  */
  const double* X = pads.x;
  const double* Y = pads.y;
  const double* DX = pads.dx;
  const double* DY = pads.dy;
  double* Xm = getX(xyDxyMasked, nxyDxyMasked);
  double* Ym = getY(xyDxyMasked, nxyDxyMasked);
  double* DXm = getDX(xyDxyMasked, nxyDxyMasked);
  double* DYm = getDY(xyDxyMasked, nxyDxyMasked);
  vectorGather(X, mask, nMask, Xm);
  vectorGather(Y, mask, nMask, Ym);
  vectorGather(DX, mask, nMask, DXm);
  vectorGather(DY, mask, nMask, DYm);
}

// Unused ???
/*
void Cluster::prepareDataForFitting0( Mask_t* maskFit[2],
        double *xyDxyFit, double *qFit, Mask_t *cathFit, Mask_t *notSaturatedFit, double *zCathTotalCharge, int nFits[2]) {
    int nFit = nFits[0] + nFits[1];
    int nbrCath0 = pads[0]->nPads;
    int nbrCath1 = pads[1]->nPads;
    int nbrCaths[2] = { nbrCath0, nbrCath1};
    int n0 = nFits[0]; int n1 = nFits[1];
    zCathTotalCharge[0] = 0; zCathTotalCharge[1] = 0;
    if (nbrOfCathodePlanes == 2) {
        // Build xyDxyFit  in group g
        //
        // Extract from cath0 the pads which belong to the group g
        maskedCopyToXYdXY( *pads[0], maskFit[0], nbrCath0, xyDxyFit, nFit );
        // Extract from cath1 the pads which belong to the group g
        maskedCopyToXYdXY( *pads[1], maskFit[1], nbrCath1, &xyDxyFit[n0], nFit );
        // Saturated pads
        // vectorPrintShort(" notSaturatedFit 0 ??? ", notSaturatedFit, nFit);
        vectorGatherShort( pads[0]->saturate, maskFit[0], nbrCath0, &notSaturatedFit[0]);
        vectorGatherShort( pads[1]->saturate, maskFit[1], nbrCath1, &notSaturatedFit[n0]);
        // vectorPrintShort(" notSaturatedFit 0 ??? ", notSaturatedFit, nFit);
        vectorNotShort( notSaturatedFit, nFit, notSaturatedFit);
        // vectorPrintShort(" MaskFit0 ??? ", maskFit0, nbrCath0);
        // vectorPrintShort(" MaskFit1 ??? ", maskFit1, nbrCath1);
        // vectorPrintShort(" saturatedFit ??? ", saturated, nFit);
        // vectorPrintShort(" notSaturatedFit ??? ", notSaturatedFit, nFit);
        // Chargei in group g
        vectorGather( pads[0]->q, maskFit[0], nbrCath0, qFit);
        vectorGather( pads[1]->q, maskFit[1], nbrCath1, &qFit[n0]);
        // saturated pads are ignored
        // ??? Don't Set to zero the sat. pads
        // vectorMaskedMult( zFit, notSaturatedFit, nFit, zFit);
        // Total Charge on both cathodes
        zCathTotalCharge[0] = vectorMaskedSum( qFit,      &notSaturatedFit[0], n0);
        zCathTotalCharge[1] = vectorMaskedSum( &qFit[n0], &notSaturatedFit[n0], n1);
        // Merge the 2 Cathodes
        vectorSetShort( cathFit,      0, nFit);
        vectorSetShort( &cathFit[n0], 1, n1);
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
        maskedCopyToXYdXY( *pads[singleCathPlaneID], maskFit[singleCathPlaneID], nbrCaths[singleCathPlaneID], xyDxyFit, nFit );
        vectorGatherShort( pads[singleCathPlaneID]->saturate, maskFit[singleCathPlaneID], nbrCaths[singleCathPlaneID], &notSaturatedFit[0]);
        vectorNotShort( notSaturatedFit, nFit, notSaturatedFit);
        vectorGather( pads[singleCathPlaneID]->q, maskFit[singleCathPlaneID], nbrCaths[singleCathPlaneID], qFit);
        zCathTotalCharge[singleCathPlaneID] = vectorMaskedSum( qFit, notSaturatedFit, nFit);
        vectorSetShort(cathFit, singleCathPlaneID, nFit);

        // Don't take into account saturated pads
        vectorMaskedMult(  qFit, notSaturatedFit, nFit, qFit);
    }
}
*/

int Cluster::filterFitModelOnClusterRegion( Pads &pads, double *theta, int K, Mask_t *maskFilteredTheta) {
  // Spatial filter
  //
  /*
  const double *x    = getConstX   ( xyDxy, N);
  const double *y    = getConstY   ( xyDxy, N);
  const double *dx   = getConstDX   ( xyDxy, N);
  const double *dy   = getConstDY   ( xyDxy, N);
  */
  double *x    = pads.x;
  double *y    = pads.y;
  double *dx    = pads.dx;
  double *dy    = pads.dy;
  int N = pads.nPads;
  // Min Max pads

  double xyTmp[N];
  int kSpacialFilter = 0;
  vectorAddVector( x, -1.0, dx, N,  xyTmp);
  double xMin = vectorMin( xyTmp, N);
  vectorAddVector( x, +1.0, dx, N,  xyTmp);
  double xMax = vectorMax( xyTmp, N);
  vectorAddVector( y, -1.0, dy, N,  xyTmp);
  double yMin = vectorMin( xyTmp, N);
  vectorAddVector( y, +1.0, dy, N,  xyTmp);
  double yMax = vectorMax( xyTmp, N);
  double *muX    = getMuX   ( theta, K);
  double *muY    = getMuY   ( theta, K);
  for( int k=0; k<K; k++) {
    maskFilteredTheta[k] = 0;
    if (( muX[k] > xMin ) && ( muX[k] < xMax )) {
      if (( muY[k] > yMin ) && ( muY[k] < yMax )) {
        maskFilteredTheta[k] = 1;
        kSpacialFilter++;
      }
    }
  }
  if ( (VERBOSE>0) && (kSpacialFilter != K) ) {
    printf("---> Spacial Filter; removing %d hit\n", K - kSpacialFilter);
  }
  // W filter
  // w cut-off
  double cutOff = 0.02 / kSpacialFilter;
  //
  double *w_    = getW   ( theta, K);
  double w[K];
  double wSum=0.0;
  // Normalize new w
  for( int k=0; k<K; k++) {
    wSum += (maskFilteredTheta[k]*w_[k]);
  }
  int kWFilter = 0;
  double norm = 1.0 / wSum;
  for( int k=0; k<K; k++) {
    w[k] = maskFilteredTheta[k] * w_[k] * norm;
    maskFilteredTheta[k] = maskFilteredTheta[k] && (w[k] > cutOff);
    kWFilter += (maskFilteredTheta[k] && (w[k] > cutOff));
  }
  if(  (VERBOSE>0) && (kSpacialFilter > kWFilter) ) {
    printf("---> At least one hit, w[k] < (0.05 / K) = %8.4f); removing %d hit\n", cutOff, kSpacialFilter - kWFilter);
  }
  return kWFilter;
}

int Cluster::filterFitModelOnSpaceVariations( const double *theta0, int K0, double *theta, int K, Mask_t *maskFilteredTheta) {
  // K is the same for theta0 & theta
  // Spatial filter
  //
  int kSpacialFilter = 0;
  const double *mu0X    = getConstMuX   ( theta0, K0);
  const double *mu0Y    = getConstMuY   ( theta0, K0);
  const double *mu0Dx   = getConstVarX  ( theta0, K0);
  const double *mu0Dy   = getConstVarY  ( theta0, K0);
  double *muX    = getMuX   ( theta, K);
  double *muY    = getMuY   ( theta, K);
  double xTmp[K0], yTmp[K0];
  for( int k=0; k<K; k++) {
    maskFilteredTheta[k] = 0;
    // Find the neighbour
    vectorAddScalar( mu0X, -muX[k], K0,  xTmp);
    vectorMultVector( xTmp, xTmp, K,  xTmp);
    vectorAddScalar( mu0Y, -muY[k], K0,  yTmp);
    vectorMultVector( yTmp, yTmp, K,  yTmp);
    vectorAddVector( xTmp, 1.0, xTmp, K, xTmp);
    int kMin = vectorArgMin( xTmp, K0);
    printf("kMin=%d, dx=%f, dy=%f\n", kMin, mu0Dx[k], mu0Dy[k]);
    double xMin = mu0X[kMin] - 4 * mu0Dx[kMin];
    double xMax = mu0X[kMin] + 4 * mu0Dx[kMin];
    double yMin = mu0Y[kMin] - 4 * mu0Dy[kMin];
    double yMax = mu0Y[kMin] + 4 * mu0Dy[kMin];
    if ( (( muX[k] > xMin ) && ( muX[k] < xMax ))
      && (( muY[k] > yMin ) && ( muY[k] < yMax )) ) {
        maskFilteredTheta[k] = 1;
        kSpacialFilter++;
    } else {
      if( VERBOSE>0) {
        printf("---> Spacial Filter; remove mu=(%f,%f)\n", muX[k], muY[k]);
        printf("     xMin=(%f,%f) xMin=(%f,%f)\n", xMin, xMax, yMin, yMax);
      }
    }
  }
  if ( (VERBOSE>0) && (kSpacialFilter != K) ) {
    printf("---> Spacial Filter; removing %d hit\n", K - kSpacialFilter);
  }
  // W filter
  // w cut-off
  double cutOff = 0.02 / kSpacialFilter;
  //
  double *w_    = getW   ( theta, K);
  double w[K];
  double wSum=0.0;
  // Normalize new w
  for( int k=0; k<K; k++) {
    wSum += (maskFilteredTheta[k]*w_[k]);
  }
  int kWFilter = 0;
  double norm = 1.0 / wSum;
  for( int k=0; k<K; k++) {
    w[k] = maskFilteredTheta[k] * w_[k] * norm;
    maskFilteredTheta[k] = maskFilteredTheta[k] && (w[k] > cutOff);
    kWFilter += (maskFilteredTheta[k] && (w[k] > cutOff));
  }
  if(  (VERBOSE>0) && (kSpacialFilter > kWFilter) ) {
    printf("---> At least one hit, w[k] < (0.05 / K) = %8.4f); removing %d hit\n", cutOff, kSpacialFilter - kWFilter);
  }
  return kWFilter;
}

DataBlock_t Cluster::fit(  double *thetaInit, int kInit) {
    // To use to avoid fitting
    // when K > aValue and ratioPadPerSeed > 10 ????
    double ratioPadPerSeed =  projectedPads->nPads / kInit;
    int nbrCath0 = getNbrOfPads(0);
    int nbrCath1 = getNbrOfPads(1);
    /*
    // ??? (111) Invalid fiting
    // Build the mask to handle pads with the g-group
    Mask_t maskFit0[nbrCath0];
    Mask_t maskFit1[nbrCath1];
    Mask_t *maskFit[2] = {maskFit0, maskFit1};
    // printf(" ???? nbrCath0=%d, nbrCath1=%d\n", nbrCath0, nbrCath1);
    // Ne laplacian ??? getMaskCathToGrpFromProj( g, maskFit0, maskFit1, nbrCath0, nbrCath1);
    vectorBuildMaskEqualShort( pads[0]->cath, g, nbrCath0, maskFit0);
    vectorBuildMaskEqualShort( pads[1]->cath, g, nbrCath1, maskFit1);
    // vectorPrintShort("maskFit0", maskFit0, nbrCath0);
    // vectorPrintShort("maskFit1", maskFit1, nbrCath1);
    int nFits[2];
    nFits[1] = vectorSumShort( maskFit1, nbrCath1);
    nFits[0] = vectorSumShort( maskFit0, nbrCath0);
    */
    int nFit = nbrCath0 + nbrCath1;
    // double *xyDxyFit;
    // double *qFit;
    int filteredK = 0;
    int finalK = 0;
    // ThetaFit (output)
    // double thetaFit[K*5];
    double *thetaFit = new double[kInit*5];
    // if ( (nFit < nbrOfPadsLimitForTheFitting) && wellSplitGroup[g] ) {
    if ( nFit < nbrOfPadsLimitForTheFitting ) {
      //
      // Preparing the fitting
      //
      /*
      xyDxyFit = new double[nFit*4];
      qFit = new double[nFit];
      Mask_t cathFit[nFit];
      Mask_t notSaturatedFit[nFit];
      */
      //
      // Invalid ??? prepareDataForFitting( maskFit, xyDxyFit, qFit, cathFit, notSaturatedFit, zCathTotalCharge, nFits);

      // Concatenate the 2 planes of the subCluster For the fitting
      Pads *fitPads = new Pads( pads[0], pads[1]);
      // khi2 (output)
      double khi2[1];
      // pError (output)
      double pError[3*kInit*3*kInit];
      if (VERBOSE > 0) {
        printf( "Starting the fitting\n");
        printf( "- # cath0, cath1 for fitting: %2d %2d\n", getNbrOfPads(0), getNbrOfPads(1));
        printTheta("- thetaInit", thetaInit, kInit);
      }
      // Fit
      if ( (kInit*3 - 1) <= nFit) {
        /*
        fitMathieson( thetaInit, xyDxyFit, qFit, cathFit, notSaturatedFit, zCathTotalCharge, K, nFit,
                         chamberId, processFitVerbose,
                         thetaFit, khi2, pError
                  );
        */
        fitMathieson( *fitPads, thetaInit, kInit,
                         processFitVerbose,
                         thetaFit, khi2, pError );
      } else {
        printf("---> Fitting parameters to large : k=%d, 3k-1=%d, nFit=%d\n", kInit, kInit * 3 -1, nFit);
        printf("     keep the EM solution\n");
        vectorCopy( thetaInit, kInit*5, thetaFit);
      }
      if (VERBOSE) {
        printTheta("- thetaFit", thetaFit, kInit);
      }
      // Filter Fitting solution
      Mask_t maskFilterFit[kInit];
      filteredK = filterFitModelOnClusterRegion( *fitPads, thetaFit, kInit, maskFilterFit);
      // int filteredK = filterFitModelOnSpaceVariations( thetaEMFinal, K, thetaFit, K, maskFilterFit);
      double filteredTheta[5*filteredK];
      if ( (filteredK != kInit) && (nFit >= filteredK ) ) {
        if (VERBOSE) {
          printf("Filtering the fitting K=%d >= K=%d\n", nFit, filteredK);
          printTheta("- filteredTheta", filteredTheta, filteredK);
        }
        if( filteredK > 0) {
          maskedCopyTheta( thetaFit, kInit, maskFilterFit, kInit, filteredTheta, filteredK);
          /*
          fitMathieson( filteredTheta, xyDxyFit, qFit, cathFit, notSaturatedFit,
                      zCathTotalCharge, filteredK, nFit,
                      chamberId, processFit,
                      filteredTheta, khi2, pError
                    );
          */
          fitMathieson( *fitPads, filteredTheta, filteredK,
                         processFitVerbose,
                         filteredTheta, khi2, pError );
          // ?????????????????????????????,              kInit ????
          copyTheta( filteredTheta, filteredK, thetaFit, kInit, filteredK);
          finalK = filteredK;
        } else {
           // No hit with the fitting
           vectorCopy( thetaInit, kInit*5, thetaFit);
           finalK = kInit;
        }
      } else {
        // ??? InvvectorCopy( thetaFit, K*5, thetaFitFinal);
        // Don't Filter, theta resul in "thetaFit"
        finalK = kInit;
      }
    } else {
      // Keep "thetaInit
      vectorCopy( thetaInit, kInit*5, thetaFit);
      finalK = kInit;
    }
    return  std::make_pair(finalK, thetaFit);
}

// ??? To remove - unused
double *Cluster::getProjPadsAsXYdXY( Groups_t group, const Mask_t* maskGrp, int nbrProjPadsInTheGroup) {
    double *xyDxyGProj = new double[nbrProjPadsInTheGroup*4];
    // double *qGProj = new double[nbrProjPadsInTheGroup*4];

    maskedCopyToXYdXY( *projectedPads, maskGrp, projectedPads->nPads, xyDxyGProj, nbrProjPadsInTheGroup );
    //maskedCopy(qProject)
    return xyDxyGProj;
}
// ??????????? Try to do the same with renumberV2
int Cluster::renumberGroups( short *grpToGrp, int nGrp ) {
  // short renumber[nGrp+1];
  // vectorSetShort( renumber, 0, nGrp+1 );
  int maxIdx = vectorMaxShort( grpToGrp, nGrp+1);
  short counters[maxIdx+1];
  vectorSetShort( counters, 0, maxIdx+1 );

  for (int g = 1; g <= nGrp; g++) {
    if ( grpToGrp[g] != 0) {
      counters[ grpToGrp[g] ]++;
    }
  }
  int curGrp = 0;
  for (int g = 1; g <= maxIdx; g++) {
    if (counters[g] != 0) {
      curGrp++;
      counters[g] = curGrp;
    }
  }
  // Now counters contains the mapping oldGrp -> newGrp
  // ??? vectorMapShort( grpToGrp, )
  for (int g = 1; g <= nGrp; g++) {
    grpToGrp[g] = counters[ grpToGrp[g]];
  }
  // vectorCopyShort( renumber, nGrp+1, grpToGrp );
  return curGrp;
}

int Cluster::renumberGroupsV2( Mask_t *cath0Grp, int nbrCath0, Mask_t *cath1Grp, int nbrCath1, Mask_t *grpToGrp, int nGrp ) {
  int currentGrp=0;
  for (int g=0; g < (nGrp+1); g++) {
    grpToGrp[g] = 0;
  }
  Mask_t *bothCathToGrp[2] = { cath0Grp, cath1Grp };
  int nbrBothCath[2] = { nbrCath0, nbrCath1 };
  for (int c=0; c <2; c++) {
    Mask_t *cathToGrp = bothCathToGrp[c];
    int nbrCath = nbrBothCath[c];
    for ( int p=0; p < nbrCath; p++) {
      int g = cathToGrp[p];
      // ??? printf(" p=%d, g[p]=%d, grpToGrp[g]=%d\n", p, g, grpToGrp[g]);
      if ( grpToGrp[g] == 0 ) {
        // It's a new Group
        currentGrp++;
        // Update the map and the cath-group
        grpToGrp[g] = currentGrp;
        cathToGrp[p] = currentGrp;
      } else {
        // The cath-pad takes the group of the map (new numbering)
        cathToGrp[p] = grpToGrp[g];
      }
    }
  }
  int newNbrGroups = currentGrp;
  if ( VERBOSE > 0) {
    printf("[renumberGroups] nbrOfGroups=%d\n", newNbrGroups);
    vectorPrintShort("  cath0ToGrp", cath0Grp, nbrCath0);
    vectorPrintShort("  cath1ToGrp", cath1Grp, nbrCath1);
  }
  return newNbrGroups;
}

int Cluster::findLocalMaxWithPET( double *thetaL, int nbrOfPadsInTheGroupCath) {

    /// ??? Verify if not already done
    // Already done if 1 group
    int verbose = 0;
    Pads *cath0 = pads[0];
    Pads *cath1 = pads[1];
    Pads *projPads = projectedPads;
    int chId = chamberId;
    PadIdx_t *neighCath0 = nullptr, *neighCath1 = nullptr;
    Pads *bPads0 = nullptr, *bPads1 = nullptr;
    if( cath0 ) {
      neighCath0 = cath0->buildFirstNeighbors();
      bPads0 = cath0->addBoundaryPads( neighCath0);
      delete [] neighCath0;
    }
    if( cath1 ) {
      neighCath1 = cath1->buildFirstNeighbors();
      bPads1 = cath1->addBoundaryPads( neighCath1);
      delete [] neighCath1;
    }
    // Pads *bPads0 = addBoundaryPads( cath0, neighborsCath0);
    //Pads displayPads = Pads( *bPads0, Pads::xydxdyMode);
    // bPads0->display("bPads0");
    // Pads *bPads1 = addBoundaryPads( cath1, neighborsCath1);
    int nMaxPads = std::fmax( getNbrOfPads(0), getNbrOfPads(1));
    Pads *pixels = Pads::refinePads( *projPads );
    printf("projPads->nPads=%d, pixel->nPads=%d", projPads->nPads, pixels->nPads);

    Pads *pads;
    if (1) {
      // Merge the pads & describe them on the boundaries
      pads = new Pads( bPads0, bPads1, Pads::xyInfSupMode);
    } else {
      pads = new Pads( *bPads0, Pads::xyInfSupMode);
    }
    if ( bPads0==nullptr ) delete bPads0;
    if ( bPads1==nullptr ) delete bPads1;

    Pads *localMax = nullptr;
    Pads *saveLocalMax = nullptr;
    double chi2=0;
    int dof, nParameters;

    // Pixel initilization
    // ??? Overxrite the the projection
    if (1) {
      for (int i=0; i<pixels->nPads; i++) {
          pixels->q[i] = 1.0;
      }
    }
    int nMacroIterations=8;
    int nIterations[nMacroIterations] = {5, 10, 10, 10, 10, 10, 10, 30 };
    double minPadResidues[nMacroIterations] = { 2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0.5};
    double previousCriteriom = DBL_MAX;
    double criteriom = DBL_MAX;
    bool goon = true;
    int macroIt = 0;
    while ( goon ) {
      if( localMax != nullptr) saveLocalMax = new Pads( *localMax, o2::mch::Pads::xydxdyMode);
      previousCriteriom = criteriom;
      chi2 = PoissonEMLoop( *pads, *pixels,  0, minPadResidues[macroIt], nIterations[macroIt], verbose );
      // PoissonEMLoop( *pads, *pixels, 0, 1.5, 1 );
      localMax = Pads::clipOnLocalMax( *pixels, true);
      nParameters = localMax->nPads;
      dof = nMaxPads - 3*nParameters +1;
      // dof = nMaxPads - 3*nParameters+2;
      if (dof == 0) dof = 1;
      if (1 || VERBOSE > 0) {
        printf("  CHI2 step %d: chi2=%8.2f, nParam=%d, sqrt(chi2)/nPads=%8.2f,  chi2/dof=%8.2f, sqrt(chi2)/dof=%8.2f\n", macroIt, chi2, nParameters, sqrt(chi2) / nMaxPads,  chi2 / dof, sqrt(chi2) / dof);
      }
      inspectSavePixels( macroIt, *pixels);
      macroIt++;
      criteriom = fabs( (chi2 / dof));
      //criteriom = 1.0 / macroIt;
      goon = ( criteriom < previousCriteriom ) && (macroIt<nMacroIterations);
      // goon = ( criteriom < previousCriteriom ) && (macroIt<1);
    }
    delete pixels;
    if (  criteriom <previousCriteriom ) {
      delete saveLocalMax;
    } else {
      delete localMax;
      localMax = saveLocalMax;
    }

    //
    // Select local Max
    // Remove local Max < 0.01 * max(LocalMax)
    //
    double cutRatio = 0.01;
    double qCut = cutRatio * vectorMax ( localMax->q, localMax->nPads);
    int k=0;
    double qSum = 0.0;
    // Remove the last hits if > (nMaxPads +1) / 3
    int nMaxSolutions = int( (std::max( getNbrOfPads(0), getNbrOfPads(1)) + 1.0 ) / 3.0);
    // if (nMaxSolutions < 1) {
    //     nMaxSolutions = 1;
    //}
    // To avoid 0 possibility and give more inputs to the fitting
    nMaxSolutions +=1;
    printf("--> Reduce the nbr max of solutions=%d, nLocMax=%d\n", nMaxSolutions, localMax->nPads );
    if (localMax->nPads > nMaxSolutions ) {
      printf("--> Reduce the nbr of solutions to fit: Take %d/%d solutions\n", nMaxSolutions, localMax->nPads );
      int index[localMax->nPads];
      for( int k=0; k<localMax->nPads; k++) { index[k]=k; }
      std::sort( index, &index[localMax->nPads], [=](int a, int b){ return (localMax->q[a] > localMax->q[b]); });
      // Reoder
      qCut = localMax->q[index[nMaxSolutions-1]] - 1.e-03;
    }
    for (int i=0; i<localMax->nPads; i++) {
      if (localMax->q[i] > qCut) {
        qSum += localMax->q[i];
        localMax->q[k] = localMax->q[i];
        localMax->x[k] = localMax->x[i];
        localMax->y[k] = localMax->y[i];
        localMax->dx[k] = localMax->dx[i];
        localMax->dy[k] = localMax->dy[i];

        k++;
      }
    }
    // Quality
    int removedLocMax = localMax->nPads - k;
    localMax->nPads = k;
    if(0 ) {
    // Quality
    if ( localMax->nPads > 1) {
      Pads copyLocalMax( *localMax, o2::mch::Pads::xydxdyMode );
      printf("Quality test\n");
      PoissonEMLoop( *pads, copyLocalMax, 0, 0.5, 60, 1 );
      Pads *testLocalMax = new Pads( copyLocalMax, Pads::xydxdyMode);
      int qMinIdx = vectorArgMin( copyLocalMax.q, copyLocalMax.nPads );
      testLocalMax->removePad( qMinIdx );
      PoissonEMLoop( *pads, *testLocalMax, 0, 0.5, 60, 1 );
      delete testLocalMax;
     }
    }
    // Remove the last hit

    // Weight normalization
    for (int i=0; i<localMax->nPads; i++) {
      // printf( "??? q[i]=%f, qSum=%f\n", localMax->q[i], qSum);
      localMax->q[i] = localMax->q[i] / qSum;
    }

    if (1 || VERBOSE > 0) {
      printf( "---> Final cut %d percent (qcut=%8.2f), number of local max removed = %d\n", int(cutRatio*100), qCut, removedLocMax);
    }
    // ??? chisq = computeChiSq( xyInfSup, q, chId, refinedTheta )

    // Store the
    int K0 = localMax->nPads;
    int K = std::min( K0, nbrOfPadsInTheGroupCath);
    double *w = getW( thetaL, nbrOfPadsInTheGroupCath);
    double *muX = getMuX( thetaL, nbrOfPadsInTheGroupCath);
    double *muY = getMuY( thetaL, nbrOfPadsInTheGroupCath);
    double *varX = getVarX( thetaL, nbrOfPadsInTheGroupCath);
    double *varY = getVarY( thetaL, nbrOfPadsInTheGroupCath);
    for (int k=0; k<K; k++) {
      w[k] = localMax->q[k];
      muX[k] = localMax->x[k];
      muY[k] = localMax->y[k];
      varX[k] = localMax->dx[k];
      varY[k] = localMax->dy[k];
    }
    delete localMax;
    return K;
}

void Cluster::updateProjectionGroups () {
    if (VERBOSE > 0) {
      printf("[updateProjectionGroups]\n");
    }
    /// Inv ??? Groups_t *projPadToGrp = projPadToGrp;;
    int nProjPads = projectedPads->nPads;
    Groups_t *cath0ToGrp = cathGroup[0];
    Groups_t *cath1ToGrp = cathGroup[1];

    // Save projPadToGrp to CHECK
    Group_t savePadGrp[nProjPads];
    if (CHECK) {
      vectorCopyShort(projPadToGrp, nProjPads, savePadGrp );
    }
    for (int k=0; k < nProjPads; k++) {
         MapKToIJ_t ij = mapKToIJ[k];
         PadIdx_t i = ij.i;
         PadIdx_t j = ij.j;
         if ((i > -1) && (j== -1)) {
           // int cath0Idx = mapPadToCathIdx[ i ];
           projPadToGrp[k] = cath0ToGrp[i];
           // printf("  projPadToGrp[k] = cath0ToGrp[cath0Idx], i=%d, j=%d, cath0Idx=%d, cath0ToGrp[cath0Idx]=%d\n", i, j, cath0Idx, cath0ToGrp[cath0Idx]);
         } else if ((i == -1) && (j > -1)) {
           // int cath1Idx = mapPadToCathIdx[ j ];
           projPadToGrp[k] = cath1ToGrp[j];
           // printf("  projPadToGrp[k] = cath1ToGrp[cath1Idx], i=%d, j=%d, cath1Idx=%d, cath1ToGrp[cath1Idx]=%d\n", i, j, cath1Idx, cath1ToGrp[cath1Idx]);
         } else if ((i > -1) && (j > -1)) {
           // projPadToGrp[k] = grpToGrp[ projPadToGrp[k] ];
           projPadToGrp[k] = cath0ToGrp[i];
           if ( CHECK && (cath0ToGrp[i] != cath1ToGrp[j]) ) {
             printf("  [updateProjectionGroups] i, cath0ToGrp[i]=(%d, %d); j, cath1ToGrp[j]=(%d, %d)\n", i, cath0ToGrp[i], j, cath1ToGrp[j]);
             // throw std::overflow_error("updateProjectionGroups cath0ToGrp[i] != cath1ToGrp[j]");
           }
           // printf("  projPadToGrp[k] = grpToGrp[ projPadToGrp[k] ], i=%d, j=%d, k=%d \n", i, j, k);
         } else {
           throw std::overflow_error("updateProjectionGroups i,j=-1");
         }
    }
    if (VERBOSE > 0) {
      vectorPrintShort("  updated projGrp", projPadToGrp, nProjPads);
    }
    if (0 && CHECK) {
        bool same=true;
        for( int p=0; p < nProjPads; p++) {
            same = same && (projPadToGrp[p] == savePadGrp[p] );
        }
        if (same == false) {
            vectorPrintShort("  WARNING: old projPadToGrp", savePadGrp, nProjPads);
            vectorPrintShort("  WARNING: new projPadToGrp", projPadToGrp, nProjPads);
            // throw std::overflow_error("updateProjectionGroups projection has changed");
        }
    }
}

// Not used in the Clustering/fitting
// Just to check hit results
int Cluster::laplacian2D( const Pads &pads_, PadIdx_t *neigh, int chId, PadIdx_t *sortedLocalMax, int kMax, double *smoothQ) {
  // ??? Place somewhere
  double eps = 1.0e-7;
  double noise = 4. * 0.22875;
  double laplacianCutOff = noise;
  // ??? Inv int atLeastOneMax = -1;
  //
  int N = pads_.nPads;
  const double *x  = pads_.x;
  const double *y  = pads_.y;
  const double *dx = pads_.dx;
  const double *dy = pads_.dy;
  const double *q = pads_.q;
  //
  // Laplacian allocation
  double lapl[N];
  // Locations not used as local max
  Mask_t unselected[N];
  vectorSetShort( unselected, 1, N );
  // printNeighbors(neigh, N);
  for (int i=0; i< N; i++) {
    int nNeigh = 0;
    double sumNeigh = 0;
    int nNeighSmaller = 0;
    // printf("  Neighbors of i=%d [", i);
    //
    // For all neighbours of i
    for( PadIdx_t *neigh_ptr = getNeighborsOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t j = *neigh_ptr;
      // printf("%d ,", j);
      // nNeighSmaller += (q[j] <= ((q[i] + noise) * unselected[i]));
      nNeighSmaller += (q[j] <= ((q[i] + noise) * unselected[j]));
      nNeigh++;
      sumNeigh += q[j];
    }
    // printf("]");
    // printf(" nNeighSmaller %d / nNeigh %d \n", nNeighSmaller, nNeigh);
    lapl[i] = float(nNeighSmaller) / nNeigh;
    if (lapl[i] < laplacianCutOff) {
      lapl[i] = 0.0;
    }
    unselected[i] = (lapl[i] != 1.0);
    smoothQ[i] =  sumNeigh / nNeigh;
    if (1 || VERBOSE) {
      printf("Laplacian i=%d, x[i]=%6.3f, y[i]=%6.3f, z[i]=%6.3f, smoothQ[i]=%6.3f, lapl[i]=%6.3f\n", i, x[i], y[i], q[i], smoothQ[i], lapl[i]);
    }
  }
  //
  // Get local maxima
  Mask_t localMaxMask[N];
  vectorBuildMaskEqual( lapl, 1.0, N, localMaxMask);
  // Get the location in lapl[]
  // Inv ??? int nSortedIdx = vectorSumShort( localMaxMask, N );
  // ??? Inv int sortPadIdx[nSortedIdx];
  int nSortedIdx = vectorGetIndexFromMask( localMaxMask, N, sortedLocalMax );
  // Sort the slected laplacian (index sorting)
  // Indexes for sorting
  // Rq: Sometimes chage the order of max
  // std::sort( sortedLocalMax, &sortedLocalMax[nSortedIdx], [=](int a, int b){ return smoothQ[a] > smoothQ[b]; });
  std::sort( sortedLocalMax, &sortedLocalMax[nSortedIdx], [=](int a, int b){ return q[a] > q[b]; });
  if (1 || VERBOSE) {
    vectorPrint("  sort w", q, N);
    vectorPrintInt("  sorted q-indexes", sortedLocalMax, nSortedIdx);
  }

  ////
  // Filtering local max
  ////

  printf("FILTERing Max\n");
  // At Least one locMax
  if ((nSortedIdx == 0) && (N!=0)) {
    // Take the first pad
    printf("-> No local Max, take the highest value < 1\n");
    sortedLocalMax[0] = 0;
    nSortedIdx = 1;
    return nSortedIdx;
  }

  // For a small number of pads
  // limit the number of max to 1 local max
  // if the aspect ratio of the cluster
  // is close to 1
  double aspectRatio=0;
  if ( (N > 0) && (N < 6) && (chId <= 6) ) {
    double xInf=DBL_MAX, xSup=DBL_MIN, yInf=DBL_MAX, ySup=DBL_MIN;
    // Compute aspect ratio of the cluster
    for (int i=0; i <N; i++) {
      xInf = fmin( xInf, x[i] - dx[i]);
      xSup = fmax( xSup, x[i] + dx[i]);
      yInf = fmin( yInf, y[i] - dy[i]);
      ySup = fmax( ySup, y[i] + dy[i]);
    }
    // Nbr of pads in x-direction
    int nX = int((xSup - xInf) / dx[0] + eps);
    // Nbr of pads in y-direction
    int nY = int((ySup - yInf) / dy[0] + eps);
    aspectRatio = fmin( nX, nY) / fmax( nX, nY);
    if ( aspectRatio > 0.6 ) {
      // Take the max
      nSortedIdx = 1;
      printf("-> Limit to one local Max, nPads=%d, chId=%d, aspect ratio=%6.3f\n", N, chId, aspectRatio);
    }
  }

  // Suppress noisy peaks  when at least 1 peak
  // is bigger than
  if ( (N > 0) && (q[ sortedLocalMax[0]] > 2*noise) ) {
    int trunkIdx=nSortedIdx;
    for ( int ik=0; ik < nSortedIdx; ik++) {
      if ( q[ sortedLocalMax[ik]] <= 2*noise) {
        trunkIdx = ik;
      }
    }
    nSortedIdx = std::max(trunkIdx, 1);
    if (trunkIdx != nSortedIdx) {
      printf("-> Suppress %d local Max. too noisy (q < %6.3f),\n", nSortedIdx-trunkIdx, 2*noise);
    }

  }
  // At most
  // int nbrOfLocalMax = floor( (N + 1) / 3.0 );
  // if  ( nSortedIdx > nbrOfLocalMax) {
  //  printf("Suppress %d local Max. the limit of number of local max %d is reached (< %d)\n", nSortedIdx-nbrOfLocalMax, nSortedIdx, nbrOfLocalMax);
  //  nSortedIdx = nbrOfLocalMax;
  //}

  return nSortedIdx;
}

// Not used in the Clustering/fitting
// Just to check hit results
int Cluster::findLocalMaxWithBothCathodes( double *thetaOut, int kMax, int verbose ) {

  int N0 = pads[0]->nPads;
  int N1 = pads[1]->nPads;
  int kMax0 = N0;
  int kMax1 = N1;

  // Number of seeds founds
  int k=0;
  //
  // Pad indexes of local max. allocation per cathode
  PadIdx_t localMax0[kMax0];
  PadIdx_t localMax1[kMax1];
  // Smoothed values of q[0/1] with neighbours
  double smoothQ0[N0];
  double smoothQ1[N1];
  // Local Maximum for each cathodes
  // There are sorted with the lissed q[O/1] values
  if (verbose > 1) {
    printf("findLocalMaxWithBothCathodes N0=%d N1=%d\n", N0, N1);
  }
  PadIdx_t *grpNeighborsCath0 = nullptr;
  PadIdx_t *grpNeighborsCath1 = nullptr;
  if ( N0 ) {
    grpNeighborsCath0 = pads[0]->buildFirstNeighbors( );
  }
  if ( N1 ) {
    grpNeighborsCath1 = pads[1]->buildFirstNeighbors( );
  }
  int K0 = laplacian2D( *pads[0], grpNeighborsCath0, chamberId, localMax0, kMax0, smoothQ0);
  int K1 = laplacian2D( *pads[1], grpNeighborsCath1, chamberId, localMax1, kMax1, smoothQ1);
  // Seed allocation
  double localXMax[K0+K1];
  double localYMax[K0+K1];
  double localQMax[K0+K1];
  //
  // Need an array to transform global index to the grp indexes
  PadIdx_t mapIToGrpIdx[N0];
  vectorSetInt(mapIToGrpIdx, -1, N0);
  PadIdx_t mapGrpIdxToI[N0];
  for (int i=0; i<N0; i++) {
    // ??? printf("mapGrpIdxToI[%d]=%d\n", i, mapGrpIdxToI[i]);
    // VPads mapIToGrpIdx[ mapGrpIdxToI[i]] = i;
    mapIToGrpIdx[i] = i;
    mapGrpIdxToI[i] = i;
  }
  PadIdx_t mapJToGrpIdx[N1];
  vectorSetInt(mapJToGrpIdx, -1, N1);
  PadIdx_t mapGrpIdxToJ[N0];
  for (int j=0; j<N1; j++) {
    // ??? printf("mapGrpIdxToJJ[%d]=%d\n", j, mapGrpIdxToJ[j]);
    // Vpads mapJToGrpIdx[ mapGrpIdxToJ[j]] = j;
    mapJToGrpIdx[ j] = j;
    mapGrpIdxToJ[ j] = j;
  }


  const double *x0  = pads[0]->x;
  const double *y0  = pads[0]->y;
  const double *dx0 = pads[0]->dx;
  const double *dy0 = pads[0]->dy;
  const double *q0  = pads[0]->q;

  const double *x1  = pads[1]->x;
  const double *y1  = pads[1]->y;
  const double *dx1 = pads[1]->dx;
  const double *dy1 = pads[1]->dy;
  const double *q1  = pads[1]->q;

  const double *xProj  = projectedPads->x;
  const double *yProj  = projectedPads->y;
  const double *dxProj = projectedPads->dx;
  const double *dyProj = projectedPads->dy;

    // ???
  // vectorPrintInt( "mapIToGrpIdx", mapIToGrpIdx, N0);
  // vectorPrintInt( "mapJToGrpIdx", mapJToGrpIdx, N1);
  if (verbose > 1) {
    vectorPrint("findLocalMax q0", q0, N0);
    vectorPrint("findLocalMax q1", q1, N1);
    vectorPrintInt("findLocalMax localMax0", localMax0, K0);
    vectorPrintInt("findLocalMax localMax1", localMax1, K1);
  }

  //
  // Make the combinatorics between the 2 cathodes
  // - Take the maxOf( N0,N1) for the external loop
  //
  if (verbose >1) {
    printf("  Local max per cathode K0=%d, K1=%d\n", K0, K1);
  }
  bool K0GreaterThanK1 = ( K0 >= K1 );
  bool K0EqualToK1 = ( K0 == K1 );
  // Choose the highest last local max.
  bool highestLastLocalMax0;
  if (K0 == 0) {
    highestLastLocalMax0 = false;
  } else if (K1 == 0){
    highestLastLocalMax0 = true;
  } else {
    // highestLastLocalMax0 = (smoothQ0[localMax0[std::max(K0-1, 0)]] >= smoothQ1[localMax1[std::max(K1-1,0)]]);
    highestLastLocalMax0 = (q0[localMax0[std::max(K0-1, 0)]] >= q1[localMax1[std::max(K1-1,0)]]);
  }
  // Permute cathodes if necessary
  int NU, NV;
  int KU, KV;
  PadIdx_t *localMaxU, *localMaxV;
  const double *qU, *qV;
  PadIdx_t *interUV;
  bool permuteIJ;
  const double *xu, *yu, *dxu, *dyu;
  const double *xv, *yv, *dxv, *dyv;
  const PadIdx_t *mapGrpIdxToU, *mapGrpIdxToV;
  PadIdx_t  *mapUToGrpIdx, *mapVToGrpIdx;

  // Do permutation between cath0/cath1 or not
  if( K0GreaterThanK1 || ( K0EqualToK1 && highestLastLocalMax0)) {
    NU = N0; NV=N1;
    KU = K0; KV = K1;
    xu=x0;  yu=y0; dxu=dx0;  dyu=dy0;
    xv=x1;  yv=y1; dxv=dx1;  dyv=dy1;
    localMaxU = localMax0; localMaxV = localMax1;
    // qU = smoothQ0; qV = smoothQ1;
    qU = q0; qV = q1;
    interUV = IInterJ;
    mapGrpIdxToU = mapGrpIdxToI;
    mapGrpIdxToV = mapGrpIdxToJ;
    mapUToGrpIdx = mapIToGrpIdx;
    mapVToGrpIdx = mapJToGrpIdx;
    permuteIJ = false;
  }
  else {
    NU = N1; NV = N0;
    KU = K1; KV = K0;
    xu=x1;  yu=y1; dxu=dx1;  dyu=dy1;
    xv=x0;  yv=y0; dxv=dx0;  dyv=dy0;
    localMaxU = localMax1; localMaxV = localMax0;
    // qU = smoothQ1; qV = smoothQ0;
    qU = q1; qV = q0;
    interUV = JInterI;
    mapGrpIdxToU = mapGrpIdxToJ;
    mapGrpIdxToV = mapGrpIdxToI;
    mapUToGrpIdx = mapJToGrpIdx;
    mapVToGrpIdx = mapIToGrpIdx;
    permuteIJ = true;
  }
  // Keep the memory of the localMaxV already assigned
  Mask_t qvAvailable[KV];
  vectorSetShort( qvAvailable, 1, KV);
  // Compact intersection matrix
  PadIdx_t *UInterV;
  //
  // Cathodes combinatorics
  if (verbose > 1) {
    printf("  Local max combinatorics: KU=%d KV=%d\n", KU, KV);
    // printXYdXY("Projection", xyDxyProj, NProj, NProj, 0, 0);
    // printf("  mapIJToK=%p, N0=%d N1=%d\n", mapIJToK, N0, N1);
    for (int i=0; i <N0; i++) {
      // VPads int ii = mapGrpIdxToI[i];
      int ii = i;
      for (int j=0; j <N1; j++) {
        // VPads int jj = mapGrpIdxToJ[j];
        int jj = j;
        //if ( (mapIJToK[ii*nbrCath1+jj] != -1))
              printf("   %d inter %d, grp : %d inter %d yes=%d\n", ii, jj, i, j, mapIJToK[ii*N1+jj]);
      }
    }
  }
  for(int u=0; u<KU; u++) {
    //
    PadIdx_t uPadIdx = localMaxU[u];
    double maxValue = 0.0;
    // Cathode-V pad index of the max (localMaxV)
    PadIdx_t maxPadVIdx = -1;
    // Index in the maxCathv
    PadIdx_t maxCathVIdx = -1;
    // Choose the best localMaxV
    // i.e. the maximum value among
    // the unselected localMaxV
    //
    // uPadIdx in index in the Grp
    // need to get the cluster index
    // to checck the intersection
    // VPads int ug = mapGrpIdxToU[uPadIdx];
    int ug = uPadIdx;
    if(verbose) {
      printf("  Cathode u=%d localMaxU[u]=%d, x,y= %6.3f,  %6.3f, q=%6.3f\n", u, localMaxU[u], xu[localMaxU[u]], yu[localMaxU[u]], qU[localMaxU[u]]);
    }
    bool interuv;
    for(int v=0; v<KV; v++) {
      PadIdx_t vPadIdx = localMaxV[v];
      // VPads int vg = mapGrpIdxToV[vPadIdx];
      int vg = vPadIdx;
      if (permuteIJ) {
         // printf("uPadIdx=%d,vPadIdx=%d, mapIJToK[vPadIdx*N0+uPadIdx]=%d permute\n",uPadIdx,vPadIdx, mapIJToK[vPadIdx*N0+uPadIdx]);
         interuv = (mapIJToK[vg*N1+ug] != -1);
      }
      else {
         //printf("uPadIdx=%d,vPadIdx=%d, mapIJToK[uPadIdx*N1+vPadIdx]=%d\n",uPadIdx,vPadIdx, mapIJToK[uPadIdx*N1+vPadIdx]);
         interuv = (mapIJToK[ug*N1+vg] != -1);
      }
      if (interuv) {
        double val = qV[vPadIdx] * qvAvailable[v];
        if (val > maxValue) {
          maxValue = val;
          maxCathVIdx = v;
          maxPadVIdx = vPadIdx;
        }
      }
    }
    // A this step, we've got (or not) an
    // intercepting pad v with u. This v is
    // the maximum of all possible values
    // ??? printf("??? maxPadVIdx=%d, maxVal=%f\n", maxPadVIdx, maxValue);
    if (maxPadVIdx != -1) {
      // Found an intersevtion and a candidate
      // add in the list of seeds
      PadIdx_t kProj;
      int vg = mapGrpIdxToV[maxPadVIdx];
      if( permuteIJ) {
        kProj = mapIJToK[vg*N1 + ug];
      }
      else {
        kProj = mapIJToK[ug*N1 + vg];
      }
      // mapIJToK and projection UNUSED ????
      localXMax[k] = xProj[kProj];
      localYMax[k] = yProj[kProj];
      // localQMax[k] = 0.5 * (qU[uPadIdx] + qV[maxPadVIdx]);
      localQMax[k] = qU[uPadIdx];
      // Cannot be selected again as a seed
      qvAvailable[ maxCathVIdx ] = 0;
      if (verbose > 1) {
        printf("    found intersection of u with v: u,v=(%d,%d) , x=%f, y=%f, w=%f\n", u, maxCathVIdx, localXMax[k],localYMax[k], localQMax[k]);
        // printf("Projection u=%d, v=%d, uPadIdx=%d, ,maxPadVIdx=%d, kProj=%d, xProj[kProj]=%f, yProj[kProj]=%f\n", u, maxCathVIdx,
        //        uPadIdx, maxPadVIdx, kProj, xProj[kProj], yProj[kProj] );
        //kProj = mapIJToK[maxPadVIdx*N0 + uPadIdx];
        //printf(" permut kProj=%d xProj[kProj], yProj[kProj] = %f %f\n", kProj, xProj[kProj], yProj[kProj] );
      }
      k++;
    }
    else {
      // No intersection u with localMaxV set
      // Approximate the seed position
      //
      // Search v pads intersepting u
      PadIdx_t *uInterV;
      PadIdx_t uPad = 0;
      if ( verbose > 1) {
        printf("  No intersection between u=%d and v-set of , approximate the location\n", u);
      }
      // Go to the mapGrpIdxToU[uPadIdx] (???? mapUToGrpIdx[uPadIdx])
      uInterV=interUV;
      if (NV != 0 ) {
        for( uInterV=interUV; uPad < ug; uInterV++) {
          if (*uInterV == -1) {
            uPad++;
          }
        }
      }
      // if (uInterV) printf("??? uPad=%d, uPadIdx=%d *uInterV=%d\n", uPad, uPadIdx, *uInterV);
      // If intercepting pads or no V-Pad
      if ((NV != 0) && (uInterV[0] != -1) ) {
        double vMin = 1.e+06;
        double vMax = -1.e+06;
        // Take the most precise direction
        if (dxu[u] < dyu[u]) {
          // x direction most precise
          // Find the y range intercepting pad u
          for( ; *uInterV != -1; uInterV++) {
            PadIdx_t idx = mapVToGrpIdx[*uInterV];
            if( verbose > 1) {
              printf("  Global upad=%d intersect global vpad=%d grpIdx=%d\n", uPad, *uInterV, idx);
            }
            if (idx != -1) {
              vMin = fmin( vMin, yv[idx] - dyv[idx]);
              vMax = fmax( vMax, yv[idx] + dyv[idx]);
            }
          }
          localXMax[k] = xu[uPadIdx];
          localYMax[k] = 0.5*(vMin+vMax);
          localQMax[k] = qU[uPadIdx];
          if( localYMax[k] ==0 ) printf("WARNING localYMax[k] == 0, meaning no intersection");
        }
        else {
          // y direction most precise
          // Find the x range intercepting pad u
          for( ; *uInterV != -1; uInterV++) {
            PadIdx_t idx = mapVToGrpIdx[*uInterV];
            if( verbose > 1) {
              printf(" Global upad=%d intersect global vpad=%d  grpIdx=%d \n", uPad, *uInterV, idx);
            }
            if (idx != -1) {
              if( verbose > 1) {
                printf("xv[idx], yv[idx], dxv[idx], dyv[idx]: %6.3f %6.3f %6.3f %6.3f\n", xv[idx], yv[idx], dxv[idx], dyv[idx]);
              }
              vMin = fmin( vMin, xv[idx] - dxv[idx]);
              vMax = fmax( vMax, xv[idx] + dxv[idx]);
            }
          }
          localXMax[k] = 0.5*(vMin+vMax);
          localYMax[k] = yu[uPadIdx];
          localQMax[k] = qU[uPadIdx];
          // printf(" uPadIdx = %d/%d\n", uPadIdx, KU);
          if( localXMax[k] ==0 ) printf("WARNING localXMax[k] == 0, meaning no intersection");
        }
        if (verbose) {
          printf("  solution found with all intersection of u=%d with all v, x more precise %d, position=(%f,%f), qU=%f\n",
                    u, (dxu[u] < dyu[u]), localXMax[k], localYMax[k], localQMax[k]);
        }
        k++;
      }
      else {
        // No interception in the v-list
        // or no V pads
        // Takes the u values
        // printf("No intersection of the v-set with u=%d, take the u location", u);

        localXMax[k] = xu[uPadIdx];
        localYMax[k] = yu[uPadIdx];
        localQMax[k] = qU[uPadIdx];
        if (verbose) {
          printf("  No intersection with u, u added in local Max: k=%d u=%d, position=(%f,%f), qU=%f\n",
                 k, u, localXMax[k], localYMax[k], localQMax[k]);
        }
        k++;
      }
    }
  }
  // Proccess unselected localMaxV
  for(int v=0; v<KV; v++) {
    if (qvAvailable[v]) {
      int l = localMaxV[v];
      localXMax[k] = xv[l];
      localYMax[k] = yv[l];
      localQMax[k] = qV[l];
      if (verbose>1) {
        printf("  Remaining VMax, v added in local Max:  v=%d, position=(%f,%f), qU=%f\n",
                 v, localXMax[k], localYMax[k], localQMax[k]);
      }
      k++;
    }
  }
  // k seeds
  double *varX  = getVarX(thetaOut, kMax);
  double *varY  = getVarY(thetaOut, kMax);
  double *muX   = getMuX(thetaOut, kMax);
  double *muY   = getMuY(thetaOut, kMax);
  double *w     = getW(thetaOut, kMax);
  //
  double wRatio = 0;
  for ( int k_=0; k_ < k; k_++) {
    wRatio += localQMax[k_];
  }
  wRatio = 1.0/wRatio;
  if (verbose > 1) {
    printf("Local max found k=%d kmax=%d\n", k, kMax);
  }
  for ( int k_=0; k_ < k; k_++) {
   muX[k_] = localXMax[k_];
   muY[k_] = localYMax[k_];
   w[k_] = localQMax[k_] * wRatio;
   if (verbose > 1) {
     printf("  w=%6.3f, mux=%7.3f, muy=%7.3f\n", w[k_], muX[k_], muY[k_]);
   }
  }
  if ( N0 ) delete [] grpNeighborsCath0;
  if ( N1 ) delete [] grpNeighborsCath1;
  return k;
}

} // namespace mch
} // namespace o2