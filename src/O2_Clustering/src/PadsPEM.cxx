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

/// \file PadPEM.cxx
/// \brief Pads representation and transformation
///
/// \author Gilles Grasseau, Subatech

#include <stdexcept>
#include <cstring>
#include <vector>

#include "MCHClustering/PadsPEM.h"
#include "mathUtil.h"

#define VERBOSE 1
#define CHECK 1

namespace o2
{
namespace mch
{

PadIdx_t *Pads::buildFirstNeighbors( double *X, double *Y, double *DX, double *DY, int N, int verbose) {
  const double eps = 1.0e-5;
  PadIdx_t *neighbors = new PadIdx_t[MaxNeighbors*N];
  for( PadIdx_t i=0; i<N; i++) {
    PadIdx_t *i_neigh = getNeighborListOf(neighbors, i);
    // Search neighbors of i
    for( PadIdx_t j=0; j<N; j++) {

      int xMask0 = ( std::fabs( X[i] - X[j]) < (DX[i] + DX[j]) + eps);
      int yMask0 = ( std::fabs( Y[i] - Y[j]) < (DY[i] + eps) );
      int xMask1 = ( std::fabs( X[i] - X[j]) < (DX[i] + eps) );
      int yMask1 = ( std::fabs( Y[i] - Y[j]) < (DY[i] + DY[j] + eps) );
      if ( (xMask0 && yMask0) || (xMask1 && yMask1) ) {
        *i_neigh = j;
        i_neigh++;
        // Check
        // printf( "pad %d neighbor %d xMask=%d yMask=%d\n", i, j, (xMask0 && yMask0), (xMask1 && yMask1));
      }
    }
    *i_neigh = -1;
    if (  CHECK && (std::fabs( i_neigh - getNeighborListOf(neighbors, i) ) > MaxNeighbors) ) {
      printf("Pad %d : nbr of neighbours %ld greater than the limit %d \n",
              i, i_neigh - getNeighborListOf(neighbors, i), MaxNeighbors );
      throw std::overflow_error("Not enough allocation");
    }
  }
  return neighbors;
}
// Build the K-neighbor list
PadIdx_t *Pads::buildKFirstsNeighbors( const Pads &pads, int kernelSize ) {
  // kernelSize must be in the interval [0:2]
  // ??? "eps" to set away
  const double eps = 1.0e-5;
  const double *X = pads.x;
  const double *Y = pads.y;
  const double *DX = pads.dx;
  const double *DY = pads.dy;
  int N=pads.nPads;
  if ( (kernelSize < 0) || (kernelSize > 2)  ) {
    // set to default values
    printf( "Warning in getNeighbors : kerneSize overwritten by the default\n");
    kernelSize = 1;
  }
  PadIdx_t *neighbors_ = new PadIdx_t[MaxNeighbors*N];
  for( PadIdx_t i=0; i<N; i++) {
    PadIdx_t *i_neigh = getNeighborListOf(neighbors_, i);
    // Search neighbors of i
    for( PadIdx_t j=0; j<N; j++) {
      int xMask0 = ( fabs( X[i] - X[j]) < ( (2*kernelSize-1)*DX[i] + DX[j] + eps));
      int yMask0 = ( fabs( Y[i] - Y[j]) < ( (2*kernelSize-1)*DY[i] + DY[j] + eps) );
      if ( (xMask0 && yMask0) ) {
        *i_neigh = j;
        i_neigh++;
        // Check
        // printf( "pad %d neighbor %d xMask=%d yMask=%d\n", i, j, (xMask0 && yMask0), (xMask1 && yMask1));
      }
    }
    // Set the End of list
    *i_neigh = -1;
    //
    if (  CHECK && (fabs( i_neigh - getNeighborListOf(neighbors_, i) ) > MaxNeighbors) ) {
      printf("Pad %d : nbr of neighbours %ld greater than the limit %d \n",
              i, i_neigh - getNeighborListOf(neighbors_, i), MaxNeighbors );
      throw std::overflow_error("Not enough allocation");
    }
  }
  if (VERBOSE > 1) Pads::printNeighbors( neighbors_, N);
  return neighbors_;
}

// ??? To integrate in Pads or else where
Pads *Pads::addBoundaryPads( const double *x_, const double *y_, const double *dx_, const double *dy_, const double *q_, const Mask_t *cath_, const Mask_t *sat_, int chamberId, int N) {

  // TODO: Remove duplicate pads
  double eps = 1.0e-4;
  //
  std::vector<double> bX;
  std::vector<double> bY;
  std::vector<double> bdX;
  std::vector<double> bdY;
  std::vector<int> bCath;
  int n1 = vectorSumShort(cath_, N);
  int nPads_[2] = {N-n1, n1};

  for (int c=0; c < 2; c++) {
    int nc = nPads_[c];
    Mask_t mask[N];
    double x[nc], y[nc], dx[nc], dy[nc], q[nc], sat[nc];
    vectorBuildMaskEqualShort( cath_, c, N, mask);
    vectorGather(x_, mask, N, x);
    vectorGather(y_, mask, N, y);
    vectorGather(dx_, mask, N, dx);
    vectorGather(dy_, mask, N, dy);

    PadIdx_t *neighC = Pads::buildFirstNeighbors( x, y, dx, dy, nc, VERBOSE );

    for (int i=0; i < nc; i++) {
      bool east = true, west = true, north = true, south = true;
      for( const PadIdx_t *neigh_ptr = getNeighborListOf(neighC, i); *neigh_ptr != -1; neigh_ptr++) {
        PadIdx_t v = *neigh_ptr;
        double xDelta = (x[v] - x[i]);
        if (fabs(xDelta) > eps) {
          if (xDelta > 0) {
            east = false;
          } else {
            west = false;
          }
        }
        double yDelta = (y[v] - y[i]);
        if (fabs(yDelta) > eps) {
          if (yDelta > 0) {
            north = false;
          } else {
            south = false;
          }
        }
      }
      // Add new pads
      if ( east ) {
        bX.push_back( x[i]+2*dx[i] );
        bY.push_back( y[i]);
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
        bCath.push_back( c );
      }
      if ( west ) {
        bX.push_back( x[i]-2*dx[i] );
        bY.push_back( y[i]);
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
        bCath.push_back( c );
      }
      if (north) {
        bX.push_back( x[i] );
        bY.push_back( y[i]+2*dy[i] );
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
        bCath.push_back( c );
      }
      if (south) {
        bX.push_back( x[i] );
        bY.push_back( y[i]-2*dy[i] );
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
        bCath.push_back( c );
      }
    }
    delete neighC;
  }

  int nPadToAdd = bX.size();
  // ??? int nTotalPads = nPadToAdd + N;
  int nTotalPads =  N+nPadToAdd;
  if (VERBOSE > 2) {
      printf("nTotalPads=%d, nPads=%d,  nPadToAdd=%d\n", nTotalPads, N, nPadToAdd);
  }
  o2::mch::Pads *padsWithBoundaries = new o2::mch::Pads( nTotalPads, chamberId );
  o2::mch::Pads *newPads = padsWithBoundaries;
  for (int i=0; i < N; i++) {
    newPads->x[ i ] = x_[i];
    newPads->y[ i ] = y_[i];
    newPads->dx[ i ] = dx_[i];
    newPads->dy[ i ] = dy_[i];
    newPads->q[ i ] = q_[i];
    newPads->cath[ i ] = cath_[i];
    newPads->saturate[ i ] = sat_[i];
  }
  for (int i=N, k=0; i < nTotalPads; i++,k++) {
      newPads->x[ i ] = bX[k];
      newPads->y[ i ] = bY[k];
      newPads->dx[ i ] = bdX[k];
      newPads->dy[ i ] = bdY[k];
      newPads->cath[ i ] = bCath[k];

      newPads->q[ i ] = 0.0;
      // Not saturated
      newPads->saturate[ i ] = 0;
    }
    //
    return padsWithBoundaries;
}

Pads *Pads::addBoundaryPads( ) {

    // TODO: Remove duplicate pads
    double eps = 1.0e-4;
    // ???neigh = getFirstNeighbours( x, y, dx, dy )
    //
    std::vector<double> bX;
    std::vector<double> bY;
    std::vector<double> bdX;
    std::vector<double> bdY;
    int N = nPads;
    // Build neigbours if required
    PadIdx_t *neigh = buildFirstNeighbors();
    for (int i=0; i < N; i++) {
      bool east = true, west = true, north = true, south = true;
      for( const PadIdx_t *neigh_ptr = getTheFirtsNeighborOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
        PadIdx_t v = *neigh_ptr;
        double xDelta = (x[v] - x[i]);
        if (fabs(xDelta) > eps) {
          if (xDelta > 0) {
            east = false;
          } else {
            west = false;
          }
        }
        double yDelta = (y[v] - y[i]);
        if (fabs(yDelta) > eps) {
          if (yDelta > 0) {
            north = false;
          } else {
            south = false;
          }
        }
      }
      // Add new pads
      if ( east ) {
        bX.push_back( x[i]+2*dx[i] );
        bY.push_back( y[i]);
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
      }
      if ( west ) {
        bX.push_back( x[i]-2*dx[i] );
        bY.push_back( y[i]);
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
      }
      if (north) {
        bX.push_back( x[i] );
        bY.push_back( y[i]+2*dy[i] );
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
      }
      if (south) {
        bX.push_back( x[i] );
        bY.push_back( y[i]-2*dy[i] );
        bdX.push_back( dx[i]);
        bdY.push_back( dy[i]);
      }
    }
    int nPadToAdd = bX.size();
    // ??? int nTotalPads = nPadToAdd + N;
    int nTotalPads =  N+nPadToAdd;
    if (VERBOSE > 2) {
      printf("nTotalPads=%d, nPads=%d,  nPadToAdd=%d\n", nTotalPads, N, nPadToAdd);
    }
    Pads *padsWithBoundaries = new Pads( nTotalPads, chamberId );
    Pads *newPads = padsWithBoundaries;
    for (int i=0; i < N; i++) {
      newPads->x[ i ] = x[i];
      newPads->y[ i ] = y[i];
      newPads->dx[ i ] = dx[i];
      newPads->dy[ i ] = dy[i];
      newPads->q[ i ] = q[i];
      newPads->saturate[i] = saturate[i];
    }
    for (int i=N, k=0; i < nTotalPads; i++,k++) {
      newPads->x[ i ] = bX[k];
      newPads->y[ i ] = bY[k];
      newPads->dx[ i ] = bdX[k];
      newPads->dy[ i ] = bdY[k];
      newPads->q[ i ] = 0.0;
      newPads->saturate[i] = 0;
    }
    //
    return padsWithBoundaries;
}

Pads::Pads( int N, int chId, int mode_ ) {
  nPads= N;
  mode = mode_;
  chamberId = chId;
  allocate();
}

// Concatenate
// Remark : pads0, pads1 correspond respectively
//  to the cath-plane 0, 1
Pads::Pads( const Pads *pads0, const Pads *pads1) {
  int n0 = Pads::getNbrOfPads(pads0);
  int n1 = Pads::getNbrOfPads(pads1);
  nPads = n0 + n1;
  if ( n0 != 0 ) {
    chamberId = pads0->chamberId;
    mode = pads0->mode;
  }
  else {
    chamberId = pads1->chamberId;
    mode = pads1->mode;
  }
  allocate();
  totalCharge = 0.0;
  // X, Y, dX, dY, q
  if( n0 ) {
    vectorCopy(pads0->x, n0, x);
    vectorCopy(pads0->y, n0, y);
    vectorCopy(pads0->dx, n0, dx);
    vectorCopy(pads0->dy, n0, dy);
    vectorCopy(pads0->q, n0, q);
   // saturate pads
    vectorCopyShort( pads0->saturate, n0, saturate);
    totalCharge += pads0->totalCharge;
 }
 if (n1) {
    vectorCopy(pads1->x, n1, &x[n0]);
    vectorCopy(pads1->y, n1, &y[n0]);
    vectorCopy(pads1->dx, n1, &dx[n0]);
    vectorCopy(pads1->dy, n1, &dy[n0]);
    vectorCopy(pads1->q, n1, &q[n0]);
  // saturate pads
    vectorCopyShort( pads1->saturate, n1, &saturate[n0]);
    totalCharge += pads1->totalCharge;

  }
  // Cathode plane
  vectorSetShort( cath, 0, n0 );
  vectorSetShort( &cath[n0], 1, n1 );
}

Pads::Pads( const Pads &pads, int mode_ ) {
  nPads= pads.nPads;
  mode = mode_;
  chamberId = pads.chamberId;
  allocate();
  if (mode == pads.mode) {
    if (mode == xydxdyMode) {
      memcpy ( x, pads.x, sizeof(double)*nPads );
      memcpy ( y, pads.y, sizeof(double)*nPads);
      memcpy ( dx, pads.dx, sizeof(double)*nPads );
      memcpy ( dy, pads.dy, sizeof(double)*nPads );
      memcpy ( q, pads.q, sizeof(double)*nPads );
    } else {
      memcpy ( xInf, pads.xInf, sizeof(double)*nPads );
      memcpy ( yInf, pads.yInf, sizeof(double)*nPads);
      memcpy ( xSup, pads.xSup, sizeof(double)*nPads );
      memcpy ( ySup, pads.ySup, sizeof(double)*nPads );
      memcpy ( q, pads.q, sizeof(double)*nPads );
    }
  } else if (mode == xydxdyMode ) {
    //  xyInfSupMode ->  xydxdyMode
    for (int i=0; i < nPads; i++ ) {
      dx[i] = 0.5 * (pads.xSup[i] - pads.xInf[i]);
      dy[i] = 0.5 * (pads.ySup[i] - pads.yInf[i]);
      x[i] = pads.xInf[i] + dx[i];
      y[i] = pads.yInf[i] + dy[i];
    }
    memcpy ( q, pads.q, sizeof(double)*nPads );
  } else {
    // xydxdyMode -> xyInfSupMode
    for (int i=0; i < nPads; i++ ) {
      xInf[i] = pads.x[i] - pads.dx[i];
      xSup[i] = pads.x[i] + pads.dx[i];
      yInf[i] = pads.y[i] - pads.dy[i];
      ySup[i] = pads.y[i] + pads.dy[i];
    }
    memcpy ( q, pads.q, sizeof(double)*nPads );
  }
  memcpy ( saturate, pads.saturate, sizeof(Mask_t)*nPads );
}

Pads::Pads( const Pads &pads, const Groups_t *mask) {
  nPads = vectorSumShort( mask, pads.nPads );
  mode = xydxdyMode;
  chamberId = pads.chamberId;
  allocate();

  vectorGather( pads.x, mask, pads.nPads, x);
  vectorGather( pads.y, mask, pads.nPads, y);
  vectorGather( pads.dx, mask, pads.nPads, dx);
  vectorGather( pads.dy, mask, pads.nPads, dy);
  vectorGather( pads.q, mask, pads.nPads, q);
  vectorGatherShort( pads.saturate, mask, pads.nPads, saturate);
}

Pads::Pads( const double *x_, const double *y_, const double *dx_, const double *dy_, const double *q_, const Mask_t *saturate_, int chId, int nPads_) {
  mode = xydxdyMode;
  nPads = nPads_;
  chamberId = chId;
  allocate();
  // Copy pads
  memcpy ( x, x_, sizeof(double)*nPads );
  memcpy ( y, y_, sizeof(double)*nPads);
  memcpy ( dx, dx_, sizeof(double)*nPads );
  memcpy ( dy, dy_, sizeof(double)*nPads );
  memcpy ( q, q_, sizeof(double)*nPads );
  if( saturate_ != nullptr ) {
    memcpy ( saturate, saturate_, sizeof(Mask_t)*nPads );
  }
}

Pads::Pads( const double *x_, const double *y_, const double *dx_, const double *dy_, const double *q_,
        const short *cathode, const Mask_t *saturate_, short selectedCath, int chId,
        PadIdx_t *mapCathPadIdxToPadIdx, int nAllPads) {
  mode = xydxdyMode;
  int nCathode1 = vectorSumShort( cathode, nAllPads );
  nPads = nCathode1;
  if ( selectedCath == 0 ) {
      nPads = nAllPads - nCathode1;
  }
  chamberId = chId;
  allocate();
  double qSum = 0;
  // Copy pads
  int k=0;
  for( int i=0; i<nAllPads; i++) {
    if( cathode[i] == selectedCath ) {
      x[k] = x_[i];
      y[k] = y_[i];
      dx[k] = dx_[i];
      dy[k] = dy_[i];
      q[k] = q_[i];
      qSum += q_[i];
      saturate[k] = saturate_[i];
      mapCathPadIdxToPadIdx[k] = i;
      k++;
    }
  }
  totalCharge = qSum;
}

Pads::Pads( const Pads *pads1, const Pads *pads2, int mode_) {
  int N1 = (pads1 == nullptr) ? 0 : pads1->nPads;
  int N2 = (pads2 == nullptr) ? 0 : pads2->nPads;
  nPads = N1+N2;
  chamberId = (N1) ? pads1->chamberId :  pads2->chamberId;
  mode = mode_;
  allocate();
  if (mode == xydxdyMode) {
    // Copy pads1
    if (N1) {
       memcpy ( x, pads1->x, sizeof(double)*N1 );
       memcpy ( y, pads1->y, sizeof(double)*N1 );
       memcpy ( dx, pads1->dx, sizeof(double)*N1 );
       memcpy ( dy, pads1->dy, sizeof(double)*N1 );
       memcpy ( q, pads1->q, sizeof(double)*N1 );
       memcpy ( saturate, pads1->saturate, sizeof(double)*N1 );
    }
    if (N2) {
      // Copy pads2
      memcpy ( &x[N1], pads2->x, sizeof(double)*N2 );
      memcpy ( &y[N1], pads2->y, sizeof(double)*N2 );
      memcpy ( &dx[N1], pads2->dx, sizeof(double)*N2 );
      memcpy ( &dy[N1], pads2->dy, sizeof(double)*N2 );
      memcpy ( &q[N1], pads2->q, sizeof(double)*N2 );
      memcpy ( &saturate[N1], pads2->saturate, sizeof(double)*N2 );
    }
  } else
    for (int i=0; i<N1; i++) {
      xInf[i] = pads1->x[i] - pads1->dx[i];
      xSup[i] = pads1->x[i] + pads1->dx[i];
      yInf[i] = pads1->y[i] - pads1->dy[i];
      ySup[i] = pads1->y[i] + pads1->dy[i];
      q[i] = pads1->q[i];
      saturate[i] = pads1->saturate[i];
    }
    for (int i=0; i<N2; i++) {
      xInf[i+N1] = pads2->x[i] - pads2->dx[i];
      xSup[i+N1] = pads2->x[i] + pads2->dx[i];
      yInf[i+N1] = pads2->y[i] - pads2->dy[i];
      ySup[i+N1] = pads2->y[i] + pads2->dy[i];
      q[i+N1] = pads2->q[i];
      saturate[i+N1] = pads2->saturate[i];
    }
}

 void Pads::removePad( int index) {

  if( (index <0) || (index >= nPads)) return;
  int nItems = nPads - index;
  if (index ==  nPads-1) {
    nPads = nPads-1;
    return;
  }
  if( mode == xydxdyMode ) {
    vectorCopy(&x[index+1], nItems, &x[index]);
    vectorCopy(&y[index+1], nItems, &y[index]);
    vectorCopy(&dx[index+1], nItems, &dx[index]);
    vectorCopy(&dy[index+1], nItems, &dy[index]);
  } else {
    vectorCopy(&xInf[index+1], nItems, &xInf[index]);
    vectorCopy(&yInf[index+1], nItems, &yInf[index]);
    vectorCopy(&xSup[index+1], nItems, &xSup[index]);
    vectorCopy(&ySup[index+1], nItems, &ySup[index]);
  }
  vectorCopy(&q[index+1], nItems, &q[index]);
  vectorCopyShort(&saturate[index+1], nItems, &saturate[index]);

  nPads = nPads-1;
}

void Pads::allocate() {
  // Note: Must be deallocated/releases if required
  x = nullptr;
  y = nullptr;
  dx = nullptr;
  dy = nullptr;
  xInf = nullptr;
  xSup = nullptr;
  yInf = nullptr;
  ySup = nullptr;
  saturate = nullptr;
  q = nullptr;
  neighbors = nullptr;
  int N = nPads;
  if (mode == xydxdyMode ) {
    x = new double[N];
    y = new double[N];
    dx = new double[N];
    dy = new double[N];
  } else {
    xInf = new double[N];
    xSup = new double[N];
    yInf = new double[N];
    ySup = new double[N];
  }
  saturate = new Mask_t[N];
  cath = new Mask_t[N];
  q = new double[N];
}

void Pads::setToZero() {
  if ( mode == xydxdyMode ) {
    for (int i=0; i<nPads; i++) {
      x[i] = 0.0;
      y[i] = 0.0;
      dx[i] = 0.0;
      dy[i] = 0.0;
      q[i] = 0.0;
    }
  } else {
    for (int i=0; i<nPads; i++) {
      xInf[i] = 0.0;
      ySup[i] = 0.0;
      yInf[i] = 0.0;
      yInf[i] = 0.0;
      q[i] = 0.0;
    }
  }
}

// Build the neighbor list
PadIdx_t *Pads::buildFirstNeighbors() {
  int N = nPads;
  if ( neighbors == nullptr ) neighbors = buildFirstNeighbors( x, y, dx, dy, N, VERBOSE );
  return neighbors;
}

int Pads::addIsolatedPadInGroups( Mask_t *cathToGrp, Mask_t *grpToGrp, int nGroups) {
  int nNewGroups = 0;
  if (nPads == 0) return nGroups;
  if ( VERBOSE >0 ) {
    printf("[addIsolatedPadInGroups]  nGroups=%d\n", nGroups);
    vectorPrintShort("  cathToGrp input", cathToGrp, nPads);
  }
  PadIdx_t *neigh = buildFirstNeighbors();

  for ( int p=0; p < nPads; p++) {
    if( cathToGrp[p] == 0 ) {
      // Neighbors
      //
      int q = -1;
      for( PadIdx_t *neigh_ptr = getNeighborListOf(neigh, p); *neigh_ptr != -1; neigh_ptr++) {
        q = *neigh_ptr;
        // printf("  Neigh of %d: %d\n", p, q);
        if ( cathToGrp[q] != 0) {
            if ( cathToGrp[p] == 0 ) {
              // Propagation
              cathToGrp[p] = cathToGrp[q];
              // printf("    Neigh=%d: Propagate the grp=%d of the neighbor to p=%d\n", q, cathToGrp[q], p);
            } else if ( cathToGrp[p] != cathToGrp[q] ) {
              // newCathToGrp[p] changed
              // Fuse Grp
              Mask_t minGrp = cathToGrp[p];
              Mask_t maxGrp = cathToGrp[q];
              if ( cathToGrp[p] > cathToGrp[q]) {
                minGrp = cathToGrp[q];
                maxGrp = cathToGrp[p];

              }

              grpToGrp[ maxGrp] = minGrp;
              // printf("    Neigh=%d: Fuse the grp=%d of the neighbor with p-Group=%d\n", q, cathToGrp[q], cathToGrp[p]);
              // Update
              cathToGrp[p] = minGrp;
            }
        }
      }
      if ( cathToGrp[p] == 0) {
        // New Group
        nGroups++;
        nNewGroups++;
        cathToGrp[p] = nGroups;
        // printf("    Grp-isolated pad p=%d, new grp=%d \n", p, nGroups);
      }
    }
  }

  // Finish the Fusion
  for (int g=0; g < (nGroups+1); g++) {
    Mask_t gBar = g;
    while ( gBar != grpToGrp[gBar]) {
        gBar = grpToGrp[gBar];
    }
    // Terminal Grp :  gBar = grpToGrp[gBar]
    grpToGrp[g] = gBar;
  }
  if (VERBOSE > 2) {
  printf("  grpToGrp\n");
    for (int g=0; g < (nGroups+1); g++) {
      printf( "  %d -> %d\n", g,  grpToGrp[g]);
    }
  }
  // Apply group to Pads
  for ( int p=0; p < nPads; p++) {
    cathToGrp[p] = grpToGrp[cathToGrp[p]];
  }
  // Save in grpToGrp
  // ???
  vectorCopyShort( grpToGrp, (nGroups+1), grpToGrp);
  //
  // vectorPrintShort("  cathToGrp", cathToGrp, nPads);
  // vectorPrintShort("  grpToGrp before renumbering", grpToGrp, nGroups+1);
  // Inv ?? return vectorMaxShort( cathToGrp, nPads);
  return nNewGroups;
}
void Pads::release() {
  if ( mode == xydxdyMode ) {
    if ( x != nullptr ) {
      delete [] x;
      x = nullptr;
    }
    if ( y != nullptr ) {
      delete [] y;
      y = nullptr;
    }
    if ( dx != nullptr ) {
      delete [] dx;
      dx = nullptr;
    }
    if ( dy != nullptr ) {
      delete [] dy;
      dy = nullptr;
    }
  } else {
   if ( xInf != nullptr ) {
      delete [] xInf;
      xInf = nullptr;
    }
    if ( xSup != nullptr ) {
      delete [] xSup;
      xSup = nullptr;
    }
    if ( yInf != nullptr ) {
      delete [] yInf;
      yInf = nullptr;
    }
    if ( ySup != nullptr ) {
      delete [] ySup;
      ySup = nullptr;
    }
  }
  if ( q != nullptr ) {
    delete [] q;
    q = nullptr;
  }
  if ( cath != nullptr ) {
    delete [] cath;
    cath = nullptr;
  }
  if ( saturate != nullptr ) {
    delete [] saturate;
    saturate = nullptr;
  }
  nPads = 0;
}

Pads *Pads::refinePads(const Pads &pads) {
  int N = pads.nPads;
  // Count pad such as q > 4 * pixCutOf
  int count=0;
  // ??? double cut = 4*0.2;
  // double cut = -1.0;
  double cut = 0.2;
  for (int i=0; i < N; i++) {
    if ( pads.q[i] > cut ) {
        count++;
    }
  }
  // Warning: ??? The charge on the projected pads are not computed
  // so count (no filtering on charge)
  cut = -1;
  count = N;
  //
  vectorPrint( "Pads::refinePads",  pads.q, N);
  printf("Pads::refinePads count(new nPads)=%d\n", count);
  Pads *rPads = new Pads( count*4, pads.chamberId );
  int k=0;
  for (int i=0; i < N; i++) {
    if ( pads.q[i] > cut ) {
      // NW
      rPads->x[k] = pads.x[i] - 0.5*pads.dx[i];
      rPads->y[k] = pads.y[i] + 0.5*pads.dy[i];
      rPads->dx[k] = 0.5 * pads.dx[i];
      rPads->dy[k] = 0.5 * pads.dy[i];
      // rPads->q[k] = 0.25 * pads.q[i];
      rPads->q[k] = pads.q[i];
      k++;

      // NE
      rPads->x[k] = pads.x[i] + 0.5*pads.dx[i];
      rPads->y[k] = pads.y[i] + 0.5*pads.dy[i];
      rPads->dx[k] = 0.5 * pads.dx[i];
      rPads->dy[k] = 0.5 * pads.dy[i];
      // rPads->q[k] = 0.25 * pads.q[i];
      rPads->q[k] = pads.q[i];
      k++;

      // SW
      rPads->x[k] = pads.x[i] - 0.5*pads.dx[i];
      rPads->y[k] = pads.y[i] - 0.5*pads.dy[i];
      rPads->dx[k] = 0.5 * pads.dx[i];
      rPads->dy[k] = 0.5 * pads.dy[i];
      // rPads->q[k] = 0.25 * pads.q[i];
      rPads->q[k] = pads.q[i];
      k++;

      // SE
      rPads->x[k] = pads.x[i] + 0.5*pads.dx[i];
      rPads->y[k] = pads.y[i] - 0.5*pads.dy[i];
      rPads->dx[k] = 0.5 * pads.dx[i];
      rPads->dy[k] = 0.5 * pads.dy[i];
      // rPads->q[k] = 0.25 * pads.q[i];
      rPads->q[k] = pads.q[i];
      k++;
    }
  }
  return rPads;
}

void Pads::display( const char *str) {
  printf("%s\n", str);
  printf("  nPads=%d, mode=%d, chId=%d \n", nPads, mode, chamberId);
  if (mode == xydxdyMode) {
    vectorPrint( "  x", x, nPads);
    vectorPrint( "  y", y, nPads);
    vectorPrint( "  dx", dx, nPads);
    vectorPrint( "  dy", dy, nPads);
  } else {
    vectorPrint( "  xInf", xInf, nPads);
    vectorPrint( "  xSup", xSup, nPads);
    vectorPrint( "  yInf", yInf, nPads);
    vectorPrint( "  ySup", ySup, nPads);
  }
  vectorPrint( "  q", q, nPads);
}

Pads *Pads::clipOnLocalMax( const Pads &pixels, bool extractLocalMax ) {
    // Option extractLocalMax
    //   - true: extraxt local maxima
    //   - false: filter pixels arround the maxima
    if (VERBOSE > 0 ) {
      printf( "  ClipOnLocalMax (extractLocalMax Flag=%d, pixels.nPads=%d)\n", extractLocalMax, pixels.nPads);
    }
    // ????
    double eps = 1.0e-7;
    double noise = 0;
    double cutoff = noise;
    // ??? inv atLeastOneMax = -1
    PadIdx_t *neigh;
    if ( extractLocalMax ) {
      neigh = buildKFirstsNeighbors( pixels, 1);
    } else {
      neigh = buildKFirstsNeighbors( pixels, 2);
    }
    int nPads = pixels.nPads;
    double *q = pixels.q;
    double qMax =    vectorMax( q, nPads );
    // Result of the Laplacian-like operator
    double morphLaplacian[nPads];
    double laplacian[nPads];
    vectorSet( morphLaplacian, -1.0, nPads);
    Mask_t alreadySelect[nPads];
    vectorSetZeroShort( alreadySelect, nPads);
    std::vector<PadIdx_t> newPixelIdx;
    // getNeighborsOf ??? depends on the kernel size
    for (int i=0; i<nPads; i++) {
      int nLess = 0;
      int count=0;
      laplacian[i] = 0.0;
      for( PadIdx_t *neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
        PadIdx_t v = *neigh_ptr;
        // Morphologic Laplacian
        nLess += ( q[v] <= (q[i] + noise));
        count++;
        // Laplacian
        double cst;
        cst = (fabs( pixels.x[v] - pixels.x[i]) > eps) ? 0.5 : 1.0;
        cst = (fabs( pixels.y[v] - pixels.y[i]) > eps) ? 0.5*cst : cst;
        cst = (cst == 1.0) ? -3.0 : cst;
        laplacian[i] += cst*q[v];

      }
      morphLaplacian[i] =  nLess / count;
      if (1 && VERBOSE) {
        printf("  Laplacian i=%d, x[i]=%6.3f, y[i]=%6.3f, z[i]=%6.3f, smoothQ[i]=%6.3f, lapl[i]=%6.3f\n", i, pixels.x[i], pixels.y[i], q[i], morphLaplacian[i], laplacian[i]);
      }
      if (morphLaplacian[i] >= 1.0 ) {
        if (extractLocalMax) {
          if ( ( q[i] > 0.015 *qMax) || (fabs(laplacian[i]) > (0.5 * q[i])) ) {
            newPixelIdx.push_back( i );
            if (VERBOSE > 0) {
               printf("  Laplacian i=%d, x[i]=%6.3f, y[i]=%6.3f, z[i]=%6.3f, smoothQ[i]=%6.3f, lapl[i]=%6.3f ", i, pixels.x[i], pixels.y[i], q[i], morphLaplacian[i], laplacian[i]);
               printf("  Selected %d\n", i);
            }
          }
        } else {
          // Select as new pixels in the vinicity of the local max
          printf("  Selected neighbors of i=%d: ", i);

          for( PadIdx_t *neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
            PadIdx_t v = *neigh_ptr;
            if( alreadySelect[v] == 0 ) {
              alreadySelect[v] = 1;
              newPixelIdx.push_back( v );
              printf("%d, ", v);
            }
          }
          printf("\n");

        }
      }

    }
    // Extract the new selected pixels
    int nNewPixels = newPixelIdx.size();
    Pads *newPixels = new Pads( nNewPixels, pixels.chamberId );
    for ( int i=0; i< nNewPixels; i++) {
       newPixels->x[i] = pixels.x[newPixelIdx[i]];
       newPixels->y[i] = pixels.y[newPixelIdx[i]];
       newPixels->dx[i] = pixels.dx[newPixelIdx[i]];
       newPixels->dy[i] = pixels.dy[newPixelIdx[i]];
       newPixels->q[i] = pixels.q[newPixelIdx[i]];
    }
    Pads *localMax = nullptr;
    if (extractLocalMax) {
      double cutRatio = 0.01;
      double qCut = cutRatio * vectorMax ( newPixels->q, newPixels->nPads);
      //
      // Refine the charge and coordinates of the local max.
      //
      localMax = new Pads( nNewPixels, pixels.chamberId);
      localMax->setToZero();
      // Sort local max by charge value
      int index[nNewPixels];
      for( int k=0; k<nNewPixels; k++) { index[k]=k; }
      std::sort( index, &index[nNewPixels], [=](int a, int b){ return (newPixels->q[a] > newPixels->q[b]); });
      // ???? Delete neigh, neigh2
      // ??? PadIdx_t *neigh2 = getFirstNeighboursWithDiag2(u, v, du, dv);
      delete [] neigh;
      neigh = buildKFirstsNeighbors( *newPixels, 1);
      // Avoid taking the same charge for 2 different localMax
      Mask_t mask[nNewPixels];
      vectorSetShort( mask, 1, nNewPixels);
      int kSelected = 0;
      for (int k=0; k<nNewPixels; k++) {
        if (mask[k] == 1) {
          for( PadIdx_t *neigh_ptr = getNeighborListOf(neigh, k); *neigh_ptr != -1; neigh_ptr++) {
            PadIdx_t v = *neigh_ptr;
            localMax->q[k] += newPixels->q[v]*mask[v];
            localMax->x[k] += newPixels->x[v]*newPixels->q[v]*mask[v];
            localMax->y[k] += newPixels->y[v]*newPixels->q[v]*mask[v];
            mask[v] = 0;
          }
          if (localMax->q[k] > qCut ) {
            localMax->q[kSelected] = localMax->q[k];
            localMax->x[kSelected] = localMax->x[k] / localMax->q[k];
            localMax->y[kSelected] = localMax->y[k] / localMax->q[k];
            localMax->dx[kSelected] = newPixels->dx[k];
            localMax->dy[kSelected] = newPixels->dy[k];
            if (VERBOSE > 1) {
              printf("  add a seed q=%9.4f, (x,y) = (%9.4f, %9.4f)\n", localMax->q[k], localMax->x[k], localMax->q[k]);
            }
            kSelected++;
          }
        }
      }
      localMax->nPads = kSelected;
    }
    delete [] neigh;
    if (extractLocalMax) {
      delete newPixels;
      return localMax;
    } else {
      return newPixels;
    }
}

void Pads::printNeighbors( const PadIdx_t *neigh, int N ) {
  printf("Neighbors %d\n", N);
  for ( int i=0; i < N; i++) {
    printf("  neigh of i=%2d: ", i);
    for( const PadIdx_t *neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t j = *neigh_ptr;
      printf("%d, ", j);
    }
    printf("\n");
  }
}

void Pads::printPads(const char* title, const Pads &pads) {
  printf("%s\n", title);
  for (int i=0; i < pads.nPads; i++) {
    printf( "  pads i=%3d: x=%3.5f, dx=%3.5f, y=%3.5f, dy=%3.5f\n",
            i, pads.x[i], pads.dx[i], pads.y[i], pads.dy[i]);
  }
}

Pads::~Pads() {
  release();
}

} // namespace mch
} // namespace o2


