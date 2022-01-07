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
#include "newPadProcessing.h"
#include "mathUtil.h"

#define VERBOSE 1
#define CHECK 1

namespace o2
{
namespace mch
{
double epsilonGeometry = 1.0e-4;

Cluster::Cluster( Pads *pads0, Pads *pads1) {
  int nPlanes = 0;
  aloneCath = -1;
  if ( pads0 != nullptr ) {
    pads[nPlanes] = pads0;
    nPlanes++;
  } else {
    aloneCath = 1;
  }
  if ( pads1 != nullptr ) {
    pads[nPlanes] = pads1;
    nPlanes++;
  } else {
    aloneCath = 0;
  }
  nbrOfCathodes = nPlanes;
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
  const double *y1Sup = pad1InfSup.xSup;
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
      l = fmax( x0Inf[i], x1Inf[j]);
      r = fmin( x0Sup[i], x1Sup[j]);
      b = fmax( y0Inf[i], y1Inf[j]);
      t = fmin( y0Sup[i], y1Sup[j]);
      projX[k] = (l+r)*0.5;
      projY[k] = (b+t)*0.5;
      projDX[k] = (r-l)*0.5;
      projDY[k] = (t-b)*0.5;
      mapKToIJ[k].i = i;
      mapKToIJ[k].j = j;
      mapIJToK[i*N1+ j] = k;
      // Debug
      // printf("newpad %d %d %d %9.3g %9.3g %9.3g %9.3g\n", i, j, k, projX[k], projY[k], projDX[k], projDY[k]);
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

void Cluster::buildProjectedGeometry( int includeSingleCathodePads) {
  if (nbrOfCathodes==1) {
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
    projNeighbors = buildFirstNeighbors( *projectedPads, VERBOSE );
    return;
  }

  int N0 = pads[0]->nPads;
  int N1 = pads[1]->nPads;
  char intersectionMatrix[N0*N1];
  vectorSetZeroChar( intersectionMatrix, N0*N1);

  // Get the pad limits
  Pads padInfSup0( *pads[0], Pads::xyInfSupMode);
  Pads padInfSup1( *pads[1], Pads::xyInfSupMode);
  mapIJToK = new PadIdx_t[N0*N1];
  // Stack allocation ????
  /*
  PadIdx_t *aloneIPads = new PadIdx_t[N0];
  PadIdx_t *aloneJPads = new PadIdx_t[N1];
  PadIdx_t *aloneKPads = new PadIdx_t[N0*N1];
  */
  PadIdx_t aloneIPads[N0];
  PadIdx_t aloneJPads[N1];
  // ??? Inv PadIdx_t aloneKPads[N0*N1];
  aloneKPads = new PadIdx_t[N0*N1];
  //
  // Build the intersection matrix
  // Looking for j pads, intercepting pad i
  //
  double xmin, xmax, ymin, ymax;
  PadIdx_t xInter, yInter;
  for( PadIdx_t i=0; i < N0; i++) {
    for( PadIdx_t j=0; j < N1; j++) {
      xmin = std::fmax( padInfSup0.xInf[i], padInfSup1.xInf[i] );
      xmax = std::fmax( padInfSup0.xSup[i], padInfSup1.xSup[i] );
      xInter = ( xmin <= (xmax - epsilonGeometry) );
      if( xInter ) {
        ymin = std::fmax( padInfSup0.yInf[i], padInfSup1.yInf[i] );
        ymax = std::fmax( padInfSup0.ySup[i], padInfSup1.ySup[i] );
        yInter = ( ymin <= (ymax - epsilonGeometry));
        intersectionMatrix[i*N1+j] =  yInter;
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
  mapKToIJ = new MapKToIJ_t[maxNbrOfProjPads];

  //
  // Build the new pads
  // ???
  computeProjectedPads( padInfSup0, padInfSup1,
                      aloneIPads, aloneJPads, aloneKPads, includeSingleCathodePads );

  if (CHECK == 1) checkConsistencyMapKToIJ( intersectionMatrix, mapKToIJ, mapIJToK, aloneIPads, aloneJPads, N0, N1, projectedPads->nPads);

  //
  // Get the isolated new pads
  // (they have no neighboring)
  //
  int thereAreIsolatedPads = 0;
  projNeighbors = buildFirstNeighbors( *projectedPads, VERBOSE );
  // printNeighbors();
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
    projNeighbors = buildFirstNeighbors( *projectedPads, VERBOSE );
  }
}

void Cluster::buildGroupOfPads() {

}

int Cluster::getConnectedComponentsOfProjPadsWOIsolatedPads(  ) {
  // Class from neighbors list of Projected pads, the pads in groups (connected components)
  // projPadToGrp is set to the group Id of the pad.
  // If the group Id is zero, the the pad is unclassified
  // Return the number of groups
  int  N = projectedPads->nPads;
  projPadToGrp = new Groups_t[N];
  PadIdx_t *neigh = projNeighbors;
  PadIdx_t neighToDo[N];
  vectorSetZeroShort( projPadToGrp, N);
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
    // New group for k - then search all neighbours of k
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
void Cluster::assignOneCathPadsToGroup( short *padGroup, int nPads, int nGrp, int nCath0, int nCath1, short *wellSplitGroup) {
  cath0ToGrpFromProj = 0;
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


// Assign a group to the original pads
int assignPadsToGroupFromProj( short *projPadToGrp, int nProjPads,
        const PadIdx_t *cath0ToPadIdx, const PadIdx_t *cath1ToPadIdx,
        int nGrp, int nPads, short *padMergedGrp ) {
// cath0ToPadIdx : pad indices of cath0 (cath0ToPadIdx[0..nCath0] -> i-pad
// outputs:
  short padToGrp[nPads];
  short matGrpGrp[ (nGrp+1)*(nGrp+1)];
  vectorSetZeroShort( padToGrp, nPads);
  //
  // vectorSetShort( wellSplitGroup, 1, nGrp+1);
  vectorSetZeroShort( matGrpGrp, (nGrp+1)*(nGrp+1) );
  //
  PadIdx_t i, j;
  short g, prevGroup;
  // Expand the projected Groups
  // 'projPadToGrp' to the pad groups 'padToGrp'
  // If there are conflicts, fuse the groups
  // Build the Group-to-Group matrix matGrpGrp
  // which describe how to fuse Groups
  // with the projected Groups
  // projPadToGrp
  for( int k=0; k < nProjPads; k++) {
    g = projPadToGrp[k];
    // give the indexes of overlapping pads
    i = mapKToIJ[k].i; j = mapKToIJ[k].j;
    //
    // Cathode 0
    //
    if ( (i >= 0) && (cath0ToPadIdx !=0) ) {
      // Remark: if i is an alone pad (j<0)
      // i is processed as well
      //
      // cath0ToPadIdx: map cathode-pad to the original pad
      PadIdx_t padIdx = cath0ToPadIdx[i];
      prevGroup = padToGrp[ padIdx ];
      if ( (prevGroup == 0) || (prevGroup == g) ) {
        // Case: no group before or same group
        //
        padToGrp[ padIdx ] = g;
        matGrpGrp[ g*(nGrp+1) +  g ] = 1;
      } else {
        // Already a Grp which differs
        // if ( prevGroup > 0) {
          // Invalid prev group
          // wellSplitGroup[ prevGroup ] = 0;
          // Store in the grp to grp matrix
          // Group to fuse
          matGrpGrp[ g*(nGrp+1) +  prevGroup ] = 1;
          matGrpGrp[ prevGroup*(nGrp+1) +  g ] = 1;
        //}
        // padToGrp[padIdx] = -g;
      }
    }
    //
    // Cathode 1
    //
    if ( (j >= 0) && (cath1ToPadIdx != 0) ) {
      // Remark: if j is an alone pad (j<0)
      // j is processed as well
      //
      // cath1ToPadIdx: map cathode-pad to the original pad
      PadIdx_t padIdx = cath1ToPadIdx[j];
      prevGroup = padToGrp[padIdx];
      if ( (prevGroup == 0) || (prevGroup == g) ){
         // No group before
         padToGrp[padIdx] = g;
         matGrpGrp[ g*(nGrp+1) +  g ] = 1;
      } else {
        // Already a Group
        // if ( prevGroup > 0) {
          matGrpGrp[ g*(nGrp+1) +  prevGroup ] = 1;
          matGrpGrp[ prevGroup*(nGrp+1) +  g ] = 1;
        // }
        // padToGrp[padIdx] = -g;
      }
    }
  }
  if (VERBOSE > 0) {
    printf( "[AssignPadsToGroupFromProj]\n");
    printMatrixShort("  Group/Group matrix", matGrpGrp, nGrp+1, nGrp+1);
    vectorPrintShort("  padToGrp", padToGrp, nPads);
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

  // Perform the mapping group -> mergedGroups
  if ( VERBOSE >0 ) {
    vectorPrintShort( "  grpToMergedGrp", grpToMergedGrp, nGrp+1);
  }
  for (int p=0; p < nPads; p++) {
    padMergedGrp[p] = grpToMergedGrp[ abs( padToGrp[p] ) ];
  }
  if (CHECK) {
    for (int p=0; p < nPads; p++) {
      if ( (VERBOSE > 0) && (padMergedGrp[p] == 0)) {
        printf("  assignPadsToGroupFromProj: pad %d with no group\n", p);
      }
    }
  }
  // Update the group of the proj-pads
  vectorMapShort(projPadToGrp, grpToMergedGrp, nProjPads);

  //
  return newGroupID;
}

} // namespace mch
} // namespace o2