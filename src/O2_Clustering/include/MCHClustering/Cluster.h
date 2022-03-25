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

/// \file Cluster.h
/// \brief Definition of a class to reconstruct clusters with the gem MLEM algorithm
///
/// \author Gilles Grasseau, Subatech

#ifndef ALICEO2_MCH_CLUSTER_H_
#define ALICEO2_MCH_CLUSTER_H_

#include "MCHClustering/PadsPEM.h"

namespace o2
{
namespace mch
{
typedef std::pair<int, const double*> DataBlock_t;

typedef struct {
  PadIdx_t i;
  PadIdx_t j;
} MapKToIJ_t;

class Cluster
{
 public:
  Cluster();
  Cluster( const double *x, const double *y, const double *dx, const double *dy, 
          const double *q, const short *cathodes, const short *saturated, int chId, int nPads);
  Cluster( Pads *pads0, Pads *pads1 );
  // Extract a sub-cluster belonging to the group g
  Cluster( Cluster &cluster, Groups_t g);
  ~Cluster();
  inline int getNbrOfPads(int c) { return ( pads[c]==nullptr ? 0 : pads[c]->nPads); };
  inline const Groups_t *getCathGroup(int c) { return cathGroup[c]; };
  inline Groups_t *getProjPadGroup( ) { return projPadToGrp; };
  inline int getNbrOfProjectedPads( ) { return (projectedPads==nullptr ? 0 : projectedPads->nPads); };
  // ??? To remove - unused
  double *getProjPadsAsXYdXY( Groups_t group, const Mask_t* maskGrp, int nbrProjPadsInTheGroup);
  const PadIdx_t *getMapCathPadToPad(int c) { return mapCathPadIdxToPadIdx[c]; };
  int getNbrOfCathPad(int c) { return pads[c]->nPads; };
  int buildProjectedGeometry( int includeAlonePads);
  const Pads *getProjectedPads() { return projectedPads;};
  double *projectChargeOnProjGeometry(int includeAlonePads);
  int buildGroupOfPads();
  int findLocalMaxWithPET( double *thetaL, int nbrOfPadsInTheGroupCath );
  DataBlock_t fit( double *thetaInit, int K);
  // Not used in the Clustering/fitting
  // Just to check hit results
  int findLocalMaxWithBothCathodes( double *thetaOut, int kMax, int verbose=0 );
  
  
 private:
  // ???
  static constexpr int SNFitClustersMax = 3;                     ///< maximum number of clusters fitted at the same time
  
  int chamberId=-1; 
  int singleCathPlaneID=-1;       // index of the unique cathode plane (0 or 1)
  int nbrOfCathodePlanes=0;   // Nbr of Cathodes Plane
  Pads *pads[2]={nullptr, nullptr};       // Two cathode-pads
  int nbrSaturated=0;       // Number of saturated pads 
  // ??? Unused probably
  PadIdx_t *mapCathPadIdxToPadIdx[2]={nullptr, nullptr}; // Map cathode-pad index to pad index
  
  // Projection
  Pads *projectedPads=nullptr;      // Projected pads
  PadIdx_t *projNeighbors=nullptr;  // Neighbors list of projected pads 
  Groups_t *projPadToGrp=nullptr;   // Groups of projected pads
  int nbrOfProjGroups=0;
  // Groups
  Groups_t *cathGroup[2]={nullptr, nullptr};
  int nbrOfCathGroups=0;          
  // Geometry
  PadIdx_t *IInterJ=nullptr;  // Compressed intersection matrix
  PadIdx_t *JInterI=nullptr;  // Compressed intersection matrix
  PadIdx_t *aloneIPads=nullptr;       // Indexes of alone pads in cathode 0
  PadIdx_t *aloneJPads=nullptr;       // Indexes of alone pads in cathode 1
  MapKToIJ_t *mapKToIJ=nullptr;  // Mapping projected pads (k) to the 2 intersection pads
  PadIdx_t *mapIJToK=nullptr;    // Inverse mapping (i,j) pads -> k (projected pads)
  PadIdx_t *aloneKPads=nullptr;  // Alone (isolate) projected pads
  
  void computeProjectedPads(
            const Pads &pad0InfSup, const Pads &pad1InfSup,
            PadIdx_t *aloneIPads, PadIdx_t *aloneJPads, PadIdx_t *aloneKPads, int includeAlonePads);
  
  int getConnectedComponentsOfProjPadsWOSinglePads( );
  // Used ???
  void assignSingleCathPadsToGroup( short *padGroup, int nPads, int nGrp, int nCath0, int nCath1, short *wellSplitGroup);
  // Used ???
  int assignPadsToGroupFromProj( int nGrp);
  // ???  short *projPadToGrp, int nProjPads,
  //      const PadIdx_t *cath0ToPadIdx, const PadIdx_t *cath1ToPadIdx,
  //      int nGrp, int nPads, short *padMergedGrp );
  int assignGroupToCathPads( );
  void maskedCopyToXYdXY(const Pads &pads, const Mask_t* mask, int nMask,
                     double* xyDxyMasked, int nxyDxyMasked);
  int filterFitModelOnClusterRegion( Pads &pads, double *theta, int K, Mask_t *maskFilteredTheta);
  int filterFitModelOnSpaceVariations( const double *theta0, int K0, double *theta, int K, Mask_t *maskFilteredTheta);
  void prepareDataForFitting( Mask_t *maskFit[2],
        double *xyDxyFit, double *zFit, Mask_t *cathFit, Mask_t *notSaturatedFit, double *zCathTotalCharge, int nFits[2]);
  void updateProjectionGroups ();
  int renumberGroups( short *grpToGrp, int nGrp );
  int renumberGroupsV2( Mask_t *cath0Grp, int nbrCath0, Mask_t *cath1Grp, int nbrCath1, Mask_t *grpToGrp, int nGrp );
  int getIndexByRow( const char *matrix, PadIdx_t N, PadIdx_t M, PadIdx_t *IIdx);
  int getIndexByColumns( const char *matrix, PadIdx_t N, PadIdx_t M, PadIdx_t *JIdx);
  int checkConsistencyMapKToIJ( const char *intersectionMatrix, const MapKToIJ_t *mapKToIJ, const PadIdx_t *mapIJToK, const PadIdx_t *aloneIPads, const PadIdx_t *aloneJPads, int N0, int N1, int nbrOfProjPads);

  // Not used in the Clustering/fitting
  // Just to check hit results
  int laplacian2D( const Pads &pads_, PadIdx_t *neigh, int chId, PadIdx_t *sortedLocalMax, int kMax, double *smoothQ);
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_CLUSTER_H_