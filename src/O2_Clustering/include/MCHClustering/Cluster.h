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

#include "Cluster.h"

namespace o2
{
namespace mch
{

typedef struct {
  PadIdx_t i;
  PadIdx_t j;
} MapKToIJ_t;

class Cluster
{
 public:
  Cluster( Pads *pads0, Pads *pads1 );
  ~Cluster();
  void buildProjectedGeometry( int includeAlonePads);
  void buildGroupOfPads();
  
 private:
  // ???
  static constexpr int SNFitClustersMax = 3;                     ///< maximum number of clusters fitted at the same time

  int aloneCath;       // index of the unique cathode (0 or 1)
  int nbrOfCathodes;   // Nbr of Cathodes
  Pads *pads[2];       // Two cathode-pads
  
  // Projection
  Pads *projectedPads;      // Projected pads
  PadIdx_t *projNeighbors;  // Neighbors list of projected pads 
  Groups_t *projPadToGrp;   // Groups of projected pads
  
  // Geometry
  PadIdx_t *IInterJ;  // Compressed intersection matrix
  PadIdx_t *JInterI;  // Compressed intersection matrix
  MapKToIJ_t *mapKToIJ;  // Mapping projected pads (k) to the 2 intersection pads
  PadIdx_t *mapIJToK;    // Inverse mapping (i,j) pads -> k (projected pads)
  PadIdx_t *aloneKPads;  // Alone (isolate) projected pads
  
  
  void computeProjectedPads(
            const Pads &pad0InfSup, const Pads &pad1InfSup,
            PadIdx_t *aloneIPads, PadIdx_t *aloneJPads, PadIdx_t *aloneKPads, int includeAlonePads);
  
  int getConnectedComponentsOfProjPadsWOIsolatedPads( );
  void assignOneCathPadsToGroup( short *padGroup, int nPads, int nGrp, int nCath0, int nCath1, short *wellSplitGroup);
  
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_CLUSTER_H_
