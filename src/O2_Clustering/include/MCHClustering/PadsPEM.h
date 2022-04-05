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

/// \file PadPEM.h
/// \brief Pads representation and transformation
///
/// \author Gilles Grasseau, Subatech

#ifndef ALICEO2_MCH_PADSPEM_H_
#define ALICEO2_MCH_PADSPEM_H_

#include "MCHClustering/dataStructure.h"


typedef int PadIdx_t;    // Pad index type
typedef short Groups_t;   // Pad index type
namespace o2
{
namespace mch
{
  // Neigbours
  static const int MaxNeighbors = 25 +13 ; // 5*5 neigbours + the center pad itself + separator (-1) 
                                           // 5x5 neighbours + 50 % 
  template <class T>
  inline static T getNeighborListOf(T neigh, PadIdx_t i) { return &neigh[MaxNeighbors * i]; };
  
  inline static PadIdx_t *getTheFirtsNeighborOf( PadIdx_t* neigh, PadIdx_t i) { return &neigh[MaxNeighbors * i]; };

struct Pads {
  enum padMode {
   xydxdyMode = 0x0,    ///< x, y, dx, dy pad coordinates
   xyInfSupMode = 0x1  ///< xInf, xSup, yInf, ySup pad coordinates
  };
  // Representation mode  (see padMode)
  int mode=xydxdyMode;
  // Mode xydxdy
  double *x = nullptr;
  double *y = nullptr;
  double *dx = nullptr;
  double *dy = nullptr;
  // Mode xyInfSupMode
  double *xInf = nullptr, *xSup = nullptr;
  double *yInf = nullptr, *ySup = nullptr;
  
  Mask_t *cath = nullptr;
  Mask_t *saturate = nullptr;
  double *q = nullptr;
  double totalCharge=0;
  int nPads=0;
  int chamberId=-1;
  PadIdx_t *neighbors = nullptr;
  
  // Building Neighbors 
  static PadIdx_t *buildFirstNeighbors( double *X, double *Y, double *DX, double *DY, int N, int verbose);
  static PadIdx_t *buildKFirstsNeighbors( const Pads &pads, int kernelSize );
  // Add zero-charged pads to the neighboring of the pads (cathode cluster)
  static Pads *addBoundaryPads( const double *x_, const double *y_, const double *dx_, const double *dy_, const double *q_, const Mask_t *cath_, const Mask_t *sat_, int chamberId, int N);
  // Split each pads in 4 smaller pads with the same sizes
  static Pads *refinePads(const Pads &pads);
  // Extract local maximima, with of without a neighboring
  static Pads *clipOnLocalMax( const Pads &pixels, bool extractLocalMax );
  // Utilities
  static void printNeighbors( const PadIdx_t *neigh, int N );
  static void printPads(const char* title, const Pads &pads);
  static inline int getNbrOfPads(const Pads *pads) { return (pads==nullptr) ? 0 : pads->nPads; };

  // Allocation constructor
  Pads( int N, int chId, int mode=xydxdyMode);
  // Concatenate 2 pads set
  Pads( const Pads *pads0,  const Pads *pads1);
  // Build a new set of pads 
  Pads( const Pads &pads, int mode_ );
  // Build a pads set from those selected by "mask"
  Pads( const Pads &pads, const Groups_t *mask);
  // TO DO ??? must be fused with 
  Pads( const Pads *pads1, const Pads *pads2, int mode);
  Pads( const double *x_, const double *y_, const double *dx_, const double *dy_, 
    const double *q_, const Mask_t *saturate_, int chId, int nPads_);
  Pads( const double *x_, const double *y_, const double *dx_, const double *dy_, 
    const double *q_, const short *cathode, const Mask_t *saturate_, short cathID, 
    int chId, PadIdx_t *mapCathPadIdxToPadIdx, int nAllPads);
  // Add zero-charged pads to the neighboring of the pads (cathode cluster)
  Pads *addBoundaryPads(); 
    // Building Neighbors 
  PadIdx_t *buildFirstNeighbors();
  // Groups
  int addIsolatedPadInGroups( Mask_t *cathToGrp, Mask_t *grpToGrp, int nGroups);
  ~Pads();
  // Memory allocation/deallocation
  void allocate(); 
  void release();
  // Utilities
  void removePad( int index);
  void setToZero();
  void display( const char *str);

};


} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_PADSPEM_H_
