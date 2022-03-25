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

#ifndef ALICEO2_MCH_INSPECTMODEL_H_
#define ALICEO2_MCH_INSPECTMODEL_H_

#include <vector>

#include "MCHClustering/clusterProcessing.h"
#include "MCHClustering/PadsPEM.h"
#include "MCHClustering/Cluster.h"

// Inspect data
typedef struct dummy_t {
    // Data on Projected Pads
    int nbrOfProjPads=0;
    double *projectedPads=nullptr;
    double *qProj=nullptr; // Projected charges
    Groups_t *projGroups=nullptr;
    // Theta init
    double *thetaInit=nullptr;
    int kThetaInit=0;
    // Data about subGroups
    int totalNbrOfSubClusterPads=0;
    int totalNbrOfSubClusterThetaEMFinal=0;
    int totalNbrOfSubClusterThetaExtra=0;
    std::vector< DataBlock_t > subClusterPadList;
    std::vector< DataBlock_t > subClusterChargeList;
    std::vector< DataBlock_t > subClusterThetaEMFinal;
    std::vector< DataBlock_t > subClusterThetaFitList;
    std::vector< DataBlock_t > subClusterThetaExtra;

    // Cath groups
    int nCathGroups=0;
    short *padToCathGrp=nullptr;
} InspectModel;
//

// PadProcessing
typedef struct dummyPad_t {
    // Data on Pixels
    const static int nStorage = 8;
    std::vector<DataBlock_t> xyDxyQPixels[nStorage];
} InspectPadProcessing_t;


extern "C" {
  void cleanThetaList();
  void cleanInspectModel();
  // ??? Internal void copyInGroupList( const double *values, int N, int item_size, std::vector< DataBlock_t > &groupList);
  // ??? void appendInThetaList( const double *values, int N, std::vector< DataBlock_t > &groupList);
  void saveThetaEMInGroupList( const double *thetaEM, int K);
  void saveThetaExtraInGroupList( const double *thetaExtra, int K);
  void saveThetaFitInGroupList( const double *thetaFit, int K);
  void collectTheta( double *theta, Groups_t *thetaToGroup, int K);
  void savePadsOfSubCluster( const double *xyDxy, const double *q,  int n);
  void finalizeInspectModel();
  int getNbrOfProjPads();
  int getNbrOfPadsInGroups();
  int getNbrOfThetaEMFinal();
  int getNbrOfThetaExtra();
  //
  void saveProjectedPads(const o2::mch::Pads *pads, double *qProj );
  void collectProjectedPads(double *xyDxy, double *chA, double *chB);
  void savePadToCathGroup( const Groups_t *cath0Grp, const PadIdx_t *mapCath1PadIdxToPadIdx, int nCath0, 
        const Groups_t *cath1Grp, const PadIdx_t *mapCath0PadIdxToPadIdx, int nCath1);
  int collectProjGroups( Groups_t *projPadToGrp);
  void saveProjPadToGroups( Groups_t *projPadToGrp, int N );
  void collectPadToCathGroup( Mask_t *padToMGrp, int nPads );
  void collectPadsAndCharges( double *xyDxy, double *z, Groups_t *padToGroup, int N);
  // Unused ??? void collectLaplacian( double *laplacian, int N);
  void collectResidual( double *residual, int N);
  int getKThetaInit();
  void collectThetaInit( double *thetai, int N);
  void collectThetaEMFinal( double *thetaEM, int K);
  void collectThetaExtra( double *thetaExtra, int K);
  void cleanInspectPadProcess();
  int collectPixels( int which, int N, double *xyDxy, double *q);
  void inspectSavePixels( int which, o2::mch::Pads &pixels);
  int getNbrProjectedPads();
  void setNbrProjectedPads( int n);
}
#endif // ALICEO2_MCH_INSPECTMODEL_H_