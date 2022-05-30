#include <algorithm>
#include <stdexcept>
#include <string.h>
#include <vector>

#include "MCHClustering/clusterProcessing.h"
#include "mathUtil.h"
#include "mathieson.h"
// Used to analyse the clustering/fitting
#include "InspectModel.h"

// To keep internal data
#define INSPECTMODEL 1
#define VERBOSE 1
#define CHECK 1

// Type of projection
// Here add single cathode pads
// (No intestersection with pads
// in other plane)
static int includeSinglePads = 1;

using namespace o2::mch;

// Total number of hits/seeds (number of mathieson)
// found in the precluster;
static int nbrOfHits = 0;
// Storage of the seeds found
std::vector<DataBlock_t> seedList;

// Release memory and reset the seed list
void cleanThetaList() {
  for (int i = 0; i < seedList.size(); i++)
    delete[] seedList[i].second;
  seedList.clear();
}

// Extract hits/seeds of a pre-cluster
int clusterProcess(const double *xyDxyi_, const Mask_t *cathi_,
                   const Mask_t *saturated_, const double *zi_, int chId,
                   int nPads) {

  nbrOfHits = 0;
  cleanThetaList();
  if (INSPECTMODEL) {
    cleanInspectModel();
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
  int nNewPads = 0;

  // Pad filter when there are a too large number of pads
  if (nPads > 800) {
    // Remove noisy event
    if (VERBOSE > 0) {
      printf("WARNING: remove noisy pads <z>=%f, min/max z=%f,%f",
             vectorSum(zi_, nPads) / nPads, vectorMin(zi_, nPads),
             vectorMax(zi_, nPads));
    }
    // Select pads which q > 2.0
    vectorBuildMaskGreater(zi_, 2.0, nPads, noiseMask);
    nNewPads = vectorSumShort(noiseMask, nPads);
    xyDxyi__ = new double[nNewPads * 4];
    zi__ = new double[nNewPads];
    saturated__ = new Mask_t[nNewPads];
    cathi__ = new Mask_t[nNewPads];
    //
    vectorGather(zi_, noiseMask, nPads, zi__);
    vectorGatherShort(saturated_, noiseMask, nPads, saturated__);
    vectorGatherShort(cathi_, noiseMask, nPads, cathi__);
    maskedCopyXYdXY(xyDxyi_, nPads, noiseMask, nPads, xyDxyi__, nNewPads);
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

  // Build a cluster object
  Cluster cluster(getConstX(xyDxyi, nPads), getConstY(xyDxyi, nPads),
                  getConstDX(xyDxyi, nPads), getConstDY(xyDxyi, nPads), zi,
                  cathi, saturated, chId, nPads);

  // Compute the underlying geometry (cathode plae superposition
  int nProjPads = cluster.buildProjectedGeometry(includeSinglePads);

  if (nProjPads == 0)
    throw std::range_error("No projected pads !!!");

  // Build geometric groups of pads
  // which constitute sub-clusters
  // A sub-cluster can contain several seeds
  int nGroups = cluster.buildGroupOfPads();

  if (INSPECTMODEL) {
    // Compute the charge on projected geometry
    double *qProj = cluster.projectChargeOnProjGeometry(includeSinglePads);
    // Save the projection with projected pads
    saveProjectedPads(cluster.getProjectedPads(), qProj);
    // Save the final groups (or cath-groups)
    savePadToCathGroup(cluster.getCathGroup(0), cluster.getMapCathPadToPad(0),
                       cluster.getNbrOfPads(0), cluster.getCathGroup(1),
                       cluster.getMapCathPadToPad(1), cluster.getNbrOfPads(1));
  }
  //
  // Sub-Cluster loop
  //
  int nbrOfProjPadsInTheGroup = 0;
  // Group allocations
  // ??? double *xyDxyGrp=nullptr;
  // ??? double *chGrp=nullptr;

  // EM allocations
  double *thetaEMFinal = 0;
  int finalK = 0;

  //
  // Find local maxima (seeds)
  //
  for (int g = 1; g <= nGroups; g++) {
    //
    //  Exctract the current group
    //
    if (ClusterConfig::groupsLog >= ClusterConfig::info) {
      printf("----------------\n");
      printf("Group %d/%d \n", g, nGroups);
      printf("----------------\n");
    }
    //
    // Number of seeds in this group
    int kEM;

    Cluster *subCluster = nullptr;
    // Extract the sub-cluster
    if (nGroups == 1) {
      subCluster = &cluster;
    } else {
      subCluster = new Cluster(cluster, g);
    }
    /* ???
    if (VERBOSE > 0) {
      printf("Start findLocalMaxWithPET\n");
    }
    */
    int nbrOfPadsInTheGroup =
        subCluster->getNbrOfPads(0) + subCluster->getNbrOfPads(1);
    // Allocation of possible nbr of seeds
    // (.i.e the nbr of Pads)
    double thetaL[nbrOfPadsInTheGroup * 5];

    if (INSPECTMODEL) {
      // Compute the local max with laplacian method
      // Used only to give insights of the cluster
      subCluster->buildProjectedGeometry(includeSinglePads);
      kEM =
          subCluster->findLocalMaxWithBothCathodes(thetaL, nbrOfPadsInTheGroup);
      double thetaExtra[kEM * 5];
      copyTheta(thetaL, nbrOfPadsInTheGroup, thetaExtra, kEM, kEM);
      saveThetaExtraInGroupList(thetaExtra, kEM);
      if (ClusterConfig::inspectModelLog > ClusterConfig::info) {
        printTheta("Theta findLocalMaxWithBothCathodes", thetaExtra, kEM);
      }
    }
    // Add null pads in the neighboring of the sub-cluster
    subCluster->addBoundaryPads();
    //
    // Search for seeds on this sub-cluster
    kEM = subCluster->findLocalMaxWithPET(thetaL, nbrOfPadsInTheGroup);
    if (kEM != 0) {
      double thetaEM[kEM * 5];
      copyTheta(thetaL, nbrOfPadsInTheGroup, thetaEM, kEM, kEM);

      if (VERBOSE > 0) {
        printf("Find %2d local maxima : \n", kEM);
        printTheta("  ThetaEM", thetaEM, kEM);
      }

      //
      // EM
      //
      // ??? double *projXc = getX( xyDxyGrp, nbrOfProjPadsInTheGroup);
      // ??? double *projYc = getY( xyDxyGrp, nbrOfProjPadsInTheGroup);
      /*
      if (VERBOSE > 1) {
        printf("projPads in the group=%d, Xmin/max = %f %f, min/max = %f %f\n",
      g, vectorMin( projXc, nbrOfProjPadsInTheGroup),vectorMax( projXc,
      nbrOfProjPadsInTheGroup), vectorMin( projYc, nbrOfProjPadsInTheGroup),
      vectorMax( projYc, nbrOfProjPadsInTheGroup));
      }
      */
      if (INSPECTMODEL) {
        // Save the seed founds by the EM algorithm
        saveThetaEMInGroupList(thetaEM, kEM);
      }
      //
      //
      //
      // Perform the fitting if the sub-cluster g
      // is well separated at the 2 planes level (cath0, cath1)
      // If not the EM result is kept
      //
      DataBlock_t newSeeds = subCluster->fit(thetaEM, kEM);
      finalK = newSeeds.first;
      printTheta("- End of fitting ???", newSeeds.second, finalK);

      nbrOfHits += finalK;
      //
      // Store result (hits/seeds)
      seedList.push_back(newSeeds);
      if (INSPECTMODEL) {
        saveThetaFitInGroupList(newSeeds.second, newSeeds.first);
      }
    } else {
      // No EM seeds
      finalK = kEM;
      nbrOfHits += finalK;
      // Save the result of EM
      DataBlock_t newSeeds = std::make_pair(finalK, nullptr);
      seedList.push_back(newSeeds);
    }
    /*
    if (INSPECTMODEL ) {
      // ??? printf("??????????????? xyDxyGrp %p\n", xyDxyGrp);
      savePadsOfSubCluster( xyDxyGrp, chGrp, nbrOfProjPadsInTheGroup);
    }
     */
    // Release pointer for group
    // deleteDouble( xyDxyGrp );
    // deleteDouble( chGrp );
    if (nGroups > 1) {
      delete subCluster;
    }
  } // next group

  // Finalise inspectModel
  if (INSPECTMODEL)
    finalizeInspectModel();

  if (nNewPads) {
    delete[] xyDxyi__;
    delete[] cathi__;
    delete[] saturated__;
    delete[] zi__;
  }
  return nbrOfHits;
}
