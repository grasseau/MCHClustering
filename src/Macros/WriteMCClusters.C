#include <iostream>

#include <TGeoManager.h>
#include <TString.h>
#include <TMath.h>

#include "AliMCEventHandler.h"
#include "AliMCEvent.h"
#include "AliVParticle.h"
#include "AliRunLoader.h"
#include "AliCDBManager.h"
#include "AliGeomManager.h"

#include "AliMUONGeometryTransformer.h"
#include "AliMUONVTrackStore.h"
#include "AliMUONRecoCheck.h"
#include "AliMUONTrack.h"
#include "AliMUONTrackParam.h"
#include "AliMUONVCluster.h"
#include "AliMUONCDB.h"

using namespace std;

AliMUONGeometryTransformer* LoadGeometry(int runNumber, TString mcOCDB);

//------------------------------------------------------------------
void WriteMCClusters(TString mcPath = "./generated",
                     TString mcOCDB = "local://$ALIROOT_OCDB_ROOT/OCDB")
{
  /// Read the MC kinematics and trackRefs and write them

  // prepare to read MC tracks and clusters
  AliMCEventHandler mcEventHandler;
  mcEventHandler.SetInputPath(mcPath.Data());
  mcEventHandler.InitIO("");

  // get the run number
  AliRunLoader* rl = AliRunLoader::Open("galice.root", "MUONLoader");
  rl->LoadHeader();
  int runNumber = rl->GetHeader()->GetRun();

  // load the geometry from the OCDB
  AliMUONGeometryTransformer* geoTransformerMC = LoadGeometry(runNumber, mcOCDB);
  if (!geoTransformerMC) {
    return;
  }

  // loop over events
  int nEvents = rl->GetNumberOfEvents();
  for (int event = 0; event < nEvents; ++event) {

    printf("\n--- processing event %d ---\n", event + 1);

    // get MC tracks and clusters
    if (!mcEventHandler.BeginEvent(event)) {
      cout << endl << "unable to read MC objects" << endl;
      return;
    }
    AliMUONRecoCheck rc(nullptr, &mcEventHandler);
    AliMUONVTrackStore* mcTrackStore = rc.TrackRefs(event);

    // loop over MC tracks
    TIter next(mcTrackStore->CreateIterator());
    AliMUONTrack* mcTrack(nullptr);
    while ((mcTrack = static_cast<AliMUONTrack*>(next()))) {

      // get the MC label of this track
      int mcLabel = mcTrack->GetUniqueID();

      // get the corresponding particle
      AliVParticle* particle = mcEventHandler.MCEvent()->GetTrack(mcLabel);

      // check if it is a muon
      bool isMuon = (TMath::Abs(particle->PdgCode()) == 13);

      printf("\nparticle Id = %d: x = %f, y = %f, z = %f, px = %f, py = %f, pz = %f, muon: %s\n",
             mcLabel,
             particle->Xv(), particle->Yv(), particle->Zv(), particle->Px(), particle->Py(), particle->Pz(),
             isMuon ? "yes" : "no");

      // loop over MC clusters
      for (int iCl = 0; iCl < mcTrack->GetNClusters(); ++iCl) {

        // get the MC cluster
        AliMUONVCluster* mcCluster = static_cast<AliMUONTrackParam*>(mcTrack->GetTrackParamAtCluster()->UncheckedAt(iCl))->GetClusterPtr();

        // get its position in the local coordinate system
        int chId = mcCluster->GetChamberId();
        int deId = mcCluster->GetDetElemId();
        double xMC(0.), yMC(0.), zMC(0.);
        geoTransformerMC->Global2Local(deId, mcCluster->GetX(), mcCluster->GetY(), mcCluster->GetZ(), xMC, yMC, zMC);

        printf("\tMC cluster on DE %d: x = %f, y = %f\n", deId, xMC, yMC);
      }
    }

    mcEventHandler.FinishEvent();
  }
}

//------------------------------------------------------------------
AliMUONGeometryTransformer* LoadGeometry(int runNumber, TString mcOCDB)
{
  /// load the geometry from the OCDB

  // set MC OCDB location
  AliCDBManager* cdbm = AliCDBManager::Instance();
  if (mcOCDB.EndsWith(".root")) {
    cdbm->SetDefaultStorage("local:///dev/null");
    cdbm->SetSnapshotMode(mcOCDB.Data());
  } else {
    cdbm->SetDefaultStorage(mcOCDB.Data());
  }
  cdbm->SetRun(runNumber);

  // load the geometry
  //cdbm->SetSpecificStorage("MUON/Align/Data", "local://$ALIROOT_OCDB_ROOT/OCDB", -1, -1);
  AliGeomManager::LoadGeometry();
  if (!AliGeomManager::GetGeometry() || !AliGeomManager::ApplyAlignObjsFromCDB("MUON")) {
    return nullptr;
  }

  // get MC geometry transformer
  AliMUONGeometryTransformer* geoTransformerMC = new AliMUONGeometryTransformer();
  geoTransformerMC->LoadGeometryData();

  return geoTransformerMC;
}
