#include <iostream>

#include <TGeoManager.h>
#include <TString.h>
#include <TMath.h>

#include "AliMCEventHandler.h"
#include "AliMCEvent.h"
#include "AliVParticle.h"
#include "AliRunLoader.h"
#include "AliLoader.h"
#include "AliCDBManager.h"
#include "AliGeomManager.h"

#include "AliMpVSegmentation.h"
#include "AliMpSegmentation.h"
#include "AliMpPad.h"

#include "AliMUONGeometryTransformer.h"
#include "AliMUONCDB.h"
#include "AliMUONRecoCheck.h"
#include "AliMUONVTrackStore.h"
#include "AliMUONTrack.h"
#include "AliMUONTrackParam.h"
#include "AliMUONVCluster.h"
#include "AliMUONVDigitStore.h"
#include "AliMUONVDigit.h"

using namespace std;

AliMUONGeometryTransformer* LoadGeometry(int runNumber, TString mcOCDB);

//------------------------------------------------------------------
void WriteMCInfo(TString mcPath = "./generated",
                 TString mcOCDB = "local://$ALIROOT_OCDB_ROOT/OCDB")
{
  /// Read the MC kinematics, trackRefs and digits and write them

  // prepare to read MC tracks and trackRefs
  AliMCEventHandler mcEventHandler;
  mcEventHandler.SetInputPath(mcPath.Data());
  if (!mcEventHandler.InitIO("")) {
    cout << "unable to load kinematics and trackRefs" << endl;
    return;
  }

  // prepare to read simulated digits
  AliRunLoader* rl = AliRunLoader::Open("galice.root", "MUONLoader");
  AliLoader* muonLoader = rl->GetDetectorLoader("MUON");
  muonLoader->SetDigitsFileName(mcPath + "/MUON.Digits.root");
  if (muonLoader->LoadDigits("READ") != 0) {
    cout << "unable to load digits" << endl;
    return;
  }
  TTree* treeD = muonLoader->TreeD();
  AliMUONVDigitStore* digitStore = AliMUONVDigitStore::Create(*treeD);

  // get the run number
  rl->LoadHeader();
  int runNumber = rl->GetHeader()->GetRun();

  // load the geometry (and the mapping) from the OCDB
  AliMUONGeometryTransformer* geoTransformerMC = LoadGeometry(runNumber, mcOCDB);
  if (!geoTransformerMC) {
    return;
  }

  // loop over events
  int nEvents = rl->GetNumberOfEvents();
  for (int event = 0; event < nEvents; ++event) {

    printf("\n--- processing event %d ---\n", event + 1);

    // get MC tracks and trackRefs
    if (!mcEventHandler.BeginEvent(event)) {
      cout << endl << "unable to read MC objects" << endl;
      return;
    }
    AliMUONRecoCheck rc(nullptr, &mcEventHandler);
    AliMUONVTrackStore* mcTrackStore = rc.TrackRefs(event);

    // loop over MC tracks
    TIter nextTrack(mcTrackStore->CreateIterator());
    AliMUONTrack* mcTrack(nullptr);
    while ((mcTrack = static_cast<AliMUONTrack*>(nextTrack()))) {

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

      // loop over MC trackRefs
      for (int iCl = 0; iCl < mcTrack->GetNClusters(); ++iCl) {

        // get the MC trackRef
        AliMUONVCluster* mcCluster = static_cast<AliMUONTrackParam*>(mcTrack->GetTrackParamAtCluster()->UncheckedAt(iCl))->GetClusterPtr();

        // get its position in the local coordinate system
        int deId = mcCluster->GetDetElemId();
        double xMC(0.), yMC(0.), zMC(0.);
        geoTransformerMC->Global2Local(deId, mcCluster->GetX(), mcCluster->GetY(), mcCluster->GetZ(), xMC, yMC, zMC);

        printf("\tMC trackRef on DE %d: x = %f, y = %f\n", deId, xMC, yMC);
      }
    }

    printf("\n---\n");

    // get simulated digits
    if (rl->GetEvent(event) != 0) {
      cout << endl << "unable to read MC objects" << endl;
      return;
    }
    treeD = muonLoader->TreeD();
    digitStore->Connect(*treeD);
    treeD->GetEvent(0);
    // get corresponding digits information if any
    Int_t nPads = 0, sumOfTracks = 0;
    AliMUONVDigit* digit(nullptr);
    // Extract dimensions
    {
      TIter next( digitStore->CreateTrackerIterator() );
      while ( ( digit = static_cast<AliMUONVDigit*>(next()) ) ) {
        nPads++;
        sumOfTracks += digit->Ntracks();
      }
    }
    cout << "Event " << event 
	 << ", start Pad part (nPads, sumOfTracks)= " << nPads 
	 << "," << sumOfTracks 
	 << endl;

    // loop over simulated digits
    TIter nextDigit(digitStore->CreateTrackerIterator());
    // GG AliMUONVDigit* digit(nullptr);
    digit = nullptr;
    while ((digit = static_cast<AliMUONVDigit*>(nextDigit()))) {

      // find the corresponding pad
      const AliMpVSegmentation* seg = AliMpSegmentation::Instance()->GetMpSegmentation(digit->DetElemId(), AliMp::GetCathodType(digit->Cathode()));
      AliMpPad pad = seg->PadByIndices(digit->PadX(), digit->PadY());


      printf("digit Id %d (DE %d): x = %f, y = %f, dx = %f, dy = %f, ADC = %d, N MC tracks = %d:\n",
             digit->GetUniqueID(), digit->DetElemId(),
             pad.GetPositionX(), pad.GetPositionY(), pad.GetDimensionX(), pad.GetDimensionY(),
             digit->ADC(), digit->Ntracks());

      // loop over contributing MC tracks
      for (int iTrack = 0; iTrack < digit->Ntracks(); ++iTrack) {
        printf("\tMC track Id %d: charge %f\n", digit->Track(iTrack), digit->TrackCharge(iTrack));
      }
    }

    // cleanup before reading next event
    mcEventHandler.FinishEvent();
    digitStore->Clear();
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
