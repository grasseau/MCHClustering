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
// GG
# include <fstream>

using namespace std;

AliMUONGeometryTransformer* LoadGeometry(int runNumber, TString mcOCDB);
//
// GG I/O
//
void  initDumpFiles ( std::ofstream *dumpFiles ) {
   dumpFiles[0].open("MCDataDump.dat", ofstream::out | ofstream::app | ios_base::binary );

  // dumpFiles[1].open("c.dat", ofstream::out | ofstream::app | ios_base::binary );
  // dumpFiles[3].open("clusterServerDump.dat", ofstream::out | ofstream::app | ios_base::binary );
}
void closeDumpFiles ( std::ofstream *dumpFiles ) {
  dumpFiles[0].close();
}

void dumpFloat32( std::ofstream *dumpFiles, int ifile, long size, const Float_t * data) {
  dumpFiles[ifile].write( (char *) &size, sizeof(long));
  dumpFiles[ifile].write( (char *) data, sizeof(float)*size );
 
}

void dumpFloat64( std::ofstream *dumpFiles, int ifile, long size, const Double_t * data) {
  
  dumpFiles[ifile].write( (char *) & size, sizeof(long) );
  dumpFiles[ifile].write( (char *) data, sizeof(double)*size );
}

void dumpInt32( std::ofstream *dumpFiles, int ifile, long size, const Int_t * data) {
  
  dumpFiles[ifile].write( (char *) &size, sizeof(long));
  dumpFiles[ifile].write( (char *) data, sizeof(int)*size );
  
}

//------------------------------------------------------------------
void GetMCData(TString mcPath = "./generated",
                 TString mcOCDB = "local://$ALIROOT_OCDB_ROOT/OCDB")
{
  // GG Init/open dump files
  std::ofstream dumpFiles[4];
  initDumpFiles( dumpFiles );  
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

    // GG Allocate buffers
   // Get the number of Tracks
    Int_t nTracks = 0;
    {     
     TIter next( mcTrackStore->CreateIterator());
     AliMUONTrack* mcTrack(nullptr);
     while ((mcTrack = static_cast<AliMUONTrack*>(next()))) { nTracks++; }
    }
    cout << "Event=" << event << ", nTracks=" << nTracks << endl;
    Int_t trackListHeader[] = { -1, event, -1, -1, 0, nTracks };
    dumpInt32( dumpFiles, 0, 6, trackListHeader );
    
    // loop over MC tracks
    TIter nextTrack(mcTrackStore->CreateIterator());
    AliMUONTrack* mcTrack(nullptr);
    Int_t trackIdx = 0;
    while ((mcTrack = static_cast<AliMUONTrack*>(nextTrack()))) {

      // get the MC label of this track
      int mcLabel = mcTrack->GetUniqueID();

      // get the corresponding particle
      AliVParticle* particle = mcEventHandler.MCEvent()->GetTrack(mcLabel);

      // check if it is a muon
      // bool isMuon = (TMath::Abs(particle->PdgCode()) == 13);
      int partCode = particle->PdgCode();

      // printf("\nparticle Id = %d: x = %f, y = %f, z = %f, px = %f, py = %f, pz = %f, muon: %s\n",
      //       mcLabel,
      //       particle->Xv(), particle->Yv(), particle->Zv(), particle->Px(), particle->Py(), particle->Pz(),
      //       isMuon ? "yes" : "no");
      
      // GG MC Track / MC "Clusters"
      Int_t nbrOfMCClusters = mcTrack->GetNClusters();
      Int_t trackHeader[] = { -1, trackIdx, mcLabel, partCode, 0, nbrOfMCClusters };
      dumpInt32( dumpFiles, 0, 6, trackHeader );
      Double_t partInfo[] = {particle->Xv(), particle->Yv(), particle->Zv(), particle->Px(), particle->Py(), particle->Pz()};
      dumpFloat64( dumpFiles, 0, 6, partInfo );
      //
      cout << "GG NEW MC track: label=" << mcLabel  
	   << ", part. code=" << partCode
	   << ", nbr of MCClusters=" << mcTrack->GetNClusters() << endl;
        if ( nbrOfMCClusters != 0 ) {
          Double_t xx[nbrOfMCClusters], yy[nbrOfMCClusters];
          Int_t chamberId[nbrOfMCClusters], detElemId[nbrOfMCClusters];      
          // loop over MC trackRefs
          for (int iCl = 0; iCl < mcTrack->GetNClusters(); ++iCl) {
            // get the MC trackRef
            AliMUONVCluster* mcCluster = static_cast<AliMUONTrackParam*>(mcTrack->GetTrackParamAtCluster()->UncheckedAt(iCl))->GetClusterPtr();

            // get its position in the local coordinate system
            int chId = mcCluster->GetChamberId();
            int deId = mcCluster->GetDetElemId();
            chamberId[iCl] = chId; detElemId[iCl] = deId;
            double xMC(0.), yMC(0.), zMC(0.);
            geoTransformerMC->Global2Local(deId, mcCluster->GetX(), mcCluster->GetY(), mcCluster->GetZ(), xMC, yMC, zMC);
            xx[iCl] = xMC; yy[iCl] = yMC;
            // printf("\tMC trackRef on DE %d: x = %f, y = %f\n", deId, xMC, yMC);
          }
          dumpFloat64( dumpFiles, 0, nbrOfMCClusters, xx);
          dumpFloat64( dumpFiles, 0, nbrOfMCClusters, yy);
          dumpInt32( dumpFiles, 0, nbrOfMCClusters, chamberId);
          dumpInt32( dumpFiles, 0, nbrOfMCClusters, detElemId); 
        }
	trackIdx++;
    }
    cout << "Double_t :" << sizeof(Double_t) << endl;
    cout << "Int_t :" << sizeof(Int_t) << endl;
    cout << "int :" << sizeof(int) << endl;
    
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
    // Extract dimensions
    {
      AliMUONVDigit* digit(nullptr);
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
    // Header and allocations
    Int_t padHeader[] = { -1, event, -1, -1, 0, nPads };
    dumpInt32( dumpFiles, 0, 6, padHeader );
    if ( nPads != 0 ) {
        Double_t xPad[ nPads ], yPad[ nPads ];
        Double_t dxPad[ nPads ], dyPad[ nPads ];
        Int_t DEId[ nPads ], cath[ nPads ];
        Int_t padId[ nPads ], padADC[ nPads ];
        Int_t hit[ nPads ];
        Int_t isSaturated[ nPads ], isCalibrated[ nPads ];
        Int_t nTracks[ nPads ];
        Double_t trackCharges[sumOfTracks];
        Int_t    trackId[sumOfTracks];
        Int_t i = 0, jaggedIdx =0;
        // loop over simulated digits
        TIter nextDigit(digitStore->CreateTrackerIterator());
        AliMUONVDigit* digit(nullptr);
        while ((digit = static_cast<AliMUONVDigit*>(nextDigit()))) {

            // find the corresponding pad
            const AliMpVSegmentation* seg = AliMpSegmentation::Instance()->GetMpSegmentation(digit->DetElemId(), AliMp::GetCathodType(digit->Cathode()));
            AliMpPad pad = seg->PadByIndices(digit->PadX(), digit->PadY());


            // printf("digit Id %d (DE %d): x = %f, y = %f, dx = %f, dy = %f, ADC = %d, N MC tracks = %d:\n",
            // digit->GetUniqueID(), digit->DetElemId(),
            // pad.GetPositionX(), pad.GetPositionY(), pad.GetDimensionX(), pad.GetDimensionY(),
            // digit->ADC(), digit->Ntracks());

            // loop over contributing MC tracks
            // for (int iTrack = 0; iTrack < digit->Ntracks(); ++iTrack) {
            // printf("\tMC track Id %d: charge %f\n", digit->Track(iTrack), digit->TrackCharge(iTrack));
            
            DEId[i] = digit->DetElemId();
            xPad[i]  = pad.GetPositionX(); yPad[i] = pad.GetPositionY();
            dxPad[i] = pad.GetDimensionX(); dyPad[i] = pad.GetDimensionY();
            //
            cath[i] = digit->Cathode(); 
            padId[i] = pad.GetUniqueID();
            padADC[i] = digit->ADC();
            isSaturated[i] = digit->IsSaturated();
            isCalibrated[i] = digit->IsCalibrated();
            nTracks[i] = digit->Ntracks();
            hit[i] = digit->Hit();
            // Jagged Array indexed on nTracks[i]
            for (int k=0; k< nTracks[i]; k++, jaggedIdx++) {
                trackCharges[jaggedIdx] = digit->TrackCharge(k);
                trackId    [jaggedIdx] = digit->Track(k);
            }
            i++;
        }
      	cout << " Store Digit/pads , ev= " << event << ", ipad = " << i << "," << jaggedIdx << endl;
        dumpFloat64( dumpFiles, 0, nPads, xPad);
        dumpFloat64( dumpFiles, 0, nPads, yPad);
        dumpFloat64( dumpFiles, 0, nPads, dxPad);
        dumpFloat64( dumpFiles, 0, nPads, dyPad);
        dumpInt32( dumpFiles, 0, nPads, padId);
        dumpInt32( dumpFiles, 0, nPads, DEId);            
        dumpInt32( dumpFiles, 0, nPads, cath);
        dumpInt32( dumpFiles, 0, nPads, padADC);
        dumpInt32( dumpFiles, 0, nPads, hit);
        dumpInt32( dumpFiles, 0, nPads, isSaturated);
        dumpInt32( dumpFiles, 0, nPads, isCalibrated);
        dumpInt32( dumpFiles, 0, nPads, nTracks);
        /// Jagged indexed on nTracks
        dumpFloat64( dumpFiles, 0, sumOfTracks, trackCharges);
        dumpInt32( dumpFiles, 0, sumOfTracks, trackId);
    }
    // cleanup before reading next event
    mcEventHandler.FinishEvent();
    digitStore->Clear();
  }    
  closeDumpFiles( dumpFiles );
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
