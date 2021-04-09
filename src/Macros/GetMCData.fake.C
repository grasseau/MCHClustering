#include <iostream>
#include <limits>

#include <TTree.h>
#include <TGeoManager.h>
#include <TFile.h>
#include <TH1F.h>
#include <TString.h>
#include <TMath.h>

#include "AliMCEventHandler.h"
#include "AliMCEvent.h"
#include "AliRunLoader.h"
#include "AliLoader.h"
#include "AliCDBManager.h"
#include "AliGeomManager.h"

#include "AliMpVSegmentation.h"
#include "AliMpSegmentation.h"
#include "AliMpPad.h"

#include "AliMUONGeometryTransformer.h"
#include "AliMUONVDigitStore.h"
#include "AliMUONVClusterStore.h"
#include "AliMUONVTrackStore.h"
#include "AliMUONRecoCheck.h"
#include "AliMUONConstants.h"
#include "AliMUONTrack.h"
#include "AliMUONTrackParam.h"
#include "AliMUONVCluster.h"
#include "AliMUONVDigit.h"
#include "AliMUONCDB.h"
// GG
# include <fstream>

using namespace std;

//GG
// GG std::ofstream dumpFiles[4];
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
AliMUONGeometryTransformer* LoadGeometry(int runNumber, TString mcOCDB);

bool LoadOCDB(int runNumber);
// void SetMCLabelByPosition(AliMUONVTrackStore* mcTrackStore, AliMUONVClusterStore* clusterStore);
// AliMUONVCluster* findReconstructedCluster(AliMUONVCluster* mcCluster, int mcLabel, AliMUONVClusterStore* clusterStore);

TString mcOCDB = "local://$ALIROOT_OCDB_ROOT/OCDB";
//TString mcOCDB = "OCDBsim.root";
TString recoOCDB = "local://$ALIROOT_OCDB_ROOT/OCDB";
//TString recoOCDB = "OCDBrec.root";

AliMUONGeometryTransformer geoTransformerMC;
AliMUONGeometryTransformer geoTransformerRec;

//------------------------------------------------------------------
void GetMCData(bool fromMuons = false, bool fromReconstructibleTracks = false, bool matchByPosition = false,
                       TString mcPath = "./generated", TString digitFileName = "") {
  /// Compare the MC trackRefs with the reconstructed clusters
  /// - fromMuons: select trackRefs from muon tracks
  /// - fromReconstructibleTracks: select trackRefs from reconstructible tracks
  /// - matchByPosition: reset the reconstructed cluster MC label to point to the closest trackRef
  /// - mcPath: relative path to kinematics and trackRefs files
  /// - digitFileName: file containing the merged TreeD (use regular MUON.Digits.root file otherwise)
    
  // GG Init/open dump files
  std::ofstream dumpFiles[4];
  initDumpFiles( dumpFiles );
  
  /*
  // prepare to read MC clusters from trackRefs
  AliMCEventHandler mcEventHandler;
  mcEventHandler.SetInputPath(mcPath.Data());
  mcEventHandler.InitIO("");

  // GG
  cout << "GG: In CompareClustersMC" << endl;

  // prepare to read reconstructed clusters from RecPoints
  AliRunLoader* rl = AliRunLoader::Open("galice.root", "MUONLoader");
  TTree* treeR(nullptr);
  AliMUONVClusterStore* clusterStore(nullptr);
  */
  // GG Removed
//  if (clusterFileName.IsNull()) {
//    muonLoader = rl->GetDetectorLoader("MUON");
//    muonLoader->LoadRecPoints("READ");
//    if (rl->GetEvent(0) != 0) {
//      cout << "unable to load event" << endl;
//      return;
//    }
//    treeR = muonLoader->TreeR();
//    clusterStore = AliMUONVClusterStore::Create(*treeR);
//  } else {
//    TFile* inFile = TFile::Open(clusterFileName.Data());
//    if (!inFile || !inFile->IsOpen()) {
//      cout << "unable to open cluster file" << endl;
//      return;
//    }
//    treeR = static_cast<TTree*>(inFile->Get("TreeR"));
//    clusterStore = AliMUONVClusterStore::Create(*treeR);
//    clusterStore->Connect(*treeR);
//  }
//
  /*
  //
  // GG Added (digits)
  // prepare to read reconstructed digits
  AliLoader* muonLoader = rl->GetDetectorLoader("MUON");
  muonLoader->SetDigitsFileName(mcPath + "/MUON.Digits.root");
  if (muonLoader->LoadDigits("READ") != 0) {
    cout << "unable to load digits" << endl;
    return;
  }
  TTree* treeD = muonLoader->TreeD();
  AliMUONVDigitStore* digitStore = AliMUONVDigitStore::Create(*treeD);
  */
  /*
  AliMUONVDigitStore* digitStore(nullptr);
  if (digitFileName.IsNull()) {
    if (muonLoader->LoadDigits("READ") == 0) {
      treeD = muonLoader->TreeD();
      digitStore = AliMUONVDigitStore::Create(*treeD);
    }
  } else {
    TFile* inFile = TFile::Open(digitFileName.Data());
    if (!inFile || !inFile->IsOpen()) {
      cout << "unable to open digit file" << endl;
      return;
    }
    treeD = static_cast<TTree*>(inFile->Get("TreeD"));
    digitStore = AliMUONVDigitStore::Create(*treeD);
    digitStore->Connect(*treeD);
  }
  */
  /*  
  // get the run number
  rl->LoadHeader();
  int runNumber = rl->GetHeader()->GetRun();

  // load OCDB objects
  if (!LoadOCDB(runNumber)) {
    return;
  }
  */

  // create output file and histograms
  /*
  TFile* histoFile = new TFile("residuals.root", "RECREATE");
  TH1F* hResidualXInCh[AliMUONConstants::NTrackingCh()];
  TH1F* hResidualYInCh[AliMUONConstants::NTrackingCh()];
  for (Int_t i = 0; i < AliMUONConstants::NTrackingCh(); i++) {
    hResidualXInCh[i] = new TH1F(Form("hResidualXInCh%d", i + 1), Form("cluster-track residual-X distribution in chamber %d;#Delta_{X} (cm)", i + 1), 4000, -2., 2.);
    hResidualYInCh[i] = new TH1F(Form("hResidualYInCh%d", i + 1), Form("cluster-track residual-Y distribution in chamber %d;#Delta_{Y} (cm)", i + 1), 4000, -2., 2.);
  }
  */
  // Get the total numer of clusters

  /*
  int nbrOfClusters = 0;
  {
    int nEvents = rl->GetNumberOfEvents();
    for (int event = 0; event < nEvents; ++event) {
      AliMUONRecoCheck rc(nullptr, &mcEventHandler);
      AliMUONVTrackStore* mcTrackStore = fromReconstructibleTracks ? rc.ReconstructibleTracks(event, 0x1F, true, true) : rc.TrackRefs(event);

      // loop over MC clusters in MC tracks
      TIter next(mcTrackStore->CreateIterator());
      AliMUONTrack* mcTrack(nullptr);
      while ((mcTrack = static_cast<AliMUONTrack*>(next()))) {
	  nbrOfClusters += mcTrack->GetNClusters();
      }
    }
  }
  cout << "GG Total number of MC Clusters=", nbrOfClusters;
  */
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
  muonLoader->SetDigitsFileName( mcPath + "/MUON.Digits.root");
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


  int nEvents = rl->GetNumberOfEvents();
  for (int event = 0; event < nEvents; ++event) {

    if ((event + 1) % 100 == 0) {
      cout << "\rEvent processing... " << event + 1 << flush;
    }

    // get MC clusters
    if (!mcEventHandler.BeginEvent(event)) {
      cout << endl << "unable to read MC objects" << endl;
      return;
    }
    AliMUONRecoCheck rc(nullptr, &mcEventHandler);
    /// GG AliMUONVTrackStore* mcTrackStore = fromReconstructibleTracks ? rc.ReconstructibleTracks(event, 0x1F, true, true) : rc.TrackRefs(event);
    // GG Force to get all TrackRefs
    AliMUONVTrackStore* mcTrackStore = rc.TrackRefs(event);

    // GG Not used here
    // get reconstructed clusters
//    if (clusterFileName.IsNull()) {
//      if (rl->GetEvent(event) != 0) {
//        cout << endl << "unable to load event" << endl;
//        return;
//      }
//      treeR = muonLoader->TreeR();
//      clusterStore->Connect(*treeR);
//      treeR->GetEvent(0);
//    } else {
//      if (treeR->GetEvent(event) <= 0) {
//        cout << endl << "unable to load event" << endl;
//        return;
//      }
//    }
    
//    // reset the reconstructed cluster MC label to point to the closest trackRef
//    if (matchByPosition) {
//      SetMCLabelByPosition(mcTrackStore, clusterStore);
//    }
    /*
    if (treeD->GetEvent(event) <= 0) {
          cout << endl << "unable to load event" << endl;
          return;
    }
    */

    //
    // loop over MC clusters in MC tracks
    //
   // Get the number of Tracks
    Int_t nTracks = 0;
    {     
     TIter next( mcTrackStore->CreateIterator());
     AliMUONTrack* mcTrack(nullptr);
     while ((mcTrack = static_cast<AliMUONTrack*>(next()))) { nTracks++; }
    }
    Int_t trackListHeader[] = { -1, event, -1, -1, 0, nTracks };
    dumpInt32( dumpFiles, 0, 6, trackListHeader );
    //
    TIter next( mcTrackStore->CreateIterator());
    AliMUONTrack* mcTrack(nullptr);
    Int_t trackIdx = 0;
    while ((mcTrack = static_cast<AliMUONTrack*>(next()))) {
        // get the MC label of this track
        int mcLabel = mcTrack->GetUniqueID();
        int partCode = mcEventHandler.MCEvent()->GetTrack(mcLabel)->PdgCode();
        cout << "GG NEW MC track: label=" << mcLabel  
	   << ", part. code=" << partCode
	   << ", nbr of MCClusters=" << mcTrack->GetNClusters() << endl;

        // GG MC Track / MC "Clusters"
        Int_t nbrOfMCClusters = mcTrack->GetNClusters();
        Int_t trackHeader[] = { -1, trackIdx, mcLabel, partCode, 0, nbrOfMCClusters };
        dumpInt32( dumpFiles, 0, 6, trackHeader );
        AliVParticle* particle = mcEventHandler.MCEvent()->GetTrack(mcLabel);
        Double_t partInfo[] = {particle->Xv(), particle->Yv(), particle->Zv(), particle->Px(), particle->Py(), particle->Pz()};
        dumpFloat64( dumpFiles, 0, 6, partInfo );
        if ( nbrOfMCClusters != 0 ) {
            Double_t xx[nbrOfMCClusters], yy[nbrOfMCClusters];
            Int_t chamberId[nbrOfMCClusters], detElemId[nbrOfMCClusters];
            for (int iCl = 0; iCl < nbrOfMCClusters; ++iCl) {
	        // cout << "GG MC cluster index " << iCl << endl;
                // get the MC cluster
                AliMUONVCluster* mcCluster = static_cast<AliMUONTrackParam*>(mcTrack->GetTrackParamAtCluster()->UncheckedAt(iCl))->GetClusterPtr();

                int chId = mcCluster->GetChamberId();
                int deId = mcCluster->GetDetElemId();
                chamberId[iCl] = chId; detElemId[iCl] = deId;
                double xMC(0.), yMC(0.), xRec(0.), yRec(0.), z(0.);
                geoTransformerMC->Global2Local(deId, mcCluster->GetX(), mcCluster->GetY(), mcCluster->GetZ(), xMC, yMC, z);
                // GG x[iCl] = xMC; y[iCl] = yMC;
                // GG geoTransformerRec.Global2Local(deId, recoCluster->GetX(), recoCluster->GetY(), recoCluster->GetZ(), xRec, yRec, z);
                // cout << "  GG hit : event=" << event << ", xyMC=" << xMC << ", " << yMC 
                //     << endl;
                // cout << "  GG hit : event=" << event << ", chId=" << chamberId[iCl] << ", DEId" << detElemId[iCl] 
                //     << endl;
                xx[iCl] = xMC; yy[iCl] = yMC;

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
        
    //
    // Contributions of MC Particles on pads
    //
    /// Change event ????
    /*
    treeD = muonLoader->TreeD();
    digitStore->Connect(*treeD);
    treeD->GetEvent(event);
    */
    // get the digits of current event
    /*
    TTree* treeD = muonLoader->TreeD();
    if (treeD) {
      digitStore = AliMUONVDigitStore::Create(*treeD);
	digitStore->Clear();
	digitStore->Connect(*treeD);
	treeD->GetEvent(0);
    } else {
      cout << "bad treeD" << endl;
      return;
    }
    */
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
    TIter next0( digitStore->CreateTrackerIterator() );
    while ( ( digit = static_cast<AliMUONVDigit*>(next0()) ) ) {
        nPads++;
        sumOfTracks += digit->Ntracks();
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
        TIter next( digitStore->CreateTrackerIterator() );
	digit = nullptr;
        Int_t i = 0, jaggedIdx =0;
        while (( digit = static_cast<AliMUONVDigit*>(next()) )) {
            // Pad position
            const AliMpVSegmentation* seg = AliMpSegmentation::Instance()->GetMpSegmentation(
                                                    digit->DetElemId(), AliMp::GetCathodType(digit->Cathode()));
            AliMpPad pad = seg->PadByIndices( digit->PadX(), digit->PadY());
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
	cout << " Digit/pads done, ev= " << event << ", ipad = " << i << "," << jaggedIdx << endl;
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
/*        
        if (digitStore) {
          nPads = recoCluster->GetNDigits();
          Double_t xPad[nPads], yPad[nPads];
          Double_t dxPad[nPads], dyPad[nPads];
          Double_t charge[nPads];
          //
          Int_t cath[nPads], padId[nPads];
          Int_t padADC[nPads], isSaturated[nPads], isCalibrated[nPads];
          // Header
          Int_t padHeader[] = { -1, event, mcLabel, partCode, iCl, nPads };
          dumpInt32( dumpFiles, 0, 6, padHeader);
          for (int i = 0; i < recoCluster->GetNDigits(); ++i) {
            AliMUONVDigit* digit = digitStore->FindObject(recoCluster->GetDigitId(i));
            if (!digit) {
              cout << endl << "missing digit "  << digitStore->GetSize() << endl;
              return;
            }
            // GG Debug 
            // digit->Print();

            // pad location
            const AliMpVSegmentation* seg = AliMpSegmentation::Instance()->GetMpSegmentation(digit->DetElemId(), AliMp::GetCathodType(digit->Cathode()));
            AliMpPad pad = seg->PadByIndices(digit->PadX(), digit->PadY());
            cerr << "pad" << pad << endl;
            cout << "NoDigit=" << i << ", X=" <<  pad.GetPositionX() << ", dy"
                 << pad.GetDimensionY() << ", c=" << digit->Charge() 
                 << ", cath=" << digit->Cathode() << endl;
            xPad[i] = pad.GetPositionX(); yPad[i] = pad.GetPositionY();
            dxPad[i] = pad.GetDimensionX(); dyPad[i] = pad.GetDimensionY();
            charge[i] = digit->Charge();
            cath[i] = digit->Cathode(); 
            padId[i] = pad.GetUniqueID();
            padADC[i] = digit->ADC();
            isSaturated[i] = digit->IsSaturated();
            isCalibrated[i] = digit->IsCalibrated();
            // cout << "padID size=" << sizeof( pad.GetUniqueID() ) 
            // << ", padADC size=" << sizeof( digit->ADC() ) 
            // << typeid( digit->ADC() ).name()
            // << endl;
	    }
	  dumpFloat64( dumpFiles, 0, nPads, xPad);
          dumpFloat64( dumpFiles, 0, nPads, yPad);
	  dumpFloat64( dumpFiles, 0, nPads, dxPad);
	  dumpFloat64( dumpFiles, 0, nPads, dyPad);
	  dumpFloat64( dumpFiles, 0, nPads, charge);
	  dumpInt32( dumpFiles, 0, nPads, cath);
	  dumpInt32( dumpFiles, 0, nPads, padId);
	  dumpInt32( dumpFiles, 0, nPads, padADC);
	  dumpInt32( dumpFiles, 0, nPads, isSaturated);
	  dumpInt32( dumpFiles, 0, nPads, isCalibrated);
	    
	} else {
	    // No pads
	    Int_t padHeader[] = { -1, event, mcLabel, partCode, iCl, nPads };
	    dumpInt32( dumpFiles, 0, 6, padHeader);
	}
*/        

  } // Event loop
  
  mcEventHandler.FinishEvent();
  // clusterStore->Clear();
  cout << "\rEvent processing... " << nEvents << " done" << endl;

  // save histograms
  // histoFile->Write();
  // histoFile->Close();
  closeDumpFiles( dumpFiles );
}

//------------------------------------------------------------------
bool LoadOCDB(int runNumber)
{
  /// load necessary objects from OCDB ending with MC geometry so that it
  /// can be used in AliMUONRecoCheck to select trackRefs on top of pads

  // set reco OCDB location
  AliCDBManager* cdbm = AliCDBManager::Instance();
  if (recoOCDB.EndsWith(".root")) {
    cdbm->SetDefaultStorage("local:///dev/null");
    cdbm->SetSnapshotMode(recoOCDB.Data());
  } else {
    cdbm->SetDefaultStorage(recoOCDB.Data());
  }
  cdbm->SetRun(runNumber);

  // get reco geometry transformer
  //cdbm->SetSpecificStorage("MUON/Align/Data", "local://$ALIROOT_OCDB_ROOT/OCDB", -1, -1);
  AliGeomManager::LoadGeometry();
  if (!AliGeomManager::GetGeometry() || !AliGeomManager::ApplyAlignObjsFromCDB("MUON")) {
    return false;
  }
  geoTransformerRec.LoadGeometryData();

  // set MC OCDB location
  cdbm->UnsetDefaultStorage();
  cdbm->UnsetSnapshotMode();
  if (mcOCDB.EndsWith(".root")) {
    cdbm->SetDefaultStorage("local:///dev/null");
    cdbm->SetSnapshotMode(mcOCDB.Data());
  } else {
    cdbm->SetDefaultStorage(mcOCDB.Data());
  }
  cdbm->SetRun(runNumber);

  // get MC geometry transformer
  //cdbm->SetSpecificStorage("MUON/Align/Data", "local://$ALIROOT_OCDB_ROOT/OCDB", -1, -1);
  AliGeomManager::GetGeometry()->UnlockGeometry();
  AliGeomManager::LoadGeometry();
  if (!AliGeomManager::GetGeometry() || !AliGeomManager::ApplyAlignObjsFromCDB("MUON")) {
    return false;
  }
  geoTransformerMC.LoadGeometryData();

  return true;
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

/*
//------------------------------------------------------------------
void SetMCLabelByPosition(AliMUONVTrackStore* mcTrackStore, AliMUONVClusterStore* clusterStore)
{
  /// set the MC label of reconstructed cluster to point to the closest trackRef

  TIter nextCl(clusterStore->CreateIterator());
  AliMUONVCluster* cluster(nullptr);
  while ((cluster = static_cast<AliMUONVCluster*>(nextCl()))) {

    int deId = cluster->GetDetElemId();
    double minDist2(std::numeric_limits<double>::max());
    int mcLabel(-1);

    TIter nextTr(mcTrackStore->CreateIterator());
    AliMUONTrack* mcTrack(nullptr);
    while ((mcTrack = static_cast<AliMUONTrack*>(nextTr()))) {
      for (int iCl = 0; iCl < mcTrack->GetNClusters(); ++iCl) {

        AliMUONVCluster* mcCluster = static_cast<AliMUONTrackParam*>(mcTrack->GetTrackParamAtCluster()->UncheckedAt(iCl))->GetClusterPtr();
        
        if (mcCluster->GetDetElemId() == deId) {
          double dx = cluster->GetX() - mcCluster->GetX();
          double dy = cluster->GetY() - mcCluster->GetY();
          double dist2 = dx * dx + dy * dy;
          if (dist2 < minDist2) {
            mcLabel = mcTrack->GetUniqueID();
            minDist2 = dist2;
          }
        }
      }
    }

    cluster->SetMCLabel(mcLabel);
  }
}

//------------------------------------------------------------------
AliMUONVCluster* findReconstructedCluster(AliMUONVCluster* mcCluster, int mcLabel, AliMUONVClusterStore* clusterStore)
{
  /// find the closest reconstructed cluster on the same DE with the same MC label as the trackRef

  AliMUONVCluster *recoCluster(nullptr);
  int chId = mcCluster->GetChamberId();
  int deId = mcCluster->GetDetElemId();
  double minDist2(std::numeric_limits<double>::max());

  TIter nextInCh(clusterStore->CreateChamberIterator(chId, chId));
  AliMUONVCluster* cluster(nullptr);
  while ((cluster = static_cast<AliMUONVCluster*>(nextInCh()))) {
    // GG cout << "GG recoClusterDetId=" << cluster->GetDetElemId() << ", MCdetID=" << deId 
    // GG << ", Reco cl. label=" << cluster->GetMCLabel() << ", MC cl. label=" << mcLabel
    // GG << endl;
  
    // GG if (cluster->GetDetElemId() == deId && cluster->GetMCLabel() == mcLabel) {
    if (cluster->GetDetElemId() == deId ) {
      double dx = cluster->GetX() - mcCluster->GetX();
      double dy = cluster->GetY() - mcCluster->GetY();
      double dist2 = dx * dx + dy * dy;
      // cout << "  GG same DetId =" << cluster->GetDetElemId() 
      // << ", MC cl. label=" << mcLabel
      // << endl;
      cout << "  GG reco cluster found in DetId=" << deId 
	   << ", dist.=" << TMath::Sqrt(dist2) << endl;
      if (dist2 < minDist2) {
        recoCluster = cluster;
        minDist2 = dist2;
      }
    }
  }

  return recoCluster;
}
*/
