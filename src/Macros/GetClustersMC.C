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

void  initDumpFiles ( std::ofstream *dumpFiles ) {
   dumpFiles[0].open("MCTrackRefDump.dat", ofstream::out | ios_base::binary );
   // GG Inv dumpFiles[0].open("MCClusterDump.dat", ofstream::out | ofstream::app | ios_base::binary );

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

bool LoadOCDB(int runNumber);
void SetMCLabelByPosition(AliMUONVTrackStore* mcTrackStore, AliMUONVClusterStore* clusterStore);
AliMUONVCluster* findReconstructedCluster(AliMUONVCluster* mcCluster, int mcLabel, AliMUONVClusterStore* clusterStore);

TString mcOCDB = "local://$ALIROOT_OCDB_ROOT/OCDB";
//TString mcOCDB = "OCDBsim.root";
TString recoOCDB = "local://$ALIROOT_OCDB_ROOT/OCDB";
//TString recoOCDB = "OCDBrec.root";

AliMUONGeometryTransformer geoTransformerMC;
AliMUONGeometryTransformer geoTransformerRec;

//------------------------------------------------------------------
void GetClustersMC(bool fromMuons = false, bool fromReconstructibleTracks = false, bool matchByPosition = false,
                       TString mcPath = "./generated", TString clusterFileName = "", TString digitFileName = "")
{
  /// Compare the MC trackRefs with the reconstructed clusters
  /// - fromMuons: select trackRefs from muon tracks
  /// - fromReconstructibleTracks: select trackRefs from reconstructible tracks
  /// - matchByPosition: reset the reconstructed cluster MC label to point to the closest trackRef
  /// - mcPath: relative path to kinematics and trackRefs files
  /// - clusterFileName: file containing the merged TreeR (use regular MUON.RecPoints.root file otherwise)
  /// - digitFileName: file containing the merged TreeD (use regular MUON.Digits.root file otherwise)

  // GG Init/open dump files
  std::ofstream dumpFiles[4];
  initDumpFiles( dumpFiles );

  // prepare to read MC clusters from trackRefs
  AliMCEventHandler mcEventHandler;
  mcEventHandler.SetInputPath(mcPath.Data());
  mcEventHandler.InitIO("");

  // prepare to read reconstructed clusters from RecPoints
  AliRunLoader* rl = AliRunLoader::Open("galice.root", "MUONLoader");
  AliLoader* muonLoader(nullptr);
  TTree* treeR(nullptr);
  AliMUONVClusterStore* clusterStore(nullptr);
  if (clusterFileName.IsNull()) {
    muonLoader = rl->GetDetectorLoader("MUON");
    if (muonLoader->LoadRecPoints("READ") != 0) {
      cout << "unable to load clusters" << endl;
      return;
    }
    treeR = muonLoader->TreeR();
    clusterStore = AliMUONVClusterStore::Create(*treeR);
  } else {
    TFile* inFile = TFile::Open(clusterFileName.Data());
    if (!inFile || !inFile->IsOpen()) {
      cout << "unable to open cluster file" << endl;
      return;
    }
    treeR = static_cast<TTree*>(inFile->Get("TreeR"));
    clusterStore = AliMUONVClusterStore::Create(*treeR);
    clusterStore->Connect(*treeR);
  }

  // prepare to read reconstructed digits
  TTree* treeD(nullptr);
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

  // get the run number
  rl->LoadHeader();
  int runNumber = rl->GetHeader()->GetRun();

  // load OCDB objects
  if (!LoadOCDB(runNumber)) {
    return;
  }

  // create output file and histograms
  /* GG
  TFile* histoFile = new TFile("residuals.root", "RECREATE");
  TH1F* hResidualXInCh[AliMUONConstants::NTrackingCh()];
  TH1F* hResidualYInCh[AliMUONConstants::NTrackingCh()];
  for (Int_t i = 0; i < AliMUONConstants::NTrackingCh(); i++) {
    hResidualXInCh[i] = new TH1F(Form("hResidualXInCh%d", i + 1), Form("cluster-track residual-X distribution in chamber %d;#Delta_{X} (cm)", i + 1), 4000, -2., 2.);
    hResidualYInCh[i] = new TH1F(Form("hResidualYInCh%d", i + 1), Form("cluster-track residual-Y distribution in chamber %d;#Delta_{Y} (cm)", i + 1), 4000, -2., 2.);
  }
  */
  int nEvents = rl->GetNumberOfEvents();
  for (int event = 0; event < nEvents; ++event) {

    if ((event + 1) % 100 == 0) {
      cout << "Event processing... " << event + 1 << flush;
    }

    // get MC clusters
    if (!mcEventHandler.BeginEvent(event)) {
      cout << endl << "unable to read MC objects" << endl;
      return;
    }
    AliMUONRecoCheck rc(nullptr, &mcEventHandler);
    AliMUONVTrackStore* mcTrackStore = fromReconstructibleTracks ? rc.ReconstructibleTracks(event, 0x1F, true, true) : rc.TrackRefs(event);

    // get reconstructed clusters
    bool eventLoaded(false);
    if (clusterFileName.IsNull()) {
      if (rl->GetEvent(event) != 0) {
        cout << endl << "unable to load event" << endl;
        return;
      }
      eventLoaded = true;
      treeR = muonLoader->TreeR();
      clusterStore->Connect(*treeR);
      treeR->GetEvent(0);
    } else {
      if (treeR->GetEvent(event) <= 0) {
        cout << endl << "unable to load event" << endl;
        return;
      }
    }

    // get reconstructed digits if any
    if (digitStore) {
      if (digitFileName.IsNull()) {
        if (!eventLoaded && rl->GetEvent(event) != 0) {
          cout << endl << "unable to load event" << endl;
          return;
        }
        treeD = muonLoader->TreeD();
        digitStore->Connect(*treeD);
        treeD->GetEvent(0);
      } else {
        if (treeD->GetEvent(event) <= 0) {
          cout << endl << "unable to load event" << endl;
          return;
        }
      }
    }

    // reset the reconstructed cluster MC label to point to the closest trackRef
    if (matchByPosition) {
      SetMCLabelByPosition(mcTrackStore, clusterStore);
    }

    // loop over MC clusters in MC tracks
    TIter next(mcTrackStore->CreateIterator());
    AliMUONTrack* mcTrack(nullptr);
    while ((mcTrack = static_cast<AliMUONTrack*>(next()))) {

      // get the MC label of this track
      int mcLabel = mcTrack->GetUniqueID();
      int partCode = mcEventHandler.MCEvent()->GetTrack(mcLabel)->PdgCode();
      cout << "GG NEW MC track: label=" << mcLabel  
	   << ", part. code=" << partCode
	   << ", nbr of MCClusters=" << mcTrack->GetNClusters() << endl;

      // select muons
      if (fromMuons && TMath::Abs(mcEventHandler.MCEvent()->GetTrack(mcLabel)->PdgCode()) != 13) {
        continue;
      }
      // GG Track header
      Int_t nbrOfClusters = mcTrack->GetNClusters();
      if ( nbrOfClusters != 0 ) {
	Double_t xx[nbrOfClusters], yy[nbrOfClusters];
	Int_t chamberId[nbrOfClusters], detElemId[nbrOfClusters];
	Int_t trackHeader[] = { -1, event, mcLabel, partCode, 0, nbrOfClusters };
	dumpInt32( dumpFiles, 0, 6, trackHeader );
        for (int iCl = 0; iCl < mcTrack->GetNClusters(); ++iCl) {

	  // get the MC cluster
	  AliMUONVCluster* mcCluster = static_cast<AliMUONTrackParam*>(mcTrack->GetTrackParamAtCluster()->UncheckedAt(iCl))->GetClusterPtr();

	  // find the corresponding reconstructed cluster if any
	  AliMUONVCluster* recoCluster = findReconstructedCluster(mcCluster, mcLabel, clusterStore);
	  cout << "recoCluster " << recoCluster 
	       << ", mcCluster " << mcCluster
	       << ", mcLabel " << mcLabel
	       << ", clusterStore " << clusterStore
	       << endl;
	  if (!recoCluster) {
	    cout << "GG Not a reco cluster" << endl;
	    continue;
	  }

	  // compare their position in the local coordinate system
	  int chId = mcCluster->GetChamberId();
	  int deId = mcCluster->GetDetElemId();
	  chamberId[iCl] = chId; detElemId[iCl] = deId;
	  double xMC(0.), yMC(0.), xRec(0.), yRec(0.), z(0.);
	  geoTransformerMC.Global2Local(deId, mcCluster->GetX(), mcCluster->GetY(), mcCluster->GetZ(), xMC, yMC, z);
	  // GG geoTransformerRec.Global2Local(deId, recoCluster->GetX(), recoCluster->GetY(), recoCluster->GetZ(), xRec, yRec, z);
	  // hResidualXInCh[chId]->Fill(xRec - xMC);
	  // hResidualYInCh[chId]->Fill(yRec - yMC);

          cout << "  GG hit : event=" << event << ", xyMC=" << xMC << ", " << yMC 
	     << endl;
          cout << "  GG hit : event=" << event << ", chId=" << chamberId[iCl] << ", DEId" << detElemId[iCl] 
	     << endl;
	  xx[iCl] = xMC; yy[iCl] = yMC;

	  //
          // Pad Storage
	  //
	  // get corresponding digits information if any
	  Int_t nPads = 0;
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
	      
	    /* GG get pad feature template
            // fill pad info
            padInfo.SetPadId(digit->GetUniqueID());
            padInfo.SetPadPlaneType(planeType);
            padInfo.SetPadXY(pad.GetPositionX(), pad.GetPositionY());
            padInfo.SetPadDimXY(pad.GetDimensionX(), pad.GetDimensionY());
            padInfo.SetPadCharge((double)digit->Charge());
            padInfo.SetPadADC(digit->ADC());
            padInfo.SetSaturated(digit->IsSaturated());
            padInfo.SetCalibrated(digit->IsCalibrated());
*/
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
	}
	dumpFloat64( dumpFiles, 0, nbrOfClusters, xx);
	dumpFloat64( dumpFiles, 0, nbrOfClusters, yy);
	dumpInt32( dumpFiles, 0, nbrOfClusters, chamberId);
	dumpInt32( dumpFiles, 0, nbrOfClusters, detElemId);
      }
    }

    mcEventHandler.FinishEvent();
    clusterStore->Clear();
    if (digitStore) {
      digitStore->Clear();
    }
  }
  cout << "\rEvent processing... " << nEvents << " done" << endl;

  // save histograms
  //histoFile->Write();
  //histoFile->Close();
  // GG
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
      // GG if (cluster->GetDetElemId() == deId && cluster->GetMCLabel() == mcLabel) {
      if (cluster->GetDetElemId() == deId) {
      double dx = cluster->GetX() - mcCluster->GetX();
      double dy = cluster->GetY() - mcCluster->GetY();
      double dist2 = dx * dx + dy * dy;
      if (dist2 < minDist2) {
        recoCluster = cluster;
        minDist2 = dist2;
      }
    }
  }

  return recoCluster;
}
