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

#include "AliMUONGeometryTransformer.h"
#include "AliMUONVClusterStore.h"
#include "AliMUONVTrackStore.h"
#include "AliMUONRecoCheck.h"
#include "AliMUONConstants.h"
#include "AliMUONTrack.h"
#include "AliMUONTrackParam.h"
#include "AliMUONVCluster.h"
#include "AliMUONCDB.h"
// GG
# include <fstream>

using namespace std;

//GG
// GG std::ofstream dumpFiles[4];
//
void  initDumpFiles ( std::ofstream *dumpFiles ) {
   dumpFiles[0].open("MCClusterDump.dat", ofstream::out | ofstream::app | ios_base::binary );

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
void CompareClustersMC(bool fromMuons = false, bool fromReconstructibleTracks = false, bool matchByPosition = false,
                       TString mcPath = "./generated", TString clusterFileName = "")
{
  /// Compare the MC trackRefs with the reconstructed clusters
  /// - fromMuons: select trackRefs from muon tracks
  /// - fromReconstructibleTracks: select trackRefs from reconstructible tracks
  /// - matchByPosition: reset the reconstructed cluster MC label to point to the closest trackRef
  /// - mcPath: relative path to kinematics and trackRefs files
  /// - clusterFileName: file containing the merged TreeR (use regular MUON.RecPoints.root file otherwise)

  // GG Init/open dump files
  std::ofstream dumpFiles[4];
  initDumpFiles( dumpFiles );
 
  // prepare to read MC clusters from trackRefs
  AliMCEventHandler mcEventHandler;
  mcEventHandler.SetInputPath(mcPath.Data());
  mcEventHandler.InitIO("");

  // GG
  cout << "GG: In CompareClustersMC" << endl;

  // prepare to read reconstructed clusters from RecPoints
  AliRunLoader* rl = AliRunLoader::Open("galice.root", "MUONLoader");
  AliLoader* muonLoader(nullptr);
  TTree* treeR(nullptr);
  AliMUONVClusterStore* clusterStore(nullptr);
  if (clusterFileName.IsNull()) {
    muonLoader = rl->GetDetectorLoader("MUON");
    muonLoader->LoadRecPoints("READ");
    if (rl->GetEvent(0) != 0) {
      cout << "unable to load event" << endl;
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

  // get the run number
  rl->LoadHeader();
  int runNumber = rl->GetHeader()->GetRun();

  // load OCDB objects
  if (!LoadOCDB(runNumber)) {
    return;
  }

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
    AliMUONVTrackStore* mcTrackStore = fromReconstructibleTracks ? rc.ReconstructibleTracks(event, 0x1F, true, true) : rc.TrackRefs(event);

    // get reconstructed clusters
    if (clusterFileName.IsNull()) {
      if (rl->GetEvent(event) != 0) {
        cout << endl << "unable to load event" << endl;
        return;
      }
      treeR = muonLoader->TreeR();
      clusterStore->Connect(*treeR);
      treeR->GetEvent(0);
    } else {
      if (treeR->GetEvent(event) <= 0) {
        cout << endl << "unable to load event" << endl;
        return;
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
	Int_t trackInfo[] = { -1, event, mcLabel, partCode, 0, nbrOfClusters };
	dumpInt32( dumpFiles, 0, 6, trackInfo );
	for (int iCl = 0; iCl < nbrOfClusters; ++iCl) {
	  cout << "GG MC cluster id " << iCl << endl;
          // get the MC cluster
          AliMUONVCluster* mcCluster = static_cast<AliMUONTrackParam*>(mcTrack->GetTrackParamAtCluster()->UncheckedAt(iCl))->GetClusterPtr();

          // find the corresponding reconstructed cluster if any
          // GG AliMUONVCluster* recoCluster = findReconstructedCluster(mcCluster, mcLabel, clusterStore);
        // if (!recoCluster) {
	//  cout << "GG Not a reco cluster" << endl;
        //  continue;
        // }

          // compare their position in the local coordinate system
          int chId = mcCluster->GetChamberId();
          int deId = mcCluster->GetDetElemId();
    	  chamberId[iCl] = chId; detElemId[iCl] = deId;
          double xMC(0.), yMC(0.), xRec(0.), yRec(0.), z(0.);
          geoTransformerMC.Global2Local(deId, mcCluster->GetX(), mcCluster->GetY(), mcCluster->GetZ(), xMC, yMC, z);
	  // GG x[iCl] = xMC; y[iCl] = yMC;
          // GG geoTransformerRec.Global2Local(deId, recoCluster->GetX(), recoCluster->GetY(), recoCluster->GetZ(), xRec, yRec, z);
        // hResidualXInCh[chId]->Fill(xRec - xMC);
        // hResidualYInCh[chId]->Fill(yRec - yMC);
        // cout << "  GG hit : event=" << event << ", xyMC=" << xMC << ", " << yMC 
	//  << ", diff=" << xRec - xMC << ", " << yRec - yMC << endl;
          cout << "  GG hit : event=" << event << ", xyMC=" << xMC << ", " << yMC 
	     << endl;
          cout << "  GG hit : event=" << event << ", chId=" << chamberId[iCl] << ", DEId" << detElemId[iCl] 
	     << endl;
	  xx[iCl] = xMC; yy[iCl] = yMC;
	  
	}
	cout << "Double_t :" << sizeof(Double_t) << endl;
	cout << "Int_t :" << sizeof(Int_t) << endl;
	cout << "int :" << sizeof(int) << endl;

	dumpFloat64( dumpFiles, 0, nbrOfClusters, xx);
	dumpFloat64( dumpFiles, 0, nbrOfClusters, yy);
	dumpInt32( dumpFiles, 0, nbrOfClusters, chamberId);
	dumpInt32( dumpFiles, 0, nbrOfClusters, detElemId);

      }
    }

    mcEventHandler.FinishEvent();
    clusterStore->Clear();
  }
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
