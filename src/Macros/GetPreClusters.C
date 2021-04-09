#include <iostream>
#include <list>
#include <set>
#include <utility>

#include <TFile.h>
#include <TTree.h>
#include <TGeoManager.h>
#include <TString.h>
#include <TMath.h>

#include "AliRunLoader.h"
#include "AliLoader.h"
#include "AliHeader.h"
#include "AliCDBManager.h"
#include "AliGeomManager.h"

#include "AliMpConstants.h"
#include "AliMpVSegmentation.h"
#include "AliMpSegmentation.h"
#include "AliMpPad.h"

#include "AliMUONGeometryTransformer.h"
#include "AliMUONCDB.h"
#include "AliMUONVClusterStore.h"
#include "AliMUONVCluster.h"
#include "AliMUONVDigitStore.h"
#include "AliMUONVDigit.h"
// GG
# include <fstream>

using namespace std;
//
// GG I/O
//
void  initDumpFiles ( std::ofstream *dumpFiles ) {
   dumpFiles[0].open("RecoDataDump.dat", ofstream::out | ofstream::app | ios_base::binary );

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

struct PreCluster {
  // PreCluster() = default;
  PreCluster() : clusters(), digitIds() {};
  ///PreCluster(AliMUONVCluster* cluster, std::set<unsigned int>& digits) : clusters(), digitIds(std::move(digits)) { clusters.push_back(cluster); }
  PreCluster(AliMUONVCluster* cluster, std::set<unsigned int>& digits) : clusters(), digitIds(std::move(digits)) { clusters.push_back(cluster); }
  std::list<AliMUONVCluster*> clusters; // link to the clusters reconstructed from this precluster
  std::set<unsigned int> digitIds;          // list of Id of digits forming this precluster
};

AliMUONGeometryTransformer* LoadGeometry(int runNumber, TString recOCDB);
void LoadPreClusters(const AliMUONVClusterStore& preclusterStore, std::list<PreCluster>& preclusters);
void LinkClusters(const AliMUONVClusterStore& clusterStore, std::list<PreCluster>& preclusters);
PreCluster* FindPreCluster(const AliMUONVCluster& cluster, std::list<PreCluster>& preclusters);
void LoadClusters(const AliMUONVClusterStore& clusterStore, std::list<PreCluster>& preclusters);
PreCluster* FindPreCluster(const std::set<unsigned int>& digitIds, std::list<PreCluster>& preclusters);

//------------------------------------------------------------------
void GetPreClusters(TString preclusterFileName = "preclusters.root",
                  TString recOCDB = "local://$ALIROOT_OCDB_ROOT/OCDB")
{
  /// Read the reconstructed digits, preclusters and clusters and write them
  /// If preclusterFileName is provided, the preclusters are read from this file and the clusters are linked to them
  /// Otherwise, the preclusters are built from the digits attached to the clusters
  // GG Init/open dump files
  std::ofstream dumpFiles[4];
  initDumpFiles( dumpFiles );  
  
  // prepare to read reconstructed digits
  AliRunLoader* rl = AliRunLoader::Open("galice.root", "MUONLoader");
  AliLoader* muonLoader = rl->GetDetectorLoader("MUON");
  if (muonLoader->LoadDigits("READ") != 0) {
    cout << "unable to load digits" << endl;
    return;
  }
  AliMUONVDigitStore* digitStore = AliMUONVDigitStore::Create(*(muonLoader->TreeD()));

  // prepare to read reconstructed preclusters
  TTree* treePR(nullptr);
  AliMUONVClusterStore* preclusterStore(nullptr);
  if (!preclusterFileName.IsNull()) {
    TFile* inFile = TFile::Open(preclusterFileName.Data());
    if (!inFile || !inFile->IsOpen()) {
      cout << "unable to load preclusters" << endl;
      return;
    }
    treePR = static_cast<TTree*>(inFile->Get("TreeR"));
    preclusterStore = AliMUONVClusterStore::Create(*treePR);
    preclusterStore->Connect(*treePR);
  }
  std::list<PreCluster> preclusters{};

  // prepare to read reconstructed clusters
  if (muonLoader->LoadRecPoints("READ") != 0) {
    cout << "unable to load clusters" << endl;
    return;
  }
  AliMUONVClusterStore* clusterStore = AliMUONVClusterStore::Create(*(muonLoader->TreeR()));

  // get the run number
  rl->LoadHeader();
  int runNumber = rl->GetHeader()->GetRun();

  // load the geometry (and the mapping) from the OCDB
  AliMUONGeometryTransformer* geoTransformerRec = LoadGeometry(runNumber, recOCDB);
  if (!geoTransformerRec) {
    return;
  }

  // loop over events
  int nEvents = rl->GetNumberOfEvents();
  for (int event = 0; event < nEvents; ++event) {

    printf("\n--- processing event %d ---\n", event + 1);

    // get reconstructed digits and clusters
    if (rl->GetEvent(event) != 0) {
      cout << endl << "unable to read digits and clusters" << endl;
      return;
    }
    TTree* treeD = muonLoader->TreeD();
    digitStore->Connect(*treeD);
    treeD->GetEvent(0);
    TTree* treeR = muonLoader->TreeR();
    clusterStore->Connect(*treeR);
    treeR->GetEvent(0);

    // get reconstructed preclusters
    if (treePR && treePR->GetEvent(event) <= 0) {
      cout << endl << "unable to read preclusters" << endl;
      return;
    }

    if (preclusterStore) {
      
      // load the preclusters
      LoadPreClusters(*preclusterStore, preclusters);

      // link the clusters to these preclusters
      LinkClusters(*clusterStore, preclusters);

    } else {

      // load the clusters and create the corresponding preclusters from the associated digits
      LoadClusters(*clusterStore, preclusters);
    }
    // GG Allocate buffers
   // Get the number of Tracks
    Int_t nPreClusters = 0;
    for (const auto &precluster : preclusters) { nPreClusters++; }

    cout << "Event=" << event << ", nPreClusters=" << nPreClusters << endl;
    Int_t prClusterListHeader[] = { -1, event, -1, -1, 0, nPreClusters };
    dumpInt32( dumpFiles, 0, 6, prClusterListHeader );
    
    // loop over preclusters
    int iPreCluster(0);
    for (const auto &precluster : preclusters) {

      printf("\nprecluster %d contains %lu digits:\n", iPreCluster, precluster.digitIds.size());

      // Get the nbr of digits and allocate
      Int_t nPads=0;
      for (unsigned int digitId : precluster.digitIds) { nPads++;}
      Int_t padHeader[] = { -1, event, iPreCluster, -1, 0, nPads };
      dumpInt32( dumpFiles, 0, 6, padHeader );
      if ( nPads != 0 ) {
        Double_t xPad[ nPads ], yPad[ nPads ];
        Double_t dxPad[ nPads ], dyPad[ nPads ];
        Double_t charge[ nPads ];
        Int_t DEId[ nPads ], cath[ nPads ];
        Int_t padId[ nPads ], padADC[ nPads ];
        Int_t hit[ nPads ];
        Int_t isSaturated[ nPads ], isCalibrated[ nPads ];
        Int_t nTracks[ nPads ];
        // Double_t trackCharges[sumOfTracks];
        // Int_t    trackId[sumOfTracks];
        Int_t i = 0, jaggedIdx =0;      
        // loop over associated digits
        for (unsigned int digitId : precluster.digitIds) {
          // find the digit
          AliMUONVDigit* digit = digitStore->FindObject(digitId);
          if (!digit) {
            cout << endl << "missing digit" << endl;
            return;
          }
          // find the corresponding pad
          const AliMpVSegmentation* seg = AliMpSegmentation::Instance()->GetMpSegmentation(digit->DetElemId(), AliMp::GetCathodType(digit->Cathode()));
          AliMpPad pad = seg->PadByIndices(digit->PadX(), digit->PadY());
          bool isNonBending = (digit->ManuId() & AliMpConstants::ManuMask(AliMp::kNonBendingPlane));
          DEId[i] = digit->DetElemId();
          xPad[i]  = pad.GetPositionX(); yPad[i] = pad.GetPositionY();
          dxPad[i] = pad.GetDimensionX(); dyPad[i] = pad.GetDimensionY();
          //
          cath[i] = digit->Cathode(); 
          padId[i] = pad.GetUniqueID();
          padADC[i] = digit->ADC();
          isSaturated[i] = digit->IsSaturated();
          isCalibrated[i] = digit->IsCalibrated();
          charge[i] = digit->Charge();
          nTracks[i] = digit->Ntracks();
          hit[i] = digit->Hit();
            
          // printf("\tdigit Id %d (DE %d %s): x = %f, y = %f, dx = %f, dy = %f, ADC = %d, charge = %f, is saturated: %s\n",
          //     digit->GetUniqueID(), digit->DetElemId(), isNonBending ? "nb" : "b",
          //     pad.GetPositionX(), pad.GetPositionY(), pad.GetDimensionX(), pad.GetDimensionY(),
          //     digit->ADC(), digit->Charge(), digit->IsSaturated() ? "yes" : "no");
          i++;
        }
      	cout << " Store Digit/pads , ev= " << event << ", ipad = " << i << endl;
        dumpFloat64( dumpFiles, 0, nPads, xPad);
        dumpFloat64( dumpFiles, 0, nPads, yPad);
        dumpFloat64( dumpFiles, 0, nPads, dxPad);
        dumpFloat64( dumpFiles, 0, nPads, dyPad);
        dumpFloat64( dumpFiles, 0, nPads, charge);
        dumpInt32( dumpFiles, 0, nPads, padId);
        dumpInt32( dumpFiles, 0, nPads, DEId);            
        dumpInt32( dumpFiles, 0, nPads, cath);
        dumpInt32( dumpFiles, 0, nPads, padADC);
        dumpInt32( dumpFiles, 0, nPads, hit);
        dumpInt32( dumpFiles, 0, nPads, isSaturated);
        dumpInt32( dumpFiles, 0, nPads, isCalibrated);
        dumpInt32( dumpFiles, 0, nPads, nTracks);
        /// Jagged indexed on nTracks
        // dumpFloat64( dumpFiles, 0, sumOfTracks, trackCharges);
        // dumpInt32( dumpFiles, 0, sumOfTracks, trackId);
      }
      printf("and %lu associated clusters:\n", precluster.clusters.size());

      Int_t nbrOfRecoClusters = precluster.clusters.size();
      Int_t clusterHeader[] = { -1, event, iPreCluster, -1, -1, nbrOfRecoClusters };
      dumpInt32( dumpFiles, 0, 6, clusterHeader );
      if ( nbrOfRecoClusters != 0 ) {
        Double_t xx[nbrOfRecoClusters], yy[nbrOfRecoClusters];
        Int_t chamberId[nbrOfRecoClusters], detElemId[nbrOfRecoClusters];  
        Int_t rClusterId[nbrOfRecoClusters];
        // loop over associated clusters
        Int_t iCl = 0;
        for (const AliMUONVCluster* rCluster : precluster.clusters) {
          int chId = rCluster->GetChamberId();
          int deId = rCluster->GetDetElemId();
          rClusterId[iCl] = rCluster->GetUniqueID();
          chamberId[iCl] = chId; detElemId[iCl] = deId;
          // get its position in the local coordinate system
          double x(0.), y(0.), z(0.);
          geoTransformerRec->Global2Local(rCluster->GetDetElemId(), rCluster->GetX(), rCluster->GetY(), rCluster->GetZ(), x, y, z);
          xx[iCl] = x; yy[iCl] = y;
          printf("\tcluster Id %d: x = %f, y = %f\n", rCluster->GetUniqueID(), x, y);
          iCl++;
        }
        dumpFloat64( dumpFiles, 0, nbrOfRecoClusters, xx);
        dumpFloat64( dumpFiles, 0, nbrOfRecoClusters, yy);
        dumpInt32( dumpFiles, 0, nbrOfRecoClusters, rClusterId); 
        dumpInt32( dumpFiles, 0, nbrOfRecoClusters, chamberId);
        dumpInt32( dumpFiles, 0, nbrOfRecoClusters, detElemId); 
      }
      iPreCluster++;
    }

    // cleanup before reading next event
    digitStore->Clear();
    if (preclusterStore) {
      preclusterStore->Clear();
    }
    preclusters.clear();
    clusterStore->Clear();
  }
  closeDumpFiles( dumpFiles );
}

//------------------------------------------------------------------
AliMUONGeometryTransformer* LoadGeometry(int runNumber, TString recOCDB)
{
  /// load the geometry from the OCDB

  // set MC OCDB location
  AliCDBManager* cdbm = AliCDBManager::Instance();
  if (recOCDB.EndsWith(".root")) {
    cdbm->SetDefaultStorage("local:///dev/null");
    cdbm->SetSnapshotMode(recOCDB.Data());
  } else {
    cdbm->SetDefaultStorage(recOCDB.Data());
  }
  cdbm->SetRun(runNumber);

  // load the geometry
  //cdbm->SetSpecificStorage("MUON/Align/Data", "local://$ALIROOT_OCDB_ROOT/OCDB", -1, -1);
  AliGeomManager::LoadGeometry();
  if (!AliGeomManager::GetGeometry() || !AliGeomManager::ApplyAlignObjsFromCDB("MUON")) {
    return nullptr;
  }

  // get MC geometry transformer
  AliMUONGeometryTransformer* geoTransformerRec = new AliMUONGeometryTransformer();
  geoTransformerRec->LoadGeometryData();

  return geoTransformerRec;
}

//------------------------------------------------------------------
void LoadPreClusters(const AliMUONVClusterStore& preclusterStore, std::list<PreCluster>& preclusters)
{
  /// read the reconstructed preclusters

  AliMUONVCluster* cluster(nullptr);
  TIter nextCluster(preclusterStore.CreateIterator());
  while ((cluster = static_cast<AliMUONVCluster*>(nextCluster()))) {

    preclusters.emplace_back();

    for (int iDigit = 0; iDigit < cluster->GetNDigits(); ++iDigit) {
      preclusters.back().digitIds.emplace(cluster->GetDigitId(iDigit));
    }
  }
}

//------------------------------------------------------------------
void LinkClusters(const AliMUONVClusterStore& clusterStore, std::list<PreCluster>& preclusters)
{
  /// read the reconstructed clusters and link them to the provided preclusters

  AliMUONVCluster* cluster(nullptr);
  TIter nextCluster(clusterStore.CreateIterator());
  while ((cluster = static_cast<AliMUONVCluster*>(nextCluster()))) {

    // find the precluster with the same digits and attach the cluster to it
    auto* precluster = FindPreCluster(*cluster, preclusters);
    if (precluster == nullptr) {
      cout << "this cluster cannot be linked to any precluster !?" << endl;
      exit(1);
    } else {
      precluster->clusters.push_back(cluster);
    }
  }
}

//------------------------------------------------------------------
PreCluster* FindPreCluster(const AliMUONVCluster& cluster, std::list<PreCluster>& preclusters)
{
  /// find the precluster that contains the digits associated to this cluster

  for (auto& precluster : preclusters) {

    if (precluster.digitIds.count(cluster.GetDigitId(0)) != 0) {

      // just a cross-check that must always be true
      for (int iDigit = 0; iDigit < cluster.GetNDigits(); ++iDigit) {
        if (precluster.digitIds.count(cluster.GetDigitId(iDigit)) != 1) {
          cout << "some digits associated to this cluster are not part of the same precluster !?" << endl;
          exit(1);
        }
      }

      return &precluster;
    }
  }

  return nullptr;
}

//------------------------------------------------------------------
void LoadClusters(const AliMUONVClusterStore& clusterStore, std::list<PreCluster>& preclusters)
{
  /// read the reconstructed clusters and make the preclusters from the associated digits

  AliMUONVCluster* cluster(nullptr);
  TIter nextCluster(clusterStore.CreateIterator());
  while ((cluster = static_cast<AliMUONVCluster*>(nextCluster()))) {

    // get the list of associated digits
    std::set<unsigned int> digitIds;
    for (int iDigit = 0; iDigit < cluster->GetNDigits(); ++iDigit) {
      digitIds.emplace(cluster->GetDigitId(iDigit));
    }

    // find the precluster with the same digits, or create it, and attach the cluster to it
    auto* precluster = FindPreCluster(digitIds, preclusters);
    if (precluster == nullptr) {
      preclusters.emplace_back(cluster, digitIds);
    } else {
      precluster->clusters.push_back(cluster);
    }
  }
}

//------------------------------------------------------------------
PreCluster* FindPreCluster(const std::set<unsigned int>& digitIds, std::list<PreCluster>& preclusters)
{
  /// find the precluster with the exact same list of digits, if any

  for (auto& precluster : preclusters) {
    
    if (precluster.digitIds.count(*(digitIds.begin())) != 0) {
      
      // just a cross-check that must always be true
      if (!(precluster.digitIds == digitIds)) {
        cout << "some clusters share only a fraction of their associated digits !?" << endl;
        exit(1);
      }

      return &precluster;
    }
  }

  return nullptr;
}

