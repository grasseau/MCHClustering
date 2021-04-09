#include <iostream>
#include <chrono>

#include <TString.h>
#include <TFile.h>
#include <TTree.h>
#include <TROOT.h>

#include "AliRunLoader.h"
#include "AliLoader.h"
#include "AliCDBManager.h"
#include "AliGeomManager.h"

#include "AliMpArea.h"

#include "AliMUONCDB.h"
#include "AliMUONGeometryTransformer.h"
#include "AliMUONRecoParam.h"
#include "AliMUONPreClusterFinder.h"
#include "AliMUONSimpleClusterServer.h"
#include "AliMUONVDigitStore.h"
#include "AliMUONClusterStoreV2.h"

using namespace std;

//------------------------------------------------------------------
void RunAliRootPreClusterizer(int runNumber = 0, TString ocdb = "local://$ALIROOT_OCDB_ROOT/OCDB",
                              TString digitFileName = "", TString clusterFileName = "preclusters.root")
{
  /// run the Aliroot preclustering
  // if digitFileName == "", read the digits from the standard MUON.Digits.root file

  // load geometry and reconstruction parameters
//  AliCDBManager::Instance()->SetDefaultStorage("raw://");
//  AliCDBManager::Instance()->SetDefaultStorage("local://./OCDB");
//  AliCDBManager::Instance()->SetRun(169099);
  AliCDBManager::Instance()->SetDefaultStorage(ocdb.Data());
  AliCDBManager::Instance()->SetRun(runNumber);
  AliGeomManager::LoadGeometry();
  if (!AliGeomManager::GetGeometry() || !AliGeomManager::ApplyAlignObjsFromCDB("MUON")) return;
  AliMUONGeometryTransformer transformer;
  transformer.LoadGeometryData();
  AliMUONRecoParam* recoParam = AliMUONCDB::LoadRecoParam();
  if (!recoParam) return;

  // prepare the precluster finder
//  gROOT->LoadMacro("$WORK/Macros/PreClustering/AliMUONPreClusterFinderV4.cxx+");
//  AliMUONVClusterFinder* clusterFinder = reinterpret_cast<AliMUONVClusterFinder*>(gROOT->ProcessLineSync("new AliMUONPreClusterFinderV4()"));
  AliMUONVClusterFinder* clusterFinder = new AliMUONPreClusterFinder();
  AliMUONVClusterServer* clusterServer = new AliMUONSimpleClusterServer(clusterFinder, transformer);

  // prepare to read reconstructed digits
  AliRunLoader* rl(nullptr);
  AliLoader* muonLoader(nullptr);
  TTree* treeD(nullptr);
  AliMUONVDigitStore* digitStore(nullptr);
  if (digitFileName.IsNull()) {
    rl = AliRunLoader::Open("galice.root", "MUONLoader");
    muonLoader = rl->GetDetectorLoader("MUON");
    if (muonLoader->LoadDigits("READ") != 0) return;
    treeD = muonLoader->TreeD();
    digitStore = AliMUONVDigitStore::Create(*treeD);
  } else {
    TFile* digitFile = TFile::Open(digitFileName);
    if (!digitFile) return;
    treeD = static_cast<TTree*>(digitFile->Get("TreeD"));
    if (!treeD) return;
    digitStore = AliMUONVDigitStore::Create(*treeD);
    if (!digitStore->Connect(*treeD)) return;
  }

  // prepare to write clusters
  AliMUONVClusterStore* clusterStore = new AliMUONClusterStoreV2();
  TFile* clusterFile = TFile::Open(clusterFileName,"RECREATE");
  TTree* treeR = new TTree("TreeR","Cluster Container");
  clusterStore->Connect(*treeR,kTRUE);

  // loop over events
  AliMpArea area; // invalid area to clusterize everything
  int nEvents = rl ? rl->GetNumberOfEvents() : treeD->GetEntries();
  std::chrono::duration<double> preclusteringTime{};
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

    // load the digits
    if (digitFileName.IsNull()) {
      rl->GetEvent(iEvent);
      treeD = muonLoader->TreeD();
      digitStore->Connect(*treeD);
      treeD->GetEvent(0);
    } else {
      treeD->GetEntry(iEvent);
    }
    TIter next(digitStore->CreateIterator());
    clusterServer->UseDigits(next);

    // clusterize every chambers
    auto tStart = std::chrono::high_resolution_clock::now();
    for (int iCh = 0; iCh < 10; ++iCh) {
      clusterServer->Clusterize(iCh, *clusterStore, area, recoParam);
    }
    auto tEnd = std::chrono::high_resolution_clock::now();
    preclusteringTime += tEnd - tStart;

    // store the clusters
    treeR->Fill();
    
    // clear the stores
    clusterStore->Clear();
    digitStore->Clear();
  }

  // write the clusters
  clusterFile->cd();
  treeR->Write();
  clusterFile->Close();

  // print timer
  cout << "preclustering duration = " << preclusteringTime.count() << " s" << endl;
}
