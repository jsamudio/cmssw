#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFRecHitHostCollection.h"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#define DEBUG false

class PFRecHitProducerTest : public DQMEDAnalyzer {
public:
  PFRecHitProducerTest(edm::ParameterSet const& conf);
  ~PFRecHitProducerTest() override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override {};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void DumpEvent(const reco::PFRecHitCollection& pfRecHitsCPU, const PFRecHitHostCollection::ConstView& pfRecHitsAlpaka);

  edm::EDGetTokenT<edm::SortedCollection<HBHERecHit>> recHitsToken;
  edm::EDGetTokenT<reco::PFRecHitCollection> pfRecHitsTokenCPU;
  edm::EDGetTokenT<PFRecHitHostCollection> pfRecHitsTokenAlpaka;
  int32_t num_events = 0, num_errors = 0;
};

PFRecHitProducerTest::PFRecHitProducerTest(const edm::ParameterSet& conf)
    : recHitsToken(
          consumes<edm::SortedCollection<HBHERecHit>>(conf.getUntrackedParameter<edm::InputTag>("recHitsSourceCPU"))),
      pfRecHitsTokenCPU(
          consumes<reco::PFRecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSourceCPU"))),
      pfRecHitsTokenAlpaka(
          consumes<PFRecHitHostCollection>(conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSourceAlpaka")))
     {}

PFRecHitProducerTest::~PFRecHitProducerTest() {
  fprintf(stderr, "PFRecHitProducerTest has compared %u events and found %u problems\n", num_events, num_errors);
}

void PFRecHitProducerTest::analyze(edm::Event const& event, edm::EventSetup const& c) {
  // Rec Hits
  //edm::Handle<edm::SortedCollection<HBHERecHit>> recHits;
  //event.getByToken(recHitsToken, recHits);
  //printf("Found %zd recHits\n", recHits->size());
  //fprintf(stderr, "Found %zd recHits\n", recHits->size());
  //for (size_t i = 0; i < recHits->size(); i++)
  //  printf("recHit %4lu %u\n", i, recHits->operator[](i).id().rawId());

  // PF Rec Hits
  edm::Handle<reco::PFRecHitCollection> pfRecHitsCPUlegacy;
  edm::Handle<PFRecHitHostCollection> pfRecHitsAlpakaSoA;
  event.getByToken(pfRecHitsTokenCPU, pfRecHitsCPUlegacy);
  event.getByToken(pfRecHitsTokenAlpaka, pfRecHitsAlpakaSoA);
  
  const reco::PFRecHitCollection& pfRecHitsCPU = *pfRecHitsCPUlegacy;
  const PFRecHitHostCollection::ConstView& pfRecHitsAlpaka = pfRecHitsAlpakaSoA->const_view();

  bool error = false;
  if(pfRecHitsCPU.size() != pfRecHitsAlpaka.size())
    error = true;
  else
  {
    for (size_t i = 0; i < pfRecHitsCPU.size(); i++)
    {
      const uint32_t detId = pfRecHitsCPU[i].detId();
      bool detId_found = false;
      for (size_t j = 0; j < pfRecHitsAlpaka.size(); j++)
      {
        if(detId == pfRecHitsAlpaka[j].detId())
        {
          if(detId_found)
            error = true;
          detId_found = true;
          if(pfRecHitsCPU[i].depth() != pfRecHitsAlpaka[j].depth()
            || pfRecHitsCPU[i].layer() != pfRecHitsAlpaka[j].layer()
            || pfRecHitsCPU[i].time() != pfRecHitsAlpaka[j].time()
            || pfRecHitsCPU[i].energy() != pfRecHitsAlpaka[j].energy()
            || pfRecHitsCPU[i].position().x() != pfRecHitsAlpaka[i].x()
            || pfRecHitsCPU[i].position().y() != pfRecHitsAlpaka[i].y()
            || pfRecHitsCPU[i].position().z() != pfRecHitsAlpaka[i].z()
            )
            error = true;
        }
      }
      if(!detId_found)
        error = true;
    }
  }

  //if(num_events == 0)
  //  DumpEvent(pfRecHitsCPU, pfRecHitsAlpaka);

  if(error)
  {
    // When enabling this, need to set number of threads to 1 to get useful output
    if(DEBUG && num_errors == 0)
      DumpEvent(pfRecHitsCPU, pfRecHitsAlpaka);
    num_errors++;
  }
  num_events++;
}

void PFRecHitProducerTest::DumpEvent(const reco::PFRecHitCollection& pfRecHitsCPU, const PFRecHitHostCollection::ConstView& pfRecHitsAlpaka) {
  printf("Found %zd/%d pfRecHits with CPU/Alpaka\n", pfRecHitsCPU.size(), pfRecHitsAlpaka.size());
  for (size_t i = 0; i < pfRecHitsCPU.size(); i++)
  {
    reco::PFRecHit::Neighbours neighbours = pfRecHitsCPU[i].neighbours();
    printf("CPU %4lu detId:%u depth:%d layer:%d time:%f energy:%f pos:%f,%f,%f neighbours:%u(",
           i,
           pfRecHitsCPU[i].detId(),
           pfRecHitsCPU[i].depth(),
           pfRecHitsCPU[i].layer(),
           pfRecHitsCPU[i].time(),
           pfRecHitsCPU[i].energy(),
           pfRecHitsCPU[i].position().x(),
           pfRecHitsCPU[i].position().y(),
           pfRecHitsCPU[i].position().z(),
           neighbours.size()
    );
    for(reco::PFRecHit::Neighbours::Pointer p = neighbours.begin(); p < neighbours.end(); p++)
      printf("%s%u", (p == neighbours.begin()) ? "" : ",", *p);
    printf(")\n");
  }
  for (size_t i = 0; i < pfRecHitsAlpaka.size(); i++)
  {
    printf("Alpaka %4lu detId:%u depth:%d layer:%d time:%f energy:%f pos:%f,%f,%f neighbours:%u(",
           i,
           pfRecHitsAlpaka[i].detId(),
           pfRecHitsAlpaka[i].depth(),
           pfRecHitsAlpaka[i].layer(),
           pfRecHitsAlpaka[i].time(),
           pfRecHitsAlpaka[i].energy(),
           pfRecHitsAlpaka[i].x(),
           pfRecHitsAlpaka[i].y(),
           pfRecHitsAlpaka[i].z(),
           pfRecHitsAlpaka[i].num_neighbours()
    );
    for(size_t j = 0; j < pfRecHitsAlpaka[i].num_neighbours(); j++)
      printf("%s%u", (j == 0) ? "" : ",", pfRecHitsAlpaka[i].neighbours()[j]);
    printf(")\n");
  }
}

void PFRecHitProducerTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("recHitsSourceCPU");
  desc.addUntracked<edm::InputTag>("pfRecHitsSourceCPU");
  desc.addUntracked<edm::InputTag>("pfRecHitsSourceAlpaka");
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFRecHitProducerTest);
