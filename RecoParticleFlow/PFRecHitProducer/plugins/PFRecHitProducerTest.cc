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
  void DumpEvent(const reco::PFRecHitCollection& pfRecHitsCPU, const reco::PFRecHitHostCollection::ConstView& pfRecHitsAlpaka);

  edm::EDGetTokenT<edm::SortedCollection<HBHERecHit>> recHitsToken;
  edm::EDGetTokenT<reco::PFRecHitCollection> pfRecHitsTokenCPU;
  edm::EDGetTokenT<reco::PFRecHitHostCollection> pfRecHitsTokenAlpaka;
  int32_t num_events = 0, num_errors = 0;
};

PFRecHitProducerTest::PFRecHitProducerTest(const edm::ParameterSet& conf)
    : recHitsToken(
          consumes<edm::SortedCollection<HBHERecHit>>(conf.getUntrackedParameter<edm::InputTag>("recHitsSourceCPU"))),
      pfRecHitsTokenCPU(
          consumes<reco::PFRecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSourceCPU"))),
      pfRecHitsTokenAlpaka(
          consumes<reco::PFRecHitHostCollection>(conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSourceAlpaka")))
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
  edm::Handle<reco::PFRecHitHostCollection> pfRecHitsAlpakaSoA;
  event.getByToken(pfRecHitsTokenCPU, pfRecHitsCPUlegacy);
  event.getByToken(pfRecHitsTokenAlpaka, pfRecHitsAlpakaSoA);
  
  const reco::PFRecHitCollection& pfRecHitsCPU = *pfRecHitsCPUlegacy;
  const reco::PFRecHitHostCollection::ConstView& pfRecHitsAlpaka = pfRecHitsAlpakaSoA->const_view();

  int error = 0;
  if(pfRecHitsCPU.size() != pfRecHitsAlpaka.size())
    error = 1;
  else
  {
    for (size_t i = 0; i < pfRecHitsCPU.size() && error == 0; i++)
    {
      const uint32_t detId = pfRecHitsCPU[i].detId();
      bool detId_found = false;
      for (size_t j = 0; j < pfRecHitsAlpaka.size() && error == 0; j++)
      {
        if(detId == pfRecHitsAlpaka[j].detId())
        {
          if(detId_found)
            error = 2;
          detId_found = true;
          if(pfRecHitsCPU[i].depth() != pfRecHitsAlpaka[j].depth()
            || pfRecHitsCPU[i].layer() != pfRecHitsAlpaka[j].layer()
            || pfRecHitsCPU[i].time() != pfRecHitsAlpaka[j].time()
            || pfRecHitsCPU[i].energy() != pfRecHitsAlpaka[j].energy()
            || pfRecHitsCPU[i].position().x() != pfRecHitsAlpaka[j].x()
            || pfRecHitsCPU[i].position().y() != pfRecHitsAlpaka[j].y()
            || pfRecHitsCPU[i].position().z() != pfRecHitsAlpaka[j].z()
            )
            error = 3;
          else
          {
            // check neighbours
            reco::PFRecHit::Neighbours pfRecHitNeighbours = pfRecHitsCPU[i].neighbours();
            std::vector<uint32_t> neighbours_cpu(pfRecHitNeighbours.begin(), pfRecHitNeighbours.end());
            for(size_t k = 0; k < neighbours_cpu.size(); k++)
              neighbours_cpu[k] = pfRecHitsCPU[neighbours_cpu[k]].detId();
            std::sort(neighbours_cpu.begin(), neighbours_cpu.end());

            std::vector<uint32_t> neighbours_alpaka;
            neighbours_alpaka.reserve(8);
            for(size_t k = 0; k < 8; k++)
              if(pfRecHitsAlpaka[j].neighbours()(k) != -1)
                neighbours_alpaka.emplace_back((uint32_t)pfRecHitsAlpaka[pfRecHitsAlpaka[j].neighbours()(k)].detId());
            std::sort(neighbours_alpaka.begin(), neighbours_alpaka.end());

            if(neighbours_cpu.size() != neighbours_alpaka.size())
              error = 4;
            else
              for(size_t k = 0; k < neighbours_cpu.size() && error == 0; k++)
                if(neighbours_cpu[k] != neighbours_alpaka[k])
                  error = 5;
          }
        }
      }
      if(!detId_found)
        error = 6;
    }
  }

  //if(num_events == 0)
  //  DumpEvent(pfRecHitsCPU, pfRecHitsAlpaka);

  if(error)
  {
    // When enabling this, need to set number of threads to 1 to get useful output
    if(DEBUG && num_errors == 0)
    {
      printf("Error: %d\n", error);
      DumpEvent(pfRecHitsCPU, pfRecHitsAlpaka);
    }
    num_errors++;
  }
  num_events++;
}

void PFRecHitProducerTest::DumpEvent(const reco::PFRecHitCollection& pfRecHitsCPU, const reco::PFRecHitHostCollection::ConstView& pfRecHitsAlpaka) {
  printf("Found %zd/%d pfRecHits with CPU/Alpaka\n", pfRecHitsCPU.size(), pfRecHitsAlpaka.size());
  for (size_t i = 0; i < pfRecHitsCPU.size(); i++)
  {
    reco::PFRecHit::Neighbours pfRecHitNeighbours4 = pfRecHitsCPU[i].neighbours4();
    reco::PFRecHit::Neighbours pfRecHitNeighbours8 = pfRecHitsCPU[i].neighbours8();
    std::vector<uint32_t> neighbours4, neighbours8;
    neighbours4.reserve(4);
    neighbours8.reserve(8);
    for(auto p = pfRecHitNeighbours8.begin(); p < pfRecHitNeighbours8.end(); p++)
        neighbours8.emplace_back(*p);
    for(auto p = pfRecHitNeighbours4.begin(); p < pfRecHitNeighbours4.end(); p++)
    {
      neighbours4.emplace_back(*p);
      auto idx = std::find(neighbours8.begin(), neighbours8.end(), *p);
      std::copy(idx+1, neighbours8.end(), idx);
    }
    neighbours8.resize(pfRecHitNeighbours8.size() - pfRecHitNeighbours4.size());

    printf("CPU %4lu detId:%u depth:%d layer:%d time:%f energy:%f pos:%f,%f,%f neighbours:%lu+%lu(",
           i,
           pfRecHitsCPU[i].detId(),
           pfRecHitsCPU[i].depth(),
           pfRecHitsCPU[i].layer(),
           pfRecHitsCPU[i].time(),
           pfRecHitsCPU[i].energy(),
           pfRecHitsCPU[i].position().x(),
           pfRecHitsCPU[i].position().y(),
           pfRecHitsCPU[i].position().z(),
           neighbours4.size(),
           neighbours8.size()
    );
    for(uint32_t j = 0; j < neighbours4.size(); j++)
      printf("%s%u", (j == 0) ? "" : ",", pfRecHitsCPU[neighbours4[j]].detId());
    printf(";");
    for(uint32_t j = 0; j < neighbours8.size(); j++)
      printf("%s%u", (j == 0) ? "" : ",", pfRecHitsCPU[neighbours8[j]].detId());
    printf(")\n");
  }
  for (size_t i = 0; i < pfRecHitsAlpaka.size(); i++)
  {
    std::vector<uint32_t> neighbours4, neighbours8;
    neighbours4.reserve(4);
    neighbours4.reserve(8);
    for(size_t k = 0; k < 4; k++)
      if(pfRecHitsAlpaka[i].neighbours()(k) != -1)
        neighbours4.emplace_back((uint32_t)pfRecHitsAlpaka[pfRecHitsAlpaka[i].neighbours()(k)].detId());
    for(size_t k = 4; k < 8; k++)
      if(pfRecHitsAlpaka[i].neighbours()(k) != -1)
        neighbours8.emplace_back((uint32_t)pfRecHitsAlpaka[pfRecHitsAlpaka[i].neighbours()(k)].detId());

    printf("Alpaka %4lu detId:%u depth:%d layer:%d time:%f energy:%f pos:%f,%f,%f neighbours:%lu+%lu(",
           i,
           pfRecHitsAlpaka[i].detId(),
           pfRecHitsAlpaka[i].depth(),
           pfRecHitsAlpaka[i].layer(),
           pfRecHitsAlpaka[i].time(),
           pfRecHitsAlpaka[i].energy(),
           pfRecHitsAlpaka[i].x(),
           pfRecHitsAlpaka[i].y(),
           pfRecHitsAlpaka[i].z(),
           neighbours4.size(),
           neighbours8.size()
    );
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
