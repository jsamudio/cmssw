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

  struct GenericPFRecHit {
    uint32_t detId;
    int depth;
    PFLayer::Layer layer;
    float time;
    float energy;
    float x, y, z;
    std::vector<uint32_t> neighbours4, neighbours8;

    GenericPFRecHit(const reco::PFRecHit& pfRecHit);  // Constructor from legacy
    GenericPFRecHit(const reco::PFRecHitHostCollection::ConstView& pfRecHitsAlpaka, size_t i);  // Constructor from Alpaka

    void Print(const char* prefix, size_t idx);
    int Compare(const GenericPFRecHit& other);
  };
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
    for (size_t i = 0; i < pfRecHitsCPU.size() && error == 0; i++)
      error = GenericPFRecHit{pfRecHitsCPU[i]}.Compare(GenericPFRecHit{pfRecHitsAlpaka, i});

  //if(num_events == 0)
  //  DumpEvent(pfRecHitsCPU, pfRecHitsAlpaka);

  if(error)
  {
    // When enabling this, need to set number of threads to 1 to get useful output
    if(DEBUG && num_errors == 0)
    {
      // Error codes:
      //  1 different number of PFRecHits
      //  2 detId different (different order?)
      //  3 depth,layer,time,energy or pos different
      //  4 different number of neighbours
      //  5 neighbours different (order?)
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
    GenericPFRecHit{pfRecHitsCPU[i]}.Print("CPU", i);
  for (size_t i = 0; i < pfRecHitsAlpaka.size(); i++)
    GenericPFRecHit{pfRecHitsAlpaka, i}.Print("Alpaka", i);
}

void PFRecHitProducerTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("recHitsSourceCPU");
  desc.addUntracked<edm::InputTag>("pfRecHitsSourceCPU");
  desc.addUntracked<edm::InputTag>("pfRecHitsSourceAlpaka");
  descriptions.addDefault(desc);
}


PFRecHitProducerTest::GenericPFRecHit::GenericPFRecHit(const reco::PFRecHit& pfRecHit) : 
    detId(pfRecHit.detId()),
    depth(pfRecHit.depth()),
    layer(pfRecHit.layer()),
    time(pfRecHit.time()),
    energy(pfRecHit.energy()),
    x(pfRecHit.position().x()),
    y(pfRecHit.position().y()),
    z(pfRecHit.position().z())
{
  // Fill neighbours4 and neighbours8, then remove elements of neighbours4 from neighbours8
  // This is necessary, because there can be duplicates in the neighbour lists
  // This procedure correctly accounts for these multiplicities
  reco::PFRecHit::Neighbours pfRecHitNeighbours4 = pfRecHit.neighbours4();
  reco::PFRecHit::Neighbours pfRecHitNeighbours8 = pfRecHit.neighbours8();
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
}

PFRecHitProducerTest::GenericPFRecHit::GenericPFRecHit(const reco::PFRecHitHostCollection::ConstView& pfRecHitsAlpaka, size_t i) : 
    detId(pfRecHitsAlpaka[i].detId()),
    depth(pfRecHitsAlpaka[i].depth()),
    layer(pfRecHitsAlpaka[i].layer()),
    time(pfRecHitsAlpaka[i].time()),
    energy(pfRecHitsAlpaka[i].energy()),
    x(pfRecHitsAlpaka[i].x()),
    y(pfRecHitsAlpaka[i].y()),
    z(pfRecHitsAlpaka[i].z())
{
  // Copy first four elements into neighbours4 and last four into neighbours8
  neighbours4.reserve(4);
  neighbours8.reserve(4);
  for(size_t k = 0; k < 4; k++)
    if(pfRecHitsAlpaka[i].neighbours()(k) != -1)
      neighbours4.emplace_back((uint32_t)pfRecHitsAlpaka[i].neighbours()(k));
  for(size_t k = 4; k < 8; k++)
    if(pfRecHitsAlpaka[i].neighbours()(k) != -1)
      neighbours8.emplace_back((uint32_t)pfRecHitsAlpaka[i].neighbours()(k));
}

void PFRecHitProducerTest::GenericPFRecHit::Print(const char* prefix, size_t idx) {
  printf("%s %4lu detId:%u depth:%d layer:%d time:%f energy:%f pos:%f,%f,%f neighbours:%lu+%lu(",
          prefix, idx,
          detId,
          depth,
          layer,
          time,
          energy,
          x, y, z,
          neighbours4.size(), neighbours8.size()
  );
  for(uint32_t j = 0; j < neighbours4.size(); j++)
    printf("%s%u", j ? "," : "", neighbours4[j]);
  printf(";");
  for(uint32_t j = 0; j < neighbours8.size(); j++)
    printf("%s%u", j ? "," : "", neighbours8[j]);
  printf(")\n");
}

int PFRecHitProducerTest::GenericPFRecHit::Compare(const GenericPFRecHit& other) {
  if(detId != other.detId)
    return 2;

  if(  depth  != other.depth
    || layer  != other.layer
    || time   != other.time
    || energy != other.energy
    || x      != other.x
    || y      != other.y
    || z      != other.z)
    return 3;

  if(neighbours4.size() != other.neighbours4.size()
    || neighbours8.size() != other.neighbours8.size())
    return 4;

  for(size_t i = 0; i < neighbours4.size(); i++)
    if(neighbours4[i] != other.neighbours4[i])
      return 5;
  for(size_t i = 0; i < neighbours8.size(); i++)
    if(neighbours8[i] != other.neighbours8[i])
      return 5;

  return 0;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFRecHitProducerTest);
