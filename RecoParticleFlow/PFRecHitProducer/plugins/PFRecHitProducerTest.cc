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
#include <variant>

class PFRecHitProducerTest : public DQMEDAnalyzer {
public:
  PFRecHitProducerTest(edm::ParameterSet const& conf);
  ~PFRecHitProducerTest() override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override{};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // Define generic types for token, handle and collection, such that origin of PFRecHits can be selected at runtime
  // Idea is to read desired format from config (pfRecHitsType1/2) and construct token accordingly. Then use handle
  // and collection corresponding to token type. Finally, construct GenericPFRecHits from each source, which are
  // independent of the format. This way this module can be used with to validate any combination of legacy and
  // Alpaka formats (and possibly CUDA?).
  using LegacyToken = edm::EDGetTokenT<reco::PFRecHitCollection>;
  using AlpakaToken = edm::EDGetTokenT<reco::PFRecHitHostCollection>;
  using GenericPFRecHitToken = std::variant<LegacyToken, AlpakaToken>;
  using LegacyHandle = edm::Handle<reco::PFRecHitCollection>;
  using AlpakaHandle = edm::Handle<reco::PFRecHitHostCollection>;
  using GenericHandle = std::variant<LegacyHandle, AlpakaHandle>;
  using LegacyCollection = const reco::PFRecHitCollection*;
  using AlpakaCollection = const reco::PFRecHitHostCollection::ConstView*;
  using GenericCollection = std::variant<LegacyCollection, AlpakaCollection>;

  static size_t GenericCollectionSize(const GenericCollection& collection) {
    if (std::holds_alternative<LegacyCollection>(collection))
      return std::get<LegacyCollection>(collection)->size();
    else
      return std::get<AlpakaCollection>(collection)->size();
  };

  edm::EDGetTokenT<edm::SortedCollection<HBHERecHit>> recHitsToken;
  GenericPFRecHitToken pfRecHitsTokens[2];

  void DumpEvent(const GenericCollection&, const GenericCollection&);
  int32_t num_events = 0, num_errors = 0;
  const std::string title;
  const bool dumpFirstEvent, dumpFirstError;

  // Container for PFRecHit, independent of how it was constructed
  struct GenericPFRecHit {
    uint32_t detId;
    int depth;
    PFLayer::Layer layer;
    float time;
    float energy;
    float x, y, z;
    std::vector<uint32_t> neighbours4, neighbours8;

    static GenericPFRecHit Construct(const GenericCollection&, size_t i);  // Generic constructor
    GenericPFRecHit(const reco::PFRecHit& pfRecHit);                       // Constructor from legacy format
    GenericPFRecHit(const reco::PFRecHitHostCollection::ConstView& pfRecHitsAlpaka,
                    size_t i);  // Constructor from Alpaka SoA

    void Print(const char* prefix, size_t idx);
  };
};

PFRecHitProducerTest::PFRecHitProducerTest(const edm::ParameterSet& conf)
    : recHitsToken(
          consumes<edm::SortedCollection<HBHERecHit>>(conf.getUntrackedParameter<edm::InputTag>("recHitsSourceCPU"))),
      title(conf.getUntrackedParameter<std::string>("title")),             // identifier added to final printout
      dumpFirstEvent(conf.getUntrackedParameter<bool>("dumpFirstEvent")),  // print PFRecHits from first event
      dumpFirstError(
          conf.getUntrackedParameter<bool>("dumpFirstError"))  // print PFRecHits from first event that yields an error
{
  const edm::InputTag input[2] = {conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSource1"),
                                  conf.getUntrackedParameter<edm::InputTag>("pfRecHitsSource2")};
  const std::string type[2] = {conf.getUntrackedParameter<std::string>("pfRecHitsType1"),
                               conf.getUntrackedParameter<std::string>("pfRecHitsType2")};
  for (int i = 0; i < 2; i++) {
    if (type[i] == "legacy")
      pfRecHitsTokens[i].emplace<LegacyToken>(consumes<LegacyHandle::element_type>(input[i]));
    else if (type[i] == "alpaka")
      pfRecHitsTokens[i].emplace<AlpakaToken>(consumes<AlpakaHandle::element_type>(input[i]));
    else {
      fprintf(stderr, "Invalid value for PFRecHitProducerTest::pfRecHitsType%d: \"%s\"\n", i + 1, type[i].c_str());
      std::exit(1);
    }
  }
}

PFRecHitProducerTest::~PFRecHitProducerTest() {
  fprintf(stderr,
          "PFRecHitProducerTest%s%s%s has compared %u events and found %u problems\n",
          title.empty() ? "" : "[",
          title.c_str(),
          title.empty() ? "" : "]",
          num_events,
          num_errors);
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
  GenericHandle pfRecHitsHandles[2];
  GenericCollection pfRecHits[2];
  for (int i = 0; i < 2; i++)
    if (std::holds_alternative<LegacyToken>(pfRecHitsTokens[i])) {
      auto& handle = pfRecHitsHandles[i].emplace<LegacyHandle>();
      event.getByToken(std::get<LegacyToken>(pfRecHitsTokens[i]), handle);
      pfRecHits[i].emplace<LegacyCollection>(&*handle);
    } else {
      auto& handle = pfRecHitsHandles[i].emplace<AlpakaHandle>();
      event.getByToken(std::get<AlpakaToken>(pfRecHitsTokens[i]), handle);
      pfRecHits[i].emplace<AlpakaCollection>(&handle->const_view());
    }

  int error = 0;

  const size_t n = GenericCollectionSize(pfRecHits[0]);
  if (n != GenericCollectionSize(pfRecHits[1]))
    error = 1;
  else {
    std::vector<GenericPFRecHit> first, second;
    std::unordered_map<uint32_t, size_t> detId2Idx;  // for second vector
    first.reserve(n);
    second.reserve(n);
    for (size_t i = 0; i < n; i++) {
      first.emplace_back(GenericPFRecHit::Construct(pfRecHits[0], i));
      second.emplace_back(GenericPFRecHit::Construct(pfRecHits[1], i));
      detId2Idx[second.at(i).detId] = i;
    }
    for (size_t i = 0; i < n && error == 0; i++) {
      const GenericPFRecHit& rh1 = first.at(i);
      if (detId2Idx.find(rh1.detId) == detId2Idx.end()) {
        error = 2;
        break;
      }

      const GenericPFRecHit& rh2 = second.at(detId2Idx.at(rh1.detId));
      assert(rh1.detId == rh2.detId);
      if (rh1.depth != rh2.depth || rh1.layer != rh2.layer || rh1.time != rh2.time || rh1.energy != rh2.energy ||
          rh1.x != rh2.x || rh1.y != rh2.y || rh1.z != rh2.z) {
        error = 3;
        break;
      }

      if (rh1.neighbours4.size() != rh2.neighbours4.size() || rh1.neighbours8.size() != rh2.neighbours8.size()) {
        error = 4;
        break;
      }

      for (size_t i = 0; i < rh1.neighbours4.size(); i++)
        if (first.at(rh1.neighbours4[i]).detId != second.at(rh2.neighbours4[i]).detId) {
          error = 5;
          break;
        }
      for (size_t i = 0; i < rh1.neighbours8.size(); i++)
        if (first.at(rh1.neighbours8[i]).detId != second.at(rh2.neighbours8[i]).detId) {
          error = 5;
          break;
        }
    }
  }

  if (num_events == 0 && dumpFirstEvent)
    DumpEvent(pfRecHits[0], pfRecHits[1]);

  if (error) {
    if (dumpFirstError && num_errors == 0) {
      // Error codes:
      //  1 different number of PFRecHits
      //  2 detId not found
      //  3 depth,layer,time,energy or pos different
      //  4 different number of neighbours
      //  5 neighbours different (different order?)
      printf("Error: %d\n", error);
      DumpEvent(pfRecHits[0], pfRecHits[1]);
    }
    num_errors++;
  }
  num_events++;
}

void PFRecHitProducerTest::DumpEvent(const GenericCollection& pfRecHits1, const GenericCollection& pfRecHits2) {
  printf("Found %zd/%ld pfRecHits from first/second origin\n",
         GenericCollectionSize(pfRecHits1),
         GenericCollectionSize(pfRecHits2));
  for (size_t i = 0; i < GenericCollectionSize(pfRecHits1); i++)
    GenericPFRecHit::Construct(pfRecHits1, i).Print("First", i);
  for (size_t i = 0; i < GenericCollectionSize(pfRecHits2); i++)
    GenericPFRecHit::Construct(pfRecHits2, i).Print("Second", i);
}

void PFRecHitProducerTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("recHitsSourceCPU");
  desc.addUntracked<edm::InputTag>("pfRecHitsSource1");
  desc.addUntracked<edm::InputTag>("pfRecHitsSource2");
  desc.addUntracked<std::string>("pfRecHitsType1", "legacy");
  desc.addUntracked<std::string>("pfRecHitsType2", "alpaka");
  desc.addUntracked<std::string>("title", "");
  desc.addUntracked<bool>("dumpFirstEvent", false);
  desc.addUntracked<bool>("dumpFirstError", false);
  descriptions.addDefault(desc);
}

PFRecHitProducerTest::GenericPFRecHit PFRecHitProducerTest::GenericPFRecHit::Construct(
    const PFRecHitProducerTest::GenericCollection& collection, size_t i) {
  if (std::holds_alternative<LegacyCollection>(collection))
    return GenericPFRecHit{(*std::get<LegacyCollection>(collection))[i]};
  else
    return GenericPFRecHit{*std::get<AlpakaCollection>(collection), i};
}

PFRecHitProducerTest::GenericPFRecHit::GenericPFRecHit(const reco::PFRecHit& pfRecHit)
    : detId(pfRecHit.detId()),
      depth(pfRecHit.depth()),
      layer(pfRecHit.layer()),
      time(pfRecHit.time()),
      energy(pfRecHit.energy()),
      x(pfRecHit.position().x()),
      y(pfRecHit.position().y()),
      z(pfRecHit.position().z()) {
  // Fill neighbours4 and neighbours8, then remove elements of neighbours4 from neighbours8
  // This is necessary, because there can be duplicates in the neighbour lists
  // This procedure correctly accounts for these multiplicities
  reco::PFRecHit::Neighbours pfRecHitNeighbours4 = pfRecHit.neighbours4();
  reco::PFRecHit::Neighbours pfRecHitNeighbours8 = pfRecHit.neighbours8();
  neighbours4.reserve(4);
  neighbours8.reserve(8);
  for (auto p = pfRecHitNeighbours8.begin(); p < pfRecHitNeighbours8.end(); p++)
    neighbours8.emplace_back(*p);
  for (auto p = pfRecHitNeighbours4.begin(); p < pfRecHitNeighbours4.end(); p++) {
    neighbours4.emplace_back(*p);
    auto idx = std::find(neighbours8.begin(), neighbours8.end(), *p);
    std::copy(idx + 1, neighbours8.end(), idx);
  }
  neighbours8.resize(pfRecHitNeighbours8.size() - pfRecHitNeighbours4.size());
}

PFRecHitProducerTest::GenericPFRecHit::GenericPFRecHit(const reco::PFRecHitHostCollection::ConstView& pfRecHitsAlpaka,
                                                       size_t i)
    : detId(pfRecHitsAlpaka[i].detId()),
      depth(pfRecHitsAlpaka[i].depth()),
      layer(pfRecHitsAlpaka[i].layer()),
      time(pfRecHitsAlpaka[i].time()),
      energy(pfRecHitsAlpaka[i].energy()),
      x(pfRecHitsAlpaka[i].x()),
      y(pfRecHitsAlpaka[i].y()),
      z(pfRecHitsAlpaka[i].z()) {
  // Copy first four elements into neighbours4 and last four into neighbours8
  neighbours4.reserve(4);
  neighbours8.reserve(4);
  for (size_t k = 0; k < 4; k++)
    if (pfRecHitsAlpaka[i].neighbours()(k) != -1)
      neighbours4.emplace_back((uint32_t)pfRecHitsAlpaka[i].neighbours()(k));
  for (size_t k = 4; k < 8; k++)
    if (pfRecHitsAlpaka[i].neighbours()(k) != -1)
      neighbours8.emplace_back((uint32_t)pfRecHitsAlpaka[i].neighbours()(k));
}

void PFRecHitProducerTest::GenericPFRecHit::Print(const char* prefix, size_t idx) {
  printf("%s %4lu detId:%u depth:%d layer:%d time:%f energy:%f pos:%f,%f,%f neighbours:%lu+%lu(",
         prefix,
         idx,
         detId,
         depth,
         layer,
         time,
         energy,
         x,
         y,
         z,
         neighbours4.size(),
         neighbours8.size());
  for (uint32_t j = 0; j < neighbours4.size(); j++)
    printf("%s%u", j ? "," : "", neighbours4[j]);
  printf(";");
  for (uint32_t j = 0; j < neighbours8.size(); j++)
    printf("%s%u", j ? "," : "", neighbours8[j]);
  printf(")\n");
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFRecHitProducerTest);
