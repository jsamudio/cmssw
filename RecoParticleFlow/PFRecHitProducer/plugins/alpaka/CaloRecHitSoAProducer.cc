#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/CalorimeterDefinitions.h"

#define DEBUG false

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace ParticleFlowRecHitProducerAlpaka;

  template <typename CAL>
  class CaloRecHitSoAProducer : public global::EDProducer<> {
  public:
    CaloRecHitSoAProducer(edm::ParameterSet const& config)
        : recHitsToken(consumes(config.getParameter<edm::InputTag>("src"))),
          deviceToken(produces()),
          synchronise(config.getUntrackedParameter<bool>("synchronise")) {}

    void produce(edm::StreamID sid, device::Event& event, device::EventSetup const&) const override {
      const edm::SortedCollection<typename CAL::CaloRecHitType>& recHits = event.get(recHitsToken);
      const int32_t num_recHits = recHits.size();
      if (DEBUG)
        printf("Found %d recHits\n", num_recHits);

      reco::CaloRecHitHostCollection hostProduct{num_recHits, event.queue()};
      auto& view = hostProduct.view();

      for (int i = 0; i < num_recHits; i++) {
        ConvertRecHit(view[i], recHits[i]);

        if (DEBUG && i < 10)
          printf("recHit %4d %u %f %f %08x\n", i, view.detId(i), view.energy(i), view.time(i), view.flags(i));
      }

      CaloRecHitDeviceCollection deviceProduct{num_recHits, event.queue()};
      alpaka::memcpy(event.queue(), deviceProduct.buffer(), hostProduct.buffer());
      if (synchronise)
        alpaka::wait(event.queue());
      event.emplace(deviceToken, std::move(deviceProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src")
        ->setComment("Input calorimeter rec hit collection");
      desc.addUntracked<bool>("synchronise", false)
        ->setComment("Add synchronisation point after execution (for benchmarking asynchronous execution)");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<edm::SortedCollection<typename CAL::CaloRecHitType>> recHitsToken;
    const device::EDPutToken<CaloRecHitDeviceCollection> deviceToken;
    const bool synchronise;

    static void ConvertRecHit(reco::CaloRecHitHostCollection::View::element to,
                              const typename CAL::CaloRecHitType& from);
  };

  template <>
  void CaloRecHitSoAProducer<HCAL>::ConvertRecHit(reco::CaloRecHitHostCollection::View::element to,
                                                  const HCAL::CaloRecHitType& from) {
    // Fill SoA from HCAL rec hit
    to.detId() = from.id().rawId();
    to.energy() = from.energy();
    to.time() = from.time();
    to.flags() = from.flags();
  }

  template <>
  void CaloRecHitSoAProducer<ECAL>::ConvertRecHit(reco::CaloRecHitHostCollection::View::element to,
                                                  const ECAL::CaloRecHitType& from) {
    // Fill SoA from ECAL rec hit
    to.detId() = from.id().rawId();
    to.energy() = from.energy();
    to.time() = from.time();
    to.flags() = from.flagsBits();
  }

  using HCALRecHitSoAProducer = CaloRecHitSoAProducer<HCAL>;
  using ECALRecHitSoAProducer = CaloRecHitSoAProducer<ECAL>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HCALRecHitSoAProducer);
DEFINE_FWK_ALPAKA_MODULE(ECALRecHitSoAProducer);
