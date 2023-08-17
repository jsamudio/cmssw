#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoParticleFlow/PFRecHitProducer/plugins/alpaka/PFRecHitProducerKernel.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/CalorimeterDefinitions.h"

#include <utility>
#include <vector>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace ParticleFlowRecHitProducerAlpaka;

  template <typename CAL>
  class PFRecHitProducerAlpaka : public stream::EDProducer<> {
  public:
    PFRecHitProducerAlpaka(edm::ParameterSet const& config)
        : topologyToken(esConsumes(config.getParameter<edm::ESInputTag>("topology"))),
          pfRecHitsToken(produces()),
          synchronise(config.getParameter<bool>("synchronise")) {
      const std::vector<edm::ParameterSet> producers = config.getParameter<std::vector<edm::ParameterSet>>("producers");
      recHitsToken.reserve(producers.size());
      for (const edm::ParameterSet& producer : producers) {
        recHitsToken.emplace_back(consumes(producer.getParameter<edm::InputTag>("src")),
                                  esConsumes(producer.getParameter<edm::ESInputTag>("params")));
      }
    }

    void produce(device::Event& event, const device::EventSetup& setup) override {
      const typename CAL::TopologyTypeDevice& topology = setup.getData(topologyToken);

      int num_recHits = 0;
      for (const auto& token : recHitsToken)
        num_recHits += event.get(token.first)->metadata().size();

      PFRecHitDeviceCollection pfRecHits{num_recHits, event.queue()};

      if (!kernel)
        kernel.emplace(event.queue());

      kernel->prepare_event(event.queue(), num_recHits);
      for (const auto& token : recHitsToken)
        kernel->process_rec_hits(event.queue(), event.get(token.first), setup.getData(token.second), pfRecHits);
      kernel->associate_topology_info(event.queue(), topology, pfRecHits);

      if (synchronise)
        alpaka::wait(event.queue());

      event.emplace(pfRecHitsToken, std::move(pfRecHits));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      edm::ParameterSetDescription producers;
      producers.add<edm::InputTag>("src");
      producers.add<edm::ESInputTag>("params");
      desc.addVPSet("producers", producers);
      desc.add<edm::ESInputTag>("topology");
      desc.add<bool>("synchronise", false);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<typename CAL::TopologyTypeDevice, typename CAL::TopologyRecordType> topologyToken;
    std::vector<std::pair<device::EDGetToken<CaloRecHitDeviceCollection>,
                          device::ESGetToken<typename CAL::ParameterType, typename CAL::ParameterRecordType>>>
        recHitsToken;
    const device::EDPutToken<PFRecHitDeviceCollection> pfRecHitsToken;
    const bool synchronise;
    std::optional<PFRecHitProducerKernel<CAL>> kernel = {};
  };

  using PFRecHitProducerAlpakaHCAL = PFRecHitProducerAlpaka<HCAL>;
  using PFRecHitProducerAlpakaECAL = PFRecHitProducerAlpaka<ECAL>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFRecHitProducerAlpakaHCAL);
DEFINE_FWK_ALPAKA_MODULE(PFRecHitProducerAlpakaECAL);
