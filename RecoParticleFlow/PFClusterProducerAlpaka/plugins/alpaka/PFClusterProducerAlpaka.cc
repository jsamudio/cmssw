#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/PFClusterParamsAlpakaESRecord.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterProducerKernel.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#define DEBUG false

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PFClusterProducerAlpaka : public stream::EDProducer<> {
  public:

    PFClusterProducerAlpaka(edm::ParameterSet const& config)
        : pfClusParamsToken(esConsumes(config.getParameter<edm::ESInputTag>("pfClusterParams"))),
          InputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
          OutputPFClusterSoA_Token_{produces()},
          OutputPFRHFractionSoA_Token_{produces()},
          synchronise(config.getParameter<bool>("synchronise")),
          _produceSoA{config.getParameter<bool>("produceSoA")},
          _produceLegacy{config.getParameter<bool>("produceLegacy")}
    {
    }

    void produce(device::Event& event, device::EventSetup const& setup) override {
      const reco::PFClusterParamsAlpakaESDataDevice& params = setup.getData(pfClusParamsToken);
      const reco::PFRecHitHostCollection& pfRecHits = event.get(InputPFRecHitSoA_Token_);
      const int nRH = pfRecHits->size();

      reco::tmpDeviceCollection tmp{nRH + 1, event.queue()};
      reco::tmpEdgeDeviceCollection tmpEdge{(nRH * 8) + 1, event.queue()};
      reco::PFClusterDeviceCollection pfClusters{nRH, event.queue()};
      reco::PFRecHitFractionDeviceCollection pfrhFractions{nRH * 120, event.queue()};

      if(!kernel)
        kernel.emplace(PFClusterProducerKernel::Construct(event.queue(), pfRecHits));
      kernel->execute(event.device(), event.queue(), params, tmp, tmpEdge, pfRecHits, pfClusters, pfrhFractions);

      if (synchronise)
        alpaka::wait(event.queue());

      if (_produceSoA) {
        event.emplace(OutputPFClusterSoA_Token_, std::move(pfClusters));
        event.emplace(OutputPFRHFractionSoA_Token_, std::move(pfrhFractions));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("PFRecHitsLabelIn");
      desc.add<std::string>("PFClustersGPUOut", "");
      desc.add<bool>("produceSoA", true);
      desc.add<bool>("produceLegacy", true);
      desc.add<edm::ESInputTag>("pfClusterParams");
      desc.add<bool>("synchronise");
      desc.setAllowAnything();
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<reco:PFClusterParamsAlpakaESDataDevice, PFClusterParamsAlpakaESRecord> pfClusParamsToken;
    const edm::EDGetTokenT<reco::PFRecHitHostCollection> InputPFRecHitSoA_Token_;
    const device::EDPutToken<reco::PFClusterDeviceCollection> OutputPFClusterSoA_Token_;
    const device::EDPutToken<reco::PFRecHitFractionDeviceCollection> OutputPFRHFractionSoA_Token_;
    const bool synchronise;
    const bool _produceSoA;
    const bool _produceLegacy;
    int nRH = 0;
    std::optional<PFClusterProducerKernel> kernel = {};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFClusterProducerAlpaka);
