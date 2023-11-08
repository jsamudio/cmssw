#include <Eigen/Core>
#include <Eigen/Dense>
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/plugins/alpaka/PFClusterProducerKernel.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#define DEBUG false

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PFClusterProducerAlpaka : public stream::EDProducer<> {
  public:
    PFClusterProducerAlpaka(edm::ParameterSet const& config)
        : pfClusParamsToken(esConsumes(config.getParameter<edm::ESInputTag>("pfClusterParams"))),
          inputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
          outputPFClusterSoA_Token_{produces()},
          outputPFRHFractionSoA_Token_{produces()},
          synchronise_(config.getParameter<bool>("synchronise")),
          produceSoA_{config.getParameter<bool>("produceSoA")} {}

    void produce(device::Event& event, device::EventSetup const& setup) override {
      const reco::PFClusterParamsAlpakaESDataDevice& params = setup.getData(pfClusParamsToken);
      const reco::PFRecHitHostCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);
      const int nRH = pfRecHits->size();

      reco::PFClusteringVarsDeviceCollection pfClusteringVars{nRH + 1, event.queue()};
      reco::PFClusteringEdgeVarsDeviceCollection pfClusteringEdgeVars{(nRH * 8) + 1, event.queue()};
      reco::PFClusterDeviceCollection pfClusters{nRH, event.queue()};
      reco::PFRecHitFractionDeviceCollection pfrhFractions{nRH * 120, event.queue()};

      PFClusterProducerKernel kernel(event.queue(), pfRecHits);
      kernel.execute(event.device(),
                     event.queue(),
                     params,
                     pfClusteringVars,
                     pfClusteringEdgeVars,
                     pfRecHits,
                     pfClusters,
                     pfrhFractions);

      if (synchronise_)
        alpaka::wait(event.queue());

      if (produceSoA_) {
        event.emplace(outputPFClusterSoA_Token_, std::move(pfClusters));
        event.emplace(outputPFRHFractionSoA_Token_, std::move(pfrhFractions));
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
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<reco::PFClusterParamsAlpakaESDataDevice, JobConfigurationGPURecord> pfClusParamsToken;
    const edm::EDGetTokenT<reco::PFRecHitHostCollection> inputPFRecHitSoA_Token_;
    const device::EDPutToken<reco::PFClusterDeviceCollection> outputPFClusterSoA_Token_;
    const device::EDPutToken<reco::PFRecHitFractionDeviceCollection> outputPFRHFractionSoA_Token_;
    const bool synchronise_;
    const bool produceSoA_;
    int nRH = 0;
    std::optional<PFClusterProducerKernel> kernel = {};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFClusterProducerAlpaka);
