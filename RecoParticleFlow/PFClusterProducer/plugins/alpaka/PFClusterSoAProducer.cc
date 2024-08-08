#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFClusterParamsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFClusterSoAProducerKernel.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PFClusterSoAProducer : public stream::SynchronizingEDProducer<> {
  public:
    PFClusterSoAProducer(edm::ParameterSet const& config)
        : pfClusParamsToken(esConsumes(config.getParameter<edm::ESInputTag>("pfClusterParams"))),
          topologyToken_(esConsumes(config.getParameter<edm::ESInputTag>("topology"))),
          inputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfRecHits"))},
          outputPFClusterSoA_Token_{produces()},
          outputPFRHFractionSoA_Token_{produces()},
          synchronise_(config.getParameter<bool>("synchronise")),
          pfRecHitFractionAllocation_(config.getParameter<int>("pfRecHitFractionAllocation")) {}

    void acquire(device::Event const& event, device::EventSetup const& setup) override {
      const reco::PFClusterParamsDeviceCollection& params = setup.getData(pfClusParamsToken);
      const reco::PFRecHitHCALTopologyDeviceCollection& topology = setup.getData(topologyToken_);
      const reco::PFRecHitHostCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);
      int nRH = 0;
      if (pfRecHits->metadata().size() != 0)
        nRH = pfRecHits->size();

      pfClusteringVars = std::make_unique<reco::PFClusteringVarsDeviceCollection>(nRH, event.queue());
      pfClusteringEdgeVars = std::make_unique<reco::PFClusteringEdgeVarsDeviceCollection>(nRH * 8, event.queue());
      pfClusters = std::make_unique<reco::PFClusterDeviceCollection>(nRH, event.queue());

      if (nRH != 0) {
        PFClusterProducerKernel kernel(event.queue(), pfRecHits);
        kernel.step1(event.queue(),
                       params,
                       topology,
                       *pfClusteringVars,
                       *pfClusteringEdgeVars,
                       pfRecHits,
                       *pfClusters);
      }

      pfClusters_h = reco::PFClusterHostCollection{nRH, event.queue()};

      alpaka::memcpy(event.queue(), pfClusters_h.buffer(), pfClusters.get()->buffer());
    }

    void produce(device::Event& event, device::EventSetup const& setup) override {

      const reco::PFClusterParamsDeviceCollection& params = setup.getData(pfClusParamsToken);
      const reco::PFRecHitHCALTopologyDeviceCollection& topology = setup.getData(topologyToken_);
      const reco::PFRecHitHostCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);
      int nRH = 0;
      if (pfRecHits->metadata().size() != 0)
        nRH = pfRecHits->size();

      if (nRH != 0) {
        pfrhFractions = std::make_unique<reco::PFRecHitFractionDeviceCollection>(pfClusters_h.view().nRHFracs(), event.queue());
        PFClusterProducerKernel kernel(event.queue(), pfRecHits);
        kernel.step2(event.queue(),
                       params,
                       topology,
                       *pfClusteringVars,
                       *pfClusteringEdgeVars,
                       pfRecHits,
                       *pfClusters,
                       *pfrhFractions);
      } else {
        pfrhFractions = std::make_unique<reco::PFRecHitFractionDeviceCollection>(0, event.queue());
      }

      
      if (synchronise_)
        alpaka::wait(event.queue());

      event.emplace(outputPFClusterSoA_Token_, std::move(*pfClusters));
      event.emplace(outputPFRHFractionSoA_Token_, std::move(*pfrhFractions));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("pfRecHits", edm::InputTag(""));
      desc.add<edm::ESInputTag>("pfClusterParams", edm::ESInputTag(""));
      desc.add<edm::ESInputTag>("topology", edm::ESInputTag(""));
      desc.add<bool>("synchronise", false);
      desc.add<int>("pfRecHitFractionAllocation", 120);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<reco::PFClusterParamsDeviceCollection, JobConfigurationGPURecord> pfClusParamsToken;
    const device::ESGetToken<reco::PFRecHitHCALTopologyDeviceCollection, PFRecHitHCALTopologyRecord> topologyToken_;
    const edm::EDGetTokenT<reco::PFRecHitHostCollection> inputPFRecHitSoA_Token_;
    const device::EDPutToken<reco::PFClusterDeviceCollection> outputPFClusterSoA_Token_;
    const device::EDPutToken<reco::PFRecHitFractionDeviceCollection> outputPFRHFractionSoA_Token_;
    unsigned int num_rhf_;
    std::unique_ptr<reco::PFClusteringVarsDeviceCollection> pfClusteringVars;
    std::unique_ptr<reco::PFClusteringEdgeVarsDeviceCollection> pfClusteringEdgeVars;
    std::unique_ptr<reco::PFClusterDeviceCollection> pfClusters;
    //reco::PFClusterDeviceCollection pfClusters;
    std::unique_ptr<reco::PFRecHitFractionDeviceCollection> pfrhFractions;
    reco::PFClusterHostCollection pfClusters_h;
    const bool synchronise_;
    const int pfRecHitFractionAllocation_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFClusterSoAProducer);
