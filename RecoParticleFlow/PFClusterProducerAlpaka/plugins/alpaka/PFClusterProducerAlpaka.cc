#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFRecHitHostCollection.h"
//#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/PFClusterDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/JobConfigurationAlpakaRecord2.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterProducerKernel.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#define DEBUG false

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PFClusterProducerAlpaka : public stream::EDProducer<> {
  public:
    std::unique_ptr<PFCPositionCalculatorBase> _positionCalc;
    std::unique_ptr<PFCPositionCalculatorBase> _allCellsPositionCalc;

    PFClusterProducerAlpaka(edm::ParameterSet const& config)
        : pfClusParamsToken(esConsumes(config.getParameter<edm::ESInputTag>("pfClusterParams"))),
          InputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
          //OutputPFClusterSoA_Token_{produces(config.getParameter<std::string>("PFClustersDeviceOut"))},
          OutputPFClusterSoA_Token_{produces()},
          OutputPFRHFractionSoA_Token_{produces()},
          synchronise(config.getParameter<bool>("synchronise")),
          _produceSoA{config.getParameter<bool>("produceSoA")},
          _produceLegacy{config.getParameter<bool>("produceLegacy")}
    //_rechitsLabel{consumes(config.getParameter<edm::InputTag>("recHitsSource"))}
    {
      edm::ConsumesCollector cc = consumesCollector();

      //setup pf cluster builder if requested
      const edm::ParameterSet& pfcConf = config.getParameterSet("pfClusterBuilder");
      if (!pfcConf.empty()) {
        if (pfcConf.exists("positionCalc")) {
          const edm::ParameterSet& acConf = pfcConf.getParameterSet("positionCalc");
          const std::string& algoac = acConf.getParameter<std::string>("algoName");
          _positionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
        }

        if (pfcConf.exists("allCellsPositionCalc")) {
          const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
          const std::string& algoac = acConf.getParameter<std::string>("algoName");
          _allCellsPositionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
        }
      }

      //legacyClusToken{produces()};
    }

    void produce(device::Event& event, device::EventSetup const& setup) override {
      const PFClusterParamsAlpakaESDataDevice& params = setup.getData(pfClusParamsToken);
      const reco::PFRecHitHostCollection& pfRecHits = event.get(InputPFRecHitSoA_Token_);
      //auto pfRecHits = event.getHandle(InputPFRecHitSoA_Token_);
      const int nRH = pfRecHits->size();
      std::cout << "nRH:"
                << " " << nRH << std::endl;

      tmpPF0DeviceCollection tmp0{nRH + 1, event.queue()};
      tmpPF1DeviceCollection tmp1{(nRH * 8) + 1, event.queue()};
      //auto tmp = std::make_unique<tmpPFDeviceCollection>((nRH, nRH*8), event.queue());
      PFClusterDeviceCollection2 pfClusters{nRH, event.queue()};
      PFRHFractionDeviceCollection pfrhFractions{nRH * 120, event.queue()};

      //if(!kernel)
      kernel.emplace(PFClusterProducerKernel::Construct(event.queue()));
      kernel->execute(event.device(), event.queue(), params, tmp0, tmp1, pfRecHits, pfClusters, pfrhFractions);

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
    const device::ESGetToken<PFClusterParamsAlpakaESDataDevice, JobConfigurationAlpakaRecord2> pfClusParamsToken;
    const edm::EDGetTokenT<reco::PFRecHitHostCollection> InputPFRecHitSoA_Token_;
    const device::EDPutToken<PFClusterDeviceCollection2> OutputPFClusterSoA_Token_;
    const device::EDPutToken<PFRHFractionDeviceCollection> OutputPFRHFractionSoA_Token_;
    const bool synchronise;
    const bool _produceSoA;
    const bool _produceLegacy;
    //const edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;
    //const device::EDPutToken<reco::PFClusterCollection> legacyClusToken;
    int nRH = 0;
    //PFClusterProducerKernel kernel;
    std::optional<PFClusterProducerKernel> kernel = {};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFClusterProducerAlpaka);
