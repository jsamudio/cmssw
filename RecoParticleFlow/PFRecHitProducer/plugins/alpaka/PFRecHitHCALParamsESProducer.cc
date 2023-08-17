#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsRecord.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/CalorimeterDefinitions.h"

#include <array>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace ParticleFlowRecHitProducerAlpaka;

  class PFRecHitHCALParamsESProducer : public ESProducer {
  public:
    PFRecHitHCALParamsESProducer(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig),
          energyThresholdsHB_(iConfig.getParameter<std::array<double, HCAL::maxDepthHB>>("energyThresholdsHB")),
          energyThresholdsHE_(iConfig.getParameter<std::array<double, HCAL::maxDepthHE>>("energyThresholdsHE")) {
      setWhatProduced(this);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("appendToDataLabel", "");
      desc.add<std::vector<double>>("energyThresholdsHB", {0.1, 0.2, 0.3, 0.3});
      desc.add<std::vector<double>>("energyThresholdsHE", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<reco::PFRecHitHCALParamsHostCollection> produce(PFRecHitHCALParamsRecord const& iRecord) {
      auto product = std::make_unique<reco::PFRecHitHCALParamsHostCollection>(HCAL::maxDepthHB + HCAL::maxDepthHE,
                                                                              cms::alpakatools::host());
      for (uint32_t idx = 0; idx < HCAL::maxDepthHB; ++idx) {
        product->view().energyThresholds()[idx] = energyThresholdsHB_[idx];
      }
      for (uint32_t idx = 0; idx < HCAL::maxDepthHE; ++idx) {
        product->view().energyThresholds()[idx + HCAL::maxDepthHB] = energyThresholdsHE_[idx];
      }
      return product;
    }

  private:
    std::array<double, HCAL::maxDepthHB> energyThresholdsHB_;
    std::array<double, HCAL::maxDepthHE> energyThresholdsHE_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFRecHitHCALParamsESProducer);
