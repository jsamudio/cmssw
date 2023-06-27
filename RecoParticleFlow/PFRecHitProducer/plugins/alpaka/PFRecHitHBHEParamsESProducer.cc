#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/JobConfigurationAlpakaRecord.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitHBHEParamsAlpakaESData.h"

#include <array>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFRecHitHBHEParamsESProducer : public ESProducer {
  public:
    PFRecHitHBHEParamsESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig),
        energyThresholdsHB_(iConfig.getParameter<std::array<double, kMaxDepthHB>>("energyThresholdsHB")),
        energyThresholdsHE_(iConfig.getParameter<std::array<double, kMaxDepthHE>>("energyThresholdsHE")) {
      setWhatProduced(this);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("appendToDataLabel", "");
      desc.add<std::vector<double>>("energyThresholdsHB", {0.1, 0.2, 0.3, 0.3});
      desc.add<std::vector<double>>("energyThresholdsHE", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<PFRecHitHBHEParamsAlpakaESDataHost> produce(JobConfigurationAlpakaRecord const& iRecord) {
      auto product = std::make_unique<PFRecHitHBHEParamsAlpakaESDataHost>(kMaxDepthHB + kMaxDepthHE, cms::alpakatools::host());
      for (int idx = 0; idx < kMaxDepthHB; ++idx) {
        product->view().energyThresholds()[idx] = energyThresholdsHB_[idx];
      }
      for (int idx = 0; idx < kMaxDepthHE; ++idx) {
        product->view().energyThresholds()[idx+kMaxDepthHB] = energyThresholdsHE_[idx];
      }
      return product;
    }

  private:
    constexpr static uint8_t kMaxDepthHB = 4;
    constexpr static uint8_t kMaxDepthHE = 7;

    std::array<double, kMaxDepthHB> energyThresholdsHB_;
    std::array<double, kMaxDepthHE> energyThresholdsHE_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFRecHitHBHEParamsESProducer);
