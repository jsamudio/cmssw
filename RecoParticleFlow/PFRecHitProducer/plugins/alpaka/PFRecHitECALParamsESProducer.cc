#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsRecord.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/CalorimeterDefinitions.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"

#include <array>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace ParticleFlowRecHitProducerAlpaka;

  class PFRecHitECALParamsESProducer : public ESProducer {
  public:
    PFRecHitECALParamsESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      thresholdsToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("appendToDataLabel", "");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<reco::PFRecHitECALParamsHostCollection> produce(const PFRecHitECALParamsRecord& iRecord) {
      const auto& thresholds = iRecord.get(thresholdsToken_);
      auto product = std::make_unique<reco::PFRecHitECALParamsHostCollection>(ECAL::SIZE, cms::alpakatools::host());
      for (uint32_t denseId = 0; denseId < ECAL::BARREL::SIZE; denseId++)
        product->view().energyThresholds()[denseId] = thresholds.barrel(denseId);
      for (uint32_t denseId = 0; denseId < ECAL::ENDCAP::SIZE; denseId++)
        product->view().energyThresholds()[denseId + ECAL::BARREL::SIZE] = thresholds.endcap(denseId);
      return product;
    }

  private:
    edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> thresholdsToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFRecHitECALParamsESProducer);
