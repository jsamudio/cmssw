#ifndef RecoParticleFlow_PFRecHitProducer_alpaka_CalorimeterDefinitions_h
#define RecoParticleFlow_PFRecHitProducer_alpaka_CalorimeterDefinitions_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace ParticleFlowRecHitProducerAlpaka {
    // structs to be used as template arguments
    struct HCAL {
      using CaloRecHitType = HBHERecHit;
    };

    struct ECAL {
      using CaloRecHitType = EcalRecHit;
    };
  }  // namespace ParticleFlowRecHitProducerAlpaka
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif