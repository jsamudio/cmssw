#ifndef RecoParticleFlow_PFRecHitProducer_PFRecHitProducerKernel_h
#define RecoParticleFlow_PFRecHitProducer_PFRecHitProducerKernel_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitHBHEParamsAlpakaESData.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFRecHitProducerKernel {
  public:
    void execute(Queue& queue,
      const PFRecHitHBHEParamsAlpakaESDataDevice& params,
      const CaloRecHitDeviceCollection& recHits,
      PFRecHitDeviceCollection& collection) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif 
