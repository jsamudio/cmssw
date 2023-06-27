#ifndef RecoParticleFlow_PFRecHitProducer_PFRecHitProducerKernel_h
#define RecoParticleFlow_PFRecHitProducer_PFRecHitProducerKernel_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitHBHEParamsAlpakaESData.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitHBHETopologyAlpakaESData.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFRecHitProducerKernel {
  public:
    static PFRecHitProducerKernel Construct(Queue& queue);

    void execute(const Device& device, Queue& queue,
      const PFRecHitHBHEParamsAlpakaESDataDevice& params,
      const PFRecHitHBHETopologyAlpakaESDataDevice& topology,
      const CaloRecHitDeviceCollection& recHits,
      PFRecHitDeviceCollection& collection);

  private:
    PFRecHitProducerKernel(cms::alpakatools::device_buffer<Device, uint32_t[]>&&);
    cms::alpakatools::device_buffer<Device, uint32_t[]> denseId2pfRecHit;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif 
