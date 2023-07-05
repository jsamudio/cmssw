#ifndef RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h
#define RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/PFClusterDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFClusterProducerAlpakaKernel {
  public:
    static PFClusterProducerAlpakaKernel Construct(Queue& queue);

    void execute(const Device& device, Queue& queue,
      const PFClusterParamsAlpakaESDataDevice& params,
      PFClusterDeviceCollection& collection);

  private:
    PFClusterProducerAlpakaKernel(cms::alpakatools::device_buffer<Device, uint32_t[]>&&);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
