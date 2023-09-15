#ifndef RecoParticleFlow_PFRecHitProducer_alpaka_PFRecHitProducerKernel_h
#define RecoParticleFlow_PFRecHitProducer_alpaka_PFRecHitProducerKernel_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/CalorimeterDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename CAL>
  class PFRecHitProducerKernel {
  public:
    PFRecHitProducerKernel(Queue& queue);

    // Prepare for processing next event
    void prepareEvent(Queue& queue, const uint32_t num_recHits);

    // Run kernel: apply filters to rec hits and construct PF rec hits
    void processRecHits(Queue& queue,
                        const typename CAL::CaloRecHitSoATypeDevice& recHits,
                        const typename CAL::ParameterType& params,
                        PFRecHitDeviceCollection& pfRecHits);

    // Run kernel: Associate topology information (position, neighbours)
    void associateTopologyInfo(Queue& queue,
                               const typename CAL::TopologyTypeDevice& topology,
                               PFRecHitDeviceCollection& pfRecHits);

  private:
    cms::alpakatools::device_buffer<Device, uint32_t[]> denseId2pfRecHit_;
    cms::alpakatools::device_buffer<Device, uint32_t> num_pfRecHits_;
    WorkDiv<Dim1D> work_div_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
