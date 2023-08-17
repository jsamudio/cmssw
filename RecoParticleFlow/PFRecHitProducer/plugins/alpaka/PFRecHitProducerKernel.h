#ifndef RecoParticleFlow_PFRecHitProducer_alpaka_PFRecHitProducerKernel_h
#define RecoParticleFlow_PFRecHitProducer_alpaka_PFRecHitProducerKernel_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/CalorimeterDefinitions.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename CAL>
  class PFRecHitProducerKernel {
  public:
    PFRecHitProducerKernel(Queue& queue);

    // Prepare for processing next event
    void prepare_event(Queue& queue, const uint32_t num_recHits);

    // Run kernel: apply filters to rec hits and construct PF rec hits
    void process_rec_hits(Queue& queue,
                          const CaloRecHitDeviceCollection& recHits,
                          const typename CAL::ParameterType& params,
                          PFRecHitDeviceCollection& pfRecHits);

    // Run kernel: Associate topology information (position, neighbours)
    void associate_topology_info(Queue& queue,
                                 const typename CAL::TopologyTypeDevice& topology,
                                 PFRecHitDeviceCollection& pfRecHits);

  private:
    cms::alpakatools::device_buffer<Device, uint32_t[]> denseId2pfRecHit;
    cms::alpakatools::device_buffer<Device, uint32_t> num_pfRecHits;
    WorkDiv<Dim1D> work_div;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
