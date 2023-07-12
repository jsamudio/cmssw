#ifndef RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h
#define RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/PFClusterDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/tmpPFDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFClusterProducerKernel {
  public:
    //static PFClusterProducerKernel Construct(Queue& queue);

    void execute(const Device&,
                 Queue& queue,
                 const PFClusterParamsAlpakaESDataDevice& params,
                 tmpPFDeviceCollection& tmp,
                 const reco::PFRecHitHostCollection& pfRecHits,
                 PFClusterDeviceCollection2& pfClusters);

  //private:
  //  PFClusterProducerKernel(cms::alpakatools::device_buffer<Device, uint32_t[]>&&);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
