#ifndef RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h
#define RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/alpaka/PFRHFractionDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/tmpPF0Device.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/tmpPF1Device.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    typedef struct float4 {
      float x;
      float y;
      float z;
      float w;
    } float4;
  class PFClusterProducerKernel {
  public:
    static PFClusterProducerKernel Construct(Queue& queue,
                 const reco::PFRecHitHostCollection& pfRecHits
            );

    void execute(const Device&,
                 Queue& queue,
                 const PFClusterParamsAlpakaESDataDevice& params,
                 tmpPF0DeviceCollection& tmp0,
                 tmpPF1DeviceCollection& tmp1,
                 const reco::PFRecHitHostCollection& pfRecHits,
                 PFClusterDeviceCollection2& pfClusters,
                 PFRHFractionDeviceCollection& pfrhFractions);


  private:
    PFClusterProducerKernel(cms::alpakatools::device_buffer<Device, uint32_t>&&,
                            cms::alpakatools::device_buffer<Device, float4[]>&&,
                            cms::alpakatools::device_buffer<Device, float4[]>&&,
                            cms::alpakatools::device_buffer<Device, float[]>&&,
                            cms::alpakatools::device_buffer<Device, float[]>&&,
                            cms::alpakatools::device_buffer<Device, int[]>&&,
                            cms::alpakatools::device_buffer<Device, int[]>&&);
    cms::alpakatools::device_buffer<Device, uint32_t> nSeeds;
    cms::alpakatools::device_buffer<Device, float4[]> globalClusterPos;
    cms::alpakatools::device_buffer<Device, float4[]> globalPrevClusterPos;
    cms::alpakatools::device_buffer<Device, float[]> globalClusterEnergy;
    cms::alpakatools::device_buffer<Device, float[]> globalRhFracSum;
    cms::alpakatools::device_buffer<Device, int[]> globalSeeds;
    cms::alpakatools::device_buffer<Device, int[]> globalRechits;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
