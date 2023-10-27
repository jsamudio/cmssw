#ifndef RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h
#define RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/tmpDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/tmpEdgeDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  typedef struct float4 {
    float x;
    float y;
    float z;
    float w;
  } float4;

  typedef struct float3 {
    float x;
    float y;
    float z;
  } float3;

  typedef struct int4 {
    int x;
    int y;
    int z;
    int w;
  } int4;

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float3 make_float3(float x, float y, float z) {
    float3 tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    return tmp;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float4 make_float4(float x, float y, float z, float w) {
    float4 tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    tmp.w = w;
    return tmp;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE int4 make_int4(int x, int y, int z, int w) {
    int4 tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    tmp.w = w;
    return tmp;
  }

  class PFClusterProducerKernel {
  public:
    static PFClusterProducerKernel Construct(Queue& queue, const reco::PFRecHitHostCollection& pfRecHits);

    void execute(const Device&,
                 Queue& queue,
                 const reco::PFClusterParamsAlpakaESDataDevice& params,
                 reco::tmpDeviceCollection& tmp0,
                 reco::tmpEdgeDeviceCollection& tmp1,
                 const reco::PFRecHitHostCollection& pfRecHits,
                 reco::PFClusterDeviceCollection& pfClusters,
                 reco::PFRecHitFractionDeviceCollection& pfrhFractions);

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
