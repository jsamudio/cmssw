#ifndef RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h
#define RecoParticleFlow_PFClusterProducerAlpaka_PFClusterProducerAlpakaKernel_h

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/ClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/ClusteringEdgeVarsDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct Position4 {
    float x;
    float y;
    float z;
    float w;
  };

  struct Position3 {
    float x;
    float y;
    float z;
  };

  struct Neighbours4 {
    int x;
    int y;
    int z;
    int w;
  };

  class PFClusterProducerKernel {
  public:
    static PFClusterProducerKernel Construct(Queue& queue, const reco::PFRecHitHostCollection& pfRecHits);

    void execute(const Device&,
                 Queue& queue,
                 const reco::PFClusterParamsAlpakaESDataDevice& params,
                 reco::ClusteringVarsDeviceCollection& clusteringVars,
                 reco::ClusteringEdgeVarsDeviceCollection& clusteringEdgeVars,
                 const reco::PFRecHitHostCollection& pfRecHits,
                 reco::PFClusterDeviceCollection& pfClusters,
                 reco::PFRecHitFractionDeviceCollection& pfrhFractions);

  private:
    PFClusterProducerKernel(cms::alpakatools::device_buffer<Device, uint32_t>&&,
                            cms::alpakatools::device_buffer<Device, Position4[]>&&,
                            cms::alpakatools::device_buffer<Device, Position4[]>&&,
                            cms::alpakatools::device_buffer<Device, float[]>&&,
                            cms::alpakatools::device_buffer<Device, float[]>&&,
                            cms::alpakatools::device_buffer<Device, int[]>&&,
                            cms::alpakatools::device_buffer<Device, int[]>&&);
    cms::alpakatools::device_buffer<Device, uint32_t> nSeeds;
    cms::alpakatools::device_buffer<Device, Position4[]> globalClusterPos;
    cms::alpakatools::device_buffer<Device, Position4[]> globalPrevClusterPos;
    cms::alpakatools::device_buffer<Device, float[]> globalClusterEnergy;
    cms::alpakatools::device_buffer<Device, float[]> globalRhFracSum;
    cms::alpakatools::device_buffer<Device, int[]> globalSeeds;
    cms::alpakatools::device_buffer<Device, int[]> globalRechits;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
