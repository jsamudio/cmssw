#ifndef RecoParticleFlow_PFClusterProducerAlpaka_test_alpaka_TestAlgo_h
#define RecoParticleFlow_PFClusterProducerAlpaka_test_alpaka_TestAlgo_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestAlgo {
  public:
    void printPFClusterESData(Queue& queue, PFClusterParamsAlpakaESDataDevice const& esParams) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoParticleFlow_PFClusterProducerAlpaka_plugins_alpaka_TestAlgo_h
