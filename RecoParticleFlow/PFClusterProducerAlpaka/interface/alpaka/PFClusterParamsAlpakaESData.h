#ifndef RecoParticleFlow_PFClusterProducerAlpaka_interface_alpaka_PFClusterParamsAlpakaESData_h
#define RecoParticleFlow_PFClusterProducerAlpaka_interface_alpaka_PFClusterParamsAlpakaESData_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/PFClusterParamsAlpakaESDataSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::PFClusterParamsAlpakaESDataHost;

  using PFClusterParamsAlpakaESDataDevice = PortableCollection<::reco::PFClusterParamsAlpakaESDataSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
