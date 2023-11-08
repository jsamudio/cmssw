#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_PFClusteringEdgeVarsDevice_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_PFClusteringEdgeVarsDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/PFClusteringEdgeVarsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using PFClusteringEdgeVarsDeviceCollection = PortableCollection<::reco::PFClusteringEdgeVarsSoA>;

}

#endif
