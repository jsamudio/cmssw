#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_ClusteringEdgeVarsDevice_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_ClusteringEdgeVarsDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/ClusteringEdgeVarsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ClusteringEdgeVarsDeviceCollection = PortableCollection<::reco::ClusteringEdgeVarsSoA>;

}

#endif
