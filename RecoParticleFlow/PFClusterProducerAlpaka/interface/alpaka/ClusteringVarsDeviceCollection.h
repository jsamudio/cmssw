#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_ClusteringVarsDevice_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_ClusteringVarsDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/ClusteringVarsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ClusteringVarsDeviceCollection = PortableCollection<::reco::ClusteringVarsSoA>;

}

#endif
