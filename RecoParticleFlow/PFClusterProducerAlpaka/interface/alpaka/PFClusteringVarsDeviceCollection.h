#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_PFClusteringVarsDevice_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_PFClusteringVarsDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/PFClusteringVarsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using PFClusteringVarsDeviceCollection = PortableCollection<::reco::PFClusteringVarsSoA>;

}

#endif
