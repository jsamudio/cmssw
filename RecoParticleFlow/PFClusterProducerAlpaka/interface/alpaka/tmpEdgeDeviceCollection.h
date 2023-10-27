#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpEdgeDevice_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpEdgeDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/tmpEdgeSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using tmpEdgeDeviceCollection = PortableCollection<::reco::tmpEdgeSoA>;

}

#endif
