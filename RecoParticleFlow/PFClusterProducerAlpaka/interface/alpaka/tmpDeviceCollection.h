#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpDevice_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/tmpSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using tmpDeviceCollection = PortableCollection<::reco::tmpSoA>;

}

#endif
