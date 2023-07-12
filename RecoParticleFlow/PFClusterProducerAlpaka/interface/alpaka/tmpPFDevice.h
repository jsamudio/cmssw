#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpPFDevice_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpPFDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/tmpPFDeviceSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    using tmpPFDeviceCollection = PortableCollection<reco::tmpPFDeviceSoA>;

}

#endif
