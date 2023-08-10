#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpPF0Device_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpPF0Device_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/tmpPF0DeviceSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

    using tmpPF0DeviceCollection = PortableCollection<reco::tmpPFDeviceSoA1>;

}

#endif
