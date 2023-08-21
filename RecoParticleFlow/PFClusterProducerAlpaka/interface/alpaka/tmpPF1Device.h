#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpPF1Device_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_alpaka_tmpPF1Device_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/tmpPF1DeviceSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using tmpPF1DeviceCollection = PortableCollection<reco::tmpPFDeviceSoA2>;

}

#endif
