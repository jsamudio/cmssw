#ifndef RecoParticleFlow_PFRecHitProducer_alpaka_TopologyDeviceCollection_h
#define RecoParticleFlow_PFRecHitProducer_alpaka_TopologyDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologySoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PFRecHitHCALTopologyDeviceCollection = PortableCollection<reco::PFRecHitHCALTopologySoA>;
  using PFRecHitECALTopologyDeviceCollection = PortableCollection<reco::PFRecHitECALTopologySoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
