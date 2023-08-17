#ifndef RecoParticleFlow_PFRecHitProducer_alpaka_ParamsDeviceCollection_h
#define RecoParticleFlow_PFRecHitProducer_alpaka_ParamsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PFRecHitHCALParamsDeviceCollection = PortableCollection<reco::PFRecHitHCALParamsSoA>;
  using PFRecHitECALParamsDeviceCollection = PortableCollection<reco::PFRecHitECALParamsSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
