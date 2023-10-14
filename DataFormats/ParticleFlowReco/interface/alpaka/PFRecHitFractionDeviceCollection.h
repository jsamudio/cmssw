#ifndef ParticleFlowReco_PFRecHitFractionDeviceCollection_h
#define ParticleFlowReco_PFRecHitFractionDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PFRecHitFractionDeviceCollection = PortableCollection<reco::PFRecHitFractionSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
