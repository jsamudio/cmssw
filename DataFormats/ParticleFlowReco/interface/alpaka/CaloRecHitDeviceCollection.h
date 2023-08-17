#ifndef DataFormats_ParticleFlowReco_CaloRecHitDeviceCollection_h
#define DataFormats_ParticleFlowReco_CaloRecHitDeviceCollection_h

#include "DataFormats/ParticleFlowReco/interface/CaloRecHitSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using CaloRecHitDeviceCollection = PortableCollection<reco::CaloRecHitSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif