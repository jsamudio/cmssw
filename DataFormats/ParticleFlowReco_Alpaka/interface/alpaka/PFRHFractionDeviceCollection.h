#ifndef ParticleFlowReco_PFRHFractionDeviceCollection_h
#define ParticleFlowReco_PFRHFractionDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFRHFractionSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PFRHFractionDeviceCollection = PortableCollection<reco::PFRHFractionSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
