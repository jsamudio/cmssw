#ifndef ParticleFlowReco_PFClusterDeviceCollection_h
#define ParticleFlowReco_PFClusterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PFClusterDeviceCollection = PortableCollection<reco::PFClusterSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

