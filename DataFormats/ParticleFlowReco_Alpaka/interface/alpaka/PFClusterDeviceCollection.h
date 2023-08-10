#ifndef ParticleFlowReco_PFClusterDeviceCollection_h
#define ParticleFlowReco_PFClusterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFClusterSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PFClusterDeviceCollection2 = PortableCollection<reco2::PFClusterSoA2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

