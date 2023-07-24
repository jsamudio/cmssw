#ifndef ParticleFlowReco_PFClusterDeviceCollection_h
#define ParticleFlowReco_PFClusterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFClusterSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using PFClusterDeviceCollection = PortableCollection2<reco::PFClusterSoA, reco::PFRHFracSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif

