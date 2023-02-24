#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitHBHETopologyAlpakaESData_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitHBHETopologyAlpakaESData_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitHBHETopologyAlpakaESData.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitHBHETopologyAlpakaESDataSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using PFRecHitHBHETopologyAlpakaESDataHost = reco::PFRecHitHBHETopologyAlpakaESDataHost;
  using PFRecHitHBHETopologyAlpakaESDataDevice = PortableCollection<reco::PFRecHitHBHETopologyAlpakaESDataSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
