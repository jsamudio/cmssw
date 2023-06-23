#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitHBHEParamsAlpakaESData_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_PFRecHitHBHEParamsAlpakaESData_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitHBHEParamsAlpakaESData.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitHBHEParamsAlpakaESDataSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using PFRecHitHBHEParamsAlpakaESDataHost = reco::PFRecHitHBHEParamsAlpakaESDataHost;
  using PFRecHitHBHEParamsAlpakaESDataDevice = PortableCollection<reco::PFRecHitHBHEParamsAlpakaESDataSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
