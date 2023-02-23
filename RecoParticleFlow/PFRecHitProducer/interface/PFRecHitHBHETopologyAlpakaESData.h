#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHETopologyAlpakaESData_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHETopologyAlpakaESData_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitHBHETopologyAlpakaESDataSoA.h"

namespace reco {

  using PFRecHitHBHETopologyAlpakaESDataHost = PortableHostCollection<PFRecHitHBHETopologyAlpakaESDataSoA>;

}

#endif
