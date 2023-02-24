#ifndef RecoParticleFlow_PFRecHitProducer_interface_AlpakaESTestData_h
#define RecoParticleFlow_PFRecHitProducer_interface_AlpakaESTestData_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitHBHEParamsAlpakaESDataSoA.h"

namespace reco {

  using PFRecHitHBHEParamsAlpakaESDataHost = PortableHostCollection<PFRecHitHBHEParamsAlpakaESDataSoA>;

}

#endif
