#ifndef RecoParticleFlow_PFClusterProducer_interface_PFClusterParamsHostCollection_h
#define RecoParticleFlow_PFClusterProducer_interface_PFClusterParamsHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterParamsSoA.h"

namespace reco {

  using PFClusterParamsHostCollection = PortableHostCollection<PFClusterParamsSoA>;

}

#endif
