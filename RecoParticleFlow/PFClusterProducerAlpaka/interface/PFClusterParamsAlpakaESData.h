#ifndef RecoParticleFlow_PFClusterProducerAlpaka_interface_PFClusterParamsAlpakaESData_h
#define RecoParticleFlow_PFClusterProducerAlpaka_interface_PFClusterParamsAlpakaESData_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/PFClusterParamsAlpakaESDataSoA.h"

namespace reco {

  using PFClusterParamsAlpakaESDataHost = PortableHostCollection<PFClusterParamsAlpakaESDataSoA>;

}

#endif
