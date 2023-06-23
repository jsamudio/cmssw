#ifndef ParticleFlowReco_CaloRecHitHostCollection_h
#define ParticleFlowReco_CaloRecHitHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/CaloRecHitSoA.h"

namespace reco {
  using CaloRecHitHostCollection = PortableHostCollection<CaloRecHitSoA>;
}

#endif