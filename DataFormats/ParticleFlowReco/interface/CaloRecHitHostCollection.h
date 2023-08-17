#ifndef DataFormats_ParticleFlowReco_CaloRecHitHostCollection_h
#define DataFormats_ParticleFlowReco_CaloRecHitHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/CaloRecHitSoA.h"

namespace reco {
  using CaloRecHitHostCollection = PortableHostCollection<CaloRecHitSoA>;
}

#endif