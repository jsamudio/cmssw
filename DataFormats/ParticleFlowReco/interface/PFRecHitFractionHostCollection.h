#ifndef ParticleFlowReco_PFRecHitFractionHostCollection_h
#define ParticleFlowReco_PFRecHitFractionHostCollection_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {
  using PFRecHitFractionHostCollection = PortableHostCollection<PFRecHitFractionSoA>;
}

#endif

