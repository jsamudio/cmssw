#ifndef RecoParticleFlow_PFRecHitProducer_ParamsHostCollection_h
#define RecoParticleFlow_PFRecHitProducer_ParamsHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsSoA.h"

namespace reco {
  using PFRecHitHCALParamsHostCollection = PortableHostCollection<PFRecHitHCALParamsSoA>;
  using PFRecHitECALParamsHostCollection = PortableHostCollection<PFRecHitECALParamsSoA>;
}  // namespace reco

#endif
