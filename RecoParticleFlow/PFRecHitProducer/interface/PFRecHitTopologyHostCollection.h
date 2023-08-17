#ifndef RecoParticleFlow_PFRecHitProducer_TopologyHostCollection_h
#define RecoParticleFlow_PFRecHitProducer_TopologyHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologySoA.h"

namespace reco {
  using PFRecHitHCALTopologyHostCollection = PortableHostCollection<PFRecHitHCALTopologySoA>;
  using PFRecHitECALTopologyHostCollection = PortableHostCollection<PFRecHitECALTopologySoA>;
}  // namespace reco

#endif
