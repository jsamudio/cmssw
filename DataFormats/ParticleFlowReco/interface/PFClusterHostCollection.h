#ifndef ParticleFlowReco_PFClusterHostCollection_h
#define ParticleFlowReco_PFClusterHostCollection_h

#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {

  using PFClusterHostCollection = PortableHostCollection<PFClusterSoA>;

}  // namespace reco

#endif

