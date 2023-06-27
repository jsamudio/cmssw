#ifndef ParticleFlowReco_PFClusterHostCollection_h
#define ParticleFlowReco_PFClusterHostCollection_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFClusterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {

  using PFClusterHostCollection = PortableHostCollection2<reco::PFClusterSoA, reco::PFRHFracSoA>;

} // namespace reco

#endif

