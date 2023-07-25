#ifndef ParticleFlowReco_PFClusterHostCollection_h
#define ParticleFlowReco_PFClusterHostCollection_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFClusterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco2 {

  using PFClusterHostCollection2 = PortableHostCollection2<PFClusterSoA2, PFRHFracSoA2>;

} // namespace reco

#endif

