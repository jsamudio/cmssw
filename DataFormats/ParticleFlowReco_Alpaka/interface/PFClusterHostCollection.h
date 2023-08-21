#ifndef ParticleFlowReco_PFClusterHostCollection_h
#define ParticleFlowReco_PFClusterHostCollection_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFClusterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco2 {

  using PFClusterHostCollection2 = PortableHostCollection<PFClusterSoA2>;

}  // namespace reco2

#endif
