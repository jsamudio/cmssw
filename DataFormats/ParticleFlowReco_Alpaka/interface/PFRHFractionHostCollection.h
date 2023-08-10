#ifndef ParticleFlowReco_PFRHFractionHostCollection_h
#define ParticleFlowReco_PFRHFractionHostCollection_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFRHFractionSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace reco {
  using PFRHFractionHostCollection = PortableHostCollection<PFRHFractionSoA>;
}

#endif
