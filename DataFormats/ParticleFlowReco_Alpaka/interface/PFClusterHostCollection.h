#ifndef ParticleFlowReco_PFClusterHostCollection_h
#define ParticleFlowReco_PFClusterHostCollection_h

#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFClusterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namspace reco {
    using PFClusterHostCollection = PortableHostCollection<PFClusterSoA>;
}

#endif
