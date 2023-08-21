#ifndef CUDADataFormats_PFClusterSoA_interface_PFClusterHostCollection_h
#define CUDADataFormats_PFClusterSoA_interface_PFClusterHostCollection_h

#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"

namespace reco {

  using PFClusterHostCollection = cms::cuda::PortableHostCollection<reco::PFClusterSoA>;
  using PFClusterHostMultiCollection = cms::cuda::PortableHostCollection2<reco::PFClusterSoA, reco::PFRHFracSoA>;

}  // namespace reco

#endif
