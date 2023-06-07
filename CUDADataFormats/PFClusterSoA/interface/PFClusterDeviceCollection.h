#ifndef CUDADataFormats_PFClusterSoA_interface_PFClusterDeviceCollection_h
#define CUDADataFormats_PFClusterSoA_interface_PFClusterDeviceCollection_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"

namespace reco {

         using PFClusterDeviceCollection = cms::cuda::PortableDeviceCollection<reco::PFClusterSoA>;
      
}  // namespace reco
      
#endif
