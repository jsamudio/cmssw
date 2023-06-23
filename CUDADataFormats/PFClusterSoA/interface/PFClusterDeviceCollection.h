#ifndef CUDADataFormats_PFClusterSoA_interface_PFClusterDeviceCollection_h
#define CUDADataFormats_PFClusterSoA_interface_PFClusterDeviceCollection_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"

namespace reco {

         using PFClusterDeviceCollection = cms::cuda::PortableDeviceCollection<reco::PFClusterSoA>;
         using PFClusterDeviceMultiCollection = cms::cuda::PortableDeviceCollection2<reco::PFClusterSoA, reco::PFRHFracSoA>;
      
}  // namespace reco
      
#endif
