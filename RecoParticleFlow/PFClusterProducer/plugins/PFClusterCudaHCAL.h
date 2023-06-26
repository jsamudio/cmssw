#ifndef RecoParticleFlow_PFClusterProducer_plugins_PFClusterCudaHCAL_h
#define RecoParticleFlow_PFClusterProducer_plugins_PFClusterCudaHCAL_h

#include <Eigen/Dense>

#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFClusteringParamsGPU.h"

#include "CudaPFCommon.h"
#include "DeclsForKernels.h"

#include "CUDADataFormats/PFClusterSoA/interface/PFClusterDeviceCollection.h"

namespace PFClusterCudaHCAL {

  void PFRechitToPFCluster_HCAL_entryPoint(cudaStream_t cudaStream,
                                           PFClusteringParamsGPU::DeviceProduct const&,
                                           ::hcal::PFRecHitCollection<::pf::common::DevStoragePolicy> const&,
                                           ::PFClustering::HCAL::OutputDataGPU&,
                                           ::PFClustering::HCAL::ScratchDataGPU&,
                                           reco::PFClusterDeviceMultiCollection&,
                                           float (&timer)[8]);

}  // namespace PFClusterCudaHCAL

#endif  // RecoParticleFlow_PFClusterProducer_plugins_PFClusterCudaHCAL_h
