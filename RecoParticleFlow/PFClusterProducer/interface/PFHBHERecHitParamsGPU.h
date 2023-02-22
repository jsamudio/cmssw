#ifndef RecoParticleFlow_PFClusterProducer_interface_PFHBHERecHitParamsGPU_h
#define RecoParticleFlow_PFClusterProducer_interface_PFHBHERecHitParamsGPU_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#endif

class PFHBHERecHitParamsGPU {
public:
  struct Product {
    ~Product();
    edm::propagate_const<cms::cuda::device::unique_ptr<int>> nDepthHB;
    edm::propagate_const<cms::cuda::device::unique_ptr<int>> nDepthHE;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> thresholdE_HB;
    edm::propagate_const_array<cms::cuda::device::unique_ptr<float[]>> thresholdE_HE;
  };

#ifndef __CUDACC__
  // rearrange reco params
  PFHBHERecHitParamsGPU(edm::ParameterSet const&);

  // will trigger deallocation of Product thru ~Product
  ~PFHBHERecHitParamsGPU() = default;

  // get device pointers
  Product const& getProduct(cudaStream_t) const;

  //using intvec = std::reference_wrapper<std::vector<int, cms::cuda::HostAllocator<int>> const>;
  //using uint32vec = std::reference_wrapper<std::vector<uint32_t, cms::cuda::HostAllocator<uint32_t>> const>;
  //using floatvec = std::reference_wrapper<std::vector<float, cms::cuda::HostAllocator<float>> const>;

  int const& getValueNDepthHB() const { return nDepthHB_; }
  int const& getValueNDepthHE() const { return nDepthHE_; }
  std::vector<float, cms::cuda::HostAllocator<float>> const& getValuesThresholdE_HB() const { return thresholdE_HB_; }
  std::vector<float, cms::cuda::HostAllocator<float>> const& getValuesThresholdE_HE() const { return thresholdE_HE_; }

private:
  int nDepthHB_;
  int nDepthHE_;
  std::vector<float, cms::cuda::HostAllocator<float>> thresholdE_HB_;
  std::vector<float, cms::cuda::HostAllocator<float>> thresholdE_HE_;

  cms::cuda::ESProduct<Product> product_;
#endif
};

#endif
