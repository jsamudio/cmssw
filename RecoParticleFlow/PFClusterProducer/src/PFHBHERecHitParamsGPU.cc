#include "RecoParticleFlow/PFClusterProducer/interface/PFHBHERecHitParamsGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

PFHBHERecHitParamsGPU::PFHBHERecHitParamsGPU(edm::ParameterSet const& ps) {
  auto const& valuesThresholdE_HB = ps.getParameter<std::vector<double>>("thresholdE_HB");
  auto const& valuesThresholdE_HE = ps.getParameter<std::vector<double>>("thresholdE_HE");
  thresholdE_HB_.resize(valuesThresholdE_HB.size());
  thresholdE_HE_.resize(valuesThresholdE_HE.size());
  std::copy(valuesThresholdE_HB.begin(), valuesThresholdE_HB.end(), thresholdE_HB_.begin());
  std::copy(valuesThresholdE_HE.begin(), valuesThresholdE_HE.end(), thresholdE_HE_.begin());
}

PFHBHERecHitParamsGPU::Product::~Product() {
  // deallocation
  cudaCheck(cudaFree(thresholdE_HB));
  cudaCheck(cudaFree(thresholdE_HE));
}

PFHBHERecHitParamsGPU::Product const& PFHBHERecHitParamsGPU::getProduct(cudaStream_t cudaStream) const {
  auto const& product = product_.dataForCurrentDeviceAsync(
      cudaStream, [this](PFHBHERecHitParamsGPU::Product& product, cudaStream_t cudaStream) {
	//
	product.thresholdE_HB = cms::cuda::make_device_unique<float[]>(thresholdE_HB_.size(), cudaStream);
	product.thresholdE_HE = cms::cuda::make_device_unique<float[]>(thresholdE_HE_.size(), cudaStream);
	//
        // malloc
        //cudaCheck(cudaMalloc((void**)&product.thresholdE_HB, this->thresholdE_HB_.size() * sizeof(float)));
        //cudaCheck(cudaMalloc((void**)&product.thresholdE_HE, this->thresholdE_HE_.size() * sizeof(float)));

        // transfer
	cms::cuda::copyAsync(product.thresholdE_HB, thresholdE_HB_, cudaStream);
	cms::cuda::copyAsync(product.thresholdE_HE, thresholdE_HE_, cudaStream);
	//
	/*
        cudaCheck(cudaMemcpyAsync(product.thresholdE_HB,
                                  this->thresholdE_HB_.data(),
                                  this->thresholdE_HB_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
        cudaCheck(cudaMemcpyAsync(product.thresholdE_HE,
                                  this->thresholdE_HE_.data(),
                                  this->thresholdE_HE_.size() * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  cudaStream));
	*/
      });

  return product;
}

TYPELOOKUP_DATA_REG(PFHBHERecHitParamsGPU);
