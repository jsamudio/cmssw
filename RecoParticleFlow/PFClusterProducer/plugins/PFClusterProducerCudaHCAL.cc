#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "CUDADataFormats/PFClusterSoA/interface/PFClusterCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterSoA.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#include "CudaPFCommon.h"
#include "DeclsForKernels.h"
#include "PFClusterCudaHCAL.h"

#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusteringParamsGPU.h"

#include "CUDADataFormats/PFClusterSoA/interface/PFClusterDeviceCollection.h"
#include "CUDADataFormats/PFClusterSoA/interface/PFClusterHostCollection.h"

class PFClusterProducerCudaHCAL : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  PFClusterProducerCudaHCAL(const edm::ParameterSet&);
  ~PFClusterProducerCudaHCAL() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  // the actual algorithm
  std::unique_ptr<PFCPositionCalculatorBase> _positionCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPositionCalc;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void acquire(edm::Event const&, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  using IProductType = cms::cuda::Product<hcal::PFRecHitCollection<pf::common::DevStoragePolicy>>;
  edm::EDGetTokenT<IProductType> InputPFRecHitSoA_Token_;
  using OProductType = cms::cuda::Product<hcal::PFClusterCollection<pf::common::DevStoragePolicy>>;
  //using OProductType = cms::cuda::Product<reco::PFClusterDeviceCollection>;
  edm::EDPutTokenT<OProductType> OutputPFClusterSoA_Token_;

  edm::ESGetToken<PFClusteringParamsGPU, JobConfigurationGPURecord> const pfClusParamsToken_;

  edm::EDPutTokenT<cms::cuda::Product<reco::PFClusterDeviceCollection>> cluster_deviceToken_;

  // PFClusters for GPU
  hcal::PFClusterCollection<pf::common::VecStoragePolicy<pf::common::CUDAHostAllocatorAlias>> tmpPFClusters;

  int nRH_ = 0;
  // int nClustersMax_ = 0;
  // int nRHFracsMax = 0;

  const bool _produceSoA;     // PFClusters in SoA format
  const bool _produceLegacy;  // PFClusters in legacy format

  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;

  PFClustering::HCAL::ConfigurationParameters cudaConfig_;
  PFClustering::HCAL::OutputDataCPU outputCPU;
  PFClustering::HCAL::OutputDataGPU outputGPU;
  PFClustering::HCAL::OutputPFClusterDataGPU outputGPU2;
  PFClustering::HCAL::ScratchDataGPU scratchGPU;

  // PFCluster Portable Collection
  reco::PFClusterHostCollection clusters_h_;
  reco::PFClusterDeviceCollection clustersGPU;

  cms::cuda::ContextState cudaState_;
};

PFClusterProducerCudaHCAL::PFClusterProducerCudaHCAL(const edm::ParameterSet& conf)
    : InputPFRecHitSoA_Token_{consumes(conf.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
      OutputPFClusterSoA_Token_{produces<OProductType>(conf.getParameter<std::string>("PFClustersGPUOut"))},
      pfClusParamsToken_{esConsumes(conf.getParameter<edm::ESInputTag>("pfClusteringParameters"))},
      cluster_deviceToken_{produces(conf.getParameter<std::string>("PFClusterDeviceCollection"))},
      _produceSoA{conf.getParameter<bool>("produceSoA")},
      _produceLegacy{conf.getParameter<bool>("produceLegacy")},
      _rechitsLabel{consumes(conf.getParameter<edm::InputTag>("recHitsSource"))} {
  edm::ConsumesCollector cc = consumesCollector();

  //setup pf cluster builder if requested
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  if (!pfcConf.empty()) {
    if (pfcConf.exists("positionCalc")) {
      const edm::ParameterSet& acConf = pfcConf.getParameterSet("positionCalc");
      const std::string& algoac = acConf.getParameter<std::string>("algoName");
      _positionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
    }

    if (pfcConf.exists("allCellsPositionCalc")) {
      const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
      const std::string& algoac = acConf.getParameter<std::string>("algoName");
      _allCellsPositionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
    }
  }

  produces<reco::PFClusterCollection>();
}

PFClusterProducerCudaHCAL::~PFClusterProducerCudaHCAL() {}

void PFClusterProducerCudaHCAL::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("PFRecHitsLabelIn", edm::InputTag("particleFlowRecHitHBHE"));
  desc.add<std::string>("PFClustersGPUOut", "");
  desc.add<std::string>("PFClustersDeviceCollection", "");
  desc.add<bool>("produceSoA", true);
  desc.add<bool>("produceLegacy", true);

  desc.add<edm::ESInputTag>("pfClusteringParameters",
                            edm::ESInputTag("pfClusteringParamsGPUESSource", "pfClusParamsOffline"));

  // Prevents the producer and navigator parameter sets from throwing an exception
  // TODO: Replace with a proper parameter set description: twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
  desc.setAllowAnything();

  cdesc.addWithDefaultLabel(desc);
}

void PFClusterProducerCudaHCAL::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  // KenH: Nothing for now. But we expect to use it soon for geometry/topology related info
}

void PFClusterProducerCudaHCAL::acquire(edm::Event const& event,
                                        edm::EventSetup const& setup,
                                        edm::WaitingTaskWithArenaHolder holder) {
  // Creates a new Cuda stream
  // TODO: Reuse stream from GPU PFRecHitProducer by passing input product as first arg
  // cmssdt.cern.ch/lxr/source/HeterogeneousCore/CUDACore/interface/ScopedContext.h#0101
  //cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};
  //cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder)};
  auto const& PFRecHitsProduct = event.get(InputPFRecHitSoA_Token_);
  cms::cuda::ScopedContextAcquire ctx{PFRecHitsProduct, std::move(holder), cudaState_};
  auto const& PFRecHits = ctx.get(PFRecHitsProduct);
  auto cudaStream = ctx.stream();
  //auto const& clusters_d = ctx.get(event, cluster_deviceToken_); 

  nRH_ = PFRecHits.size;
  if (nRH_ == 0)
    return;
  if (nRH_ > 4000)
    std::cout << "nRH(PFRecHitSize)>4000: " << nRH_ << std::endl;

  // Allocate outputGPU & scratchGPU data
  scratchGPU.allocate(cudaConfig_, nRH_, cudaStream);
  outputGPU.allocate(nRH_, cudaStream);

  float kernelTimers[8] = {0.0};

  auto const& pfClusParamsProduct = setup.getData(pfClusParamsToken_).getProduct(cudaStream);
  reco::PFClusterDeviceCollection clustersGPU;

  // Calling cuda kernels
  PFClusterCudaHCAL::PFRechitToPFCluster_HCAL_entryPoint(
      cudaStream, pfClusParamsProduct, PFRecHits, outputGPU2, outputGPU, scratchGPU, clustersGPU, kernelTimers);
      //cudaStream, pfClusParamsProduct, PFRecHits, clustersGPU, outputGPU, scratchGPU, kernelTimers);

  if (!_produceLegacy)
    return;  // do device->host transfer only when we are producing Legacy data

  //
  // --- Data transfers for array
  //

  //outputCPU.allocate(nRH_, cudaStream);

  // Data transfer from GPU
  /*
  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));
  */

  //int nTopos_h;
  //int nSeeds_h;
  //int nRHFracs_h;
  //cudaCheck(cudaMemcpyAsync(&nTopos_h, scratchGPU.nTopos.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  //cudaCheck(cudaMemcpyAsync(&nSeeds_h, scratchGPU.nSeeds.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  //cudaCheck(cudaMemcpyAsync(&nRHFracs_h, scratchGPU.nRHFracs.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));

  // cms::cuda::copyAsync(outputCPU.pcrhFracSize, outputGPU.pcrhFracSize, 1, cudaStream);

  // const Int_t nFracs2 = outputCPU.pcrhFracSize[0];

  // if (cudaStreamQuery(cudaStream) != cudaSuccess)
  //   cudaCheck(cudaStreamSynchronize(cudaStream));

  // Total size of allocated rechit fraction arrays (includes some extra padding for rechits that don't end up passing cuts)
  //const Int_t nFracs = outputCPU.pcrhFracSize[0];
  //std::cout << nRHFracs_h << " " << nFracs << " " << nFracs2 << std::endl;

  // allocate outputCPU - will go away soon and we will use tmpPFClusters + OutputPFClusterSoA_Token_ + outputGPU2.PFClusters
  //outputCPU.allocate_rhfrac(nFracs, cudaStream);

  //
  // --- Traditional SoA from Mark
  //

  /*
  cms::cuda::copyAsync(outputCPU.topoSeedCount, outputGPU.topoSeedCount, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.topoRHCount, outputGPU.topoRHCount, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.seedFracOffsets, outputGPU.seedFracOffsets, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.pfrh_isSeed, outputGPU.pfrh_isSeed, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.pfrh_topoId, outputGPU.pfrh_topoId, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.pcrh_fracInd, outputGPU.pcrh_fracInd, nFracs, cudaStream);
  cms::cuda::copyAsync(outputCPU.pcrh_frac, outputGPU.pcrh_frac, nFracs, cudaStream);
  */

  //
  // --- Newer SoA with proper length
  //

  // --- SoA transfers -----
  // Copy back PFCluster SoA data to CPU

  const int nSeeds_h = outputGPU2.PFClusters.size;
  const int nRHFracs_h = outputGPU2.PFClusters.sizeCleaned;
  tmpPFClusters.resize(outputGPU2.PFClusters.size);
  tmpPFClusters.resizeRecHitFrac(outputGPU2.PFClusters.sizeCleaned);
  auto lambdaToTransferSize = [&ctx](auto& dest, auto* src, auto size) {
    using vector_type = typename std::remove_reference<decltype(dest)>::type;
    using src_data_type = typename std::remove_pointer<decltype(src)>::type;
    using type = typename vector_type::value_type;
    static_assert(std::is_same<src_data_type, type>::value && "Dest and Src data types do not match");
    cudaCheck(cudaMemcpyAsync(dest.data(), src, size * sizeof(type), cudaMemcpyDeviceToHost, ctx.stream()));
  };
  lambdaToTransferSize(tmpPFClusters.pfc_seedRHIdx, outputGPU2.PFClusters.pfc_seedRHIdx.get(), nSeeds_h);
  lambdaToTransferSize(tmpPFClusters.pfc_topoId, outputGPU2.PFClusters.pfc_topoId.get(), nSeeds_h);
  //lambdaToTransferSize(tmpPFClusters.pfc_depth, outputGPU2.PFClusters.pfc_depth.get(), nSeeds_h);
  lambdaToTransferSize(tmpPFClusters.pfc_rhfracOffset, outputGPU2.PFClusters.pfc_rhfracOffset.get(), nSeeds_h);
  lambdaToTransferSize(tmpPFClusters.pfc_rhfracSize, outputGPU2.PFClusters.pfc_rhfracSize.get(), nSeeds_h);
  //lambdaToTransferSize(tmpPFClusters.pfc_energy, outputGPU2.PFClusters.pfc_energy.get(), nSeeds_h);
  //lambdaToTransferSize(tmpPFClusters.pfc_x, outputGPU2.PFClusters.pfc_x.get(), nSeeds_h);
  //lambdaToTransferSize(tmpPFClusters.pfc_y, outputGPU2.PFClusters.pfc_y.get(), nSeeds_h);
  //lambdaToTransferSize(tmpPFClusters.pfc_z, outputGPU2.PFClusters.pfc_z.get(), nSeeds_h);
  lambdaToTransferSize(tmpPFClusters.pcrh_frac, outputGPU2.PFClusters.pcrh_frac.get(), nRHFracs_h);
  lambdaToTransferSize(tmpPFClusters.pcrh_pfrhIdx, outputGPU2.PFClusters.pcrh_pfrhIdx.get(), nRHFracs_h);
  //lambdaToTransferSize(tmpPFClusters.pcrh_pfcIdx, outputGPU2.PFClusters.pcrh_pfcIdx.get(), nRHFracs_h);
  //cms::cuda::copyAsync(tmpPFClusters.pfc_seedRHIdx, outputGPU2.PFClusters.pfc_seedRHIdx, nSeeds_h, cudaStream);
  
  clusters_h_ = reco::PFClusterHostCollection(tmpPFClusters.size, ctx.stream());
}

void PFClusterProducerCudaHCAL::produce(edm::Event& event, const edm::EventSetup& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};
  if (_produceSoA)
    ctx.emplace(event,
                OutputPFClusterSoA_Token_,
                std::move(outputGPU2.PFClusters));  // SoA "PFClusters" still need to be defined.
                //std::move(clusters_h_));  // SoA "PFClusters" still need to be defined.

  if (_produceLegacy) {
    auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
    pfClustersFromCuda->reserve(nRH_);

    auto const rechitsHandle = event.getHandle(_rechitsLabel);

    std::vector<int> seedlist1;
    std::vector<int> seedlist2;
    tmpPFClusters.size = outputGPU2.PFClusters.size;
    for (unsigned i = 0; i < tmpPFClusters.size; i++) {
      seedlist1.push_back(tmpPFClusters.pfc_seedRHIdx[i]);
    }

    // Build PFClusters in legacy format
    std::unordered_map<int, int> nTopoSeeds;

    for (uint32_t i = 0; i < tmpPFClusters.size; i++) {
      nTopoSeeds[tmpPFClusters.pfc_topoId[i]]++;
    }

    // Looping over SoA PFClusters to produce legacy PFCluster collection
    for (uint32_t i = 0; i < tmpPFClusters.size; i++) {
      int n = tmpPFClusters.pfc_seedRHIdx[i];
      reco::PFCluster temp;
      temp.setSeed((*rechitsHandle)[n].detId());  // Pulling the detId of this PFRecHit from the legacy format input
      seedlist2.push_back(n);
      int offset = tmpPFClusters.pfc_rhfracOffset[i];
      for (int k = offset; k < (offset + tmpPFClusters.pfc_rhfracSize[i]);
           k++) {  // Looping over PFRecHits in the same topo cluster
        if (tmpPFClusters.pcrh_pfrhIdx[k] > -1 && tmpPFClusters.pcrh_frac[k] > 0.0) {
          const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechitsHandle, tmpPFClusters.pcrh_pfrhIdx[k]);
          temp.addRecHitFraction(reco::PFRecHitFraction(refhit, tmpPFClusters.pcrh_frac[k]));
        }
      }
      // Now PFRecHitFraction of this PFCluster is set. Now compute calculateAndSetPosition (energy, position etc)
      if (nTopoSeeds.count(tmpPFClusters.pfc_topoId[i]) == 1 && _allCellsPositionCalc) {
        _allCellsPositionCalc->calculateAndSetPosition(temp);
      } else {
        _positionCalc->calculateAndSetPosition(temp);
      }
      pfClustersFromCuda->emplace_back(std::move(temp));
      //}
    }

    event.put(std::move(pfClustersFromCuda));

    sort(seedlist1.begin(), seedlist1.end());
    sort(seedlist2.begin(), seedlist2.end());
    /*
    std::cout << seedlist1.size() << " " << seedlist2.size() << std::endl;
    for (unsigned int i = 0; i < seedlist1.size(); i++){
      std::cout << i << " " << seedlist1[i] << " " << seedlist2[i] << std::endl;
    }
    */
  }
}

DEFINE_FWK_MODULE(PFClusterProducerCudaHCAL);
