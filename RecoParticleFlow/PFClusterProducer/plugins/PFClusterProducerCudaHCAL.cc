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

  edm::EDGetTokenT<cms::cuda::Product<hcal::PFRecHitCollection<pf::common::DevStoragePolicy>>> InputPFRecHitSoA_Token_;
  edm::EDPutTokenT<cms::cuda::Product<hcal::PFClusterCollection<pf::common::DevStoragePolicy>>>
      OutputPFClusterSoA_Token_;

  edm::ESGetToken<PFClusteringParamsGPU, JobConfigurationGPURecord> const pfClusParamsToken_;

  // PFClusters for GPU
  hcal::PFClusterCollection<pf::common::VecStoragePolicy<pf::common::CUDAHostAllocatorAlias>> tmpPFClusters;

  int nRH_ = 0;

  const bool _produceSoA;     // PFClusters in SoA format
  const bool _produceLegacy;  // PFClusters in legacy format

  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;

  //cms::cuda::ContextState cudaState_;

  PFClustering::HCAL::ConfigurationParameters cudaConfig_;
  PFClustering::HCAL::OutputDataCPU outputCPU;
  PFClustering::HCAL::OutputDataGPU outputGPU;
  PFClustering::HCAL::OutputPFClusterDataGPU outputGPU2;
  PFClustering::HCAL::ScratchDataGPU scratchGPU;
};

PFClusterProducerCudaHCAL::PFClusterProducerCudaHCAL(const edm::ParameterSet& conf)
    : InputPFRecHitSoA_Token_{consumes(conf.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
      pfClusParamsToken_{esConsumes(conf.getParameter<edm::ESInputTag>("pfClusteringParameters"))},
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

  desc.add<edm::InputTag>("PFRecHitsLabelIn", edm::InputTag("hltParticleFlowRecHitHBHE"));
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
  cms::cuda::ScopedContextAcquire ctx{PFRecHitsProduct, std::move(holder)};
  auto const& PFRecHits = ctx.get(PFRecHitsProduct);
  auto cudaStream = ctx.stream();

  nRH_ = PFRecHits.size;
  if (nRH_ == 0)
    return;
  if (nRH_ > 4000)
    std::cout << "nRH(PFRecHitSize)>4000: " << nRH_ << std::endl;

  // Allocate scratchGPU data
  scratchGPU.allocate(cudaConfig_, nRH_, cudaStream);

  float kernelTimers[8] = {0.0};

  auto const& pfClusParamsProduct = setup.getData(pfClusParamsToken_).getProduct(cudaStream);

  // Calling cuda kernels
  PFClusterCudaHCAL::PFRechitToPFCluster_HCAL_entryPoint(
      cudaStream, pfClusParamsProduct, PFRecHits, outputGPU2, outputGPU, scratchGPU, kernelTimers);

  if (!_produceLegacy)
    return;  // do device->host transfer only when we are producing Legacy data

  //
  // --- Data transfers for array
  //

  outputCPU.allocate(nRH_, cudaStream);

  // Data transfer from GPU
  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

  int nTopos_h;
  int nSeeds_h;
  int nRHFracs_h;
  cudaCheck(cudaMemcpyAsync(&nTopos_h, scratchGPU.nTopos.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(&nSeeds_h, scratchGPU.nSeeds.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(&nRHFracs_h, scratchGPU.nRHFracs.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));

  cms::cuda::copyAsync(outputCPU.pcrhFracSize, outputGPU.pcrhFracSize, 1, cudaStream);

  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

  // Total size of allocated rechit fraction arrays (includes some extra padding for rechits that don't end up passing cuts)
  const Int_t nFracs = outputCPU.pcrhFracSize[0];
  // same as

  // allocate outputCPU - will go away soon and we will use tmpPFClusters + OutputPFClusterSoA_Token_ + outputGPU2.PFClusters
  outputCPU.allocate_rhfrac(nRHFracs_h, cudaStream);
  // CPU side nRHFracs_h. Keep both for now.

  //
  // --- Traditional SoA from Mark
  //

  cms::cuda::copyAsync(outputCPU.topoSeedCount, outputGPU.topoSeedCount, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.topoRHCount, outputGPU.topoRHCount, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.seedFracOffsets, outputGPU.seedFracOffsets, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.pfrh_isSeed, outputGPU.pfrh_isSeed, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.pfrh_topoId, outputGPU.pfrh_topoId, nRH_, cudaStream);
  cms::cuda::copyAsync(outputCPU.pcrh_fracInd, outputGPU.pcrh_fracInd, nRHFracs_h, cudaStream);
  cms::cuda::copyAsync(outputCPU.pcrh_frac, outputGPU.pcrh_frac, nRHFracs_h, cudaStream);

  //
  // --- Newer SoA with proper length
  //

  // --- SoA transfers -----
  // Copy back PFCluster SoA data to CPU

  tmpPFClusters.resize(nSeeds_h);
  tmpPFClusters.resizeRecHitFrac(nFracs);
  //lambdaToTransferSize(tmpPFClusters.pfc_rhfrac, outputGPU.pcrh_fracInd.get(), nFracs);
}

void PFClusterProducerCudaHCAL::produce(edm::Event& event, const edm::EventSetup& setup) {
  // cms::cuda::ScopedContextProduce ctx{cudaState_};
  // if (_produceSoA)
  //   ctx.emplace(event, OutputPFClusterSoA_Token_, std::move(outputGPU.PFClusters)); // SoA "PFClusters" still need to be defined.

  if (_produceLegacy) {
    auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
    pfClustersFromCuda->reserve(nRH_);

    auto const rechitsHandle = event.getHandle(_rechitsLabel);

    // Build PFClusters in legacy format
    std::unordered_map<int, std::vector<int>> nTopoRechits;
    std::unordered_map<int, int> nTopoSeeds;

    for (int rh = 0; rh < nRH_; rh++) {
      int topoId = outputCPU.pfrh_topoId[rh];
      if (topoId > -1) {
        // Valid topo id
        nTopoRechits[topoId].push_back(rh);
        if (outputCPU.pfrh_isSeed[rh] > 0) {
          nTopoSeeds[topoId]++;
        }
      }
    }

    // Looping over PFRecHits for creating PFClusters
    for (int n = 0; n < nRH_; n++) {
      if (outputCPU.pfrh_isSeed[n] ==
          1) {  // If this PFRecHit is a seed, this should form a PFCluster. Compute necessary information.
        reco::PFCluster temp;
        temp.setSeed((*rechitsHandle)[n].detId());  // Pulling the detId of this PFRecHit from the legacy format input
        int offset = outputCPU.seedFracOffsets[n];
        int topoId = outputCPU.pfrh_topoId[n];
        int nSeeds = outputCPU.topoSeedCount[topoId];
        for (int k = offset; k < (offset + outputCPU.topoRHCount[topoId] - nSeeds + 1);
             k++) {  // Looping over PFRecHits in the same topo cluster
          if (outputCPU.pcrh_fracInd[k] > -1 && outputCPU.pcrh_frac[k] > 0.0) {
            const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechitsHandle, outputCPU.pcrh_fracInd[k]);
            temp.addRecHitFraction(reco::PFRecHitFraction(refhit, outputCPU.pcrh_frac[k]));
          }
        }
        // Now PFRecHitFraction of this PFCluster is set. Now compute calculateAndSetPosition (energy, position etc)
        // Check if this topoId has one only one seed
        if (nTopoSeeds.count(outputCPU.pfrh_topoId[n]) && nTopoSeeds[outputCPU.pfrh_topoId[n]] == 1 &&
            _allCellsPositionCalc) {
          _allCellsPositionCalc->calculateAndSetPosition(temp);
        } else {
          _positionCalc->calculateAndSetPosition(temp);
        }
        pfClustersFromCuda->emplace_back(std::move(temp));
      }
    }

    event.put(std::move(pfClustersFromCuda));
  }
}

DEFINE_FWK_MODULE(PFClusterProducerCudaHCAL);
