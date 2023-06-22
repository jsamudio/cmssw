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
  //using OProductType = cms::cuda::Product<hcal::PFClusterCollection<pf::common::DevStoragePolicy>>;
  using OProductType = cms::cuda::Product<reco::PFClusterDeviceMultiCollection>;
  edm::EDPutTokenT<OProductType> OutputPFClusterSoA_Token_;

  edm::ESGetToken<PFClusteringParamsGPU, JobConfigurationGPURecord> const pfClusParamsToken_;

  //edm::EDPutTokenT<cms::cuda::Product<reco::PFClusterDeviceCollection>> cluster_deviceToken_;


  int nRH_ = 0;
  // int nClustersMax_ = 0;
  // int nRHFracsMax = 0;

  const bool _produceSoA;     // PFClusters in SoA format
  const bool _produceLegacy;  // PFClusters in legacy format

  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;

  PFClustering::HCAL::ConfigurationParameters cudaConfig_;
  PFClustering::HCAL::OutputDataCPU outputCPU;
  PFClustering::HCAL::OutputDataGPU outputGPU;
  PFClustering::HCAL::ScratchDataGPU scratchGPU;

  // PFCluster Portable Collection
  reco::PFClusterDeviceMultiCollection multiCollGPU;
  reco::PFClusterHostMultiCollection multiCollCPU;

  cms::cuda::ContextState cudaState_;
};

PFClusterProducerCudaHCAL::PFClusterProducerCudaHCAL(const edm::ParameterSet& conf)
    : InputPFRecHitSoA_Token_{consumes(conf.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
      OutputPFClusterSoA_Token_{produces<OProductType>(conf.getParameter<std::string>("PFClustersGPUOut"))},
      pfClusParamsToken_{esConsumes(conf.getParameter<edm::ESInputTag>("pfClusteringParameters"))},
      //cluster_deviceToken_{produces(conf.getParameter<std::string>("PFClusterDeviceCollection"))},
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
  //desc.add<std::string>("PFClustersDeviceCollection", "PFClusterDev");
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

  multiCollGPU = reco::PFClusterDeviceMultiCollection({{nRH_, nRH_*120}}, cudaStream);

  // Calling cuda kernels
  PFClusterCudaHCAL::PFRechitToPFCluster_HCAL_entryPoint(
      cudaStream, pfClusParamsProduct, PFRecHits, outputGPU, scratchGPU, multiCollGPU, kernelTimers);

  if (!_produceLegacy)
    return;  // do device->host transfer only when we are producing Legacy data

  //
  // --- Data transfers for array
  //
  
  multiCollCPU = reco::PFClusterHostMultiCollection({{nRH_, nRH_*120}}, cudaStream);

  cms::cuda::copyAsync(multiCollCPU.buffer(), multiCollGPU.const_buffer(), multiCollGPU.bufferSize(), ctx.stream());
  
}

void PFClusterProducerCudaHCAL::produce(edm::Event& event, const edm::EventSetup& setup) {
  cms::cuda::ScopedContextProduce ctx{cudaState_};
  if (_produceSoA)
    ctx.emplace(event,
                OutputPFClusterSoA_Token_,
                std::move(multiCollGPU));  // SoA "PFClusters" still need to be defined.

  if (_produceLegacy) {
    auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
    pfClustersFromCuda->reserve(nRH_);

    auto const rechitsHandle = event.getHandle(_rechitsLabel);

    std::vector<int> seedlist1;
    std::vector<int> seedlist2;
    for (int i = 0; i < multiCollCPU.view<0>().nSeeds(); i++) {
      seedlist1.push_back(multiCollCPU.view<0>()[i].pfc_seedRHIdx());
    }

    // Build PFClusters in legacy format
    std::unordered_map<int, int> nTopoSeeds;

    for (int i = 0; i < multiCollCPU.view<0>().nSeeds(); i++) {
      nTopoSeeds[multiCollCPU.view<0>()[i].pfc_topoId()]++;
    }

    // Looping over SoA PFClusters to produce legacy PFCluster collection
    for (int i = 0; i < multiCollCPU.view<0>().nSeeds(); i++) {
      int n = multiCollCPU.view<0>()[i].pfc_seedRHIdx();
      reco::PFCluster temp;
      temp.setSeed((*rechitsHandle)[n].detId());  // Pulling the detId of this PFRecHit from the legacy format input
      seedlist2.push_back(n);
      int offset = multiCollCPU.view<0>()[i].pfc_rhfracOffset();
      for (int k = offset; k < (offset + multiCollCPU.view<0>()[i].pfc_rhfracSize());
           k++) {  // Looping over PFRecHits in the same topo cluster
        if (multiCollCPU.view<1>()[k].pcrh_pfrhIdx() > -1 && multiCollCPU.view<1>()[k].pcrh_frac() > 0.0) {
          const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechitsHandle, multiCollCPU.view<1>()[k].pcrh_pfrhIdx());
          temp.addRecHitFraction(reco::PFRecHitFraction(refhit, multiCollCPU.view<1>()[k].pcrh_frac()));
        }
      }
      // Now PFRecHitFraction of this PFCluster is set. Now compute calculateAndSetPosition (energy, position etc)
      if (nTopoSeeds.count(multiCollCPU.view<0>()[i].pfc_topoId()) == 1 && _allCellsPositionCalc) {
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
