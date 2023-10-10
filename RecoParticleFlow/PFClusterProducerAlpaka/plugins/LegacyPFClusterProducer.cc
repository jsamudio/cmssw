#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include <Eigen/Core>
#include <Eigen/Dense>
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
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFClusterHostCollection.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFRHFractionHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/JobConfigurationAlpakaRecord2.h"

class LegacyPFClusterProducer : public edm::stream::EDProducer<> {
public:
  LegacyPFClusterProducer(edm::ParameterSet const& config)
      : alpakaPfClustersToken(consumes(config.getParameter<edm::InputTag>("src"))),
        alpakaPfRHFractionToken(consumes(config.getParameter<edm::InputTag>("src"))),
        InputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
        pfClusParamsToken(esConsumes(config.getParameter<edm::ESInputTag>("pfClusterParams"))),
        legacyPfClustersToken(produces()),
        _recHitsLabel(consumes(config.getParameter<edm::InputTag>("recHitsSource"))) {
    edm::ConsumesCollector cc = consumesCollector();

    //setup pf cluster builder if requested
    const edm::ParameterSet& pfcConf = config.getParameterSet("pfClusterBuilder");
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
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    desc.add<edm::InputTag>("PFRecHitsLabelIn");
    desc.add<edm::ESInputTag>("pfClusterParams");
    desc.add<edm::InputTag>("recHitsSource");
    desc.setAllowAnything();
    descriptions.addWithDefaultLabel(desc);
  }

  // the actual algorithm
  std::unique_ptr<PFCPositionCalculatorBase> _positionCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPositionCalc;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  const edm::EDGetTokenT<reco2::PFClusterHostCollection2> alpakaPfClustersToken;
  const edm::EDGetTokenT<reco::PFRHFractionHostCollection> alpakaPfRHFractionToken;
  const edm::EDGetTokenT<reco::PFRecHitHostCollection> InputPFRecHitSoA_Token_;
  const edm::ESGetToken<reco::PFClusterParamsAlpakaESDataHost, JobConfigurationAlpakaRecord2> pfClusParamsToken;
  const edm::EDPutTokenT<reco::PFClusterCollection> legacyPfClustersToken;
  const edm::EDGetTokenT<reco::PFRecHitCollection> _recHitsLabel;

  int nRH = 0;
};

// Sort by cluster depth
bool sortClusterDepth(const reco::PFCluster cluster1, const reco::PFCluster cluster2) {
    return (cluster2.depth() < cluster1.depth());
}

void LegacyPFClusterProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  //edm::Handle<reco2::PFClusterHostCollection2> pfClustersAlpakaSoA;
  //event.getByToken(alpakaPfClustersToken, pfClustersAlpakaSoA);
  //const reco2::PFClusterHostCollection2::ConstView<0>& alpakaPfClusters = pfClustersAlpakaSoA->const_view<0>();
  //const reco2::PFClusterHostCollection2::ConstView<1>& alpakaPfRhFrac = pfClustersAlpakaSoA->const_view<1>();
  const reco::PFRecHitHostCollection& pfRecHits = event.get(InputPFRecHitSoA_Token_);

  reco2::PFClusterHostCollection2 const& clusterSoA = event.get(alpakaPfClustersToken);
  reco::PFRHFractionHostCollection const& fractionSoA = event.get(alpakaPfRHFractionToken);

  auto const& alpakaPfClusters = clusterSoA.const_view();
  auto const& alpakaPfRhFrac = fractionSoA.const_view();

  nRH = pfRecHits.view().size();
  //auto out = std::make_unique<reco::PFClusterCollection>();
  reco::PFClusterCollection out;
  out.reserve(nRH);
  //out->reserve(nRH);

  auto const rechitsHandle = event.getHandle(_recHitsLabel);

  std::vector<int> seedlist1;
  std::vector<int> seedlist2;
  for (int i = 0; i < alpakaPfClusters.nSeeds(); i++) {
    seedlist1.push_back(alpakaPfClusters[i].pfc_seedRHIdx());
  }

  // Build PFClusters in legacy format
  std::unordered_map<int, int> nTopoSeeds;

  for (int i = 0; i < alpakaPfClusters.nSeeds(); i++) {
    nTopoSeeds[alpakaPfClusters[i].pfc_topoId()]++;
  }

  for (int i = 0; i < alpakaPfClusters.nSeeds(); i++) {
    const reco::PFRecHitRef& refhitTest = reco::PFRecHitRef(rechitsHandle, alpakaPfClusters[i].pfc_seedRHIdx());
    if (refhitTest->energy() == 0)
      printf("PROBLEM\n");
  }

  // Looping over SoA PFClusters to produce legacy PFCluster collection
  for (int i = 0; i < alpakaPfClusters.nSeeds(); i++) {
    unsigned int n = alpakaPfClusters[i].pfc_seedRHIdx();
    reco::PFCluster temp;
    temp.setSeed((*rechitsHandle)[n].detId());  // Pulling the detId of this PFRecHit from the legacy format input
    seedlist2.push_back(n);
    int offset = alpakaPfClusters[i].pfc_rhfracOffset();
    for (int k = offset; k < (offset + alpakaPfClusters[i].pfc_rhfracSize()) && k >= 0;
         k++) {  // Looping over PFRecHits in the same topo cluster
      if (alpakaPfRhFrac[k].pcrh_pfrhIdx() < nRH && alpakaPfRhFrac[k].pcrh_pfrhIdx() > -1 &&
          alpakaPfRhFrac[k].pcrh_frac() > 0.0) {
        //printf("For seed: %d\t adding pcrh_pfrhIdx: %d\t with frac: %f\t at k=%d\t \n", n, alpakaPfRhFrac[k].pcrh_pfrhIdx(), alpakaPfRhFrac[k].pcrh_frac(), k);
        const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechitsHandle, alpakaPfRhFrac[k].pcrh_pfrhIdx());
        temp.addRecHitFraction(reco::PFRecHitFraction(refhit, alpakaPfRhFrac[k].pcrh_frac()));
      }
    }

    if (temp.hitsAndFractions().empty())
      printf("empty with seed: %d\t where rhfracsize == %d\t and offset is %d\t\n",
             alpakaPfClusters[i].pfc_seedRHIdx(),
             alpakaPfClusters[i].pfc_rhfracSize(),
             offset);
    // Now PFRecHitFraction of this PFCluster is set. Now compute calculateAndSetPosition (energy, position etc)
    if (nTopoSeeds[alpakaPfClusters[i].pfc_topoId()] == 1 && _allCellsPositionCalc) {
      _allCellsPositionCalc->calculateAndSetPosition(temp);
    } else {
      _positionCalc->calculateAndSetPosition(temp);
    }
    out.emplace_back(std::move(temp));
    //out.emplace(out.begin(), std::move(temp));
  }


  //sort(out.rbegin(), out.rend());
  //std::reverse(out.begin(), out.end());

  event.emplace(legacyPfClustersToken, std::move(out));

  sort(seedlist1.begin(), seedlist1.end());
  sort(seedlist2.begin(), seedlist2.end());
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LegacyPFClusterProducer);
