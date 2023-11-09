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

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionHostCollection.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterParamsHostCollection.h"

class LegacyPFClusterProducer : public edm::stream::EDProducer<> {
public:
  LegacyPFClusterProducer(edm::ParameterSet const& config)
      : alpakaPfClustersToken(consumes(config.getParameter<edm::InputTag>("src"))),
        alpakaPfRHFractionToken(consumes(config.getParameter<edm::InputTag>("src"))),
        InputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
        pfClusParamsToken(esConsumes(config.getParameter<edm::ESInputTag>("pfClusterParams"))),
        legacyPfClustersToken(produces()),
        recHitsLabel(consumes(config.getParameter<edm::InputTag>("recHitsSource"))) {
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
    {
      edm::ParameterSetDescription pfClusterBuilder;
      pfClusterBuilder.add<unsigned int>("maxIterations", 5);
      pfClusterBuilder.add<double>("minFracTot", 1e-20);
      pfClusterBuilder.add<double>("minFractionToKeep", 1e-7);
      pfClusterBuilder.add<bool>("excludeOtherSeeds", true);
      pfClusterBuilder.add<double>("showerSigma", 10.);
      pfClusterBuilder.add<double>("stoppingTolerance", 1e-8);
      pfClusterBuilder.add<double>("timeSigmaEB", 10.);
      pfClusterBuilder.add<double>("timeSigmaEE", 10.);
      pfClusterBuilder.add<double>("maxNSigmaTime", 10.);
      pfClusterBuilder.add<double>("minChi2Prob", 0.);
      pfClusterBuilder.add<bool>("clusterTimeResFromSeed", false);
      pfClusterBuilder.add<std::string>("algoName", "");
      {
        edm::ParameterSetDescription validator;
        validator.add<std::string>("detector", "");
        validator.add<std::vector<int>>("depths", {});
        validator.add<std::vector<double>>("recHitEnergyNorm", {});
        std::vector<edm::ParameterSet> vDefaults(2);
        vDefaults[0].addParameter<std::string>("detector", "HCAL_BARREL1");
        vDefaults[0].addParameter<std::vector<int>>("depths", {1, 2, 3, 4});
        vDefaults[0].addParameter<std::vector<double>>("recHitEnergyNorm", {0.1, 0.2, 0.3, 0.3});
        vDefaults[1].addParameter<std::string>("detector", "HCAL_ENDCAP");
        vDefaults[1].addParameter<std::vector<int>>("depths", {1, 2, 3, 4, 5, 6, 7});
        vDefaults[1].addParameter<std::vector<double>>("recHitEnergyNorm", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
        pfClusterBuilder.addVPSet("recHitEnergyNorms", validator, vDefaults);
      }
      {
        edm::ParameterSetDescription bar;
        bar.add<std::string>("algoName", "Basic2DGenericPFlowPositionCalc");
        bar.add<double>("minFractionInCalc", 1e-9);
        bar.add<int>("posCalcNCrystals", 5);
        {
          edm::ParameterSetDescription validator;
          validator.add<std::string>("detector", "");
          validator.add<std::vector<int>>("depths", {});
          validator.add<std::vector<double>>("logWeightDenominator", {});
          std::vector<edm::ParameterSet> vDefaults(2);
          vDefaults[0].addParameter<std::string>("detector", "HCAL_BARREL1");
          vDefaults[0].addParameter<std::vector<int>>("depths", {1, 2, 3, 4});
          vDefaults[0].addParameter<std::vector<double>>("logWeightDenominator", {0.1, 0.2, 0.3, 0.3});
          vDefaults[1].addParameter<std::string>("detector", "HCAL_ENDCAP");
          vDefaults[1].addParameter<std::vector<int>>("depths", {1, 2, 3, 4, 5, 6, 7});
          vDefaults[1].addParameter<std::vector<double>>("logWeightDenominator", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
          bar.addVPSet("logWeightDenominatorByDetector", validator, vDefaults);
        }
        bar.add<double>("minAllowedNormalization", 1e-9);
        pfClusterBuilder.add("positionCalc", bar);
      }
      {
        edm::ParameterSetDescription bar;
        bar.add<std::string>("algoName", "Basic2DGenericPFlowPositionCalc");
        bar.add<double>("minFractionInCalc", 1e-9);
        bar.add<int>("posCalcNCrystals", -1);
        {
          edm::ParameterSetDescription validator;
          validator.add<std::string>("detector", "");
          validator.add<std::vector<int>>("depths", {});
          validator.add<std::vector<double>>("logWeightDenominator", {});
          std::vector<edm::ParameterSet> vDefaults(2);
          vDefaults[0].addParameter<std::string>("detector", "HCAL_BARREL1");
          vDefaults[0].addParameter<std::vector<int>>("depths", {1, 2, 3, 4});
          vDefaults[0].addParameter<std::vector<double>>("logWeightDenominator", {0.1, 0.2, 0.3, 0.3});
          vDefaults[1].addParameter<std::string>("detector", "HCAL_ENDCAP");
          vDefaults[1].addParameter<std::vector<int>>("depths", {1, 2, 3, 4, 5, 6, 7});
          vDefaults[1].addParameter<std::vector<double>>("logWeightDenominator", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
          bar.addVPSet("logWeightDenominatorByDetector", validator, vDefaults);
        }
        bar.add<double>("minAllowedNormalization", 1e-9);
        pfClusterBuilder.add("allCellsPositionCalc", bar);
      }
      {
        edm::ParameterSetDescription bar;
        bar.add<double>("corrTermLowE", 0.);
        bar.add<double>("threshLowE", 6.);
        bar.add<double>("noiseTerm", 21.86);
        bar.add<double>("constantTermLowE", 4.24);
        bar.add<double>("noiseTermLowE", 8.);
        bar.add<double>("threshHighE", 15.);
        bar.add<double>("constantTerm", 2.82);
        pfClusterBuilder.add("timeResolutionCalcBarrel", bar);
      }
      {
        edm::ParameterSetDescription bar;
        bar.add<double>("corrTermLowE", 0.);
        bar.add<double>("threshLowE", 6.);
        bar.add<double>("noiseTerm", 21.86);
        bar.add<double>("constantTermLowE", 4.24);
        bar.add<double>("noiseTermLowE", 8.);
        bar.add<double>("threshHighE", 15.);
        bar.add<double>("constantTerm", 2.82);
        pfClusterBuilder.add("timeResolutionCalcEndcap", bar);
      }
      {
        edm::ParameterSetDescription bar;
        pfClusterBuilder.add("positionReCalc", bar);
      }
      {
        edm::ParameterSetDescription bar;
        pfClusterBuilder.add("energyCorrector", bar);
      }
      desc.add("pfClusterBuilder", pfClusterBuilder);
    }
    descriptions.addWithDefaultLabel(desc);
  }

  // the actual algorithm
  std::unique_ptr<PFCPositionCalculatorBase> _positionCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPositionCalc;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  const edm::EDGetTokenT<reco::PFClusterHostCollection> alpakaPfClustersToken;
  const edm::EDGetTokenT<reco::PFRecHitFractionHostCollection> alpakaPfRHFractionToken;
  const edm::EDGetTokenT<reco::PFRecHitHostCollection> InputPFRecHitSoA_Token_;
  const edm::ESGetToken<reco::PFClusterParamsHostCollection, JobConfigurationGPURecord> pfClusParamsToken;
  const edm::EDPutTokenT<reco::PFClusterCollection> legacyPfClustersToken;
  const edm::EDGetTokenT<reco::PFRecHitCollection> recHitsLabel;

  int nRH = 0;
};

void LegacyPFClusterProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  const reco::PFRecHitHostCollection& pfRecHits = event.get(InputPFRecHitSoA_Token_);

  reco::PFClusterHostCollection const& clusterSoA = event.get(alpakaPfClustersToken);
  reco::PFRecHitFractionHostCollection const& fractionSoA = event.get(alpakaPfRHFractionToken);

  auto const& alpakaPfClusters = clusterSoA.const_view();
  auto const& alpakaPfRhFrac = fractionSoA.const_view();

  nRH = pfRecHits.view().size();
  reco::PFClusterCollection out;
  out.reserve(nRH);

  auto const rechitsHandle = event.getHandle(recHitsLabel);

  std::vector<int> seedlist1;
  std::vector<int> seedlist2;
  seedlist1.reserve(alpakaPfClusters.nSeeds());
  for (int i = 0; i < alpakaPfClusters.nSeeds(); i++) {
    seedlist1.push_back(alpakaPfClusters[i].seedRHIdx());
  }

  // Build PFClusters in legacy format
  std::unordered_map<int, int> nTopoSeeds;

  for (int i = 0; i < alpakaPfClusters.nSeeds(); i++) {
    nTopoSeeds[alpakaPfClusters[i].topoId()]++;
  }

  // Looping over SoA PFClusters to produce legacy PFCluster collection
  for (int i = 0; i < alpakaPfClusters.nSeeds(); i++) {
    unsigned int n = alpakaPfClusters[i].seedRHIdx();
    reco::PFCluster temp;
    temp.setSeed((*rechitsHandle)[n].detId());  // Pulling the detId of this PFRecHit from the legacy format input
    seedlist2.push_back(n);
    int offset = alpakaPfClusters[i].rhfracOffset();
    for (int k = offset; k < (offset + alpakaPfClusters[i].rhfracSize()) && k >= 0;
         k++) {  // Looping over PFRecHits in the same topo cluster
      if (alpakaPfRhFrac[k].pfrhIdx() < nRH && alpakaPfRhFrac[k].pfrhIdx() > -1 && alpakaPfRhFrac[k].frac() > 0.0) {
        const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechitsHandle, alpakaPfRhFrac[k].pfrhIdx());
        temp.addRecHitFraction(reco::PFRecHitFraction(refhit, alpakaPfRhFrac[k].frac()));
      }
    }

    // Now PFRecHitFraction of this PFCluster is set. Now compute calculateAndSetPosition (energy, position etc)
    if (nTopoSeeds[alpakaPfClusters[i].topoId()] == 1 && _allCellsPositionCalc) {
      _allCellsPositionCalc->calculateAndSetPosition(temp);
    } else {
      _positionCalc->calculateAndSetPosition(temp);
    }
    out.emplace_back(std::move(temp));
  }

  event.emplace(legacyPfClustersToken, std::move(out));

  sort(seedlist1.begin(), seedlist1.end());
  sort(seedlist2.begin(), seedlist2.end());
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LegacyPFClusterProducer);
