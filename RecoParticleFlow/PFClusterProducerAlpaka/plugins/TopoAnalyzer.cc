#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/ParticleFlowReco_Alpaka/interface/PFClusterHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <variant>

class TopoAnalyzer : public DQMEDAnalyzer {
public:
  TopoAnalyzer(edm::ParameterSet const& conf);
  ~TopoAnalyzer() override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  //static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<reco2::PFClusterHostCollection2> AlpakaToken;
  edm::EDGetTokenT<reco::PFClusterCollection> LegacyToken;
  edm::Handle<reco2::PFClusterHostCollection2> AlpakaHandle;
  edm::Handle<reco::PFClusterCollection> LegacyHandle;

  MonitorElement* pfCluster_TopoMultiplicity_GPUvsCPU_;
  MonitorElement* pfCluster_TopoMemberMultiplicity_GPUvsCPU_;

  std::string pfCaloGPUCompDir;
};

TopoAnalyzer::TopoAnalyzer(const edm::ParameterSet& conf) {
  AlpakaToken = consumes<reco2::PFClusterHostCollection2>(conf.getUntrackedParameter<edm::InputTag>("AlpakaToken"));
  LegacyToken = consumes<reco::PFClusterCollection>(conf.getParameter<edm::InputTag>("LegacyToken"));
  pfCaloGPUCompDir = conf.getUntrackedParameter<std::string>("pfCaloGPUCompDir");
}

TopoAnalyzer::~TopoAnalyzer() {}

void TopoAnalyzer::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& irun, edm::EventSetup const& isetup) {
  constexpr auto size = 100;
  char histo[size];
  ibooker.setCurrentFolder("ParticleFlow/" + pfCaloGPUCompDir);

  strncpy(histo, "pfCluster_TopoMultiplicity_GPUvsCPU_", size);
  pfCluster_TopoMultiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 2000, 100, 0, 2000);
  strncpy(histo, "pfCluster_TopoMemberMultiplicity_GPUvsCPU_", size);
  pfCluster_TopoMemberMultiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 2000, 100, 0, 2000);
}

void TopoAnalyzer::analyze(edm::Event const& event, edm::EventSetup const& c) {
  event.getByToken(AlpakaToken, AlpakaHandle);
  event.getByToken(LegacyToken, LegacyHandle);

  const reco2::PFClusterHostCollection2::ConstView& alpakaClusters = AlpakaHandle->const_view();

  int counter = 0;
  for (unsigned i = 0; i < (unsigned)alpakaClusters.size(); i++) {
    if (alpakaClusters[i].topoRHCount() > 0)
      counter++;
  }

  if (LegacyHandle->size() != (unsigned int)counter)
    printf("multiplicity problem\n");
  pfCluster_TopoMultiplicity_GPUvsCPU_->Fill((float)LegacyHandle->size(), (float)alpakaClusters.nTopos());
  // Develop matching criteria between portable collection and initial clusters
  std::vector<int> matched_idx;
  for (unsigned i = 0; i < LegacyHandle->size(); ++i) {
    bool matched = false;
    for (unsigned j = 0; j < (unsigned)alpakaClusters.size(); ++j) {
      if (alpakaClusters[j].topoRHCount() > 0)
        if ((unsigned)alpakaClusters[j].topoRHCount() == LegacyHandle->at(i).recHitFractions().size()) {
          if (!matched) {
            matched = true;
            matched_idx.push_back((int)j);
          } else {
            printf("Another matching %ld rechits?\n", LegacyHandle->at(i).recHitFractions().size());
          }
        }
    }
    if (!matched)
      matched_idx.push_back(-1);  // no match -> dummy number
  }

  // Plot matching topo clusters
  for (unsigned i = 0; i < LegacyHandle->size(); ++i) {
    if (matched_idx[i] >= 0) {
      unsigned int j = matched_idx[i];
      pfCluster_TopoMemberMultiplicity_GPUvsCPU_->Fill((float)alpakaClusters[j].topoRHCount(),
                                                       (float)LegacyHandle->at(i).recHitFractions().size());
    }
  }
}
/*
void TopoAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("LegacyToken", edm::InputTag("initialClusters"));
    desc.addUntracked<edm::InputTag>("AlpakaToken");
    desc.addUntracked<std::string>("pfCaloGPUCompDir", "pfClusterHBHEGPUv");
    descriptions.addDefault(desc);
}*/

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TopoAnalyzer);
