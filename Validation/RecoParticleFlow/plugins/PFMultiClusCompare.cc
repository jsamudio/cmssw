#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#else
#define LOGVERB(x) LogTrace(x)
#endif

// Helper function to calculate DeltaR between two clusters
float calculateDeltaR(const reco::PFCluster& clusterA, const reco::PFCluster& clusterB) {
  // reco::PFCluster inherits eta() and phi() from reco::CaloCluster/reco::ClusterMixin
  float etaA = clusterA.eta();
  float phiA = clusterA.phi();
  float etaB = clusterB.eta();
  float phiB = clusterB.phi();
  
  // Use the standard CMS DeltaR function
  return reco::deltaR(etaA, phiA, etaB, phiB); 
}

// You will also need to define a threshold for the DeltaR match (e.g., a common value like 0.01)
const float DELTAR_THRESHOLD = 0.01;

// Define the type for a unique constituent key
using ConstituentKey = std::pair<DetId, float>;

bool isConstituentMatch(const reco::PFCluster& clusterA, const reco::PFCluster& clusterB) {
  
  // 1. Helper function to build the unique set of constituents
  auto buildConstituentSet = [](const reco::PFCluster& cluster) {
    std::set<ConstituentKey> constituents;
    const auto& fractions = cluster.recHitFractions();

    for (const auto& fraction : fractions) {
      // Must check if the RecHit reference is valid!
      if (fraction.recHitRef().isAvailable() && fraction.recHitRef().isNonnull()) {
        constituents.insert({fraction.recHitRef()->detId(), fraction.fraction()});
      }
    }
    return constituents;
  };

  // 2. Build the sets for both clusters
  std::set<ConstituentKey> constituentsA = buildConstituentSet(clusterA);
  std::set<ConstituentKey> constituentsB = buildConstituentSet(clusterB);

  // 3. Compare the sets for exact equality (size and content)
  return constituentsA == constituentsB;
}


class PFMultiClusCompare : public DQMEDAnalyzer {
public:
  PFMultiClusCompare(edm::ParameterSet const& conf);
  ~PFMultiClusCompare() override = default;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterTok_ref_;
  edm::EDGetTokenT<reco::PFClusterCollection> pfClusterTok_target_;

  MonitorElement* pfCluster_Multiplicity_GPUvsCPU_;
  MonitorElement* pfCluster_Energy_GPUvsCPU_;
  MonitorElement* pfCluster_RecHitMultiplicity_GPUvsCPU_;
  MonitorElement* pfCluster_Layer_GPUvsCPU_;
  MonitorElement* pfCluster_Depth_GPUvsCPU_;
  MonitorElement* pfCluster_Eta_GPUvsCPU_;
  MonitorElement* pfCluster_Phi_GPUvsCPU_;
  MonitorElement* pfCluster_DuplicateMatches_GPUvsCPU_;

  std::string pfCaloGPUCompDir_;
};

PFMultiClusCompare::PFMultiClusCompare(const edm::ParameterSet& conf)
    : pfClusterTok_ref_{consumes<reco::PFClusterCollection>(
          conf.getUntrackedParameter<edm::InputTag>("pfClusterToken_ref"))},
      pfClusterTok_target_{
          consumes<reco::PFClusterCollection>(conf.getUntrackedParameter<edm::InputTag>("pfClusterToken_target"))},
      pfCaloGPUCompDir_{conf.getUntrackedParameter<std::string>("pfCaloGPUCompDir")} {}

void PFMultiClusCompare::bookHistograms(DQMStore::IBooker& ibooker,
                                             edm::Run const& irun,
                                             edm::EventSetup const& isetup) {
  const char* histo;

  ibooker.setCurrentFolder("ParticleFlow/" + pfCaloGPUCompDir_);

  histo = "pfCluster_Multiplicity_GPUvsCPU";
  pfCluster_Multiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 2000, 100, 0, 2000);

  histo = "pfCluster_Energy_GPUvsCPU";
  pfCluster_Energy_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 500, 100, 0, 500);

  histo = "pfCluster_RecHitMultiplicity_GPUvsCPU";
  pfCluster_RecHitMultiplicity_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);

  histo = "pfCluster_Layer_GPUvsCPU";
  pfCluster_Layer_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);

  histo = "pfCluster_Depth_GPUvsCPU";
  pfCluster_Depth_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, 0, 100, 100, 0, 100);

  histo = "pfCluster_Eta_GPUvsCPU";
  pfCluster_Eta_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, -5, 5, 100, -5, 5);

  histo = "pfCluster_Phi_GPUvsCPU";
  pfCluster_Phi_GPUvsCPU_ = ibooker.book2D(histo, histo, 100, -3.2, 3.2, 100, -3.2, 3.2);

  histo = "pfCluster_DuplicateMatches_GPUvsCPU";
  pfCluster_DuplicateMatches_GPUvsCPU_ = ibooker.book1D(histo, histo, 100, 0., 1000);
}
void PFMultiClusCompare::analyze(edm::Event const& event, edm::EventSetup const& c) {
  edm::Handle<reco::PFClusterCollection> pfClusters_ref;
  event.getByToken(pfClusterTok_ref_, pfClusters_ref);

  edm::Handle<reco::PFClusterCollection> pfClusters_target;
  event.getByToken(pfClusterTok_target_, pfClusters_target);

  //
  // Compare per-event PF cluster multiplicity

  if (pfClusters_ref->size() != pfClusters_target->size())
    LOGVERB("PFMultiClusCompare") << " PFCluster multiplicity " << pfClusters_ref->size() << " "
                                       << pfClusters_target->size();
  pfCluster_Multiplicity_GPUvsCPU_->Fill((float)pfClusters_ref->size(), (float)pfClusters_target->size());

  // --- Counters initialized before the matching loop ---
  unsigned int N_Ref = pfClusters_ref->size();
  unsigned int N_Constituent = 0;
  unsigned int N_DeltaR = 0;
  // ---------------------------------------------------

  //
  // Stage 1: Constituent-Based Matching (as before)
  // Ref: CPU clusters, Target: GPU clusters
  std::vector<int> matched_idx(pfClusters_ref->size(), -1); // Initialize to -1 (unmatched)
  std::vector<bool> is_target_matched(pfClusters_target->size(), false);

  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    for (unsigned j = 0; j < pfClusters_target->size(); ++j) {
      if (is_target_matched[j]) continue;

      // isConstituentMatch is the gold standard (exact identity)
      if (isConstituentMatch(pfClusters_ref->at(i), pfClusters_target->at(j))) {
        if (matched_idx[i] == -1) { 
          matched_idx[i] = (int)j;        // Store match
          is_target_matched[j] = true;    // Mark target as used
          N_Constituent++;
        } else {
          edm::LogWarning("PFMultiClusCompare") << "Duplicate Constituent Match for Ref Cluster " << i;
        }
        break; // Break inner loop: constituent match is unique and final
      }
    }
  }

  // ---
  // Stage 2: DeltaR Matching for Unmatched Clusters (the backup)
  // ---
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    
    // Only process clusters that were NOT matched in Stage 1
    if (matched_idx[i] != -1) continue; 

    int best_target_idx = -1;
    float min_deltaR = DELTAR_THRESHOLD; // Start with the maximum allowed DeltaR

    for (unsigned j = 0; j < pfClusters_target->size(); ++j) {

      // Only match to target clusters not yet used in Stage 1
      if (is_target_matched[j]) continue; 

      float current_deltaR = calculateDeltaR(pfClusters_ref->at(i), pfClusters_target->at(j));

      if (current_deltaR < min_deltaR) {
        min_deltaR = current_deltaR;
        best_target_idx = (int)j;
      }
    }

    // If a match within the threshold was found
    if (best_target_idx != -1) {
      matched_idx[i] = best_target_idx;   // Store the DeltaR match
      is_target_matched[best_target_idx] = true; // Mark target as used
      N_DeltaR++;
      
      // Log the backup match for monitoring purposes
      edm::LogInfo("PFMultiClusCompare") 
          << "DeltaR Match: Ref " << i << " to Target " << best_target_idx 
          << " with dR=" << min_deltaR;
          
      // You may want to fill a separate histogram here to track dR-matched clusters.
    } else {
      edm::LogWarning("PFMultiClusCompare") 
          << "Ref Cluster " << i << " remains unmatched after both stages.";
    }
  }


  /*
  //
  // Find matching PF cluster pairs
  std::vector<int> matched_idx;
  matched_idx.reserve(pfClusters_ref->size());
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    bool matched = false;
    for (unsigned j = 0; j < pfClusters_target->size(); ++j) {
      if (pfClusters_ref->at(i).seed() == pfClusters_target->at(j).seed()) {
        if (!matched) {
          matched = true;
          matched_idx.push_back((int)j);
        } else {
          edm::LogWarning("PFMultiClusCompare") << "Found duplicate match";
          pfCluster_DuplicateMatches_GPUvsCPU_->Fill((int)j);
        }
      }
    }
    if (!matched)
      matched_idx.push_back(-1);  // if you don't find a match, put a dummy number
      edm::LogWarning("PFMultiClusCompare") << "Found unmatched";
  }

  //
  // Match multi-depth clusters based on constituents
  // Get hits from both clusters
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    bool matched = false;
    for (unsigned j = 0; j < pfClusters_target->size(); ++j) {
        const auto& hits1 = pfClusters_ref->at(i).hitsAndFractions();
        const auto& hits2 = pfClusters_target->at(j).hitsAndFractions();
        for (const auto& h1 : hits1) {
            for (const auto& h2 : hits2) {
                // Compare the DetIds of the hits
                if (h1.first == h2.first) {
                    if (!matched) {
                      matched = true;
                      matched_idx.push_back((int)j);
                    }
                }
            }
        }
    }
    if (!matched) {
      matched_idx.push_back(-1);
      edm::LogWarning("PFMultiClusCompare") << "Found unmatched";
    }
  }
  */

  if (N_Ref > 0) {
    float F_Constituent = (float)N_Constituent / N_Ref;
    float F_DeltaR = (float)N_DeltaR / N_Ref;
    float F_Total = F_Constituent + F_DeltaR;

    edm::LogPrint("PFMultiClusCompare") 
        << "\n--- Cluster Matching Summary ---\n"
        << "Total Reference Clusters (CPU): " << N_Ref << "\n"
        << "1. Matched by Constituents: " << N_Constituent << " (" << F_Constituent * 100.0 << "%)\n"
        << "2. Matched by DeltaR: " << N_DeltaR << " (" << F_DeltaR * 100.0 << "%)\n"
        << "Total Matched: " << (N_Constituent + N_DeltaR) << " (" << F_Total * 100.0 << "%)\n"
        << "Unmatched: " << (N_Ref - N_Constituent - N_DeltaR) << " (" << (1.0 - F_Total) * 100.0 << "%)\n";
  }

  //
  // Plot matching PF cluster variables
  for (unsigned i = 0; i < pfClusters_ref->size(); ++i) {
    if (matched_idx[i] >= 0) {
      unsigned int j = matched_idx[i];
      int ref_energy_bin = pfCluster_Energy_GPUvsCPU_->getTH2F()->GetXaxis()->FindBin(pfClusters_ref->at(i).energy());
      int target_energy_bin =
          pfCluster_Energy_GPUvsCPU_->getTH2F()->GetXaxis()->FindBin(pfClusters_target->at(j).energy());
      if (ref_energy_bin != target_energy_bin)
        edm::LogPrint("PFMultiClusCompare")
            << "Off-diagonal energy bin entries: " << pfClusters_ref->at(i).energy() << " "
            << pfClusters_ref->at(i).eta() << " " << pfClusters_ref->at(i).phi() << " "
            << pfClusters_target->at(j).energy() << " " << pfClusters_target->at(j).eta() << " "
            << pfClusters_target->at(j).phi() << std::endl;
      pfCluster_Energy_GPUvsCPU_->Fill(pfClusters_ref->at(i).energy(), pfClusters_target->at(j).energy());
      pfCluster_Layer_GPUvsCPU_->Fill(pfClusters_ref->at(i).layer(), pfClusters_target->at(j).layer());
      pfCluster_Eta_GPUvsCPU_->Fill(pfClusters_ref->at(i).eta(), pfClusters_target->at(j).eta());
      pfCluster_Phi_GPUvsCPU_->Fill(pfClusters_ref->at(i).phi(), pfClusters_target->at(j).phi());
      pfCluster_Depth_GPUvsCPU_->Fill(pfClusters_ref->at(i).depth(), pfClusters_target->at(j).depth());
      pfCluster_RecHitMultiplicity_GPUvsCPU_->Fill((float)pfClusters_ref->at(i).recHitFractions().size(),
                                                   (float)pfClusters_target->at(j).recHitFractions().size());
    }
    if (matched_idx[i] < 0) {
        edm::LogPrint("PFMultiClusCompare")
            << "Unmatched cluster: " << pfClusters_ref->at(i).energy() << " "
            << pfClusters_ref->at(i).eta() << " " << pfClusters_ref->at(i).phi() << " " << std::endl;

    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFMultiClusCompare);
