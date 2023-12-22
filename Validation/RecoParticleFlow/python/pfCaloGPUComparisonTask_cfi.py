import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
pfClusterHBHEAlpakaComparison = DQMEDAnalyzer("PFCaloGPUComparisonTask",
                                                    pfClusterToken_ref = cms.untracked.InputTag('particleFlowClusterHBHEOnly'),
                                                    pfClusterToken_target = cms.untracked.InputTag('legacyPFClusterProducer'),
                                                    pfCaloGPUCompDir = cms.untracked.string("pfClusterHBHEAlpakaV")
)
