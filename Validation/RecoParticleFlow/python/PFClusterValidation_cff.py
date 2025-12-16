import FWCore.ParameterSet.Config as cms
from Validation.RecoParticleFlow.pfClusterValidation_cfi import pfClusterValidation
from Validation.RecoParticleFlow.pfCaloGPUComparisonTask_cfi import pfClusterHBHEOnlyAlpakaComparison, pfClusterHBHEAlpakaComparison

pfClusterValidationSequence = cms.Sequence( pfClusterValidation )

pfClusterAlpakaComparisonSequence = cms.Sequence( pfClusterHBHEAlpakaComparison )

pfClusterCaloOnlyValidation = pfClusterValidation.clone(
    pflowClusterHCAL = 'particleFlowClusterHCALOnly'
)
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

pfClusterHCALOnlyAlpakaComparison = DQMEDAnalyzer("PFMultiClusCompare",
                                                    pfClusterToken_ref = cms.untracked.InputTag('particleFlowClusterHCALOnlyLegacy'),
                                                    pfClusterToken_target = cms.untracked.InputTag('particleFlowClusterHCALOnly'),
                                                    pfCaloGPUCompDir = cms.untracked.string("pfClusterHCALAlpakaV")
)


pfClusterCaloOnlyValidationSequence = cms.Sequence( pfClusterCaloOnlyValidation )

pfClusterHBHEOnlyAlpakaComparisonSequence = cms.Sequence( pfClusterHBHEOnlyAlpakaComparison + pfClusterHCALOnlyAlpakaComparison )
