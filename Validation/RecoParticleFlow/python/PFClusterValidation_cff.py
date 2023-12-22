import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.alpakaValidationParticleFlow_cff import alpakaValidationParticleFlow
from Validation.RecoParticleFlow.pfClusterValidation_cfi import pfClusterValidation
from Validation.RecoParticleFlow.pfCaloGPUComparisonTask_cfi import pfClusterHBHEAlpakaComparison

pfClusterValidationSequence = cms.Sequence( pfClusterValidation )

pfClusterCaloOnlyValidation = pfClusterValidation.clone(
    pflowClusterHCAL = 'particleFlowClusterHCALOnly'
)

pfClusterCaloOnlyValidationSequence = cms.Sequence( pfClusterCaloOnlyValidation )

pfClusterAlpakaValidationTask = cms.Task ( pfClusterCaloOnlyValidation,
        pfClusterHBHEAlpakaComparison )
pfClusterAlpakaValidationSequence = cms.Sequence( pfClusterAlpakaValidationTask )


alpakaValidationParticleFlow.toReplaceWith(pfClusterCaloOnlyValidationSequence, pfClusterAlpakaValidationSequence)
