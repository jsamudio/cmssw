import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.alpakaValidationParticleFlow_cff import alpakaValidationParticleFlow
from Validation.RecoParticleFlow.pfClusterValidation_cfi import pfClusterValidation
from Validation.RecoParticleFlow.pfCaloGPUComparisonTask_cfi import pfClusterHBHEOnlyAlpakaComparison, pfClusterHBHEAlpakaComparison

pfClusterValidationSequence = cms.Sequence( pfClusterValidation )

pfClusterAlpakaValidationTask = cms.Task( pfClusterValidation,
        pfClusterHBHEAlpakaComparison )
pfClusterAlpakaValidationSequence = cms.Sequence( pfClusterAlpakaValidationTask )

alpakaValidationParticleFlow.toReplaceWith(pfClusterValidationSequence, pfClusterAlpakaValidationSequence)

pfClusterCaloOnlyValidation = pfClusterValidation.clone(
    pflowClusterHCAL = 'particleFlowClusterHCALOnly'
)

pfClusterCaloOnlyValidationSequence = cms.Sequence( pfClusterCaloOnlyValidation )

pfClusterAlpakaCaloOnlyValidationTask = cms.Task ( pfClusterCaloOnlyValidation,
        pfClusterHBHEOnlyAlpakaComparison )
pfClusterAlpakaCaloOnlyValidationSequence = cms.Sequence( pfClusterAlpakaCaloOnlyValidationTask )


alpakaValidationParticleFlow.toReplaceWith(pfClusterCaloOnlyValidationSequence, pfClusterAlpakaCaloOnlyValidationSequence)
