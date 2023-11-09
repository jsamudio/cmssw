import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.alpaka_cff import alpaka

from RecoParticleFlow.PFRecHitProducer.hcalRecHitSoAProducer_cfi import hcalRecHitSoAProducer as _hcalRecHitSoAProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitHCALParamsESProducer_cfi import pfRecHitHCALParamsESProducer as _pfRecHitHCALParamsESProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitHCALTopologyESProducer_cfi import pfRecHitHCALTopologyESProducer as _pfRecHitHCALTopologyESProducer
from RecoParticleFlow.PFRecHitProducer.pfRecHitSoAProducerHCAL_cfi import pfRecHitSoAProducerHCAL as _pfRecHitSoAProducerHCAL
from RecoParticleFlow.PFRecHitProducer.legacyPFRecHitProducer_cfi import legacyPFRecHitProducer as _legacyPFRecHitProducer
from RecoParticleFlow.PFClusterProducerAlpaka.pfClusterParamsESProducer_cfi import pfClusterParamsESProducer as _pfClusterParamsESProducer
from RecoParticleFlow.PFClusterProducerAlpaka.pfClusterProducerAlpaka_cfi import pfClusterProducerAlpaka as _pfClusterProducerAlpaka
from RecoParticleFlow.PFClusterProducerAlpaka.legacyPFClusterProducer_cfi import legacyPFClusterProducer as _legacyPFClusterProducer

from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import pfClusteringHBHEHFTask, particleFlowClusterHBHE, particleFlowRecHitHBHE, particleFlowClusterHCAL

_alpaka_pfClusteringHBHEHFTask = pfClusteringHBHEHFTask.copy()

pfRecHitHCALParamsRecordSource = cms.ESSource('EmptyESSource',
            recordName = cms.string('PFRecHitHCALParamsRecord'),
            iovIsRunNotTime = cms.bool(True),
            firstValid = cms.vuint32(1)
    )

pfRecHitHCALTopologyRecordSource = cms.ESSource('EmptyESSource',
            recordName = cms.string('PFRecHitHCALTopologyRecord'),
            iovIsRunNotTime = cms.bool(True),
            firstValid = cms.vuint32(1)
    )

pfClusterParamsAlpakaESRcdSource = cms.ESSource('EmptyESSource',
            recordName = cms.string('JobConfigurationGPURecord'),
            iovIsRunNotTime = cms.bool(True),
            firstValid = cms.vuint32(1)
    )

hbheRecHitToSoA = _hcalRecHitSoAProducer.clone(
        src = "hbhereco"
    )

pfRecHitHCALParamsESProducer = _pfRecHitHCALParamsESProducer.clone(
        energyThresholdsHB = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
        energyThresholdsHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 )
    )

pfRecHitHCALTopologyESProducer = _pfRecHitHCALTopologyESProducer.clone()
pfRecHitSoAProducerHCAL = _pfRecHitSoAProducerHCAL.clone(
        producers = cms.VPSet(
            cms.PSet(
                src = cms.InputTag("hbheRecHitToSoA"),
                params = cms.ESInputTag("pfRecHitHCALParamsESProducer:"),
            )
        ),
        topology = "pfRecHitHCALTopologyESProducer:",
        synchronise = cms.untracked.bool(False)
    )

legacyPFRecHitFromAlpaka = _legacyPFRecHitProducer.clone(
        src = "pfRecHitSoAProducerHCAL"
    )

pfClusterParamsESProducer = _pfClusterParamsESProducer.clone()
pfClusterProducerAlpaka = _pfClusterProducerAlpaka.clone(
        PFRecHitsLabelIn = 'pfRecHitSoAProducerHCAL',
        pfClusterParams = 'pfClusterParamsESProducer:',
        synchronise = cms.bool(False)
    )


legacyPFClusterFromAlpaka = _legacyPFClusterProducer.clone(
        src = 'pfClusterProducerAlpaka',
        pfClusterParams = 'pfClusterParamsESProducer:',
        pfClusterBuilder = particleFlowClusterHBHE.pfClusterBuilder,
        recHitsSource = 'legacyPFRecHitFromAlpaka',
        PFRecHitsLabelIn = 'pfRecHitSoAProducerHCAL'
    )


_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALParamsRecordSource)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALTopologyRecordSource)
_alpaka_pfClusteringHBHEHFTask.add(pfClusterParamsAlpakaESRcdSource)
_alpaka_pfClusteringHBHEHFTask.add(hbheRecHitToSoA)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitHCALParamsESProducer)
_alpaka_pfClusteringHBHEHFTask.add(pfRecHitSoAProducerHCAL)
_alpaka_pfClusteringHBHEHFTask.add(legacyPFRecHitFromAlpaka)
_alpaka_pfClusteringHBHEHFTask.add(pfClusterParamsESProducer)
_alpaka_pfClusteringHBHEHFTask.add(pfClusterProducerAlpaka)

_alpaka_pfClusteringHBHEHFTask.remove(particleFlowRecHitHBHE)
_alpaka_pfClusteringHBHEHFTask.remove(particleFlowClusterHBHE)
_alpaka_pfClusteringHBHEHFTask.remove(particleFlowClusterHCAL)
_alpaka_pfClusteringHBHEHFTask.add(particleFlowClusterHBHE)
_alpaka_pfClusteringHBHEHFTask.add(particleFlowClusterHCAL)

alpaka.toReplaceWith(particleFlowClusterHBHE, legacyPFClusterFromAlpaka)

alpaka.toReplaceWith(pfClusteringHBHEHFTask, _alpaka_pfClusteringHBHEHFTask)
