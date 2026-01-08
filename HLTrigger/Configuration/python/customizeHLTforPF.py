import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *

def customizeHLTforAlpakaPFMultidepthClustering(process):
    ''' Includes customization to run alpaka MDPF clustering
    '''
    old_hltParticleFlowClusterHCAL = process.hltParticleFlowClusterHCAL

    process.hltPFClusterSoAPositionUpdater = cms.EDProducer('PFClusterSoAPositionUpdater',
        #pfClusterBuilder = particleFlowClusterHBHE.pfClusterBuilder,
        src = cms.InputTag('hltParticleFlowClusterHBHESoA'), # Check: ensure this source exists
        recHitsSource = cms.InputTag('hltParticleFlowRecHitHBHESoA'),
        PFRecHitsLabelIn = cms.InputTag('hltParticleFlowRecHitHBHESoA')
    )

    process.hltPFMultiDepthClusterSoA = cms.EDProducer('PFMultiDepthClusterSoAProducer@alpaka',
        clustersSrc = cms.InputTag("hltPFClusterSoAPositionUpdater"), # Connects to Module 1
        rhfracSrc   = cms.InputTag('hltPFClusterSoAPositionUpdater'),
        rechitSrc   = cms.InputTag('hltParticleFlowRecHitHBHESoA')
    )

    process.hltParticleFlowClusterHCAL = cms.EDProducer('LegacyMultiDepthPFClusterProducer',
        src              = cms.InputTag('hltPFMultiDepthClusterSoA'), # Connects to Module 2
        recHitsSource    = cms.InputTag('hltParticleFlowRecHitHBHE'), # Check this label
        PFRecHitsLabelIn = cms.InputTag('hltParticleFlowRecHitHBHESoA') # Check this label
    )

    process.HLTPFHcalClustering = cms.Sequence(
        process.hltParticleFlowRecHitHBHESoA +
        process.hltParticleFlowRecHitHBHE +
        process.hltParticleFlowClusterHBHESoA +
        process.hltParticleFlowClusterHBHE +
        process.hltPFClusterSoAPositionUpdater +
        process.hltPFMultiDepthClusterSoA +
        process.hltParticleFlowClusterHCAL  # This now refers to your NEW LegacyMultiDepth producer
    )

    def replaceItemsInSequence(process, itemsToReplace, replacingSequence):
        for sequence, items in process.sequences.items():
            #Find Sequences containing all the items in itemsToReplace
            containsAll = all(items.contains(item) for item in itemsToReplace)
            if(containsAll):
                for item in itemsToReplace:
                    #remove items that will be replaced by replacingSequence
                    if(item != itemsToReplace[-1]):
                        items.remove(item)
                    else:
                        #if last item, replace it with the Sequence
                        items.replace(item, replacingSequence)
    return process

    itemsList = [ old_hltParticleFlowClusterHCAL ]

    process = replaceItemsInSequence(process, itemsList, process.HLTPFHcalClustering)

    return process

def customizeHLTforPF(process):
    process = customizeHLTforAlpakaPFMultidepthClustering(process)

    return process
