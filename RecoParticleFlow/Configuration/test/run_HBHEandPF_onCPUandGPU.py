# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: reHLT --processName reHLT -s HLT:@relval2021 --conditions auto:phase1_2021_realistic --datatier GEN-SIM-DIGI-RAW -n 5 --eventcontent FEVTDEBUGHLT --geometry DB:Extended --era Run3 --customise=HLTrigger/Configuration/customizeHLTforPatatrack.customizeHLTforPatatrack --filein /store/relval/CMSSW_12_3_0_pre5/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/123X_mcRun3_2021_realistic_v6-v1/10000/2639d8f2-aaa6-4a78-b7c2-9100a6717e6c.root
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('rereHLT',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(5),
    #input = cms.untracked.int32(1),
    input = cms.untracked.int32(1000),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('/store/relval/CMSSW_13_0_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_130X_mcRun3_2022_realistic_v2_HS-v4/2590000/05ad6501-815f-4df6-b115-03ad028f9b76.root'),
    #fileNames = cms.untracked.vstring('/store/relval/CMSSW_13_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_130X_mcRun3_2022_realistic_v2_HS-v4/2590000/85c3b693-68ce-478e-b1bd-dfed8be747ad.root'),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_13_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_130X_mcRun3_2022_realistic_v2_HS-v4/2590000/85c3b693-68ce-478e-b1bd-dfed8be747ad.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    FailPath = cms.untracked.vstring(),
    IgnoreCompletely = cms.untracked.vstring(),
    Rethrow = cms.untracked.vstring(),
    SkipEvent = cms.untracked.vstring(),
    allowUnscheduled = cms.obsolete.untracked.bool,
    canDeleteEarly = cms.untracked.vstring(),
    deleteNonConsumedUnscheduledModules = cms.untracked.bool(True),
    dumpOptions = cms.untracked.bool(False),
    emptyRunLumiMode = cms.obsolete.untracked.string,
    eventSetup = cms.untracked.PSet(
        forceNumberOfConcurrentIOVs = cms.untracked.PSet(
            allowAnyLabel_=cms.required.untracked.uint32
        ),
        numberOfConcurrentIOVs = cms.untracked.uint32(0)
    ),
    fileMode = cms.untracked.string('FULLMERGE'),
    forceEventSetupCacheClearOnNewRun = cms.untracked.bool(False),
    makeTriggerResults = cms.obsolete.untracked.bool,
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(0),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
    throwIfIllegalParameter = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(False)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('reHLT nevts:5'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('reHLT_HLT.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from HLTrigger.Configuration.CustomConfigs import ProcessName
process = ProcessName(process)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
# process.schedule imported from cff in HLTrigger.Configuration
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforPatatrack
#from HLTrigger.Configuration.customizeHLTforPatatrack import customizeHLTforPatatrack, customiseCommon, customiseHcalLocalReconstruction

# only enable Hcal GPU
#process = customiseCommon(process)
#process = customiseHcalLocalReconstruction(process)

#call to customisation function customizeHLTforPatatrack imported from HLTrigger.Configuration.customizeHLTforPatatrack
#process = customizeHLTforPatatrack(process)

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC

#call to customisation function customizeHLTforMC imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforMC(process)

# End of customisation functions


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

process.load( "HLTrigger.Timer.FastTimerService_cfi" )
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.TriggerSummaryProducerAOD = cms.untracked.PSet()
    process.MessageLogger.L1GtTrigReport = cms.untracked.PSet()
    process.MessageLogger.L1TGlobalSummary = cms.untracked.PSet()
    process.MessageLogger.HLTrigReport = cms.untracked.PSet()
    process.MessageLogger.FastReport = cms.untracked.PSet()
    process.MessageLogger.ThroughputService = cms.untracked.PSet()
    process.MessageLogger.cerr.FastReport = cms.untracked.PSet( limit = cms.untracked.int32( 10000000 ) )

#####################################
## Configure CPU and GPU producers ##
#####################################

# Run as:
# cmsRun RecoParticleFlow/Configuration/test/run_HBHEandPF_onCPUandGPU.py -- [--cpu] [--gpu]
import sys
use_cpu = "--cpu" in sys.argv or "--gpu" not in sys.argv
use_gpu = "--gpu" in sys.argv or "--cpu" not in sys.argv
if "--gpu" in sys.argv and "--cpu" in sys.argv:
    print("Running onf CPU and GPU")
elif "--cpu" in sys.argv and "--gpu" not in sys.argv:
    print("Running on CPU")
elif "--gpu" in sys.argv or "--cpu" not in sys.argv:
    print("Running on GPU")
else:
    print("Running onf CPU and GPU. Pass option --cpu or --gpu to run only on one.")


if use_gpu:  # Inclusion of ESSource fr GPU
    process.load("RecoParticleFlow.PFClusterProducer.pfhbheRecHitParamsGPUESProducer_cfi")
    process.load("RecoParticleFlow.PFClusterProducer.pfhbheTopologyGPUESProducer_cfi")
    from RecoParticleFlow.PFClusterProducer.pfClusteringParamsGPUESSource_cfi import pfClusteringParamsGPUESSource as _pfClusteringParamsGPUESSource
    process.pfClusteringParamsGPUESSource = _pfClusteringParamsGPUESSource.clone()

if use_gpu:  # Setup GPU RecHit producer
    producersGPU = process.hltParticleFlowRecHitHBHE.clone().producers
    for idx, x in enumerate(producersGPU):
        if x.src.moduleLabel == "hltHbhereco":
            x.src.moduleLabel = "hltHbherecoGPU" # use GPU version as input instead of legacy version
        for idy, y in enumerate(x.qualityTests):
            if y.name._value == "PFRecHitQTestHCALThresholdVsDepth": # apply phase1 depth-dependent HCAL thresholds
                for idz, z in enumerate(y.cuts): # convert signed to unsigned
                    if z.detectorEnum == 1: # HB
                        z.detectorEnum = cms.uint32( 1 )
                        z.depth = cms.vuint32( 1, 2, 3, 4 )
                        process.pfhbheRecHitParamsGPUESProducer.thresholdE_HB = z.threshold # propagate to process.pfhbheRecHitParamsGPUESProducer
                    if z.detectorEnum == 2: # HE
                        z.detectorEnum = cms.uint32( 2 )
                        z.depth = cms.vuint32( 1, 2, 3, 4, 5, 6, 7  )
                        process.pfhbheRecHitParamsGPUESProducer.thresholdE_HE = z.threshold # propagate to process.pfhbheRecHitParamsGPUESProducer
    process.hltParticleFlowRecHitHBHEonGPU = cms.EDProducer("PFHBHERecHitProducerGPU", # instead of "PFRecHitProducer"
                                                            producers = producersGPU,
                                                            navigator = process.hltParticleFlowRecHitHBHE.navigator)

if use_cpu:  # Setup CPU RecHit producer
    producersCPU = process.hltParticleFlowRecHitHBHE.producers
    for idx, x in enumerate(producersCPU):
        #if x.src.moduleLabel == "hltHbhereco":
        #    x.src.moduleLabel = "hltHbherecoFromGPU" # use GPU version as input instead of legacy version
        for idy, y in enumerate(x.qualityTests):
            if y.name._value == "PFRecHitQTestThreshold":
                y.name._value = "PFRecHitQTestHCALThresholdVsDepth" # apply phase1 depth-dependent HCAL thresholds
    process.hltParticleFlowRecHitHBHE = cms.EDProducer("PFRecHitProducer",
                                                    producers = producersCPU,
                                                    navigator = process.hltParticleFlowRecHitHBHE.navigator)
else:
    del process.hltParticleFlowRecHitHBHE

if use_gpu:  # Copy PFCluster parameters for CPU to GPU ES-based ones
    for idx, x in enumerate(process.pfClusteringParamsGPUESSource.initialClusteringStep.thresholdsByDetector):
        for idy, y in enumerate(process.hltParticleFlowClusterHBHE.initialClusteringStep.thresholdsByDetector):
            if x.detector == y.detector:
                x.gatheringThreshold = y.gatheringThreshold
    for idx, x in enumerate(process.pfClusteringParamsGPUESSource.pfClusterBuilder.recHitEnergyNorms):
        for idy, y in enumerate(process.hltParticleFlowClusterHBHE.pfClusterBuilder.recHitEnergyNorms):
            if x.detector == y.detector:
                x.recHitEnergyNorm = y.recHitEnergyNorm
    for idx, x in enumerate(process.pfClusteringParamsGPUESSource.seedFinder.thresholdsByDetector):
        for idy, y in enumerate(process.hltParticleFlowClusterHBHE.seedFinder.thresholdsByDetector):
            if x.detector == y.detector:
                x.seedingThreshold = y.seedingThreshold

if use_gpu:  # Setup GPU cluster producer
    process.hltParticleFlowClusterHBHEonGPU = cms.EDProducer("PFClusterProducerCudaHCAL", # instead of "PFClusterProducer"
                                                        pfClusterBuilder = process.hltParticleFlowClusterHBHE.pfClusterBuilder,
                                                        positionReCalc = process.hltParticleFlowClusterHBHE.positionReCalc,
                                                        recHitCleaners = process.hltParticleFlowClusterHBHE.recHitCleaners,
                                                        recHitsSource = cms.InputTag("hltParticleFlowRecHitHBHEonGPU"), # Use GPU version of input
                                                        seedCleaners = process.hltParticleFlowClusterHBHE.seedCleaners,
                                                        seedFinder = process.hltParticleFlowClusterHBHE.seedFinder,
                                                        energyCorrector = process.hltParticleFlowClusterHBHE.energyCorrector,
                                                        initialClusteringStep = process.hltParticleFlowClusterHBHE.initialClusteringStep)
    process.hltParticleFlowClusterHBHEonGPU.PFRecHitsLabelIn = cms.InputTag("hltParticleFlowRecHitHBHEonGPU","")
    process.hltParticleFlowClusterHBHEonGPU.PFClustersGPUOut = cms.string("hltParticleFlowClusterHBHEonGPU")
    #process.hltParticleFlowClusterHBHEonGPU.PFClusterDeviceCollection = cms.string("hltParticleFlowClusterHBHEonGPU")

# value before recent optimizations
if use_cpu:
    process.hltParticleFlowClusterHBHE.pfClusterBuilder.maxIterations = 50
if use_gpu:
    process.hltParticleFlowClusterHBHEonGPU.pfClusterBuilder.maxIterations = 50

#
# Additional customization
#process.maxEvents.input = 5
process.FEVTDEBUGHLToutput.outputCommands = cms.untracked.vstring('drop  *_*_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*ParticleFlow*HBHE*_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*HbherecoLegacy*_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*HbherecoFromGPU*_*_*')
process.FEVTDEBUGHLToutput.outputCommands.append('keep *_*Hbhereco*_*_*')
#process.FEVTDEBUGHLToutput.outputCommands.append('keep *_genParticles_*_*')
#process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hltFastPrimaryVertex_*_*')
#process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hltParticleFlow_*_*')
#process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hltAK4PFJets_*_*')
#process.FEVTDEBUGHLToutput.outputCommands.append('keep *_hltAK8PFJets_*_*')

#
# Run only localreco, PFRecHit and PFCluster producers for HBHE only
#process.source.fileNames = cms.untracked.vstring('file:/cms/data/hatake/ana/PF/GPU/CMSSW_12_4_0_v2/src/test/v21/GPU/reHLT_HLT.root ')

# Path/sequence definitions
if use_cpu and use_gpu:
    process.HBHEPFCPUGPUTask = cms.Path(
         process.hltHcalDigis
        +process.hltHcalDigisGPU
        +process.hltHbherecoLegacy
        +process.hltHbherecoGPU
        #+process.hltHbherecoFromGPU
        +process.hltParticleFlowRecHitHBHE
        +process.hltParticleFlowClusterHBHE
        +process.hltParticleFlowRecHitHBHEonGPU
        +process.hltParticleFlowClusterHBHEonGPU)
    process.schedule = cms.Schedule(process.HBHEPFCPUGPUTask)
elif use_cpu:
    process.HBHEPFCPUTask = cms.Path(
         process.hltHcalDigis
        +process.hltHbherecoLegacy
        +process.hltParticleFlowRecHitHBHE
        +process.hltParticleFlowClusterHBHE)
    process.schedule = cms.Schedule(process.HBHEPFCPUTask)
elif use_gpu:
    process.HBHEPFGPUTask = cms.Path(
         process.hltHcalDigis
        +process.hltHcalDigisGPU
        +process.hltHbherecoGPU
        +process.hltParticleFlowRecHitHBHEonGPU
        +process.hltParticleFlowClusterHBHEonGPU)
    process.schedule = cms.Schedule(process.HBHEPFGPUTask)

process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])

process.options.numberOfThreads = cms.untracked.uint32(8)
