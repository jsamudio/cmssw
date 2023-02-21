import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

###
### EDM Input and job configuration
###
process.source = cms.Source("EmptySource")

# limit the number of events to be processed
process.maxEvents.input = 50

# enable TrigReport, TimeReport and MultiThreading
process.options.wantSummary = True
process.options.numberOfThreads = 4
process.options.numberOfStreams = 0

###
### ESModules, EDModules, Sequences, Tasks, Paths, EndPaths and Schedule
###
process.load("Configuration.StandardSequences.Accelerators_cff")

process.PFClusteringParamsGPUESSource = cms.ESSource("PFClusteringParamsGPUESSource",
  seedFinder = cms.PSet(
    nNeighbours = cms.int32( 4 ),
    thresholdsByDetector = cms.VPSet(
      cms.PSet(
        detector = cms.string( "HCAL_BARREL1" ),
        depths = cms.vint32( 1, 2, 3, 4 ),
        seedingThreshold = cms.vdouble( 0.125, 0.25, 0.35, 0.35 ),
        seedingThresholdPt = cms.double( 0.0 )
      ),
      cms.PSet(  detector = cms.string( "HCAL_ENDCAP" ),
        depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
        seedingThreshold = cms.vdouble( 0.1375, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275 ),
        seedingThresholdPt = cms.double( 0.0 )
      )
    ),
  ),
  initialClusteringStep = cms.PSet(
    thresholdsByDetector = cms.VPSet(
      cms.PSet(
        detector = cms.string( "HCAL_BARREL1" ),
        gatheringThreshold = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
      ),
      cms.PSet(
        detector = cms.string( "HCAL_ENDCAP" ),
        gatheringThreshold = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
      )
    ),
  ),
  pfClusterBuilder = cms.PSet(
    maxIterations = cms.uint32( 5 ),
    minFracTot = cms.double( 1.0E-20 ),
    minFractionToKeep = cms.double( 1.0E-7 ),
    excludeOtherSeeds = cms.bool( True ),
    showerSigma = cms.double( 10.0 ),
    stoppingTolerance = cms.double( 1.0E-8 ),
    positionCalc = cms.PSet(
      minAllowedNormalization = cms.double( 1.0E-9 ),
      posCalcNCrystals = cms.int32( 5 ),
      minFractionInCalc = cms.double( 1.0E-9 ),
      logWeightDenominatorByDetector = cms.VPSet(
        cms.PSet(  depths = cms.vint32( 1, 2, 3, 4 ),
          detector = cms.string( "HCAL_BARREL1" ),
          logWeightDenominator = cms.vdouble( 0.1, 0.2, 0.3, 0.3 )
        ),
        cms.PSet(  depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
          detector = cms.string( "HCAL_ENDCAP" ),
          logWeightDenominator = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 )
        )
      )
    ),
    recHitEnergyNorms = cms.VPSet(
      cms.PSet(  detector = cms.string( "HCAL_BARREL1" ),
        recHitEnergyNorm = cms.vdouble( 0.1, 0.2, 0.3, 0.3 )
      ),
      cms.PSet(  detector = cms.string( "HCAL_ENDCAP" ),
        recHitEnergyNorm = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 )
      )
    ),
    timeResolutionCalcBarrel = cms.PSet(
      corrTermLowE = cms.double( 0.0 ),
      threshLowE = cms.double( 6.0 ),
      noiseTerm = cms.double( 21.86 ),
      constantTermLowE = cms.double( 4.24 ),
      noiseTermLowE = cms.double( 8.0 ),
      threshHighE = cms.double( 15.0 ),
      constantTerm = cms.double( 2.82 )
    ),
    timeResolutionCalcEndcap = cms.PSet(
      corrTermLowE = cms.double( 0.0 ),
      threshLowE = cms.double( 6.0 ),
      noiseTerm = cms.double( 21.86 ),
      constantTermLowE = cms.double( 4.24 ),
      noiseTermLowE = cms.double( 8.0 ),
      threshHighE = cms.double( 15.0 ),
      constantTerm = cms.double( 2.82 )
    ),
  )
)

process.theProducer = cms.EDProducer("TestDumpPFClusteringParamsGPU")

process.theSequence = cms.Sequence( process.theProducer )

process.thePath = cms.Path( process.theSequence )

process.schedule = cms.Schedule( process.thePath )
