import FWCore.ParameterSet.Config as cms

import sys
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test a simple workflow with Alpaka modules')

parser.add_argument('-a', '--accelerators', type=str, default='*',
                    help='Comma-separated string used to set process.options.accelerators (default: "*")')

parser.add_argument('-l', '--esProductLabel', type=str, default='',
                    help='Value of "appendToDataLabel" parameter of the test ESProducer (default: "")')

parser.add_argument('-s', '--useSequence', action='store_true', default=False,
                    help='Put the Alpaka EDProducer in a cms.Sequence instead of a cms.Task (default: False)')

parser.add_argument('-an', '--useAnalyzer', action='store_true', default=False,
                    help='Use EDAnalyzer to consume outputs of Alpaka EDProducer (default: False)')

parser.add_argument('-r', '--run', type=int, default=361054,
                    help='Run number (default: 361054)')

parser.add_argument('-d', '--dumpPython', type=str, default=None,
                    help='Path to file containing output of process.dumpPython() (disabled by default)')

parser.add_argument('-v', '--logVerbosityLevel', type=str, default='FWKINFO',
                    help='Value of process.MessageLogger.cerr.threshold (default: "FWKINFO"; examples: "INFO", "DEBUG")')

argv = sys.argv[:]
if '--' in argv: argv.remove('--')
args, unknown = parser.parse_known_args(argv)

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('TEST', Run3)

process.options.accelerators = args.accelerators.split(',')
print('accelerators:', process.options.accelerators.value())

if '"' in args.esProductLabel:
  args.esProductLabel = args.esProductLabel.replace('"', '')
if "'" in args.esProductLabel:
  args.esProductLabel = args.esProductLabel.replace("'", '')
print('esProductLabel:', '"'+args.esProductLabel+'"')

print('useSequence:', args.useSequence)
print('useAnalyzer:', args.useAnalyzer)

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')

process.source = cms.Source('EmptySource',
  firstRun = cms.untracked.uint32(args.run)
)

process.maxEvents.input = 10

process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

process.jobConfAlpakaRcdESSource = cms.ESSource('EmptyESSource',
  recordName = cms.string('JobConfigurationAlpakaRecord'),
  iovIsRunNotTime = cms.bool(True),
  firstValid = cms.vuint32(1)
)

process.pfRecHitHBHETopologyAlpakaESRcdESSource = cms.ESSource('EmptyESSource',
  recordName = cms.string('PFRecHitHBHETopologyAlpakaESRcd'),
  iovIsRunNotTime = cms.bool(True),
  firstValid = cms.vuint32(1)
)

from RecoParticleFlow.PFRecHitProducer.pfRecHitHBHEParamsESProducer_cfi import pfRecHitHBHEParamsESProducer as _pfRecHitHBHEParamsESProducer
process.pfRecHitHBHEParamsESProducer = _pfRecHitHBHEParamsESProducer.clone(
  appendToDataLabel = args.esProductLabel,
  energyThresholdsHB = [0.1, 0.2, 0.3, 0.4],
  energyThresholdsHE = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
)

from RecoParticleFlow.PFRecHitProducer.pfRecHitHBHETopologyESProducer_cfi import pfRecHitHBHETopologyESProducer as _pfRecHitHBHETopologyESProducer
process.pfRecHitHBHETopologyESProducer = _pfRecHitHBHETopologyESProducer.clone(
  appendToDataLabel = args.esProductLabel
)

from RecoParticleFlow.PFRecHitProducer.testPFRecHitHBHETestProducer_cfi import testPFRecHitHBHETestProducer as _testPFRecHitHBHETestProducer
process.testPFRecHitHBHETestProducer = _testPFRecHitHBHETestProducer.clone(
  pfRecHitParams = 'pfRecHitHBHEParamsESProducer:'+args.esProductLabel,
  pfRecHitTopology = 'pfRecHitHBHETopologyESProducer:'+args.esProductLabel
)

if args.useAnalyzer:
  from RecoParticleFlow.PFRecHitProducer.testEmptyAnalyzer_cfi import testEmptyAnalyzer as _testEmptyAnalyzer
  process.testEmptyAnalyzer = _testEmptyAnalyzer.clone(
    source = 'testPFRecHitHBHETestProducer'
  )

if args.useSequence:
  process.testSequence = cms.Sequence( process.testPFRecHitHBHETestProducer )
  if args.useAnalyzer:
    process.testSequence += process.testEmptyAnalyzer
  process.testPath = cms.Path( process.testSequence )
else:
  process.testTask = cms.Task( process.pfRecHitHBHEParamsESProducer, process.testPFRecHitHBHETestProducer )
  process.testSequence = cms.Sequence()
  if args.useAnalyzer:
    process.testSequence += process.testEmptyAnalyzer
  process.testPath = cms.Path( process.testSequence , process.testTask )

process.output = cms.OutputModule('PoolOutputModule',
  fileName = cms.untracked.string('tmp.root'),
  outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_testPFRecHitHBHETestProducer*_*_*',
  )
)
process.testEndPath = cms.EndPath( process.output )

## MessageLogger
process.load('FWCore.MessageLogger.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1 # only report every Nth event start
process.MessageLogger.cerr.FwkReport.limit = -1      # max number of reported messages (all if -1)
process.MessageLogger.cerr.enableStatistics = False  # enable "MessageLogger Summary" message
process.MessageLogger.cerr.threshold = args.logVerbosityLevel
setattr(process.MessageLogger.cerr, args.logVerbosityLevel,
  cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1), # every event!
    limit = cms.untracked.int32(-1)       # no limit! (default is limit=0, i.e. no messages reported)
  )
)
if args.logVerbosityLevel == 'DEBUG':
  process.MessageLogger.debugModules = ['*']

# dump content of cms.Process to python file
if args.dumpPython != None:
  open(args.dumpPython, 'w').write(process.dumpPython())
