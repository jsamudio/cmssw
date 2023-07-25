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

from RecoParticleFlow.PFClusterProducerAlpaka.pfClusterParamsESProducer_cfi import pfClusterParamsESProducer as _pfClusterParamsESProducer
process.pfClusterParamsESProducer = _pfClusterParamsESProducer.clone(
  appendToDataLabel = args.esProductLabel,
  pfClusterBuilder = cms.PSet(
      stoppingTolerance = cms.double(2.0)
      ),
)


from RecoParticleFlow.PFClusterProducerAlpaka.testPFClusterTestProducer_cfi import testPFClusterTestProducer as _testPFClusterTestProducer
process.testPFClusterTestProducer = _testPFClusterTestProducer.clone(
  pfClusterParams = 'pfClusterParamsESProducer:'+args.esProductLabel
)

if args.useAnalyzer:
  from RecoParticleFlow.PFClusterProducerAlpaka.testEmptyAnalyzer2_cfi import testEmptyAnalyzer2 as _testEmptyAnalyzer2
  process.testEmptyAnalyzer2 = _testEmptyAnalyzer2.clone(
    source = 'testPFClusterTestProducer'
  )

if args.useSequence:
  process.testSequence = cms.Sequence( process.testPFClusterTestProducer )
  if args.useAnalyzer:
    process.testSequence += process.testEmptyAnalyzer2
  process.testPath = cms.Path( process.testSequence )
else:
  process.testTask = cms.Task( process.pfClusterParamsESProducer, process.testPFClusterTestProducer )
  process.testSequence = cms.Sequence()
  if args.useAnalyzer:
    process.testSequence += process.testEmptyAnalyzer2
  process.testPath = cms.Path( process.testSequence , process.testTask )

process.output = cms.OutputModule('PoolOutputModule',
  fileName = cms.untracked.string('tmp.root'),
  outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_testPFClusterTestProducer*_*_*',
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
