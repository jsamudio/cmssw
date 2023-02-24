#!/bin/bash

EXECMD="cmsRun RecoParticleFlow/PFRecHitProducer/test/testPFRecHitHBHETestProducers_cfg.py"

function run_test(){
  echo "TEST:" $@
  $EXECMD -- $@
  printf "%s" "=================================="
  printf "%s\n" "=================================="
}

run_test -l '""'
run_test -l '""' -s
run_test -l '""' -s -an

run_test -l 'A'
run_test -l 'A' -s
run_test -l 'A' -s -an
