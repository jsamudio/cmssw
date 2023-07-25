#ifndef RecoParticleFlow_PFClusterProducerAlpaka_interface_PFRecHitHBHEParamsAlpakaESDataSoA_h
#define RecoParticleFlow_PFClusterProducerAlpaka_interface_PFRecHitHBHEParamsAlpakaESDataSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFClusterParamsAlpakaESDataSoALayout,
                      SOA_SCALAR(int32_t, nNeigh),
                      SOA_SCALAR(float, seedPt2ThresholdEB),
                      SOA_SCALAR(float, seedPt2ThresholdEE),
                      SOA_COLUMN(float, seedEThresholdEB_vec),
                      SOA_COLUMN(float, seedEThresholdEE_vec),
                      SOA_COLUMN(float, topoEThresholdEB_vec),
                      SOA_COLUMN(float, topoEThresholdEE_vec),
                      SOA_SCALAR(float, showerSigma2),
                      SOA_SCALAR(float, minFracToKeep),
                      SOA_SCALAR(float, minFracTot),
                      SOA_SCALAR(uint32_t, maxIterations),
                      SOA_SCALAR(bool, excludeOtherSeeds),
                      SOA_SCALAR(float, stoppingTolerance),
                      SOA_SCALAR(float, minFracInCalc),
                      SOA_SCALAR(float, minAllowedNormalization),
                      SOA_COLUMN(float, recHitEnergyNormInvEB_vec),
                      SOA_COLUMN(float, recHitEnergyNormInvEE_vec),
                      SOA_SCALAR(float, barrelTimeResConsts_corrTermLowE),
                      SOA_SCALAR(float, barrelTimeResConsts_threshLowE),
                      SOA_SCALAR(float, barrelTimeResConsts_noiseTerm),
                      SOA_SCALAR(float, barrelTimeResConsts_constantTermLowE2),
                      SOA_SCALAR(float, barrelTimeResConsts_noiseTermLowE),
                      SOA_SCALAR(float, barrelTimeResConsts_threshHighE),
                      SOA_SCALAR(float, barrelTimeResConsts_constantTerm2),
                      SOA_SCALAR(float, barrelTimeResConsts_resHighE2),
                      SOA_SCALAR(float, endcapTimeResConsts_corrTermLowE),
                      SOA_SCALAR(float, endcapTimeResConsts_threshLowE),
                      SOA_SCALAR(float, endcapTimeResConsts_noiseTerm),
                      SOA_SCALAR(float, endcapTimeResConsts_constantTermLowE2),
                      SOA_SCALAR(float, endcapTimeResConsts_noiseTermLowE),
                      SOA_SCALAR(float, endcapTimeResConsts_threshHighE),
                      SOA_SCALAR(float, endcapTimeResConsts_constantTerm2),
                      SOA_SCALAR(float, endcapTimeResConsts_resHighE2))

  using PFClusterParamsAlpakaESDataSoA = PFClusterParamsAlpakaESDataSoALayout<>;

}

#endif
