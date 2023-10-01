#ifndef RecoParticleFlow_PFRecHitProducer_ParamsSoA_h
#define RecoParticleFlow_PFRecHitProducer_ParamsSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

// This data structure is an implementation detail of the RecoParticleFlow/PFRecHitProducer subpackage. Due to Alpaka build rules, it has to be located in the interface+src directories.
namespace reco {
  // Currently, the SoA layouts are identical for HCAL and ECAL,
  // but they should be kept separate, in case they diverge in the future
  GENERATE_SOA_LAYOUT(PFRecHitHCALParamsSoALayout, SOA_COLUMN(float, energyThresholds))
  GENERATE_SOA_LAYOUT(PFRecHitECALParamsSoALayout, SOA_COLUMN(float, energyThresholds))

  using PFRecHitHCALParamsSoA = PFRecHitHCALParamsSoALayout<>;
  using PFRecHitECALParamsSoA = PFRecHitECALParamsSoALayout<>;
}  // namespace reco

#endif
