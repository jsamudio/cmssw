#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHEParamsAlpakaESDataSoA_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHEParamsAlpakaESDataSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFRecHitHBHEParamsAlpakaESDataSoALayout,
                      SOA_COLUMN(float, energyThresholds))

  using PFRecHitHBHEParamsAlpakaESDataSoA = PFRecHitHBHEParamsAlpakaESDataSoALayout<>;

}

#endif
