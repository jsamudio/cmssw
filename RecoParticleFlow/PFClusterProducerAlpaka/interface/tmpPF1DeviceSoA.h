#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_tmpPF1DeviceSoA_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_tmpPF1DeviceSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(tmpPFDeviceSoA2Layout,
                      SOA_COLUMN(int, pfrh_edgeIdx),   // needs nRH + 1 allocation
                      SOA_COLUMN(int, pfrh_edgeList))  // needs nRH + maxNeighbors allocation

  using tmpPFDeviceSoA2 = tmpPFDeviceSoA2Layout<>;
}  // namespace reco

#endif
