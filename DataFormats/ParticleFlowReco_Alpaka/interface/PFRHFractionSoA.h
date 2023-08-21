#ifndef ParticleFlowReco_PFRHFractionSoA_h
#define ParticleFlowReco_PFRHFractionSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFRHFractionSoALayout,
                      SOA_COLUMN(float, pcrh_frac),
                      SOA_COLUMN(int, pcrh_pfrhIdx),
                      SOA_COLUMN(int, pcrh_pfcIdx))

  using PFRHFractionSoA = PFRHFractionSoALayout<>;
}  // namespace reco

#endif
