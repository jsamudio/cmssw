#ifndef ParticleFlowReco_PFClusterSoA_h
#define ParticleFlowReco_PFClusterSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {
    
    GENERATE_SOA_LAYOUT(PFClusterSoALayout,
            SOA_COLUMN(int, pfc_depth),
            SOA_COLUMN(int, pfc_seedRHIdx),
            SOA_COLUMN(int, pfc_topoId),
            SOA_COLUMN(int, pfc_rhfracSize),
            SOA_COLUMN(int, pfc_rhfracOffset),
            SOA_COLUMN(float, pfc_energy),
            SOA_COLUMN(float, pfc_x),
            SOA_COLUMN(float, pfc_y),
            SOA_COLUMN(float, pfc_z)
    )
    using PFClusterSoA = PFClusterSoALayout<>;

    GENERATE_SOA_LAYOUT(PFRHFracSoALayout,
            SOA_COLUMN(float, pcrh_frac),
            SOA_COLUMN(int, pcrh_pfrhIdx),
            SOA_COLUMN(int, pcrh_pfcIdx)
    )
    using PFRHFracSoA = PFRHFracSoALayout<>;
}

#endif
