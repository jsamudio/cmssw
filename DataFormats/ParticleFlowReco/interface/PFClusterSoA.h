#ifndef DataFormats_ParticleFlowReco_interface_PFClusterSoA_h
#define DataFormats_ParticleFlowReco_interface_PFClusterSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFClusterSoALayout,
                      SOA_COLUMN(int, depth),
                      SOA_COLUMN(int, seedRHIdx),     // index of seed in PF rechits SoA
                      SOA_COLUMN(int, topoId),        // topo cluster ID, used in legacy cluster formation
                      SOA_COLUMN(int, rhfracSize),    // number of rechit fractions in cluster
                      SOA_COLUMN(int, rhfracOffset),  // start position of cluster in rechit fraction SoA
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_SCALAR(int, nTopos),  // number of topological clusters in event
                      SOA_SCALAR(int, nSeeds)   // number of seeds in event
  )
  using PFClusterSoA = PFClusterSoALayout<>;
}  // namespace reco

#endif  // DataFormats_ParticleFlowReco_interface_PFClusterSoA_h
