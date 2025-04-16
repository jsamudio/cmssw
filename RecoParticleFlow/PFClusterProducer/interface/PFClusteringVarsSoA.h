#ifndef RecoParticleFlow_PFClusterProducer_interface_PFClusteringVarsSoA_h
#define RecoParticleFlow_PFClusterProducer_interface_PFClusteringVarsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFClusteringVarsSoALayout,
                      SOA_COLUMN(int, pfrh_topoId),          // topo ID of individual PF rechit
                      SOA_COLUMN(int, pfrh_isSeed),          // is a PF rechit a seed
                      SOA_COLUMN(int, pfrh_passTopoThresh),  // does the PF rechit pass the energy noise threshold
                      SOA_COLUMN(int, topoSeedCount),        // how many seeds in a topo cluster
                      SOA_COLUMN(int, topoRHCount),          // how many non-seed rechits in a topo cluster
                      SOA_COLUMN(int, seedFracOffsets),      // offsets of seeds (start of cluster) in fraction SoA
                      SOA_COLUMN(int, topoSeedOffsets),      // offset of seed in in this SoA
                      SOA_COLUMN(int, topoSeedList),         // the seed rechit index
                      SOA_COLUMN(int, rhCount),
                      SOA_SCALAR(int, nEdges),
                      SOA_SCALAR(int, nTopos),
                      SOA_COLUMN(int, rhIdxToSeedIdx))  // given a PF rechit index, get the seed index

  using PFClusteringVarsSoA = PFClusteringVarsSoALayout<>;
}  // namespace reco

#endif
