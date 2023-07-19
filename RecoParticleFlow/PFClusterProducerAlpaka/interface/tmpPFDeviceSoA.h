#ifndef RecoParticleFlow_PFRecHitProducerAlpaka_interface_tmpPFDeviceSoA_h
#define RecoParticleFlow_PFRecHitProducerAlpaka_interface_tmpPFDeviceSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

      GENERATE_SOA_LAYOUT(tmpPFDeviceSoA1Layout,
                                    SOA_COLUMN(int, pfrh_topoId),
                                    SOA_COLUMN(int, pfrh_isSeed),
                                    SOA_COLUMN(int, pfrh_passTopoThresh),
                                    SOA_COLUMN(int, topoSeedCount),
                                    SOA_COLUMN(int, topoRHCount),
                                    SOA_COLUMN(int, seedFracOffsets),
                                    SOA_COLUMN(int, topoSeedOffsets),
                                    SOA_COLUMN(int, topoSeedList),
                                    SOA_COLUMN(int, pfc_iter),
                                    SOA_SCALAR(int, topoIter),
                                    SOA_SCALAR(int, pcrhFracSize),
                                    SOA_COLUMN(int, rhCount),
                                    SOA_SCALAR(int, nSeeds),
                                    SOA_SCALAR(int, nEdges),
                                    SOA_COLUMN(int, pfc_energy),
                                    SOA_COLUMN(int, rhcount),
                                    SOA_COLUMN(int, wl_d),
                                    SOA_SCALAR(int, topL),
                                    SOA_SCALAR(int, posL),
                                    SOA_SCALAR(int, topH),
                                    SOA_SCALAR(int, posH),
                                    SOA_SCALAR(int, nTopos),
                                    SOA_COLUMN(int, topoIds),
                                    SOA_SCALAR(int, nRHFracs_tmp),
                                    SOA_SCALAR(int, nRHFracs),
                                    SOA_COLUMN(int, rhIdxToSeedIdx),
                                    SOA_COLUMN(float, pcrh_fracSum))

            using tmpPFDeviceSoA1 = tmpPFDeviceSoA1Layout<>;


      GENERATE_SOA_LAYOUT(tmpPFDeviceSoA2Layout,
                                    SOA_COLUMN(int, pfrh_edgeIdx), // needs nRH + 1 allocation
                                    SOA_COLUMN(int, pfrh_edgeList)) // needs nRH + maxNeighbors allocation

            using tmpPFDeviceSoA2 = tmpPFDeviceSoA2Layout<>;
}

#endif
