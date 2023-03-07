#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHETopologyAlpakaESDataSoA_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHETopologyAlpakaESDataSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFRecHitHBHETopologyAlpakaESDataSoALayout,
                      SOA_COLUMN(float, positionX),
                      SOA_COLUMN(float, positionY),
                      SOA_COLUMN(float, positionZ),
                      SOA_COLUMN(int32_t, neighbour0),
                      SOA_COLUMN(int32_t, neighbour1),
                      SOA_COLUMN(int32_t, neighbour2),
                      SOA_COLUMN(int32_t, neighbour3),
                      SOA_COLUMN(int32_t, neighbour4),
                      SOA_COLUMN(int32_t, neighbour5),
                      SOA_COLUMN(int32_t, neighbour6),
                      SOA_COLUMN(int32_t, neighbour7),
                      SOA_SCALAR(uint32_t, denseId_min),
                      SOA_SCALAR(uint32_t, denseId_max))

  using PFRecHitHBHETopologyAlpakaESDataSoA = PFRecHitHBHETopologyAlpakaESDataSoALayout<>;

}

#endif
