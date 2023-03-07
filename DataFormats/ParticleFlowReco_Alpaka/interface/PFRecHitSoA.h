#ifndef ParticleFlowReco_PFRecHitSoA_h
#define ParticleFlowReco_PFRecHitSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

namespace reco {
  
  using PFRecHitsNeighbours = std::array<uint32_t, 8>;
  GENERATE_SOA_LAYOUT(PFRecHitSoALayout,
    SOA_COLUMN(uint32_t, detId),
    SOA_COLUMN(float, energy),
    SOA_COLUMN(float, time),
    SOA_COLUMN(int, depth),
    SOA_COLUMN(PFLayer::Layer, layer),
    SOA_COLUMN(uint32_t, num_neighbours),
    SOA_COLUMN(PFRecHitsNeighbours, neighbours),
    SOA_COLUMN(float, x),
    SOA_COLUMN(float, y),
    SOA_COLUMN(float, z),
    SOA_SCALAR(uint32_t, size)
  )

  using PFRecHitSoA = PFRecHitSoALayout<>;
}

#endif
