#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHETopologyAlpakaESDataSoA_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHETopologyAlpakaESDataSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  using PFRecHitsTopologyNeighbours = Eigen::Matrix<uint32_t, 8, 1>;
  GENERATE_SOA_LAYOUT(PFRecHitHBHETopologyAlpakaESDataSoALayout,
                      SOA_COLUMN(float, positionX),
                      SOA_COLUMN(float, positionY),
                      SOA_COLUMN(float, positionZ),
                      SOA_EIGEN_COLUMN(PFRecHitsTopologyNeighbours, neighbours),
                      SOA_SCALAR(uint32_t, denseId_min),
                      SOA_SCALAR(uint32_t, denseId_max))

  using PFRecHitHBHETopologyAlpakaESDataSoA = PFRecHitHBHETopologyAlpakaESDataSoALayout<>;

}  // namespace reco

#endif
