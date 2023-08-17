#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/CalorimeterDefinitions.h"

#include <memory>
#include <variant>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using namespace ParticleFlowRecHitProducerAlpaka;

  template <typename CAL>
  class PFRecHitTopologyESProducer : public ESProducer {
  public:
    PFRecHitTopologyESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      geomToken_ = cc.consumes();
      if constexpr (std::is_same_v<CAL, HCAL>)
        hcalToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("appendToDataLabel", "");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<typename CAL::TopologyTypeHost> produce(const typename CAL::TopologyRecordType& iRecord) {
      const auto& geom = iRecord.get(geomToken_);
      auto product = std::make_unique<typename CAL::TopologyTypeHost>(CAL::SIZE, cms::alpakatools::host());
      auto view = product->view();

      const int calEnums[2] = {CAL::SubdetectorBarrelId, CAL::SubdetectorEndcapId};
      for (const auto subdet : calEnums) {
        // Construct topology
        //  for HCAL: using dedicated record
        //  for ECAL: from CaloGeometry (separate for barrel and endcap)
        const CaloSubdetectorGeometry* geo = geom.getSubdetectorGeometry(CAL::DetectorId, subdet);
        const CaloSubdetectorTopology* topo;
        std::variant<EcalBarrelTopology, EcalEndcapTopology> topoVar;  // need to store ECAL topology temporarily
        if constexpr (std::is_same_v<CAL, HCAL>)
          topo = &iRecord.get(hcalToken_);
        else if (subdet == EcalSubdetector::EcalBarrel)
          topo = &topoVar.emplace<EcalBarrelTopology>(geom);
        else
          topo = &topoVar.emplace<EcalEndcapTopology>(geom);

        // Fill product
        for (auto const detId : geom.getValidDetIds(CAL::DetectorId, subdet)) {
          const uint32_t denseId = CAL::detId2denseId(detId);
          assert(denseId <= CAL::SIZE);

          const GlobalPoint pos = geo->getGeometry(detId)->getPosition();
          view.positionX(denseId) = pos.x();
          view.positionY(denseId) = pos.y();
          view.positionZ(denseId) = pos.z();

          for (uint32_t n = 0; n < 8; n++) {
            uint32_t neighDetId = GetNeighbourDetId(detId, n, *topo);
            if (CAL::detIdInRange(neighDetId))
              view.neighbours(denseId)(n) = CAL::detId2denseId(neighDetId);
            else
              view.neighbours(denseId)(n) = 0xffffffff;
          }
        }
      }

      // Remove neighbours that are not backward compatible (only for HCAL)
      if (std::is_same_v<CAL, HCAL>)
        for (const auto subdet : calEnums)
          for (auto const detId : geom.getValidDetIds(CAL::DetectorId, subdet)) {
            const uint32_t denseId = CAL::detId2denseId(detId);
            for (uint32_t n = 0; n < 8; n++) {
              const reco::PFRecHitsTopologyNeighbours& neighboursOfNeighbour =
                  view.neighbours(view.neighbours(denseId)[n]);
              if (std::find(neighboursOfNeighbour.begin(), neighboursOfNeighbour.end(), denseId) ==
                  neighboursOfNeighbour.end())
                view.neighbours(denseId)[n] = 0xffffffff;
            }
          }

      //// Print results (for debugging)
      //for(const auto subdet : calEnums)
      //  for (const auto detId : geom.getValidDetIds(CAL::DetectorId, subdet)) {
      //    const uint32_t denseId = CAL::detId2denseId(detId);
      //    printf("PFRecHitTopologyESProducer: detId:%u denseId:%u pos:%f,%f,%f neighbours:%d,%d,%d,%d;%d,%d,%d,%d\n",
      //      (uint32_t)detId, denseId,
      //      view[denseId].positionX(), view[denseId].positionY(), view[denseId].positionZ(),
      //      view[denseId].neighbours()(0), view[denseId].neighbours()(1), view[denseId].neighbours()(2), view[denseId].neighbours()(3),
      //      view[denseId].neighbours()(4), view[denseId].neighbours()(5), view[denseId].neighbours()(6), view[denseId].neighbours()(7));
      //  }

      return product;
    }

  private:
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
    edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;

    // specialised for HCAL/ECAL, because non-nearest neighbours are defined differently
    uint32_t GetNeighbourDetId(const uint32_t detId, const uint32_t direction, const CaloSubdetectorTopology& topo);
  };

  template <>
  uint32_t PFRecHitTopologyESProducer<ECAL>::GetNeighbourDetId(const uint32_t detId,
                                                               const uint32_t direction,
                                                               const CaloSubdetectorTopology& topo) {
    // desired order for PF: NORTH, SOUTH, EAST, WEST, NORTHEAST, SOUTHWEST, SOUTHEAST, NORTHWEST
    if (detId == 0)
      return 0;

    if (direction == 0)            // NORTH
      return topo.goNorth(detId);  // larger iphi values (except phi boundary)
    if (direction == 1)            // SOUTH
      return topo.goSouth(detId);  // smaller iphi values (except phi boundary)
    if (direction == 2)            // EAST
      return topo.goEast(detId);   // smaller ieta values
    if (direction == 3)            // WEST
      return topo.goWest(detId);   // larger ieta values

    if (direction == 4) {  // NORTHEAST
      const uint32_t NE = GetNeighbourDetId(GetNeighbourDetId(detId, 0, topo), 2, topo);
      if (NE)
        return NE;
      return GetNeighbourDetId(GetNeighbourDetId(detId, 2, topo), 0, topo);
    }
    if (direction == 5) {  // SOUTHWEST
      const uint32_t SW = GetNeighbourDetId(GetNeighbourDetId(detId, 1, topo), 3, topo);
      if (SW)
        return SW;
      return GetNeighbourDetId(GetNeighbourDetId(detId, 3, topo), 1, topo);
    }
    if (direction == 6) {  // SOUTHEAST
      const uint32_t ES = GetNeighbourDetId(GetNeighbourDetId(detId, 2, topo), 1, topo);
      if (ES)
        return ES;
      return GetNeighbourDetId(GetNeighbourDetId(detId, 1, topo), 2, topo);
    }
    if (direction == 7) {  // NORTHWEST
      const uint32_t WN = GetNeighbourDetId(GetNeighbourDetId(detId, 3, topo), 0, topo);
      if (WN)
        return WN;
      return GetNeighbourDetId(GetNeighbourDetId(detId, 0, topo), 3, topo);
    }
    return 0;
  }

  template <>
  uint32_t PFRecHitTopologyESProducer<HCAL>::GetNeighbourDetId(const uint32_t detId,
                                                               const uint32_t direction,
                                                               const CaloSubdetectorTopology& topo) {
    // desired order for PF: NORTH, SOUTH, EAST, WEST, NORTHEAST, SOUTHWEST, SOUTHEAST, NORTHWEST
    if (detId == 0)
      return 0;

    if (direction == 0)            // NORTH
      return topo.goNorth(detId);  // larger iphi values (except phi boundary)
    if (direction == 1)            // SOUTH
      return topo.goSouth(detId);  // smaller iphi values (except phi boundary)
    if (direction == 2)            // EAST
      return topo.goEast(detId);   // smaller ieta values
    if (direction == 3)            // WEST
      return topo.goWest(detId);   // larger ieta values

    if (direction == 4) {             // NORTHEAST
      if (HCAL::getZside(detId) > 0)  // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
        return GetNeighbourDetId(GetNeighbourDetId(detId, 2, topo), 0, topo);
      else  // negative eta: move in phi first then move to east (coarser phi granularity)
        return GetNeighbourDetId(GetNeighbourDetId(detId, 0, topo), 2, topo);
    }
    if (direction == 5) {             // SOUTHWEST
      if (HCAL::getZside(detId) > 0)  // positive eta: move in phi first then move to west (coarser phi granularity)
        return GetNeighbourDetId(GetNeighbourDetId(detId, 1, topo), 3, topo);
      else  // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
        return GetNeighbourDetId(GetNeighbourDetId(detId, 3, topo), 1, topo);
    }
    if (direction == 6) {             // SOUTHEAST
      if (HCAL::getZside(detId) > 0)  // positive eta: east -> move to smaller |ieta| (finner phi granularity) first
        return GetNeighbourDetId(GetNeighbourDetId(detId, 2, topo), 1, topo);
      else  // negative eta: move in phi first then move to east (coarser phi granularity)
        return GetNeighbourDetId(GetNeighbourDetId(detId, 1, topo), 2, topo);
    }
    if (direction == 7) {             // NORTHWEST
      if (HCAL::getZside(detId) > 0)  // positive eta: move in phi first then move to west (coarser phi granularity)
        return GetNeighbourDetId(GetNeighbourDetId(detId, 0, topo), 3, topo);
      else  // negative eta: west -> move to smaller |ieta| (finner phi granularity) first
        return GetNeighbourDetId(GetNeighbourDetId(detId, 3, topo), 0, topo);
    }
    return 0;
  }

  using PFRecHitECALTopologyESProducer = PFRecHitTopologyESProducer<ECAL>;
  using PFRecHitHCALTopologyESProducer = PFRecHitTopologyESProducer<HCAL>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFRecHitECALTopologyESProducer);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFRecHitHCALTopologyESProducer);
