#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigatorCore.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/JobConfigurationAlpakaRecord.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitHBHETopologyAlpakaESRcd.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitHBHETopologyAlpakaESData.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFRecHitHBHETopologyESProducer : public ESProducer {
  public:
    PFRecHitHBHETopologyESProducer(edm::ParameterSet const& iConfig)
      : hcalEnums_(iConfig.getParameter<std::vector<int>>("hcalEnums")) {
      auto cc = setWhatProduced(this);
      hcalToken_ = cc.consumes();
      geomToken_ = cc.consumes();
    }

  private:
    std::vector<int> const hcalEnums_;
    edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalToken_;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;

    constexpr static const char* kProducerName = "PFRecHitHBHETopologyESProducer";

    auto logDebug() const {
      return LogTrace(kProducerName) << "[" << kProducerName << "] ";
    }

  public:
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("appendToDataLabel", "");
      desc.add<std::vector<int>>("hcalEnums", {1, 2});
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<PFRecHitHBHETopologyAlpakaESDataHost> produce(PFRecHitHBHETopologyAlpakaESRcd const& iRecord) {
      auto const& geom = iRecord.get(geomToken_);
      auto const& topo = iRecord.get(hcalToken_);

      auto const* hcalBarrelGeo = geom.getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
      auto const* hcalEndcapGeo = geom.getSubdetectorGeometry(DetId::Hcal, HcalEndcap);

      auto navicore = std::make_unique<PFHCALDenseIdNavigatorCore>(hcalEnums_, geom, topo);

      // Filling HCAL DenseID vectors
      auto const& denseIds = navicore.get()->getValidDenseIds();

      // Filling information to define arrays for all relevant HBHE DetIds
      auto const denseIdMax = *std::max_element(denseIds.begin(), denseIds.end());
      auto const denseIdMin = *std::min_element(denseIds.begin(), denseIds.end());
      auto const productSize = denseIdMax - denseIdMin + 1;

      logDebug() << "denseIdMax = " << denseIdMax;
      logDebug() << "denseIdMin = " << denseIdMin;
      logDebug() << "denseIds.size() = " << denseIds.size();
      logDebug() << "productSize = " << productSize;

      auto product = std::make_unique<PFRecHitHBHETopologyAlpakaESDataHost>(productSize, cms::alpakatools::host());
      auto view = product->view();
      view.denseId_min() = denseIdMin;
      view.denseId_max() = denseIdMax;

      for (auto const denseId : denseIds) {
        DetId const detid = topo.denseId2detId(denseId);
        HcalDetId const hid = HcalDetId(detid);
        GlobalPoint pos;
        if (hid.subdet() == HcalBarrel)
          pos = hcalBarrelGeo->getGeometry(detid)->getPosition();
        else if (hid.subdet() == HcalEndcap)
          pos = hcalEndcapGeo->getGeometry(detid)->getPosition();
        else
          edm::LogWarning(kProducerName) << "Unexpected subdetector found for detId: rawId="
            << hid.rawId() << " subdet=" << hid.subdet();

        auto const index = denseId - denseIdMin;
        view.positionX()[index] = pos.x();
        view.positionY()[index] = pos.y();
        view.positionZ()[index] = pos.z();

        auto neigh = navicore.get()->getNeighbours(denseId);

        for (uint32_t n = 0; n < 8; ++n) {
          view[index].neighbours()(n) = 0xffffffff;

          // cmssdt.cern.ch/lxr/source/RecoParticleFlow/PFClusterProducer/interface/PFHCALDenseIdNavigator.h#0087
          // Order: CENTER(NONE),SOUTH,SOUTHEAST,SOUTHWEST,EAST,WEST,NORTHEAST,NORTHWEST,NORTH
          // neigh[0] is the rechit itself. Skip for neighbour array
          // If no neighbour exists in a direction, the value will be 0
          // Some neighbors from HF included! Need to test if these are included in the map!
          auto neighDetId = neigh[n + 1].rawId();
          if (neighDetId <= 0)
            continue;

          auto const neighDenseId = topo.detId2denseId(neighDetId);
          if (neighDenseId < denseIdMin or neighDenseId > denseIdMax)
            continue;

          view[index].neighbours()(n) = neighDenseId - denseIdMin;
        }

        logDebug() << "detId: rawId=" << hid.rawId() << " subdet=" << hid.subdet() << " index=" << index;
        logDebug() << "  position: x=" << pos.x() << " y=" << pos.y() << " z=" << pos.z();
        for (uint32_t idx = 0; idx < 8; ++idx) {
          logDebug() << "  neighbour[" << idx << "]: " << view[index].neighbours()(idx);
        }
      }

      navicore.release();

      return product;
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PFRecHitHBHETopologyESProducer);
