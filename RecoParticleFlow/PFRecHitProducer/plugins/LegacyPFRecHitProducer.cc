#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"

class LegacyPFRecHitProducer : public edm::stream::EDProducer<> {
public:
  LegacyPFRecHitProducer(edm::ParameterSet const& config)
      : alpakaPfRecHitsToken(consumes(config.getParameter<edm::InputTag>("src"))),
        legacyPfRecHitsToken(produces()),
        geomToken(esConsumes<edm::Transition::BeginRun>()) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<reco::PFRecHitHostCollection> alpakaPfRecHitsToken;
  const edm::EDPutTokenT<reco::PFRecHitCollection> legacyPfRecHitsToken;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken;
  edm::ESHandle<CaloGeometry> geoHandle;
  std::unordered_map<PFLayer::Layer, const CaloSubdetectorGeometry*> caloGeo;
};

void LegacyPFRecHitProducer::beginRun(edm::Run const&, const edm::EventSetup& setup) {
  geoHandle = setup.getHandle(geomToken);
  caloGeo[PFLayer::HCAL_BARREL1] = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalSubdetector::HcalBarrel);
  caloGeo[PFLayer::HCAL_ENDCAP] = geoHandle->getSubdetectorGeometry(DetId::Hcal, HcalSubdetector::HcalEndcap);
  caloGeo[PFLayer::ECAL_BARREL] = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalSubdetector::EcalBarrel);
  caloGeo[PFLayer::ECAL_ENDCAP] = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalSubdetector::EcalEndcap);
}

void LegacyPFRecHitProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  edm::Handle<reco::PFRecHitHostCollection> pfRecHitsAlpakaSoA;
  event.getByToken(alpakaPfRecHitsToken, pfRecHitsAlpakaSoA);
  const reco::PFRecHitHostCollection::ConstView& alpakaPfRecHits = pfRecHitsAlpakaSoA->const_view();

  reco::PFRecHitCollection out;
  out.reserve(alpakaPfRecHits.size());

  for (size_t i = 0; i < alpakaPfRecHits.size(); i++) {
    reco::PFRecHit& pfrh =
        out.emplace_back(caloGeo.at(alpakaPfRecHits[i].layer())->getGeometry(alpakaPfRecHits[i].detId()),
                         alpakaPfRecHits[i].detId(),
                         alpakaPfRecHits[i].layer(),
                         alpakaPfRecHits[i].energy());
    pfrh.setTime(alpakaPfRecHits[i].time());
    pfrh.setDepth(alpakaPfRecHits[i].depth());

    // order in Alpaka:   N, S, E, W,NE,SW,SE,NW
    const short eta[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const short phi[8] = {1, -1, 0, 0, 1, -1, -1, 1};
    for (size_t k = 0; k < 8; k++)
      if (alpakaPfRecHits[i].neighbours()(k) != -1)
        pfrh.addNeighbour(eta[k], phi[k], 0, alpakaPfRecHits[i].neighbours()(k));
  }

  event.emplace(legacyPfRecHitsToken, out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LegacyPFRecHitProducer);
