#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHETopologyAlpakaESRcd_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHETopologyAlpakaESRcd_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class PFRecHitHBHETopologyAlpakaESRcd : public edm::eventsetup::DependentRecordImplementation<
                                            PFRecHitHBHETopologyAlpakaESRcd,
                                            edm::mpl::Vector<HcalRecNumberingRecord, CaloGeometryRecord>> {};

#endif  // RecoParticleFlow_PFRecHitProducer_interface_PFRecHitHBHETopologyAlpakaESRcd_h
