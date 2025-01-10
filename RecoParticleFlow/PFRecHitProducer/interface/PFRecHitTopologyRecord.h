#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologyRecord_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologyRecord_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"

class PFRecHitHCALTopologyRecord : public edm::eventsetup::DependentRecordImplementation<
                                       PFRecHitHCALTopologyRecord,
                                       edm::mpl::Vector<HcalRecNumberingRecord, CaloGeometryRecord, HcalPFCutsRcd>> {};

class PFRecHitECALTopologyRecord
    : public edm::eventsetup::DependentRecordImplementation<PFRecHitECALTopologyRecord,
                                                            edm::mpl::Vector<HcalRecNumberingRecord, CaloGeometryRecord, EcalPFRecHitThresholdsRcd>> {};
                                                            //edm::mpl::Vector<CaloGeometryRecord, EcalPFRecHitThresholdsRcd>> {};

#endif  // RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologyRecord_h
