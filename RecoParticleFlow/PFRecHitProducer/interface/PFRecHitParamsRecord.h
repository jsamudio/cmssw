#ifndef RecoParticleFlow_PFRecHitProducer_ParamsRecord_h
#define RecoParticleFlow_PFRecHitProducer_ParamsRecord_h

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"

class PFRecHitHCALParamsRecord : public edm::eventsetup::EventSetupRecordImplementation<PFRecHitHCALParamsRecord> {};

class PFRecHitECALParamsRecord
    : public edm::eventsetup::DependentRecordImplementation<PFRecHitECALParamsRecord,
                                                            edm::mpl::Vector<EcalPFRecHitThresholdsRcd>> {};

#endif  // RecoParticleFlow_PFRecHitProducer_ParamsRecord_h
