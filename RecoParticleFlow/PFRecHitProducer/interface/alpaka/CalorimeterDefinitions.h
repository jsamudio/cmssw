#ifndef RecoParticleFlow_PFRecHitProducer_interface_alpaka_CalorimeterDefinitions_h
#define RecoParticleFlow_PFRecHitProducer_interface_alpaka_CalorimeterDefinitions_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/CaloRecHitDeviceCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitParamsRecord.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyHostCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyRecord.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitParamsDeviceCollection.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitTopologyDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ParticleFlowRecHitProducer {

  // Get subdetector encoded in detId to narrow the range of reference table values to search
  // https://cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0048
  constexpr inline uint32_t getSubdet(uint32_t detId) { return ((detId >> DetId::kSubdetOffset) & DetId::kSubdetMask); }

  // structs to be used as template arguments
  struct HCAL {
    using CaloRecHitType = HBHERecHit;
    using CaloRecHitSoATypeHost = reco::CaloRecHitHostCollection;
    using CaloRecHitSoATypeDevice = reco::CaloRecHitDeviceCollection;
    using ParameterType = reco::PFRecHitHCALParamsDeviceCollection;
    using ParameterRecordType = PFRecHitHCALParamsRecord;
    using TopologyTypeHost = reco::PFRecHitHCALTopologyHostCollection;
    using TopologyTypeDevice = reco::PFRecHitHCALTopologyDeviceCollection;
    using TopologyRecordType = PFRecHitHCALTopologyRecord;

    static constexpr DetId::Detector DetectorId = DetId::Detector::Hcal;
    static constexpr int SubdetectorBarrelId = HcalSubdetector::HcalBarrel;
    static constexpr int SubdetectorEndcapId = HcalSubdetector::HcalEndcap;

    static constexpr uint32_t maxDepthHB = 4;
    static constexpr uint32_t maxDepthHE = 7;
    static constexpr uint32_t firstHBRing = 1;
    static constexpr uint32_t lastHBRing = 16;
    static constexpr uint32_t firstHERing = 16;
    static constexpr uint32_t lastHERing = 29;
    static constexpr uint32_t IPHI_MAX = 72;
    static constexpr uint32_t SIZE_BARREL = maxDepthHB * (lastHBRing - firstHBRing + 1) * IPHI_MAX * 2;
    static constexpr uint32_t SIZE_ENDCAP = maxDepthHE * (lastHERing - firstHERing + 1) * IPHI_MAX * 2;
    static constexpr uint32_t SIZE = SIZE_BARREL + SIZE_ENDCAP;  // maximum possible HCAL denseId (=23328)

    static constexpr bool detIdInRange(uint32_t detId) {
      return detId != 0 && DetId(detId).det() == DetId::Detector::Hcal &&
             (getSubdet(detId) == HcalSubdetector::HcalBarrel || getSubdet(detId) == HcalSubdetector::HcalEndcap);
    }

    //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0163
    static constexpr uint32_t getDepth(uint32_t detId) {
      return ((detId >> HcalDetId::kHcalDepthOffset2) & HcalDetId::kHcalDepthMask2);
    }

    //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0148
    static constexpr uint32_t getIetaAbs(uint32_t detId) {
      return ((detId >> HcalDetId::kHcalEtaOffset2) & HcalDetId::kHcalEtaMask2);
    }

    //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0157
    static constexpr uint32_t getIphi(uint32_t detId) { return (detId & HcalDetId::kHcalPhiMask2); }

    //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0141
    static constexpr int getZside(uint32_t detId) { return ((detId & HcalDetId::kHcalZsideMask2) ? (1) : (-1)); }

    // https://cmssdt.cern.ch/lxr/source/Geometry/CaloTopology/src/HcalTopology.cc#1170
    static constexpr uint32_t detId2denseIdHB(uint32_t detId) {
      const uint32_t nEtaHB = (lastHBRing - firstHBRing + 1);
      const uint32_t ip = getIphi(detId);
      const uint32_t ie = getIetaAbs(detId);
      const uint32_t dp = getDepth(detId);
      const int zn = getZside(detId);
      uint32_t retval = (dp - 1) + maxDepthHB * (ip - 1);
      if (zn > 0)
        retval += maxDepthHB * IPHI_MAX * (ie * zn - firstHBRing);
      else
        retval += maxDepthHB * IPHI_MAX * (ie * zn + lastHBRing + nEtaHB);

      return retval;
    }

    // https://cmssdt.cern.ch/lxr/source/Geometry/CaloTopology/src/HcalTopology.cc#1189
    static constexpr uint32_t detId2denseIdHE(uint32_t detId) {
      const uint32_t nEtaHE = (lastHERing - firstHERing + 1);
      const uint32_t ip = getIphi(detId);
      const uint32_t ie = getIetaAbs(detId);
      const uint32_t dp = getDepth(detId);
      const int zn = getZside(detId);
      uint32_t retval = (dp - 1) + maxDepthHE * (ip - 1);
      if (zn > 0)
        retval += maxDepthHE * IPHI_MAX * (ie * zn - firstHERing);
      else
        retval += maxDepthHE * IPHI_MAX * (ie * zn + lastHERing + nEtaHE);

      return retval + SIZE_BARREL;
    }

    static constexpr uint32_t detId2denseId(uint32_t detId) {
      const uint32_t subdet = getSubdet(detId);
      if (subdet == HcalBarrel)
        return detId2denseIdHB(detId);
      if (subdet == HcalEndcap)
        return detId2denseIdHE(detId);

      printf("invalid detId: %u\n", detId);
      return -1;
    }
  };

  struct ECAL {
    using CaloRecHitType = EcalRecHit;
    using CaloRecHitSoATypeHost = reco::CaloRecHitHostCollection;
    using CaloRecHitSoATypeDevice = reco::CaloRecHitDeviceCollection;
    using ParameterType = reco::PFRecHitECALParamsDeviceCollection;
    using ParameterRecordType = PFRecHitECALParamsRecord;
    using TopologyTypeHost = reco::PFRecHitECALTopologyHostCollection;
    using TopologyTypeDevice = reco::PFRecHitECALTopologyDeviceCollection;
    using TopologyRecordType = PFRecHitECALTopologyRecord;

    static constexpr DetId::Detector DetectorId = DetId::Detector::Ecal;
    static constexpr int SubdetectorBarrelId = EcalSubdetector::EcalBarrel;
    static constexpr int SubdetectorEndcapId = EcalSubdetector::EcalEndcap;

    // https://cmssdt.cern.ch/lxr/source/DataFormats/EcalRecHit/interface/EcalRecHit.h#0021
    enum Flags {
      kGood = 0,   // channel ok, the energy and time measurement are reliable
      kPoorReco,   // the energy is available from the UncalibRecHit, but approximate (bad shape, large chi2)
      kOutOfTime,  // the energy is available from the UncalibRecHit (sync reco), but the event is out of time
      kFaultyHardware,  // The energy is available from the UncalibRecHit, channel is faulty at some hardware level (e.g. noisy)
      kNoisy,           // the channel is very noisy
      kPoorCalib,  // the energy is available from the UncalibRecHit, but the calibration of the channel is poor
      kSaturated,  // saturated channel (recovery not tried)
      kLeadingEdgeRecovered,  // saturated channel: energy estimated from the leading edge before saturation
      kNeighboursRecovered,   // saturated/isolated dead: energy estimated from neighbours
      kTowerRecovered,        // channel in TT with no data link, info retrieved from Trigger Primitive
      kDead,                  // channel is dead and any recovery fails
      kKilled,                // MC only flag: the channel is killed in the real detector
      kTPSaturated,           // the channel is in a region with saturated TP
      kL1SpikeFlag,           // the channel is in a region with TP with sFGVB = 0
      kWeird,                 // the signal is believed to originate from an anomalous deposit (spike)
      kDiWeird,               // the signal is anomalous, and neighbors another anomalous signal
      kHasSwitchToGain6,      // at least one data frame is in G6
      kHasSwitchToGain1,      // at least one data frame is in G1
                              //
      kUnknown                // to ease the interface with functions returning flags.
    };

    // https://cmssdt.cern.ch/lxr/source/DataFormats/EcalDetId/interface/EBDetId.h
    struct BARREL {
      static constexpr int MAX_IETA = 85;
      static constexpr int MAX_IPHI = 360;
      static constexpr int SIZE = 2 * MAX_IPHI * MAX_IETA;

      static constexpr int ietaAbs(uint32_t detId) { return (detId >> 9) & 0x7F; }
      static constexpr int iphi(uint32_t detId) { return detId & 0x1FF; }
      static constexpr bool positiveZ(uint32_t detId) { return detId & 0x10000; }
      static constexpr uint32_t denseIndex(uint32_t detId) {
        return (MAX_IETA + (positiveZ(detId) ? ietaAbs(detId) - 1 : -ietaAbs(detId))) * MAX_IPHI + iphi(detId) - 1;
      }
    };

    // https://cmssdt.cern.ch/lxr/source/DataFormats/EcalDetId/interface/EEDetId.h
    struct ENDCAP {
      static constexpr uint32_t kEEhalf = 7324;
      static constexpr uint32_t SIZE = kEEhalf * 2;

      static constexpr int ix(uint32_t detId) { return (detId >> 7) & 0x7F; }
      static constexpr int iy(uint32_t detId) { return detId & 0x7F; }
      static constexpr bool positiveZ(uint32_t detId) { return detId & 0x4000; }

      static constexpr uint32_t denseIndex(uint32_t detId) {
        const unsigned short kxf[] = {
            41, 51, 41, 51, 41, 51, 36, 51, 36, 51, 26, 51, 26, 51, 26, 51, 21, 51, 21, 51, 21, 51, 21, 51, 21,
            51, 16, 51, 16, 51, 14, 51, 14, 51, 14, 51, 14, 51, 14, 51, 9,  51, 9,  51, 9,  51, 9,  51, 9,  51,
            6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 4,  51, 4,  51, 4,
            51, 4,  51, 4,  56, 1,  58, 1,  59, 1,  60, 1,  61, 1,  61, 1,  62, 1,  62, 1,  62, 1,  62, 1,  62,
            1,  62, 1,  62, 1,  62, 1,  62, 1,  62, 1,  61, 1,  61, 1,  60, 1,  59, 1,  58, 4,  56, 4,  51, 4,
            51, 4,  51, 4,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51, 6,  51,
            9,  51, 9,  51, 9,  51, 9,  51, 9,  51, 14, 51, 14, 51, 14, 51, 14, 51, 14, 51, 16, 51, 16, 51, 21,
            51, 21, 51, 21, 51, 21, 51, 21, 51, 26, 51, 26, 51, 26, 51, 36, 51, 36, 51, 41, 51, 41, 51, 41, 51};

        const unsigned short kdi[] = {
            0,    10,   20,   30,   40,   50,   60,   75,   90,   105,  120,  145,  170,  195,  220,  245,  270,
            300,  330,  360,  390,  420,  450,  480,  510,  540,  570,  605,  640,  675,  710,  747,  784,  821,
            858,  895,  932,  969,  1006, 1043, 1080, 1122, 1164, 1206, 1248, 1290, 1332, 1374, 1416, 1458, 1500,
            1545, 1590, 1635, 1680, 1725, 1770, 1815, 1860, 1905, 1950, 1995, 2040, 2085, 2130, 2175, 2220, 2265,
            2310, 2355, 2400, 2447, 2494, 2541, 2588, 2635, 2682, 2729, 2776, 2818, 2860, 2903, 2946, 2988, 3030,
            3071, 3112, 3152, 3192, 3232, 3272, 3311, 3350, 3389, 3428, 3467, 3506, 3545, 3584, 3623, 3662, 3701,
            3740, 3779, 3818, 3857, 3896, 3935, 3974, 4013, 4052, 4092, 4132, 4172, 4212, 4253, 4294, 4336, 4378,
            4421, 4464, 4506, 4548, 4595, 4642, 4689, 4736, 4783, 4830, 4877, 4924, 4969, 5014, 5059, 5104, 5149,
            5194, 5239, 5284, 5329, 5374, 5419, 5464, 5509, 5554, 5599, 5644, 5689, 5734, 5779, 5824, 5866, 5908,
            5950, 5992, 6034, 6076, 6118, 6160, 6202, 6244, 6281, 6318, 6355, 6392, 6429, 6466, 6503, 6540, 6577,
            6614, 6649, 6684, 6719, 6754, 6784, 6814, 6844, 6874, 6904, 6934, 6964, 6994, 7024, 7054, 7079, 7104,
            7129, 7154, 7179, 7204, 7219, 7234, 7249, 7264, 7274, 7284, 7294, 7304, 7314};

        const uint32_t jx = ix(detId);
        const uint32_t jd = 2 * (iy(detId) - 1) + (jx - 1) / 50;
        return ((positiveZ(detId) ? kEEhalf : 0) + kdi[jd] + jx - kxf[jd]);
      }
    };

    static constexpr bool checkFlag(uint32_t flagBits, int flag) { return flagBits & (0x1 << flag); }

    static constexpr uint32_t detId2denseId(uint32_t detId) {
      const uint32_t subdet = getSubdet(detId);
      if (subdet == EcalBarrel)
        return BARREL::denseIndex(detId);
      if (subdet == EcalEndcap)
        return BARREL::SIZE + ENDCAP::denseIndex(detId);

      printf("invalid detId: %u\n", detId);
      return 0;
    }

    static constexpr bool detIdInRange(uint32_t detId) {
      return detId != 0 && DetId(detId).det() == DetId::Detector::Ecal &&
             (getSubdet(detId) == EcalSubdetector::EcalBarrel || getSubdet(detId) == EcalSubdetector::EcalEndcap);
    }

    static constexpr int getZside(uint32_t detId) {
      return ((getSubdet(detId) == EcalSubdetector::EcalBarrel) ? BARREL::positiveZ(detId) : ENDCAP::positiveZ(detId))
                 ? (1)
                 : (-1);
    }

    static constexpr uint32_t SIZE = BARREL::SIZE + ENDCAP::SIZE;  // maximum possible ECAL denseId (=75848)
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ParticleFlowRecHitProducerAlpaka

#endif  // RecoParticleFlow_PFRecHitProducer_interface_alpaka_CalorimeterDefinitions_h
