#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFRecHitProducer/interface/alpaka/PFRecHitProducerKernel.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

namespace {
  constexpr int maxDepthHB = 4;
  constexpr int maxDepthHE = 7;
  constexpr int firstHBRing = 1;
  constexpr int lastHBRing = 16;
  constexpr int firstHERing = 16;
  constexpr int lastHERing = 29;
  constexpr int IPHI_MAX = 72;

  // Get subdetector encoded in detId to narrow the range of reference table values to search
  // https://cmssdt.cern.ch/lxr/source/DataFormats/DetId/interface/DetId.h#0048
  constexpr uint32_t getSubdet(uint32_t detId) { return ((detId >> DetId::kSubdetOffset) & DetId::kSubdetMask); }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0163
  constexpr uint32_t getDepth(uint32_t detId) {
    return ((detId >> HcalDetId::kHcalDepthOffset2) & HcalDetId::kHcalDepthMask2);
  }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0148
  constexpr uint32_t getIetaAbs(uint32_t detId) {
    return ((detId >> HcalDetId::kHcalEtaOffset2) & HcalDetId::kHcalEtaMask2);
  }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0157
  constexpr uint32_t getIphi(uint32_t detId) { return (detId & HcalDetId::kHcalPhiMask2); }

  //https://cmssdt.cern.ch/lxr/source/DataFormats/HcalDetId/interface/HcalDetId.h#0141
  constexpr int getZside(uint32_t detId) { return ((detId & HcalDetId::kHcalZsideMask2) ? (1) : (-1)); }

  // https://cmssdt.cern.ch/lxr/source/Geometry/CaloTopology/src/HcalTopology.cc#1170
  constexpr uint32_t detId2denseIdHB(uint32_t detId) {
    const int nEtaHB = (lastHBRing - firstHBRing + 1);
    const int ip = getIphi(detId);
    const int ie = getIetaAbs(detId);
    const int dp = getDepth(detId);
    const int zn = getZside(detId);
    unsigned int retval = 0xFFFFFFFFu;
    retval = (dp - 1) + maxDepthHB * (ip - 1);
    if (zn > 0)
      retval += maxDepthHB * IPHI_MAX * (ie * zn - firstHBRing);
    else
      retval += maxDepthHB * IPHI_MAX * (ie * zn + lastHBRing + nEtaHB);

    return retval;
  }

  // https://cmssdt.cern.ch/lxr/source/Geometry/CaloTopology/src/HcalTopology.cc#1189
  constexpr uint32_t detId2denseIdHE(uint32_t detId) {
    const int nEtaHE = (lastHERing - firstHERing + 1);
    const int maxPhiHE = IPHI_MAX;
    const int ip = getIphi(detId);
    const int ie = getIetaAbs(detId);
    const int dp = getDepth(detId);
    const int zn = getZside(detId);
    unsigned int retval = 0xFFFFFFFFu;
    const int HBSize = maxDepthHB * 16 * IPHI_MAX * 2;
    retval = (dp - 1) + maxDepthHE * (ip - 1) + HBSize;
    if (zn > 0)
      retval += maxDepthHE * maxPhiHE * (ie * zn - firstHERing);
    else
      retval += maxDepthHE * maxPhiHE * (ie * zn + lastHERing + nEtaHE);

    return retval;
  }

  constexpr uint32_t detId2denseId(uint32_t detId) {
    const uint32_t subdet = getSubdet(detId);
    if (subdet == HcalBarrel)
      return detId2denseIdHB(detId);
    if (subdet == HcalEndcap)
      return detId2denseIdHE(detId);

    printf("invalid detId\n");
    return 0;
  }
}  // namespace

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class PFRecHitProducerKernelImpl1 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const PFRecHitHBHEParamsAlpakaESDataDevice::ConstView params,
                                  const PFRecHitHBHETopologyAlpakaESDataDevice::ConstView topology,
                                  const CaloRecHitDeviceCollection::ConstView recHits,
                                  int32_t num_recHits,
                                  PFRecHitDeviceCollection::View pfRecHits,
                                  uint32_t* __restrict__ denseId2pfRecHit,
                                  uint32_t* __restrict__ num_pfRecHits) const {
      const int32_t num_blocks = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];

      // HCAL barrel and endcap energy thresholds
      const float* thresholdE_HB = params.energyThresholds();      // length 4
      const float* thresholdE_HE = params.energyThresholds() + 4;  // length 7

      // Srided loop over CaloRecHits
      for (int32_t i : elements_with_stride(acc, num_recHits)) {
        const uint32_t detId = recHits[i].detId();
        const uint32_t subdet = getSubdet(detId);
        const uint32_t depth = getDepth(detId);
        const float energy = recHits[i].energy();

        float threshold = 9999.;
        if (subdet == HcalBarrel) {
          threshold = thresholdE_HB[depth - 1];
        } else if (subdet == HcalEndcap) {
          threshold = thresholdE_HE[depth - 1];
        } else {
          printf("Rechit with detId %u has invalid subdetector %u!\n", detId, subdet);
        }

        // CaloRecHits above energy threshold become PFRecHits
        if (energy >= threshold) {
          // Use the appropriate synchronisation for the kernel layout
          const uint32_t j = (num_blocks == 1) ? alpaka::atomicAdd(acc, num_pfRecHits, 1u, alpaka::hierarchy::Blocks{})
                                               : alpaka::atomicAdd(acc, num_pfRecHits, 1u, alpaka::hierarchy::Grids{});
          pfRecHits[j].detId() = detId;
          pfRecHits[j].energy() = recHits[i].energy();
          pfRecHits[j].time() = recHits[i].time();
          pfRecHits[j].depth() = depth;

          if (subdet == HcalBarrel)
            pfRecHits[j].layer() = PFLayer::HCAL_BARREL1;
          else if (subdet == HcalEndcap)
            pfRecHits[j].layer() = PFLayer::HCAL_ENDCAP;
          else
            pfRecHits[j].layer() = PFLayer::NONE;

          const uint32_t denseId_raw = detId2denseId(detId);
          if (topology.denseId_min() <= denseId_raw && denseId_raw <= topology.denseId_max())
            denseId2pfRecHit[denseId_raw - topology.denseId_min()] = j;
          else
            printf("detId %u leads to invalid denseId %u. Allowed range [%u,%u]\n",
                   detId,
                   denseId_raw,
                   topology.denseId_min(),
                   topology.denseId_max());
        }
      }
    }
  };

  class PFRecHitProducerKernelImpl2 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const PFRecHitHBHETopologyAlpakaESDataDevice::ConstView topology,
                                  const CaloRecHitDeviceCollection::ConstView recHits,
                                  int32_t num_recHits,
                                  PFRecHitDeviceCollection::View pfRecHits,
                                  const uint32_t* __restrict__ denseId2pfRecHit,
                                  uint32_t* __restrict__ num_pfRecHits) const {
      // First thread updates size field pfRecHits SoA
      if (const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]; thread == 0)
        pfRecHits.size() = *num_pfRecHits;

      // Assign position information and associate neighbours
      for (int32_t i : elements_with_stride(acc, *num_pfRecHits)) {
        const uint32_t denseId = detId2denseId(pfRecHits[i].detId()) - topology.denseId_min();

        pfRecHits[i].x() = topology[denseId].positionX();
        pfRecHits[i].y() = topology[denseId].positionY();
        pfRecHits[i].z() = topology[denseId].positionZ();

        for (uint32_t n = 0; n < 8; n++) {
          pfRecHits[i].neighbours()(n) = -1;
          const uint32_t denseId_neighbour = topology[denseId].neighbours()(n);
          if (denseId_neighbour != 0xffffffff) {
            const uint32_t pfRecHit_neighbour = denseId2pfRecHit[denseId_neighbour];

            if (pfRecHit_neighbour != 0xffffffff)
              pfRecHits[i].neighbours()(n) = (int32_t)pfRecHit_neighbour;
          }
        }
      }
    }
  };

  PFRecHitProducerKernel::PFRecHitProducerKernel(cms::alpakatools::device_buffer<Device, uint32_t[]>&& buffer1,
                                                 cms::alpakatools::device_buffer<Device, uint32_t>&& buffer2)
      : denseId2pfRecHit(std::move(buffer1)), num_pfRecHits(std::move(buffer2)) {}

  PFRecHitProducerKernel PFRecHitProducerKernel::Construct(Queue& queue) {
    // TODO copy these from device or use ESProduct on host
    const uint32_t denseId_min = 0, denseId_max = 25000;
    //alpaka::memcpy(queue, &denseId_min, topology.view().denseId_min(), 1);
    //auto event = alpaka::Event<Queue>(device);
    //alpaka::wait(event);
    //printf("denseId range: %u %u\n", denseId_min, denseId_max);

    return PFRecHitProducerKernel{
        cms::alpakatools::make_device_buffer<uint32_t[]>(queue, denseId_max - denseId_min + 1),
        cms::alpakatools::make_device_buffer<uint32_t>(queue)};
  }

  void PFRecHitProducerKernel::execute(const Device& device,
                                       Queue& queue,
                                       const PFRecHitHBHEParamsAlpakaESDataDevice& params,
                                       const PFRecHitHBHETopologyAlpakaESDataDevice& topology,
                                       const CaloRecHitDeviceCollection& recHits,
                                       PFRecHitDeviceCollection& pfRecHits) {
    alpaka::memset(queue, denseId2pfRecHit, 0xff);  // Reset denseId -> pfRecHit index map
    alpaka::memset(queue, num_pfRecHits, 0x00);     // Reset global pfRecHit counter

    // Use only one block on the synchronous CPU backend, because there is no
    // performance gain in using multiple blocks, but there is a significant
    // penalty due to the more complex synchronisation.
    const uint32_t items = 64;
    const uint32_t groups =
        std::is_same_v<Device, alpaka::DevCpu> ? 1 : divide_up_by(recHits->metadata().size(), items);

    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(groups, items),
                        PFRecHitProducerKernelImpl1{},
                        params.view(),
                        topology.view(),
                        recHits.view(),
                        recHits->metadata().size(),
                        pfRecHits.view(),
                        denseId2pfRecHit.data(),
                        num_pfRecHits.data());

    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(groups, items),
                        PFRecHitProducerKernelImpl2{},
                        topology.view(),
                        recHits.view(),
                        recHits->metadata().size(),
                        pfRecHits.view(),
                        denseId2pfRecHit.data(),
                        num_pfRecHits.data());
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
