#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/plugins/alpaka/PFClusterProducerKernel.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/AlpakaPFCommon.h"

// The following comment block is required in using the ECL-CC algorithm for topological clustering

/*
  ECL-CC code: ECL-CC is a connected components graph algorithm. The CUDA
  implementation thereof is quite fast. It operates on graphs stored in
  binary CSR format.

  Copyright (c) 2017-2020, Texas State University. All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of Texas State University nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  Authors: Jayadharini Jaiganesh and Martin Burtscher

  URL: The latest version of this code is available at
  https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/.

  Publication: This work is described in detail in the following paper.
  Jayadharini Jaiganesh and Martin Burtscher. A High-Performance Connected
  Components Implementation for GPUs. Proceedings of the 2018 ACM International
  Symposium on High-Performance Parallel and Distributed Computing, pp. 92-104.
  June 2018.
*/

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  using PFClustering::common::PFLayer;

  constexpr const float PI_F = 3.141592654f;
  static const int threadsPerBlockForClustering = std::is_same_v<Device, alpaka::DevCpu> ? 32 : 512;
  static const int warpsize = 32;

  // Preparation of topo inputs. Initializing topoId, egdeIdx, nEdges, edgeList
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void prepareTopoInputs(const TAcc& acc,
                                       const unsigned int nRH,
                                       reco::PFRecHitHostCollection::ConstView pfRecHits,
                                       reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                       reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars,
                                       uint32_t* __restrict__ nSeeds) {
    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0 &&
        alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] == 0) {
      clusteringVars.nEdges() = nRH * 8;
      clusteringEdgeVars[nRH].pfrh_edgeIdx() = nRH * 8;
    }

    for (uint32_t i = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                          alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] +
                      alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
         i < nRH;
         i += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u] *
              alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]) {
      clusteringEdgeVars[i].pfrh_edgeIdx() = i * 8;
      clusteringVars[i].pfrh_topoId() = 0;
      for (int j = 0; j < 8; j++) {  // checking if neighbours exist and assigning neighbours as edges
        if (pfRecHits[i].neighbours()(j) == -1)
          clusteringEdgeVars[i * 8 + j].pfrh_edgeList() = i;
        else
          clusteringEdgeVars[i * 8 + j].pfrh_edgeList() = pfRecHits[i].neighbours()(j);
      }
    }

    return;
  }

  // Initial step of ECL-CC. Uses ID of first neighbour in edgeList with a smaller ID
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void ECLCC_init(const TAcc& acc,
                                const int nodes,
                                reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) {
    const int from = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] +
                     alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                         alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    const int incr = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                     alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

    for (int v = from; v < nodes; v += incr) {
      const int beg = clusteringEdgeVars[v].pfrh_edgeIdx();
      const int end = clusteringEdgeVars[v + 1].pfrh_edgeIdx();
      int m = v;
      int i = beg;
      while ((m == v) && (i < end)) {
        m = std::min(m, clusteringEdgeVars[i].pfrh_edgeList());
        i++;
      }
      clusteringVars[v].pfrh_topoId() = m;
    }

    if (from == 0) {
      clusteringVars.topL() = 0;
      clusteringVars.posL() = 0;
      clusteringVars.topH() = nodes - 1;
      clusteringVars.posH() = nodes - 1;
    }
  }

  /* intermediate pointer jumping */

  ALPAKA_FN_ACC int representative(const int idx, reco::ClusteringVarsDeviceCollection::View clusteringVars) {
    int curr = clusteringVars[idx].pfrh_topoId();
    if (curr != idx) {
      int next, prev = idx;
      while (curr > (next = clusteringVars[curr].pfrh_topoId())) {
        clusteringVars[prev].pfrh_topoId() = next;
        prev = curr;
        curr = next;
      }
    }
    return curr;
  }

  // First edge processing kernel of ECL-CC
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void ECLCC_compute1(const TAcc& acc,
                                    const int nodes,
                                    reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                    reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) {
    const int from = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] +
                     alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                         alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    const int incr = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                     alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    for (int v = from; v < nodes; v += incr) {
      const int vstat = clusteringVars[v].pfrh_topoId();
      if (v != vstat) {
        const int beg = clusteringEdgeVars[v].pfrh_edgeIdx();
        const int end = clusteringEdgeVars[v + 1].pfrh_edgeIdx();
        int deg = end - beg;
        if (deg > 16) {
          int idx;
          if (deg <= 352) {
            idx = alpaka::atomicAdd(acc, &clusteringVars.topL(), 1);
          } else {
            idx = alpaka::atomicAdd(acc, &clusteringVars.topH(), -1);
          }
          clusteringVars[idx].wl_d() = v;
        } else {
          int vstat = representative(v, clusteringVars);
          for (int i = beg; i < end; i++) {
            const int nli = clusteringEdgeVars[i].pfrh_edgeList();
            if (v > nli) {
              int ostat = representative(nli, clusteringVars);
              bool repeat;
              do {
                repeat = false;
                if (vstat != ostat) {
                  int ret;
                  if (vstat < ostat) {
                    if ((ret = alpaka::atomicCas(acc, &clusteringVars[ostat].pfrh_topoId(), ostat, vstat)) != ostat) {
                      ostat = ret;
                      repeat = true;
                    }
                  } else {
                    if ((ret = alpaka::atomicCas(acc, &clusteringVars[vstat].pfrh_topoId(), vstat, ostat)) != vstat) {
                      vstat = ret;
                      repeat = true;
                    }
                  }
                }
              } while (repeat);
            }
          }
        }
      }
    }
  }

  /* process medium-degree vertices at warp granularity */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void ECLCC_compute2(const TAcc& acc,
                                    const int nodes,
                                    reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                    reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) {
    const int lane = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] % warpsize;

    int32_t idx = 0;
    if (lane == 0)
      idx = alpaka::atomicAdd(acc, &clusteringVars.posL(), 1);
    idx = alpaka::warp::shfl(acc, idx, 0);
    while (idx < clusteringVars.topL()) {
      const int v = clusteringVars[idx].wl_d();
      int vstat = representative(v, clusteringVars);
      for (int i = clusteringEdgeVars[v].pfrh_edgeIdx() + lane; i < clusteringEdgeVars[v + 1].pfrh_edgeIdx(); i += warpsize) {
        const int nli = clusteringEdgeVars[i].pfrh_edgeList();
        if (v > nli) {
          int ostat = representative(nli, clusteringVars);
          bool repeat;
          do {
            repeat = false;
            if (vstat != ostat) {
              int ret;
              if (vstat < ostat) {
                if ((ret = alpaka::atomicCas(acc, &clusteringVars[ostat].pfrh_topoId(), ostat, vstat)) != ostat) {
                  ostat = ret;
                  repeat = true;
                }
              } else {
                if ((ret = alpaka::atomicCas(acc, &clusteringVars[vstat].pfrh_topoId(), vstat, ostat)) != vstat) {
                  vstat = ret;
                  repeat = true;
                }
              }
            }
          } while (repeat);
        }
      }
      if (lane == 0)
        idx = alpaka::atomicAdd(acc, &clusteringVars.posL(), 1);
      idx = alpaka::warp::shfl(acc, idx, 0);
    }
  }

  /* process high-degree vertices at block granularity */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void ECLCC_compute3(const TAcc& acc,
                                    const int nodes,
                                    reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                    reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) {
    int& vB = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
      const int idx = alpaka::atomicAdd(acc, &clusteringVars.posH(), -1);
      vB = (idx > clusteringVars.topH()) ? clusteringVars[idx].wl_d() : -1;
    }
    alpaka::syncBlockThreads(acc);
    while (vB >= 0) {
      const int v = vB;
      alpaka::syncBlockThreads(acc);
      int vstat = representative(v, clusteringVars);
      for (int i = clusteringEdgeVars[v].pfrh_edgeIdx() + alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
           i < clusteringEdgeVars[v + 1].pfrh_edgeIdx();
           i += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
        const int nli = clusteringEdgeVars[i].pfrh_edgeList();
        if (v > nli) {
          int ostat = representative(nli, clusteringVars);
          bool repeat;
          do {
            repeat = false;
            if (vstat != ostat) {
              int ret;
              if (vstat < ostat) {
                if ((ret = alpaka::atomicCas(acc, &clusteringVars[ostat].pfrh_topoId(), ostat, vstat)) != ostat) {
                  ostat = ret;
                  repeat = true;
                }
              } else {
                if ((ret = alpaka::atomicCas(acc, &clusteringVars[vstat].pfrh_topoId(), vstat, ostat)) != vstat) {
                  vstat = ret;
                  repeat = true;
                }
              }
            }
          } while (repeat);
        }
        alpaka::syncBlockThreads(acc);
      }
      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        const int idx = alpaka::atomicAdd(acc, &clusteringVars.posH(), -1);
        vB = (idx > clusteringVars.topH()) ? clusteringVars[idx].wl_d() : -1;
      }
      alpaka::syncBlockThreads(acc);
    }
  }

  /* link all vertices to sink */

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void ECLCC_flatten(const TAcc& acc,
                                   const int nodes,
                                   reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                   reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) {
    const int from = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] +
                     alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                         alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    const int incr = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                     alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];

    for (int v = from; v < nodes; v += incr) {
      int next, vstat = clusteringVars[v].pfrh_topoId();
      const int old = vstat;
      while (vstat > (next = clusteringVars[vstat].pfrh_topoId())) {
        vstat = next;
      }
      if (old != vstat)
        clusteringVars[v].pfrh_topoId() = vstat;
    }
  }

  // ECL-CC ends

  // Contraction in a single block
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void topoClusterContraction(const TAcc& acc,
                                            const int size,
                                            reco::PFRecHitHostCollection::ConstView pfRecHits,
                                            reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                            reco::PFClusterDeviceCollection::View clusterView,
                                            uint32_t* __restrict__ nSeeds) {
    int& totalSeedOffset = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& totalSeedFracOffset = alpaka::declareSharedVar<int, __COUNTER__>(acc);

    // rhCount, topoRHCount, topoSeedCount initialized earlier
    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
      clusteringVars.nTopos() = 0;
      clusteringVars.nRHFracs() = 0;
      totalSeedOffset = 0;
      totalSeedFracOffset = 0;
      clusteringVars.pcrhFracSize() = 0;
    }

    alpaka::syncBlockThreads(acc);

    // Now determine the number of seeds and rechits in each topo cluster [topoRHCount, topoSeedCount]
    // Also get the list of topoIds (smallest rhIdx of each topo cluser)
    for (int rhIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; rhIdx < size;
         rhIdx += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
      clusteringVars[rhIdx].rhIdxToSeedIdx() = -1;
      int topoId = clusteringVars[rhIdx].pfrh_topoId();
      if (topoId > -1) {
        // Valid topo cluster
        alpaka::atomicAdd(acc, &clusteringVars[topoId].topoRHCount(), 1);
        // Valid topoId not counted yet
        if (topoId == rhIdx) {  // For every topo cluster, there is one rechit that meets this condition.
          int topoIdx = alpaka::atomicAdd(acc, &clusteringVars.nTopos(), 1);
          clusteringVars[topoIdx].topoIds() = topoId;  // topoId: the smallest index of rechits that belong to a topo cluster.
        }
        // This is a cluster seed
        if (clusteringVars[rhIdx].pfrh_isSeed()) {  // # of seeds in this topo cluster
          alpaka::atomicAdd(acc, &clusteringVars[topoId].topoSeedCount(), 1);
        }
      }
    }

    alpaka::syncBlockThreads(acc);

    // Determine offsets for topo ID seed array [topoSeedOffsets]
    for (int topoId = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; topoId < size;
         topoId += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
      if (clusteringVars[topoId].topoSeedCount() > 0) {
        // This is a valid topo ID
        int offset = alpaka::atomicAdd(acc, &totalSeedOffset, clusteringVars[topoId].topoSeedCount());
        clusteringVars[topoId].topoSeedOffsets() = offset;
      }
    }
    alpaka::syncBlockThreads(acc);

    // Fill arrays of rechit indicies for each seed [topoSeedList] and rhIdx->seedIdx conversion for each seed [rhIdxToSeedIdx]
    // Also fill pfc_seedRHIdx, pfc_topoId, pfc_depth
    for (int rhIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; rhIdx < size;
         rhIdx += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
      int topoId = clusteringVars[rhIdx].pfrh_topoId();
      if (clusteringVars[rhIdx].pfrh_isSeed()) {
        // Valid topo cluster and this rhIdx corresponds to a seed
        int k = alpaka::atomicAdd(acc, &clusteringVars[topoId].rhCount(), 1);
        int seedIdx = clusteringVars[topoId].topoSeedOffsets() + k;
        if ((unsigned int)seedIdx >= *nSeeds)
          printf("Warning(contraction) %8d > %8d should not happen, check topoId: %d has %d rh\n",
                 seedIdx,
                 *nSeeds,
                 topoId,
                 k);
        clusteringVars[seedIdx].topoSeedList() = rhIdx;
        clusteringVars[rhIdx].rhIdxToSeedIdx() = seedIdx;
        clusterView[seedIdx].pfc_topoId() = topoId;
        clusterView[seedIdx].pfc_seedRHIdx() = rhIdx;
        clusterView[seedIdx].pfc_depth() = pfRecHits[rhIdx].depth();
      }
    }

    alpaka::syncBlockThreads(acc);

    // Determine seed offsets for rechit fraction array
    for (int rhIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; rhIdx < size;
         rhIdx += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
      clusteringVars[rhIdx].rhCount() = 1;  // Reset this counter array

      int topoId = clusteringVars[rhIdx].pfrh_topoId();
      if (clusteringVars[rhIdx].pfrh_isSeed() && topoId > -1) {
        // Allot the total number of rechits for this topo cluster for rh fractions
        int offset = alpaka::atomicAdd(acc, &totalSeedFracOffset, clusteringVars[topoId].topoRHCount());

        // Add offset for this PF cluster seed
        clusteringVars[rhIdx].seedFracOffsets() = offset;

        // Store recHitFraction offset & size information for each seed
        clusterView[clusteringVars[rhIdx].rhIdxToSeedIdx()].pfc_rhfracOffset() = clusteringVars[rhIdx].seedFracOffsets();
        clusterView[clusteringVars[rhIdx].rhIdxToSeedIdx()].pfc_rhfracSize() =
            clusteringVars[topoId].topoRHCount() - clusteringVars[topoId].topoSeedCount() + 1;
      }
    }

    alpaka::syncBlockThreads(acc);

    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
      clusteringVars.pcrhFracSize() = totalSeedFracOffset;
      clusteringVars.nRHFracs() = totalSeedFracOffset;
      clusterView.nRHFracs() = totalSeedFracOffset;
      clusterView.nSeeds() = *nSeeds;
      clusterView.nTopos() = clusteringVars.nTopos();

      for (int i = 0; i < size; i++) {
        clusterView[i].topoRHCount() = clusteringVars[i].topoRHCount();
      }
      if (clusteringVars.pcrhFracSize() > 200000)  // Warning in case the fraction is too large
        printf("At the end of topoClusterContraction, found large *pcrhFracSize = %d\n", clusteringVars.pcrhFracSize());
    }

    alpaka::syncBlockThreads(acc);
  }

  // Prefill the rechit index for all PFCluster fractions
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void fillRhfIndex(const TAcc& acc,
                                  size_t nRH,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::PFRecHitFractionDeviceCollection::View fracView) {
    unsigned int i = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] +
                     alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                         alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];  // i is the seed index
    unsigned int j = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1u] +
                     alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[1u] *
                         alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[1u];  // j is NOT a seed

    if (i < nRH && j < nRH) {
      int topoId = clusteringVars[i].pfrh_topoId();
      if (topoId == clusteringVars[j].pfrh_topoId() && topoId > -1 && clusteringVars[i].pfrh_isSeed()) {
        if (!clusteringVars[j].pfrh_isSeed()) {  // NOT a seed
          int k = alpaka::atomicAdd(
              acc, &clusteringVars[i].rhCount(), 1);  // Increment the number of rechit fractions for this seed
          fracView[clusteringVars[i].seedFracOffsets() + k].pcrh_pfrhIdx() = j;
          fracView[clusteringVars[i].seedFracOffsets() + k].pcrh_pfcIdx() = clusteringVars[i].rhIdxToSeedIdx();
        } else if (i == j) {  // i==j is a seed rechit index
          fracView[clusteringVars[i].seedFracOffsets()].pcrh_pfrhIdx() = j;
          fracView[clusteringVars[i].seedFracOffsets()].pcrh_frac() = 1;
          fracView[clusteringVars[i].seedFracOffsets()].pcrh_pfcIdx() = clusteringVars[i].rhIdxToSeedIdx();
        }
      }
    }
  }

  // Serial code to prefill the rechit index for all PFCluster fractions using nested for loop
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void fillRhfIndexSerial(const TAcc& acc,
                                        size_t nRH,
                                        reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                        reco::PFRecHitFractionDeviceCollection::View fracView) {
    for (unsigned int i = 0; i < nRH; i++) {
      for (unsigned int j = 0; j < nRH; j++) {
        if (i < nRH && j < nRH) {
          int topoId = clusteringVars[i].pfrh_topoId();
          if (topoId == clusteringVars[j].pfrh_topoId() && topoId > -1 && clusteringVars[i].pfrh_isSeed()) {
            if (!clusteringVars[j].pfrh_isSeed()) {  // NOT a seed
              int k = alpaka::atomicAdd(
                  acc, &clusteringVars[i].rhCount(), 1);  // Increment the number of rechit fractions for this seed
              fracView[clusteringVars[i].seedFracOffsets() + k].pcrh_pfrhIdx() = j;
              fracView[clusteringVars[i].seedFracOffsets() + k].pcrh_pfcIdx() = clusteringVars[i].rhIdxToSeedIdx();
            } else if (i == j) {  // i==j is a seed rechit index
              fracView[clusteringVars[i].seedFracOffsets()].pcrh_pfrhIdx() = j;
              fracView[clusteringVars[i].seedFracOffsets()].pcrh_frac() = 1;
              fracView[clusteringVars[i].seedFracOffsets()].pcrh_pfcIdx() = clusteringVars[i].rhIdxToSeedIdx();
            }
          }
        }
      }
    }
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float timeResolution2Endcap(
      reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams, const float energy) {
    float res2 = 10000.;

    if (energy <= 0.)
      return res2;
    else if (energy < pfClusParams.endcapTimeResConsts_threshLowE()) {
      if (pfClusParams.endcapTimeResConsts_corrTermLowE() > 0.) {  // different parametrisation
        const float res = pfClusParams.endcapTimeResConsts_noiseTermLowE() / energy +
                          pfClusParams.endcapTimeResConsts_corrTermLowE() / (energy * energy);
        res2 = res * res;
      } else {
        const float noiseDivE = pfClusParams.endcapTimeResConsts_noiseTermLowE() / energy;
        res2 = noiseDivE * noiseDivE + pfClusParams.endcapTimeResConsts_constantTermLowE2();
      }
    } else if (energy < pfClusParams.endcapTimeResConsts_threshHighE()) {
      const float noiseDivE = pfClusParams.endcapTimeResConsts_noiseTerm() / energy;
      res2 = noiseDivE * noiseDivE + pfClusParams.endcapTimeResConsts_constantTerm2();
    } else  // if (energy >=threshHighE_)
      res2 = pfClusParams.endcapTimeResConsts_resHighE2();

    if (res2 > 10000.)
      return 10000.;
    return res2;
  }

  ALPAKA_FN_ACC ALPAKA_FN_INLINE float timeResolution2Barrel(
      reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams, const float energy) {
    float res2 = 10000.;

    if (energy <= 0.)
      return res2;
    else if (energy < pfClusParams.barrelTimeResConsts_threshLowE()) {
      if (pfClusParams.barrelTimeResConsts_corrTermLowE() > 0.) {  // different parametrisation
        const float res = pfClusParams.barrelTimeResConsts_noiseTermLowE() / energy +
                          pfClusParams.barrelTimeResConsts_corrTermLowE() / (energy * energy);
        res2 = res * res;
      } else {
        const float noiseDivE = pfClusParams.barrelTimeResConsts_noiseTermLowE() / energy;
        res2 = noiseDivE * noiseDivE + pfClusParams.barrelTimeResConsts_constantTermLowE2();
      }
    } else if (energy < pfClusParams.barrelTimeResConsts_threshHighE()) {
      const float noiseDivE = pfClusParams.barrelTimeResConsts_noiseTerm() / energy;
      res2 = noiseDivE * noiseDivE + pfClusParams.barrelTimeResConsts_constantTerm2();
    } else  // if (energy >=threshHighE_)
      res2 = pfClusParams.barrelTimeResConsts_resHighE2();

    if (res2 > 10000.)
      return 10000.;
    return res2;
  }

  // Calculation of dR2 for Clustering
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float dR2(Position4 pos1, Position4 pos2) {
    float mag1 = sqrtf(pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z);
    float cosTheta1 = mag1 > 0.0 ? pos1.z / mag1 : 1.0;
    float eta1 = 0.5 * logf((1.0 + cosTheta1) / (1.0 - cosTheta1));
    float phi1 = atan2f(pos1.y, pos1.x);

    float mag2 = sqrtf(pos2.x * pos2.x + pos2.y * pos2.y + pos2.z * pos2.z);
    float cosTheta2 = mag2 > 0.0 ? pos2.z / mag2 : 1.0;
    float eta2 = 0.5 * logf((1.0 + cosTheta2) / (1.0 - cosTheta2));
    float phi2 = atan2f(pos2.y, pos2.x);

    float deta = eta2 - eta1;
    float dphi = fabsf(fabsf(phi2 - phi1) - PI_F) - PI_F;
    return (deta * deta + dphi * dphi);
  }

  // Helper function used in atomicMaxF
  ALPAKA_FN_ACC ALPAKA_FN_INLINE int floatAsInt(float fval) {
    union {
      float fval;
      int ival;
    } u;

    u.fval = fval;

    return u.ival;
  }

  // Helper function used in atomicMaxF
  ALPAKA_FN_ACC ALPAKA_FN_INLINE float intAsFloat(int ival) {
    union {
      int ival;
      float fval;
    } u;

    u.ival = ival;

    return u.fval;
  }

  // Atomic Max for Floats
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE static float atomicMaxF(const TAcc& acc, float* address, float val) {
    int ret = floatAsInt(*address);
    while (val > intAsFloat(ret)) {
      int old = ret;
      if ((ret = alpaka::atomicCas(acc, (int*)address, old, floatAsInt(val))) == old)
        break;
    }
    return intAsFloat(ret);
  }

  // Get index of seed
  ALPAKA_FN_ACC auto dev_getSeedRhIdx(int* seeds, int seedNum) { return seeds[seedNum]; }

  // Get index of rechit fraction
  ALPAKA_FN_ACC auto dev_getRhFracIdx(int* rechits, int rhNum) {
    if (rhNum <= 0) {
      printf("Invalid rhNum (%d) for get RhFracIdx!\n", rhNum);
    }
    return rechits[rhNum - 1];
  }

  // Get rechit fraction of a given rechit for a given seed
  ALPAKA_FN_ACC auto dev_getRhFrac(reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                   int topoSeedBegin,
                                   reco::PFRecHitFractionDeviceCollection::View fracView,
                                   int seedNum,
                                   int rhNum) {
    int seedIdx = clusteringVars[topoSeedBegin + seedNum].topoSeedList();
    return fracView[clusteringVars[seedIdx].seedFracOffsets() + rhNum].pcrh_frac();
  }

  // Cluster position calculation
  ALPAKA_FN_ACC auto dev_computeClusterPos(reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                           Position4& pos4,
                                           float frac,
                                           int rhInd,
                                           bool isDebug,
                                           reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                           float rhENormInv) {
    Position4 rechitPos = Position4{pfRecHits[rhInd].x(), pfRecHits[rhInd].y(), pfRecHits[rhInd].z(), 1.0};
    const auto rh_energy = pfRecHits[rhInd].energy() * frac;
    const auto norm = (frac < pfClusParams.minFracInCalc() ? 0.0f : std::max(0.0f, logf(rh_energy * rhENormInv)));
    if (isDebug)
      printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n",
             rhInd,
             norm,
             frac,
             rh_energy,
             rechitPos.x,
             rechitPos.y,
             rechitPos.z);

    pos4.x += rechitPos.x * norm;
    pos4.y += rechitPos.y * norm;
    pos4.z += rechitPos.z * norm;
    pos4.w += norm;  //  position_norm
  }

  // Seeding using local energy maxima
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void seedingTopoThreshKernel_HCAL(const TAcc& acc,
                                                  reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                                  reco::PFRecHitHostCollection::ConstView pfRecHits,
                                                  reco::PFClusterDeviceCollection::View clusterView,
                                                  reco::PFRecHitFractionDeviceCollection::View fracView,
                                                  size_t size,
                                                  uint32_t* __restrict__ nSeeds) {
    unsigned int i = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] +
                     alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                         alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    clusterView.size() = size;

    if (i < size) {
      // Initialize arrays
      clusteringVars[i].pfrh_topoId() = i;
      clusteringVars[i].pfrh_isSeed() = 0;
      clusteringVars[i].rhCount() = 0;
      clusteringVars[i].topoSeedCount() = 0;
      clusteringVars[i].topoRHCount() = 0;
      clusterView[i].topoRHCount() = 0;
      clusteringVars[i].seedFracOffsets() = -1;
      clusteringVars[i].topoSeedOffsets() = -1;
      clusteringVars[i].topoSeedList() = -1;
      clusteringVars[i].pfc_iter() = -1;
      clusterView[i].pfc_seedRHIdx() = -1;

      int layer = pfRecHits[i].layer();
      int depthOffset = pfRecHits[i].depth() - 1;
      float energy = pfRecHits[i].energy();
      Position3 pos = Position3{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z()};

      // cmssdt.cern.ch/lxr/source/DataFormats/ParticleFlowReco/interface/PFRecHit.h#0108
      float pt2 = energy * energy * (pos.x * pos.x + pos.y * pos.y) / (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

      // Seeding threshold test
      if ((layer == PFLayer::HCAL_BARREL1 && energy > pfClusParams.seedEThresholdEB_vec()[depthOffset] &&
           pt2 > pfClusParams.seedPt2ThresholdEB()) ||
          (layer == PFLayer::HCAL_ENDCAP && energy > pfClusParams.seedEThresholdEE_vec()[depthOffset] &&
           pt2 > pfClusParams.seedPt2ThresholdEE())) {
        clusteringVars[i].pfrh_isSeed() = 1;
        for (int k = 0; k < 4; k++) {  // Does this seed candidate have a higher energy than four neighbours
          if (pfRecHits[i].neighbours()(k) < 0)
            continue;
          if (energy < pfRecHits[pfRecHits[i].neighbours()(k)].energy()) {
            clusteringVars[i].pfrh_isSeed() = 0;
            break;
          }
        }
        if (clusteringVars[i].pfrh_isSeed())
          alpaka::atomicAdd(acc, nSeeds, 1u);
      } else {
        clusteringVars[i].pfrh_isSeed() = 0;
      }

      // Topo clustering threshold test
      if ((layer == PFLayer::HCAL_ENDCAP && energy > pfClusParams.topoEThresholdEE_vec()[depthOffset]) ||
          (layer == PFLayer::HCAL_BARREL1 && energy > pfClusParams.topoEThresholdEB_vec()[depthOffset])) {
        clusteringVars[i].pfrh_passTopoThresh() = true;
      } else {
        clusteringVars[i].pfrh_passTopoThresh() = false;
        clusteringVars[i].pfrh_topoId() = -1;
      }
    }
  }

  // Processing single seed clusters
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void dev_hcalFastCluster_optimizedSimple(const TAcc& acc,
                                                         reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                                         int topoId,   // from selection
                                                         int nRHTopo,  // from selection
                                                         reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                                         reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                                         reco::PFClusterDeviceCollection::View clusterView,
                                                         reco::PFRecHitFractionDeviceCollection::View fracView) {
    int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];  // thread index is rechit number
    // Declaration of shared variables
    int& i = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& nRHOther = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    unsigned int& iter = alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
    float& tol = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& clusterEnergy = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& rhENormInv = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& seedEnergy = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    Position4& clusterPos = alpaka::declareSharedVar<Position4, __COUNTER__>(acc);
    Position4& prevClusterPos = alpaka::declareSharedVar<Position4, __COUNTER__>(acc);
    Position4& seedPos = alpaka::declareSharedVar<Position4, __COUNTER__>(acc);
    bool& notDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    bool& debug = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    if (tid == 0) {
      i = clusteringVars[clusteringVars[topoId].topoSeedOffsets()].topoSeedList();  // i is the seed rechit index
      nRHOther = nRHTopo - 1;                                       // number of non-seed rechits
      seedPos = Position4{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z(), 1.};
      clusterPos = seedPos;  // Initial cluster position is just the seed
      prevClusterPos = seedPos;
      seedEnergy = pfRecHits[i].energy();
      clusterEnergy = seedEnergy;
      tol = pfClusParams.stoppingTolerance();  // stopping tolerance * tolerance scaling

      if (pfRecHits[i].layer() == PFLayer::HCAL_BARREL1)
        rhENormInv = pfClusParams.recHitEnergyNormInvEB_vec()[pfRecHits[i].depth() - 1];
      else if (pfRecHits[i].layer() == PFLayer::HCAL_ENDCAP)
        rhENormInv = pfClusParams.recHitEnergyNormInvEE_vec()[pfRecHits[i].depth() - 1];
      else {
        rhENormInv = 0.;
        printf("Rechit %d has invalid layer %d!\n", i, pfRecHits[i].layer());
      }

      iter = 0;
      notDone = true;
      debug = false;
    }
    alpaka::syncBlockThreads(acc);

    int j = -1;  // j is the rechit index for this thread
    int rhFracOffset = -1;
    Position4 rhPos;
    float rhEnergy = -1., rhPosNorm = -1.;

    if (tid < nRHOther) {
      rhFracOffset = clusteringVars[i].seedFracOffsets() + tid + 1;  // Offset for this rechit in pcrhfrac, pcrhfracidx arrays
      j = fracView[rhFracOffset].pcrh_pfrhIdx();             // rechit index for this thread
      rhPos = Position4{pfRecHits[j].x(), pfRecHits[j].y(), pfRecHits[j].z(), 1.};
      rhEnergy = pfRecHits[j].energy();
      rhPosNorm = fmaxf(0., logf(rhEnergy * rhENormInv));
    }
    alpaka::syncBlockThreads(acc);

    do {
      if (debug && tid == 0) {
        printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }
      float dist2 = -1., d2 = -1., fraction = -1.;
      if (tid < nRHOther) {
        // Rechit distance calculation
        dist2 = (clusterPos.x - rhPos.x) * (clusterPos.x - rhPos.x) +
                (clusterPos.y - rhPos.y) * (clusterPos.y - rhPos.y) +
                (clusterPos.z - rhPos.z) * (clusterPos.z - rhPos.z);

        d2 = dist2 / pfClusParams.showerSigma2();
        fraction = clusterEnergy * rhENormInv * expf(-0.5 * d2);

        // For single seed clusters, rechit fraction is either 1 (100%) or -1 (not included)
        if (fraction > pfClusParams.minFracTot() && d2 < 100.)
          fraction = 1.;
        else
          fraction = -1.;
        fracView[rhFracOffset].pcrh_frac() = fraction;
      }
      alpaka::syncBlockThreads(acc);

      if (debug && tid == 0)
        printf("Computing cluster position for topoId %d\n", topoId);

      if (tid == 0) {
        // Reset cluster position and energy
        clusterPos = seedPos;
        clusterEnergy = seedEnergy;
      }
      alpaka::syncBlockThreads(acc);

      // Recalculate cluster position and energy
      if (fraction > -0.5) {
        alpaka::atomicAdd(acc, &clusterEnergy, rhEnergy, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &clusterPos.x, rhPos.x * rhPosNorm, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &clusterPos.y, rhPos.y * rhPosNorm, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &clusterPos.z, rhPos.z * rhPosNorm, alpaka::hierarchy::Blocks{});
        alpaka::atomicAdd(acc, &clusterPos.w, rhPosNorm, alpaka::hierarchy::Blocks{});  // position_norm
      }
      alpaka::syncBlockThreads(acc);

      if (tid == 0) {
        // Normalize the seed postiion
        if (clusterPos.w >= pfClusParams.minAllowedNormalization()) {
          // Divide by position norm
          clusterPos.x /= clusterPos.w;
          clusterPos.y /= clusterPos.w;
          clusterPos.z /= clusterPos.w;

          if (debug)
            printf("\tPF cluster (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   i,
                   clusterEnergy,
                   clusterPos.x,
                   clusterPos.y,
                   clusterPos.z);
        } else {
          if (debug)
            printf("\tPF cluster (seed %d) position norm (%f) less than minimum (%f)\n",
                   i,
                   clusterPos.w,
                   pfClusParams.minAllowedNormalization());
          clusterPos.x = 0.;
          clusterPos.y = 0.;
          clusterPos.z = 0.;
        }
        float diff2 = dR2(prevClusterPos, clusterPos);
        if (debug)
          printf("\tPF cluster (seed %d) has diff2 = %f\n", i, diff2);
        prevClusterPos = clusterPos;  // Save clusterPos

        float diff = sqrtf(diff2);
        iter++;
        notDone = (diff > tol) && (iter < pfClusParams.maxIterations());
        if (debug) {
          if (diff > tol)
            printf("\tTopoId %d has diff = %f greater than tolerance %f (continuing)\n", topoId, diff, tol);
          else if (debug)
            printf("\tTopoId %d has diff = %f LESS than tolerance %f (terminating!)\n", topoId, diff, tol);
        }
      }
      alpaka::syncBlockThreads(acc);
    } while (notDone);
    if (tid == 0) {  // Cluster is finalized, assign cluster information to te SoA
      int rhIdx = clusteringVars[clusteringVars[topoId].topoSeedOffsets()].topoSeedList();  // i is the seed rechit index
      int seedIdx = clusteringVars[rhIdx].rhIdxToSeedIdx();
      clusteringVars[topoId].pfc_iter() = iter;
      clusterView[seedIdx].pfc_energy() = clusterEnergy;
      clusterView[seedIdx].pfc_x() = clusterPos.x;
      clusterView[seedIdx].pfc_y() = clusterPos.y;
      clusterView[seedIdx].pfc_z() = clusterPos.z;
    }
  }

  // Processing clusters up to 100 seeds and 512 non-seed rechits using shared memory accesses
  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void dev_hcalFastCluster_optimizedComplex(
      const TAcc& acc,
      reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
      int topoId,   // from selection
      int nSeeds,   // from selection
      int nRHTopo,  // from selection
      reco::PFRecHitDeviceCollection::ConstView pfRecHits,
      reco::ClusteringVarsDeviceCollection::View clusteringVars,
      reco::PFClusterDeviceCollection::View clusterView,
      reco::PFRecHitFractionDeviceCollection::View fracView) {
    int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(
        acc)[0u];  // Thread index corresponds to a single rechit of the topo cluster

    int& nRHNotSeed = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& topoSeedBegin = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& gridStride = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& iter = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    float& tol = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& diff2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& rhENormInv = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    bool& notDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    bool& debug = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    auto& clusterPos = alpaka::declareSharedVar<Position4[100], __COUNTER__>(acc);
    auto& prevClusterPos = alpaka::declareSharedVar<Position4[100], __COUNTER__>(acc);
    auto& clusterEnergy = alpaka::declareSharedVar<float[100], __COUNTER__>(acc);
    auto& rhFracSum = alpaka::declareSharedVar<float[threadsPerBlockForClustering], __COUNTER__>(acc);
    auto& seeds = alpaka::declareSharedVar<int[100], __COUNTER__>(acc);
    auto& rechits = alpaka::declareSharedVar<int[threadsPerBlockForClustering], __COUNTER__>(acc);

    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
      nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
      topoSeedBegin = clusteringVars[topoId].topoSeedOffsets();
      tol = pfClusParams.stoppingTolerance() *
            powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
      gridStride = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      iter = 0;
      notDone = true;
      debug = false;

      int i = clusteringVars[topoSeedBegin].topoSeedList();
      if (pfRecHits[i].layer() == PFLayer::HCAL_BARREL1)
        rhENormInv = pfClusParams.recHitEnergyNormInvEB_vec()[pfRecHits[i].depth() - 1];
      else if (pfRecHits[i].layer() == PFLayer::HCAL_ENDCAP)
        rhENormInv = pfClusParams.recHitEnergyNormInvEE_vec()[pfRecHits[i].depth() - 1];
      else
        printf("Rechit %d has invalid layer %d!\n", i, pfRecHits[i].layer());
    }
    alpaka::syncBlockThreads(acc);

    if (tid < nSeeds)
      seeds[tid] = clusteringVars[topoSeedBegin + tid].topoSeedList();
    if (tid < nRHNotSeed - 1)
      rechits[tid] = fracView[clusteringVars[clusteringVars[topoSeedBegin].topoSeedList()].seedFracOffsets() + tid + 1].pcrh_pfrhIdx();

    if (debug) {
      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        printf("\n===========================================================================================\n");
        printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
        for (int s = 0; s < nSeeds; s++) {
          if (s != 0)
            printf(", ");
          printf("%d", dev_getSeedRhIdx(seeds, s));
        }
        if (nRHTopo == nSeeds) {
          printf(")\n\n");
        } else {
          printf(") and other rechits (");
          for (int r = 1; r < nRHNotSeed; r++) {
            if (r != 1)
              printf(", ");
            printf("%d", dev_getRhFracIdx(rechits, r));
          }
          printf(")\n\n");
        }
      }
      alpaka::syncBlockThreads(acc);
    }

    // Set initial cluster position (energy) to seed rechit position (energy)
    //for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
    if (tid < nSeeds) {
      int i = dev_getSeedRhIdx(seeds, tid);
      clusterPos[tid] = Position4{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z(), 1.0};
      prevClusterPos[tid] = clusterPos[tid];
      clusterEnergy[tid] = pfRecHits[i].energy();
      for (int r = 0; r < (nRHNotSeed - 1); r++) {
        fracView[clusteringVars[i].seedFracOffsets() + r + 1].pcrh_pfrhIdx() = rechits[r];
        fracView[clusteringVars[i].seedFracOffsets() + r + 1].pcrh_frac() = -1.;
      }
    }
    alpaka::syncBlockThreads(acc);

    int rhThreadIdx = -1;
    Position4 rhThreadPos;
    if (tid < (nRHNotSeed - 1)) {
      rhThreadIdx = rechits[tid];  // Index when thread represents rechit
      rhThreadPos = Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
    }

    // Neighbors when threadIdx represents seed
    int seedThreadIdx = -1;
    Neighbours4 seedNeighbors = Neighbours4{-9, -9, -9, -9};
    float seedEnergy = -1.;
    Position4 seedInitClusterPos = Position4{0., 0., 0., 0.};
    if (tid < nSeeds) {
      if (debug)
        printf("tid: %d\n", tid);
      seedThreadIdx = dev_getSeedRhIdx(seeds, tid);
      seedNeighbors = Neighbours4{pfRecHits[seedThreadIdx].neighbours()(0),
                                pfRecHits[seedThreadIdx].neighbours()(1),
                                pfRecHits[seedThreadIdx].neighbours()(2),
                                pfRecHits[seedThreadIdx].neighbours()(3)};
      seedEnergy = pfRecHits[seedThreadIdx].energy();

      // Compute initial cluster position shift for seed
      dev_computeClusterPos(pfClusParams, seedInitClusterPos, 1., seedThreadIdx, debug, pfRecHits, rhENormInv);
    }

    do {
      if (debug && alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }

      // Reset rhFracSum
      rhFracSum[tid] = 0.;
      if (tid == 0)
        diff2 = -1;

      if (tid < (nRHNotSeed - 1)) {
        for (int s = 0; s < nSeeds; s++) {
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          rhFracSum[tid] += fraction;
        }
      }
      alpaka::syncBlockThreads(acc);

      if (tid < (nRHNotSeed - 1)) {
        for (int s = 0; s < nSeeds; s++) {
          int i = seeds[s];
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          if (rhFracSum[tid] > pfClusParams.minFracTot()) {
            float fracpct = fraction / rhFracSum[tid];
            if (fracpct > 0.9999 || (d2 < 85. && fracpct > pfClusParams.minFracToKeep())) {
              //if (iter == 0 && d2 > 80.)
              //fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = -2;
              //else
              fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = fracpct;
            } else {
              fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = -1;
            }
          } else {
            fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = -1;
          }
        }
      }
      alpaka::syncBlockThreads(acc);

      if (debug && tid == 0)
        printf("Computing cluster position for topoId %d\n", topoId);

      // Reset cluster position and energy
      if (tid < nSeeds) {
        clusterPos[tid] = seedInitClusterPos;
        clusterEnergy[tid] = seedEnergy;
        if (debug) {
          printf("Cluster %d (seed %d) has energy %f\tpos = (%f, %f, %f, %f)\n",
                 tid,
                 seeds[tid],
                 clusterEnergy[tid],
                 clusterPos[tid].x,
                 clusterPos[tid].y,
                 clusterPos[tid].z,
                 clusterPos[tid].w);
        }
      }
      alpaka::syncBlockThreads(acc);

      // Recalculate position
      if (tid < nSeeds) {
        for (int r = 0; r < nRHNotSeed - 1; r++) {
          int j = rechits[r];
          float frac = dev_getRhFrac(clusteringVars, topoSeedBegin, fracView, tid, r + 1);

          if (frac > -0.5) {
            clusterEnergy[tid] += frac * pfRecHits[j].energy();

            if (nSeeds == 1 || j == seedNeighbors.x || j == seedNeighbors.y || j == seedNeighbors.z ||
                j == seedNeighbors.w)
              dev_computeClusterPos(pfClusParams, clusterPos[tid], frac, j, debug, pfRecHits, rhENormInv);
          }
        }
      }
      alpaka::syncBlockThreads(acc);

      // Position normalization
      if (tid < nSeeds) {
        if (clusterPos[tid].w >= pfClusParams.minAllowedNormalization()) {
          // Divide by position norm
          clusterPos[tid].x /= clusterPos[tid].w;
          clusterPos[tid].y /= clusterPos[tid].w;
          clusterPos[tid].z /= clusterPos[tid].w;

          if (debug)
            printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   tid,
                   seedThreadIdx,
                   clusterEnergy[tid],
                   clusterPos[tid].x,
                   clusterPos[tid].y,
                   clusterPos[tid].z);
        } else {
          if (debug)
            printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                   tid,
                   seedThreadIdx,
                   clusterPos[tid].w,
                   pfClusParams.minAllowedNormalization());
          clusterPos[tid].x = 0.0;
          clusterPos[tid].y = 0.0;
          clusterPos[tid].z = 0.0;
        }
      }
      alpaka::syncBlockThreads(acc);

      if (tid < nSeeds) {
        float delta2 = dR2(prevClusterPos[tid], clusterPos[tid]);
        if (debug)
          printf("\tCluster %d (seed %d) has delta2 = %f\n", tid, seeds[tid], delta2);
        atomicMaxF(acc, &diff2, delta2);
        prevClusterPos[tid] = clusterPos[tid];  // Save clusterPos
      }
      alpaka::syncBlockThreads(acc);

      if (tid == 0) {
        float diff = sqrtf(diff2);
        iter++;
        notDone = (diff > tol) && ((unsigned int)iter < pfClusParams.maxIterations());
        if (debug) {
          if (diff > tol)
            printf("\tTopoId %d has diff = %f greater than tolerance %f (continuing)\n", topoId, diff, tol);
          else if (debug)
            printf("\tTopoId %d has diff = %f LESS than tolerance %f (terminating!)\n", topoId, diff, tol);
        }
      }
      alpaka::syncBlockThreads(acc);
    } while (notDone);
    if (tid == 0)
      clusteringVars[topoId].pfc_iter() = iter;
    // Fill PFCluster-level info
    // KenH:
    if (tid < nSeeds) {
      int rhIdx = clusteringVars[tid + clusteringVars[topoId].topoSeedOffsets()].topoSeedList();
      int seedIdx = clusteringVars[rhIdx].rhIdxToSeedIdx();
      clusterView[seedIdx].pfc_energy() = clusterEnergy[tid];
      clusterView[seedIdx].pfc_x() = clusterPos[tid].x;
      clusterView[seedIdx].pfc_y() = clusterPos[tid].y;
      clusterView[seedIdx].pfc_z() = clusterPos[tid].z;
    }
  }

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void dev_hcalFastCluster_originalGlobal(const TAcc& acc,
                                                        reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                                        int topoId,
                                                        int nSeeds,
                                                        int nRHTopo,
                                                        reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                                        reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                                        reco::PFClusterDeviceCollection::View clusterView,
                                                        reco::PFRecHitFractionDeviceCollection::View fracView,
                                                        Position4* __restrict__ globalClusterPos,
                                                        Position4* __restrict__ globalPrevClusterPos,
                                                        float* __restrict__ globalClusterEnergy,
                                                        float* __restrict__ globalRhFracSum,
                                                        int* __restrict__ globalSeeds,
                                                        int* __restrict__ globalRechits) {
    int& nRHNotSeed = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& topoSeedBegin = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& gridStride = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& iter = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    float& tol = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& diff2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& rhENormInv = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    bool& notDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    bool& debug = alpaka::declareSharedVar<bool, __COUNTER__>(acc);

    //auto& clusterPos = alpaka::declareSharedVar<Position4[1000], __COUNTER__>(acc);
    //auto& prevClusterPos = alpaka::declareSharedVar<Position4[1000], __COUNTER__>(acc);
    auto& clusterEnergy = alpaka::declareSharedVar<float[1000], __COUNTER__>(acc);
    auto& rhFracSum = alpaka::declareSharedVar<float[1000], __COUNTER__>(acc);
    auto& seeds = alpaka::declareSharedVar<int[1000], __COUNTER__>(acc);
    auto& rechits = alpaka::declareSharedVar<int[1000], __COUNTER__>(acc);
    // alpaka::trait::BlockSharedMemDynSizeBytes
    //Position4 * sharedArr = alpaka::getDynSharedMem<Position4>(acc);

    //Position4* clusterPos = sharedArr;                          //nSeeds
    // Position4* prevClusterPos = (Position4*)&clusterPos[nSeeds];   //nSeeds
    // float* clusterEnergy = (float*)&prevClusterPos[nSeeds];  //nSeeds
    // float* rhFracSum = (float*)&clusterEnergy[nSeeds];       //nRHTopo - nSeeds
    // int* seeds = (int*)&rhFracSum[nRHTopo - nSeeds];         //nSeeds
    // int* rechits = (int*)&seeds[nSeeds];                     //nRHTopo - nSeeds

    int blockIdx = 1000 * alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
    Position4* clusterPos = &globalClusterPos[blockIdx];
    Position4* prevClusterPos = &globalPrevClusterPos[blockIdx];
    //float* clusterEnergy = &globalClusterEnergy[blockIdx];
    //float* rhFracSum = &globalRhFracSum[blockIdx];
    //int* seeds = &globalSeeds[blockIdx];
    //int* rechits = &globalRechits[blockIdx];

    alpaka::syncBlockThreads(acc);

    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
      nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
      topoSeedBegin = clusteringVars[topoId].topoSeedOffsets();
      tol = pfClusParams.stoppingTolerance() *
            powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
      gridStride = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      iter = 0;
      notDone = true;
      debug = false;
      //debug = (nSeeds == 62) ? true : false;

      int i = clusteringVars[topoSeedBegin].topoSeedList();
      if (pfRecHits[i].layer() == PFLayer::HCAL_BARREL1)
        rhENormInv = pfClusParams.recHitEnergyNormInvEB_vec()[pfRecHits[i].depth() - 1];
      else if (pfRecHits[i].layer() == PFLayer::HCAL_ENDCAP)
        rhENormInv = pfClusParams.recHitEnergyNormInvEE_vec()[pfRecHits[i].depth() - 1];
      else
        printf("Rechit %d has invalid layer %d!\n", i, pfRecHits[i].layer());
    }
    alpaka::syncBlockThreads(acc);

    for (int n = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; n < nRHTopo; n += gridStride) {
      if (n < nSeeds)
        seeds[n] = clusteringVars[topoSeedBegin + n].topoSeedList();
      if (n < nRHNotSeed - 1)
        rechits[n] = fracView[clusteringVars[clusteringVars[topoSeedBegin].topoSeedList()].seedFracOffsets() + n + 1].pcrh_pfrhIdx();
    }
    alpaka::syncBlockThreads(acc);

    if (debug) {
      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        printf("\n===========================================================================================\n");
        printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
        for (int s = 0; s < nSeeds; s++) {
          if (s != 0)
            printf(", ");
          printf("%d", dev_getSeedRhIdx(seeds, s));
        }
        if (nRHTopo == nSeeds) {
          printf(")\n\n");
        } else {
          printf(") and other rechits (");
          for (int r = 1; r < nRHNotSeed; r++) {
            if (r != 1)
              printf(", ");
            printf("%d", dev_getRhFracIdx(rechits, r));
          }
          printf(")\n\n");
        }
      }
      alpaka::syncBlockThreads(acc);
    }

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
      int i = seeds[s];
      clusterPos[s] = Position4{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z(), 1.0};
      prevClusterPos[s] = clusterPos[s];
      clusterEnergy[s] = pfRecHits[i].energy();
      for (int r = 0; r < (nRHNotSeed - 1); r++) {
        fracView[clusteringVars[i].seedFracOffsets() + r + 1].pcrh_pfrhIdx() = rechits[r];
        fracView[clusteringVars[i].seedFracOffsets() + r + 1].pcrh_frac() = -1.;
      }
    }
    alpaka::syncBlockThreads(acc);

    do {
      if (debug && alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }

      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0)
        diff2 = -1;
      // Reset rhFracSum
      for (int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; tid < nRHNotSeed - 1; tid += gridStride) {
        rhFracSum[tid] = 0.;
        int rhThreadIdx = rechits[tid];
        Position4 rhThreadPos =
            Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
        for (int s = 0; s < nSeeds; s++) {
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          rhFracSum[tid] += fraction;
        }
      }
      alpaka::syncBlockThreads(acc);

      for (int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; tid < nRHNotSeed - 1; tid += gridStride) {
        int rhThreadIdx = rechits[tid];
        Position4 rhThreadPos =
            Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
        for (int s = 0; s < nSeeds; s++) {
          int i = seeds[s];
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          if (rhFracSum[tid] > pfClusParams.minFracTot()) {
            float fracpct = fraction / rhFracSum[tid];
            if (fracpct > 0.9999 || (d2 < 100. && fracpct > pfClusParams.minFracToKeep())) {
              fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = fracpct;
            } else {
              fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = -1;
            }
          } else {
            fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = -1;
          }
        }
      }
      alpaka::syncBlockThreads(acc);

      if (debug && alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0)
        printf("Computing cluster position for topoId %d\n", topoId);

      // Reset cluster position and energy
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
        int seedRhIdx = dev_getSeedRhIdx(seeds, s);
        float norm = logf(pfRecHits[seedRhIdx].energy() * rhENormInv);
        clusterPos[s] = Position4{
            pfRecHits[seedRhIdx].x() * norm, pfRecHits[seedRhIdx].y() * norm, pfRecHits[seedRhIdx].z() * norm, norm};
        clusterEnergy[s] = pfRecHits[seedRhIdx].energy();
        if (debug) {
          printf("Cluster %d (seed %d) has energy %f\tpos = (%f, %f, %f, %f)\n",
                 s,
                 seeds[s],
                 clusterEnergy[s],
                 clusterPos[s].x,
                 clusterPos[s].y,
                 clusterPos[s].z,
                 clusterPos[s].w);
        }
      }
      alpaka::syncBlockThreads(acc);

      // Recalculate position
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
        int seedRhIdx = dev_getSeedRhIdx(seeds, s);
        for (int r = 0; r < nRHNotSeed - 1; r++) {
          int j = rechits[r];
          float frac = dev_getRhFrac(clusteringVars, topoSeedBegin, fracView, s, r + 1);

          if (frac > -0.5) {
            clusterEnergy[s] += frac * pfRecHits[j].energy();

            if (nSeeds == 1 || j == pfRecHits[seedRhIdx].neighbours()(0) || j == pfRecHits[seedRhIdx].neighbours()(1) ||
                j == pfRecHits[seedRhIdx].neighbours()(2) || j == pfRecHits[seedRhIdx].neighbours()(3))
              dev_computeClusterPos(pfClusParams, clusterPos[s], frac, j, debug, pfRecHits, rhENormInv);
          }
        }
      }
      alpaka::syncBlockThreads(acc);

      // Position normalization
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
        if (clusterPos[s].w >= pfClusParams.minAllowedNormalization()) {
          // Divide by position norm
          clusterPos[s].x /= clusterPos[s].w;
          clusterPos[s].y /= clusterPos[s].w;
          clusterPos[s].z /= clusterPos[s].w;

          if (debug)
            printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   s,
                   seeds[s],
                   clusterEnergy[s],
                   clusterPos[s].x,
                   clusterPos[s].y,
                   clusterPos[s].z);
        } else {
          if (debug)
            printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                   s,
                   seeds[s],
                   clusterPos[s].w,
                   pfClusParams.minAllowedNormalization());
          clusterPos[s].x = 0.0;
          clusterPos[s].y = 0.0;
          clusterPos[s].z = 0.0;
        }
      }
      alpaka::syncBlockThreads(acc);

      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
        float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
        if (debug)
          printf("\tCluster %d (seed %d) has delta2 = %f\n", s, seeds[s], delta2);
        atomicMaxF(acc, &diff2, delta2);
        prevClusterPos[s] = clusterPos[s];  // Save clusterPos
      }
      alpaka::syncBlockThreads(acc);

      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        float diff = sqrtf(diff2);
        iter++;
        notDone = (diff > tol) && ((unsigned int)iter < pfClusParams.maxIterations());
        if (debug) {
          if (diff > tol)
            printf("\tTopoId %d has diff = %f greater than tolerance %f (continuing)\n", topoId, diff, tol);
          else if (debug)
            printf("\tTopoId %d has diff = %f LESS than tolerance %f (terminating!)\n", topoId, diff, tol);
        }
      }
      alpaka::syncBlockThreads(acc);
    } while (notDone);
    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0)
      clusteringVars[topoId].pfc_iter() = iter;
    for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
      int rhIdx = clusteringVars[s + clusteringVars[topoId].topoSeedOffsets()].topoSeedList();
      int seedIdx = clusteringVars[rhIdx].rhIdxToSeedIdx();
      clusterView[seedIdx].pfc_energy() = pfRecHits[s].energy();
      clusterView[seedIdx].pfc_x() = pfRecHits[s].x();
      clusterView[seedIdx].pfc_y() = pfRecHits[s].y();
      clusterView[seedIdx].pfc_z() = pfRecHits[s].z();
    }
  }

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void dev_hcalFastCluster_originalShared(const TAcc& acc,
                                                        reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                                        int topoId,
                                                        int nSeeds,
                                                        int nRHTopo,
                                                        reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                                        reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                                        reco::PFClusterDeviceCollection::View clusterView,
                                                        reco::PFRecHitFractionDeviceCollection::View fracView) {
    // extern __shared__ Position4 sharedArr[];
    // Position4* clusterPos = sharedArr;                          //nSeeds
    // Position4* prevClusterPos = (Position4*)&clusterPos[nSeeds];   //nSeeds
    // float* clusterEnergy = (float*)&prevClusterPos[nSeeds];  //nSeeds
    // float* rhFracSum = (float*)&clusterEnergy[nSeeds];       //nRHTopo - nSeeds
    // int* seeds = (int*)&rhFracSum[nRHTopo - nSeeds];         //nSeeds
    // int* rechits = (int*)&seeds[nSeeds];                     //nRHTopo - nSeeds

    int& nRHNotSeed = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& topoSeedBegin = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& gridStride = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& iter = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    float& tol = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& diff2 = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float& rhENormInv = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    bool& notDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    bool& debug = alpaka::declareSharedVar<bool, __COUNTER__>(acc);

    // alpaka::trait::BlockSharedMemDynSizeBytes
    //Position4 * sharedArr = alpaka::getDynSharedMem<Position4>(acc);

    //Position4* clusterPos = sharedArr;                          //nSeeds
    // Position4* prevClusterPos = (Position4*)&clusterPos[nSeeds];   //nSeeds
    // float* clusterEnergy = (float*)&prevClusterPos[nSeeds];  //nSeeds
    // float* rhFracSum = (float*)&clusterEnergy[nSeeds];       //nRHTopo - nSeeds
    // int* seeds = (int*)&rhFracSum[nRHTopo - nSeeds];         //nSeeds
    // int* rechits = (int*)&seeds[nSeeds];                     //nRHTopo - nSeeds

    auto& clusterPos = alpaka::declareSharedVar<Position4[300], __COUNTER__>(acc);
    auto& prevClusterPos = alpaka::declareSharedVar<Position4[300], __COUNTER__>(acc);
    auto& clusterEnergy = alpaka::declareSharedVar<float[300], __COUNTER__>(acc);
    auto& rhFracSum = alpaka::declareSharedVar<float[1500], __COUNTER__>(acc);
    auto& seeds = alpaka::declareSharedVar<int[300], __COUNTER__>(acc);
    auto& rechits = alpaka::declareSharedVar<int[1500], __COUNTER__>(acc);

    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
      nRHNotSeed = nRHTopo - nSeeds + 1;  // 1 + (# rechits per topoId that are NOT seeds)
      topoSeedBegin = clusteringVars[topoId].topoSeedOffsets();
      tol = pfClusParams.stoppingTolerance() *
            powf(fmaxf(1.0, nSeeds - 1.0), 2.0);  // stopping tolerance * tolerance scaling
      gridStride = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      iter = 0;
      notDone = true;
      debug = false;
      //debug = (nSeeds == 62) ? true : false;

      int i = clusteringVars[topoSeedBegin].topoSeedList();
      if (pfRecHits[i].layer() == PFLayer::HCAL_BARREL1)
        rhENormInv = pfClusParams.recHitEnergyNormInvEB_vec()[pfRecHits[i].depth() - 1];
      else if (pfRecHits[i].layer() == PFLayer::HCAL_ENDCAP)
        rhENormInv = pfClusParams.recHitEnergyNormInvEE_vec()[pfRecHits[i].depth() - 1];
      else
        printf("Rechit %d has invalid layer %d!\n", i, pfRecHits[i].layer());
    }
    alpaka::syncBlockThreads(acc);

    for (int n = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; n < nRHTopo; n += gridStride) {
      if (n < nSeeds)
        seeds[n] = clusteringVars[topoSeedBegin + n].topoSeedList();
      if (n < nRHNotSeed - 1)
        rechits[n] = fracView[clusteringVars[clusteringVars[topoSeedBegin].topoSeedList()].seedFracOffsets() + n + 1].pcrh_pfrhIdx();
    }
    alpaka::syncBlockThreads(acc);

    if (debug) {
      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        printf("\n===========================================================================================\n");
        printf("Processing topo cluster %d with nSeeds = %d nRHTopo = %d and seeds (", topoId, nSeeds, nRHTopo);
        for (int s = 0; s < nSeeds; s++) {
          if (s != 0)
            printf(", ");
          printf("%d", dev_getSeedRhIdx(seeds, s));
        }
        if (nRHTopo == nSeeds) {
          printf(")\n\n");
        } else {
          printf(") and other rechits (");
          for (int r = 1; r < nRHNotSeed; r++) {
            if (r != 1)
              printf(", ");
            printf("%d", dev_getRhFracIdx(rechits, r));
          }
          printf(")\n\n");
        }
      }
      alpaka::syncBlockThreads(acc);
    }

    // Set initial cluster position (energy) to seed rechit position (energy)
    for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
      int i = seeds[s];
      clusterPos[s] = Position4{pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z(), 1.0};
      prevClusterPos[s] = clusterPos[s];
      clusterEnergy[s] = pfRecHits[i].energy();
      for (int r = 0; r < (nRHNotSeed - 1); r++) {
        fracView[clusteringVars[i].seedFracOffsets() + r + 1].pcrh_pfrhIdx() = rechits[r];
        fracView[clusteringVars[i].seedFracOffsets() + r + 1].pcrh_frac() = -1.;
      }
    }
    alpaka::syncBlockThreads(acc);

    do {
      if (debug && alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }

      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0)
        diff2 = -1;
      // Reset rhFracSum
      for (int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; tid < nRHNotSeed - 1; tid += gridStride) {
        rhFracSum[tid] = 0.;
        int rhThreadIdx = rechits[tid];
        Position4 rhThreadPos =
            Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
        for (int s = 0; s < nSeeds; s++) {
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          rhFracSum[tid] += fraction;
        }
      }
      alpaka::syncBlockThreads(acc);

      for (int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; tid < nRHNotSeed - 1; tid += gridStride) {
        int rhThreadIdx = rechits[tid];
        Position4 rhThreadPos =
            Position4{pfRecHits[rhThreadIdx].x(), pfRecHits[rhThreadIdx].y(), pfRecHits[rhThreadIdx].z(), 1.};
        for (int s = 0; s < nSeeds; s++) {
          int i = seeds[s];
          float dist2 = (clusterPos[s].x - rhThreadPos.x) * (clusterPos[s].x - rhThreadPos.x) +
                        (clusterPos[s].y - rhThreadPos.y) * (clusterPos[s].y - rhThreadPos.y) +
                        (clusterPos[s].z - rhThreadPos.z) * (clusterPos[s].z - rhThreadPos.z);

          float d2 = dist2 / pfClusParams.showerSigma2();
          float fraction = clusterEnergy[s] * rhENormInv * expf(-0.5 * d2);

          if (rhFracSum[tid] > pfClusParams.minFracTot()) {
            float fracpct = fraction / rhFracSum[tid];
            if (fracpct > 0.9999 || (d2 < 100. && fracpct > pfClusParams.minFracToKeep())) {
              fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = fracpct;
            } else {
              fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = -1;
            }
          } else {
            fracView[clusteringVars[i].seedFracOffsets() + tid + 1].pcrh_frac() = -1;
          }
        }
      }
      alpaka::syncBlockThreads(acc);

      if (debug && alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0)
        printf("Computing cluster position for topoId %d\n", topoId);

      // Reset cluster position and energy
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
        int seedRhIdx = dev_getSeedRhIdx(seeds, s);
        float norm = logf(pfRecHits[seedRhIdx].energy() * rhENormInv);
        clusterPos[s] = Position4{
            pfRecHits[seedRhIdx].x() * norm, pfRecHits[seedRhIdx].y() * norm, pfRecHits[seedRhIdx].z() * norm, norm};
        clusterEnergy[s] = pfRecHits[seedRhIdx].energy();
        if (debug) {
          printf("Cluster %d (seed %d) has energy %f\tpos = (%f, %f, %f, %f)\n",
                 s,
                 seeds[s],
                 clusterEnergy[s],
                 clusterPos[s].x,
                 clusterPos[s].y,
                 clusterPos[s].z,
                 clusterPos[s].w);
        }
      }
      alpaka::syncBlockThreads(acc);

      // Recalculate position
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
        int seedRhIdx = dev_getSeedRhIdx(seeds, s);
        for (int r = 0; r < nRHNotSeed - 1; r++) {
          int j = rechits[r];
          float frac = dev_getRhFrac(clusteringVars, topoSeedBegin, fracView, s, r + 1);

          if (frac > -0.5) {
            clusterEnergy[s] += frac * pfRecHits[j].energy();

            if (nSeeds == 1 || j == pfRecHits[seedRhIdx].neighbours()(0) || j == pfRecHits[seedRhIdx].neighbours()(1) ||
                j == pfRecHits[seedRhIdx].neighbours()(2) || j == pfRecHits[seedRhIdx].neighbours()(3))
              dev_computeClusterPos(pfClusParams, clusterPos[s], frac, j, debug, pfRecHits, rhENormInv);
          }
        }
        /*
        if (clusterPos[s].w >= pfClusParams.minAllowedNormalization()) {
          // Divide by position norm
          clusterPos[s].x /= clusterPos[s].w;
          clusterPos[s].y /= clusterPos[s].w;
          clusterPos[s].z /= clusterPos[s].w;

          if (debug)
            printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   s,
                   seeds[s],
                   clusterEnergy[s],
                   clusterPos[s].x,
                   clusterPos[s].y,
                   clusterPos[s].z);
        } else {
          if (debug)
            printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                   s,
                   seeds[s],
                   clusterPos[s].w,
                   pfClusParams.minAllowedNormalization());
          clusterPos[s].x = 0.0;
          clusterPos[s].y = 0.0;
          clusterPos[s].z = 0.0;
        }

        float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
        if (debug)
          printf("\tCluster %d (seed %d) has delta2 = %f\n", s, seeds[s], delta2);
        atomicMaxF(acc, &diff2, delta2);
        prevClusterPos[s] = clusterPos[s];  // Save clusterPos*/
      }
      alpaka::syncBlockThreads(acc);

      // Position normalization
      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
        if (clusterPos[s].w >= pfClusParams.minAllowedNormalization()) {
          // Divide by position norm
          clusterPos[s].x /= clusterPos[s].w;
          clusterPos[s].y /= clusterPos[s].w;
          clusterPos[s].z /= clusterPos[s].w;

          if (debug)
            printf("\tCluster %d (seed %d) energy = %f\tposition = (%f, %f, %f)\n",
                   s,
                   seeds[s],
                   clusterEnergy[s],
                   clusterPos[s].x,
                   clusterPos[s].y,
                   clusterPos[s].z);
        } else {
          if (debug)
            printf("\tCluster %d (seed %d) position norm (%f) less than minimum (%f)\n",
                   s,
                   seeds[s],
                   clusterPos[s].w,
                   pfClusParams.minAllowedNormalization());
          clusterPos[s].x = 0.0;
          clusterPos[s].y = 0.0;
          clusterPos[s].z = 0.0;
        }
      }
      alpaka::syncBlockThreads(acc);

      for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
        float delta2 = dR2(prevClusterPos[s], clusterPos[s]);
        if (debug)
          printf("\tCluster %d (seed %d) has delta2 = %f\n", s, seeds[s], delta2);
        atomicMaxF(acc, &diff2, delta2);
        prevClusterPos[s] = clusterPos[s];  // Save clusterPos
      }
      alpaka::syncBlockThreads(acc);

      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        float diff = sqrtf(diff2);
        iter++;
        notDone = (diff > tol) && ((unsigned int)iter < pfClusParams.maxIterations());
        if (debug) {
          if (diff > tol)
            printf("\tTopoId %d has diff = %f greater than tolerance %f (continuing)\n", topoId, diff, tol);
          else if (debug)
            printf("\tTopoId %d has diff = %f LESS than tolerance %f (terminating!)\n", topoId, diff, tol);
        }
      }
      alpaka::syncBlockThreads(acc);
    } while (notDone);
    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0)
      clusteringVars[topoId].pfc_iter() = iter;
    for (int s = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]; s < nSeeds; s += gridStride) {
      int rhIdx = clusteringVars[s + clusteringVars[topoId].topoSeedOffsets()].topoSeedList();
      int seedIdx = clusteringVars[rhIdx].rhIdxToSeedIdx();
      clusterView[seedIdx].pfc_energy() = pfRecHits[s].energy();
      clusterView[seedIdx].pfc_x() = pfRecHits[s].x();
      clusterView[seedIdx].pfc_y() = pfRecHits[s].y();
      clusterView[seedIdx].pfc_z() = pfRecHits[s].z();
    }
  }

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void hcalFastCluster_selection(const TAcc& acc,
                                               reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                               size_t nRH,
                                               reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                               reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                               reco::PFClusterDeviceCollection::View clusterView,
                                               reco::PFRecHitFractionDeviceCollection::View fracView,
                                               Position4* __restrict__ globalClusterPos,
                                               Position4* __restrict__ globalPrevClusterPos,
                                               float* __restrict__ globalClusterEnergy,
                                               float* __restrict__ globalRhFracSum,
                                               int* __restrict__ globalSeeds,
                                               int* __restrict__ globalRechits) {
    int& topoId = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& nRHTopo = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& nSeeds = alpaka::declareSharedVar<int, __COUNTER__>(acc);

    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
      topoId = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      nRHTopo = clusteringVars[topoId].topoRHCount();
      nSeeds = clusteringVars[topoId].topoSeedCount();
      clusteringVars[topoId].processedTopo() = false;
    }

    alpaka::syncBlockThreads(acc);

    if ((unsigned int)topoId < nRH && nRHTopo > 0 && nSeeds > 0) {
      clusteringVars[topoId].processedTopo() = true;
      //alpaka::syncBlockThreads(acc);
      if (nRHTopo == nSeeds) {
        // PF cluster is isolated seed. No iterations needed
        if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
          clusteringVars[topoId].pfc_iter() = 0;
          // KenH: Fill PFCluster-level information
          int rhIdx = clusteringVars[clusteringVars[topoId].topoSeedOffsets()].topoSeedList();  // i is the seed rechit index
          int seedIdx = clusteringVars[rhIdx].rhIdxToSeedIdx();
          clusterView[seedIdx].pfc_energy() = pfRecHits[rhIdx].energy();
          clusterView[seedIdx].pfc_x() = pfRecHits[rhIdx].x();
          clusterView[seedIdx].pfc_y() = pfRecHits[rhIdx].y();
          clusterView[seedIdx].pfc_z() = pfRecHits[rhIdx].z();
        }
      } else if (nSeeds == 1) {
        // Single seed cluster
        dev_hcalFastCluster_optimizedSimple(
            acc, pfClusParams, topoId, nRHTopo, pfRecHits, clusteringVars, clusterView, fracView);
      } else if (nSeeds <= 100 && nRHTopo - nSeeds < threadsPerBlockForClustering) {
        dev_hcalFastCluster_optimizedComplex(
            acc, pfClusParams, topoId, nSeeds, nRHTopo, pfRecHits, clusteringVars, clusterView, fracView);
      } else if (nSeeds <= 400 && nRHTopo - nSeeds <= 1500) {
        dev_hcalFastCluster_originalShared(
            acc, pfClusParams, topoId, nSeeds, nRHTopo, pfRecHits, clusteringVars, clusterView, fracView);
      } else if (nSeeds <= 1000 && nRHTopo - nSeeds <= 1000) {
        dev_hcalFastCluster_originalGlobal(acc,
                                           pfClusParams,
                                           topoId,
                                           nSeeds,
                                           nRHTopo,
                                           pfRecHits,
                                           clusteringVars,
                                           clusterView,
                                           fracView,
                                           globalClusterPos,
                                           globalPrevClusterPos,
                                           globalClusterEnergy,
                                           globalRhFracSum,
                                           globalSeeds,
                                           globalRechits);
      } else {
        if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0)
          printf("ERROR: Topo cluster %d has %d seeds and %d rechits. SKIPPING!!\n", topoId, nSeeds, nRHTopo);
      }
    } else
      return;
  }

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void hcalFastCluster_serial(const TAcc& acc,
                                            reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                            size_t nRH,
                                            reco::PFRecHitDeviceCollection::ConstView pfRecHits,
                                            reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                            reco::PFClusterDeviceCollection::View clusterView,
                                            reco::PFRecHitFractionDeviceCollection::View fracView,
                                            Position4* __restrict__ globalClusterPos,
                                            Position4* __restrict__ globalPrevClusterPos,
                                            float* __restrict__ globalClusterEnergy,
                                            float* __restrict__ globalRhFracSum,
                                            int* __restrict__ globalSeeds,
                                            int* __restrict__ globalRechits) {
    int& topoId = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& nRHTopo = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    int& nSeeds = alpaka::declareSharedVar<int, __COUNTER__>(acc);

    if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
      topoId = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
      nRHTopo = clusteringVars[topoId].topoRHCount();
      nSeeds = clusteringVars[topoId].topoSeedCount();
    }

    alpaka::syncBlockThreads(acc);

    if ((unsigned int)topoId < nRH && nRHTopo > 0 && nSeeds > 0) {
      if (nRHTopo == nSeeds) {
        // PF cluster is isolated seed. No iterations needed
        if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
          clusteringVars[topoId].pfc_iter() = 0;
          // KenH: Fill PFCluster-level information
          int rhIdx = clusteringVars[clusteringVars[topoId].topoSeedOffsets()].topoSeedList();  // i is the seed rechit index
          int seedIdx = clusteringVars[rhIdx].rhIdxToSeedIdx();
          clusterView[seedIdx].pfc_energy() = pfRecHits[rhIdx].energy();
          clusterView[seedIdx].pfc_x() = pfRecHits[rhIdx].x();
          clusterView[seedIdx].pfc_y() = pfRecHits[rhIdx].y();
          clusterView[seedIdx].pfc_z() = pfRecHits[rhIdx].z();
        }
      } else if (nSeeds <= 400 && nRHTopo - nSeeds <= 1500) {
        dev_hcalFastCluster_originalShared(
            acc, pfClusParams, topoId, nSeeds, nRHTopo, pfRecHits, clusteringVars, clusterView, fracView);
      } else if (nSeeds <= 1000 && nRHTopo - nSeeds <= 1000) {
        dev_hcalFastCluster_originalGlobal(acc,
                                           pfClusParams,
                                           topoId,
                                           nSeeds,
                                           nRHTopo,
                                           pfRecHits,
                                           clusteringVars,
                                           clusterView,
                                           fracView,
                                           globalClusterPos,
                                           globalPrevClusterPos,
                                           globalClusterEnergy,
                                           globalRhFracSum,
                                           globalSeeds,
                                           globalRechits);
      } else {
        if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0)
          printf("ERROR: Topo cluster %d has %d seeds and %d rechits. SKIPPING!!\n", topoId, nSeeds, nRHTopo);
      }
    } else
      return;
  }

  class seedingTopoThreshKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  const reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::PFClusterDeviceCollection::View clusterView,
                                  reco::PFRecHitFractionDeviceCollection::View fracView,
                                  uint32_t* __restrict__ nSeeds) const {
      const int nRH = pfRecHits.size();
      seedingTopoThreshKernel_HCAL(acc, pfClusParams, clusteringVars, pfRecHits, clusterView, fracView, nRH, nSeeds);
    }
  };

  class prepareTopoInputsKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars,
                                  uint32_t* __restrict__ nSeeds) const {
      const int nRH = pfRecHits.size();
      prepareTopoInputs(acc, nRH, pfRecHits, clusteringVars, clusteringEdgeVars, nSeeds);
    }
  };

  class eclccInitKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) const {
      const int nRH = pfRecHits.size();
      ECLCC_init(acc, nRH, clusteringVars, clusteringEdgeVars);
    }
  };

  class eclccCompute1Kernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) const {
      const int nRH = pfRecHits.size();
      ECLCC_compute1(acc, nRH, clusteringVars, clusteringEdgeVars);
    }
  };

  class eclccCompute2Kernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) const {
      const int nRH = pfRecHits.size();
      ECLCC_compute2(acc, nRH, clusteringVars, clusteringEdgeVars);
    }
  };

  class eclccCompute3Kernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) const {
      const int nRH = pfRecHits.size();
      ECLCC_compute3(acc, nRH, clusteringVars, clusteringEdgeVars);
    }
  };

  class eclccFlattenKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::ClusteringEdgeVarsDeviceCollection::View clusteringEdgeVars) const {
      const int nRH = pfRecHits.size();
      ECLCC_flatten(acc, nRH, clusteringVars, clusteringEdgeVars);
    }
  };

  class topoClusterContractionKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::PFClusterDeviceCollection::View clusterView,
                                  uint32_t* __restrict__ nSeeds) const {
      const int nRH = pfRecHits.size();
      topoClusterContraction(acc, nRH, pfRecHits, clusteringVars, clusterView, nSeeds);
    }
  };

  class fillRhfIndexKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::PFRecHitFractionDeviceCollection::View fracView) const {
      const int nRH = pfRecHits.size();
      fillRhfIndex(acc, nRH, clusteringVars, fracView);
    }
  };

  class fillRhfIndexSerialKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::PFRecHitFractionDeviceCollection::View fracView) const {
      const int nRH = pfRecHits.size();
      fillRhfIndexSerial(acc, nRH, clusteringVars, fracView);
    }
  };

  class fastClusterKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  const reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::PFClusterDeviceCollection::View clusterView,
                                  reco::PFRecHitFractionDeviceCollection::View fracView,
                                  Position4* __restrict__ globalClusterPos,
                                  Position4* __restrict__ globalPrevClusterPos,
                                  float* __restrict__ globalClusterEnergy,
                                  float* __restrict__ globalRhFracSum,
                                  int* __restrict__ globalSeeds,
                                  int* __restrict__ globalRechits) const {
      const int nRH = pfRecHits.size();
      hcalFastCluster_selection(acc,
                                pfClusParams,
                                nRH,
                                pfRecHits,
                                clusteringVars,
                                clusterView,
                                fracView,
                                globalClusterPos,
                                globalPrevClusterPos,
                                globalClusterEnergy,
                                globalRhFracSum,
                                globalSeeds,
                                globalRechits);
    }
  };

  class fastClusterSerialKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  const reco::PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                  reco::ClusteringVarsDeviceCollection::View clusteringVars,
                                  reco::PFClusterDeviceCollection::View clusterView,
                                  reco::PFRecHitFractionDeviceCollection::View fracView,
                                  Position4* __restrict__ globalClusterPos,
                                  Position4* __restrict__ globalPrevClusterPos,
                                  float* __restrict__ globalClusterEnergy,
                                  float* __restrict__ globalRhFracSum,
                                  int* __restrict__ globalSeeds,
                                  int* __restrict__ globalRechits) const {
      const int nRH = pfRecHits.size();
      hcalFastCluster_serial(acc,
                             pfClusParams,
                             nRH,
                             pfRecHits,
                             clusteringVars,
                             clusterView,
                             fracView,
                             globalClusterPos,
                             globalPrevClusterPos,
                             globalClusterEnergy,
                             globalRhFracSum,
                             globalSeeds,
                             globalRechits);
    }
  };

  PFClusterProducerKernel::PFClusterProducerKernel(cms::alpakatools::device_buffer<Device, uint32_t>&& buffer1,
                                                   //cms::alpakatools::host_buffer<uint32_t>&& buffer2,
                                                   cms::alpakatools::device_buffer<Device, Position4[]>&& buffer3,
                                                   cms::alpakatools::device_buffer<Device, Position4[]>&& buffer4,
                                                   cms::alpakatools::device_buffer<Device, float[]>&& buffer5,
                                                   cms::alpakatools::device_buffer<Device, float[]>&& buffer6,
                                                   cms::alpakatools::device_buffer<Device, int[]>&& buffer7,
                                                   cms::alpakatools::device_buffer<Device, int[]>&& buffer8)
      : nSeeds(std::move(buffer1)),
        //nTopos(std::move(buffer2)),
        globalClusterPos(std::move(buffer3)),
        globalPrevClusterPos(std::move(buffer4)),
        globalClusterEnergy(std::move(buffer5)),
        globalRhFracSum(std::move(buffer6)),
        globalSeeds(std::move(buffer7)),
        globalRechits(std::move(buffer8)) {}

  PFClusterProducerKernel PFClusterProducerKernel::Construct(Queue& queue,
                                                             const reco::PFRecHitHostCollection& pfRecHits) {
    const int nRH = pfRecHits->size();

    return PFClusterProducerKernel{cms::alpakatools::make_device_buffer<uint32_t>(queue),
                                   //cms::alpakatools::make_host_buffer<uint32_t>(queue),
                                   cms::alpakatools::make_device_buffer<Position4[]>(queue, 3000 * 1000),
                                   cms::alpakatools::make_device_buffer<Position4[]>(queue, 3000 * 1000),
                                   cms::alpakatools::make_device_buffer<float[]>(queue, 1),
                                   cms::alpakatools::make_device_buffer<float[]>(queue, 1),
                                   cms::alpakatools::make_device_buffer<int[]>(queue, 1),
                                   cms::alpakatools::make_device_buffer<int[]>(queue, 1)};
  }

  void PFClusterProducerKernel::execute(const Device& device,
                                        Queue& queue,
                                        const reco::PFClusterParamsAlpakaESDataDevice& params,
                                        reco::ClusteringVarsDeviceCollection& clusteringVars,
                                        reco::ClusteringEdgeVarsDeviceCollection& clusteringEdgeVars,
                                        const reco::PFRecHitHostCollection& pfRecHits,
                                        reco::PFClusterDeviceCollection& pfClusters,
                                        reco::PFRecHitFractionDeviceCollection& pfrhFractions) {
    const int nRH = pfRecHits->size();

    const int threadsPerBlock = std::is_same_v<Device, alpaka::DevCpu> ? 32 : 256;
    const int blocks = std::is_same_v<Device, alpaka::DevCpu> ? nRH : (nRH + threadsPerBlock - 1) / threadsPerBlock;
    const int threadsPerBlockForClustering = std::is_same_v<Device, alpaka::DevCpu> ? 32 : 512;

    alpaka::memset(queue, nSeeds, 0x00);  // Reset nSeeds

    // seedingTopoThresh
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        seedingTopoThreshKernel{},
                        clusteringVars.view(),
                        params.view(),
                        pfRecHits.view(),
                        pfClusters.view(),
                        pfrhFractions.view(),
                        nSeeds.data());
    // prepareTopoInputs
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        prepareTopoInputsKernel{},
                        pfRecHits.view(),
                        clusteringVars.view(),
                        clusteringEdgeVars.view(),
                        nSeeds.data());
    // ECLCC
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        eclccInitKernel{},
                        pfRecHits.view(),
                        clusteringVars.view(),
                        clusteringEdgeVars.view());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        eclccCompute1Kernel{},
                        pfRecHits.view(),
                        clusteringVars.view(),
                        clusteringEdgeVars.view());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        eclccCompute2Kernel{},
                        pfRecHits.view(),
                        clusteringVars.view(),
                        clusteringEdgeVars.view());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        eclccCompute3Kernel{},
                        pfRecHits.view(),
                        clusteringVars.view(),
                        clusteringEdgeVars.view());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocks, threadsPerBlock),
                        eclccFlattenKernel{},
                        pfRecHits.view(),
                        clusteringVars.view(),
                        clusteringEdgeVars.view());
    // topoClusterContraction
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(1, threadsPerBlockForClustering),
                        topoClusterContractionKernel{},
                        pfRecHits.view(),
                        clusteringVars.view(),
                        pfClusters.view(),
                        nSeeds.data());

    // Run fillRhfIndex on serial or parallel
    if (std::is_same_v<Device, alpaka::DevCpu>) {
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(1, 32),
                          fillRhfIndexSerialKernel{},
                          pfRecHits.view(),
                          clusteringVars.view(),
                          pfrhFractions.view());
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(nRH, threadsPerBlockForClustering),
                          fastClusterSerialKernel{},
                          pfRecHits.view(),
                          params.view(),
                          clusteringVars.view(),
                          pfClusters.view(),
                          pfrhFractions.view(),
                          globalClusterPos.data(),
                          globalPrevClusterPos.data(),
                          globalClusterEnergy.data(),
                          globalRhFracSum.data(),
                          globalSeeds.data(),
                          globalRechits.data());
    } else {
      alpaka::exec<Acc2D>(queue,
                          make_workdiv<Acc2D>({(nRH + 31) / 32, (nRH + 31) / 32}, {32, 32}),
                          fillRhfIndexKernel{},
                          pfRecHits.view(),
                          clusteringVars.view(),
                          pfrhFractions.view());

      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(nRH, threadsPerBlockForClustering),
                          fastClusterKernel{},
                          pfRecHits.view(),
                          params.view(),
                          clusteringVars.view(),
                          pfClusters.view(),
                          pfrhFractions.view(),
                          globalClusterPos.data(),
                          globalPrevClusterPos.data(),
                          globalClusterEnergy.data(),
                          globalRhFracSum.data(),
                          globalSeeds.data(),
                          globalRechits.data());
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
