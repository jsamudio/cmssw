#ifndef RecoParticleFlow_PFClusterProducerAlpaka_plugins_alpaka_PFClusterECLCC_h
#define RecoParticleFlow_PFClusterProducerAlpaka_plugins_alpaka_PFClusterECLCC_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusteringEdgeVarsDeviceCollection.h"

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

/*
 The code is modified for the specific use-case of generating topological clusters
 for PFClustering. It is adapted to work with the Alpaka portability library.
 
*/

namespace ALPAKA_ACCELERATOR_NAMESPACE {


  /* intermediate pointer jumping */

  ALPAKA_FN_ACC int representative(const int idx, reco::PFClusteringVarsDeviceCollection::View pfClusteringVars) {
    int curr = pfClusteringVars[idx].pfrh_topoId();
    if (curr != idx) {
      int next, prev = idx;
      while (curr > (next = pfClusteringVars[curr].pfrh_topoId())) {
        pfClusteringVars[prev].pfrh_topoId() = next;
        prev = curr;
        curr = next;
      }
    }
    return curr;
  }

  // Initial step of ECL-CC. Uses ID of first neighbour in edgeList with a smaller ID
  class ECLCCInit {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();
      const int from = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] +
                       alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] *
                       alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      for (int v : cms::alpakatools::elements_with_stride(acc, nRH)) {
        const int beg = pfClusteringEdgeVars[v].pfrh_edgeIdx();
        const int end = pfClusteringEdgeVars[v + 1].pfrh_edgeIdx();
        int m = v;
        int i = beg;
        while ((m == v) && (i < end)) {
          m = std::min(m, pfClusteringEdgeVars[i].pfrh_edgeList());
          i++;
        }
        pfClusteringVars[v].pfrh_topoId() = m;
      }

      if (from == 0) {
        pfClusteringVars.topL() = 0;
        pfClusteringVars.posL() = 0;
        pfClusteringVars.topH() = nRH - 1;
        pfClusteringVars.posH() = nRH - 1;
      }
    }
  };

  // First edge processing kernel of ECL-CC
  class ECLCCCompute1 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();

      for (int v : cms::alpakatools::elements_with_stride(acc, nRH)) {
        const int vstat = pfClusteringVars[v].pfrh_topoId();
        if (v != vstat) {
          const int beg = pfClusteringEdgeVars[v].pfrh_edgeIdx();
          const int end = pfClusteringEdgeVars[v + 1].pfrh_edgeIdx();
          int deg = end - beg;
          if (deg > 16) {
            int idx;
            if (deg <= 352) {
              idx = alpaka::atomicAdd(acc, &pfClusteringVars.topL(), 1);
            } else {
              idx = alpaka::atomicAdd(acc, &pfClusteringVars.topH(), -1);
            }
            pfClusteringVars[idx].wl_d() = v;
          } else {
            int vstat = representative(v, pfClusteringVars);
            for (int i = beg; i < end; i++) {
              const int nli = pfClusteringEdgeVars[i].pfrh_edgeList();
              if (v > nli) {
                int ostat = representative(nli, pfClusteringVars);
                bool repeat;
                do {
                  repeat = false;
                  if (vstat != ostat) {
                    int ret;
                    if (vstat < ostat) {
                      if ((ret = alpaka::atomicCas(acc, &pfClusteringVars[ostat].pfrh_topoId(), ostat, vstat)) != ostat) {
                        ostat = ret;
                        repeat = true;
                      }
                    } else {
                      if ((ret = alpaka::atomicCas(acc, &pfClusteringVars[vstat].pfrh_topoId(), vstat, ostat)) != vstat) {
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
  };

  /* process medium-degree vertices at warp granularity */
  class ECLCCCompute2 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();

      const int lane = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] % alpaka::warp::getSize(acc);

      int32_t idx = 0;
      if (lane == 0)
        idx = alpaka::atomicAdd(acc, &pfClusteringVars.posL(), 1);
      idx = alpaka::warp::shfl(acc, idx, 0);
      while (idx < pfClusteringVars.topL()) {
        const int v = pfClusteringVars[idx].wl_d();
        int vstat = representative(v, pfClusteringVars);
        for (int i = pfClusteringEdgeVars[v].pfrh_edgeIdx() + lane; i < pfClusteringEdgeVars[v + 1].pfrh_edgeIdx();
             i += alpaka::warp::getSize(acc)) {
          const int nli = pfClusteringEdgeVars[i].pfrh_edgeList();
          if (v > nli) {
            int ostat = representative(nli, pfClusteringVars);
            bool repeat;
            do {
              repeat = false;
              if (vstat != ostat) {
                int ret;
                if (vstat < ostat) {
                  if ((ret = alpaka::atomicCas(acc, &pfClusteringVars[ostat].pfrh_topoId(), ostat, vstat)) != ostat) {
                    ostat = ret;
                    repeat = true;
                  }
                } else {
                  if ((ret = alpaka::atomicCas(acc, &pfClusteringVars[vstat].pfrh_topoId(), vstat, ostat)) != vstat) {
                    vstat = ret;
                    repeat = true;
                  }
                }
              }
            } while (repeat);
          }
        }
        if (lane == 0)
          idx = alpaka::atomicAdd(acc, &pfClusteringVars.posL(), 1);
        idx = alpaka::warp::shfl(acc, idx, 0);
      }
    }
  };

  /* process high-degree vertices at block granularity */
  class ECLCCCompute3 {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();

      int& vB = alpaka::declareSharedVar<int, __COUNTER__>(acc);
      if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
        const int idx = alpaka::atomicAdd(acc, &pfClusteringVars.posH(), -1);
        vB = (idx > pfClusteringVars.topH()) ? pfClusteringVars[idx].wl_d() : -1;
      }
      alpaka::syncBlockThreads(acc); // all threads call sync
      while (vB >= 0) {
        const int v = vB;
        alpaka::syncBlockThreads(acc); // all threads call sync
        int vstat = representative(v, pfClusteringVars);
        for (int i = pfClusteringEdgeVars[v].pfrh_edgeIdx() + alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
             i < pfClusteringEdgeVars[v + 1].pfrh_edgeIdx();
             i += alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]) {
          const int nli = pfClusteringEdgeVars[i].pfrh_edgeList();
          if (v > nli) {
            int ostat = representative(nli, pfClusteringVars);
            bool repeat;
            do {
              repeat = false;
              if (vstat != ostat) {
                int ret;
                if (vstat < ostat) {
                  if ((ret = alpaka::atomicCas(acc, &pfClusteringVars[ostat].pfrh_topoId(), ostat, vstat)) != ostat) {
                    ostat = ret;
                    repeat = true;
                  }
                } else {
                  if ((ret = alpaka::atomicCas(acc, &pfClusteringVars[vstat].pfrh_topoId(), vstat, ostat)) != vstat) {
                    vstat = ret;
                    repeat = true;
                  }
                }
              }
            } while (repeat);
          }
          alpaka::syncBlockThreads(acc); // all threads call sync
        }
        if (alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] == 0) {
          const int idx = alpaka::atomicAdd(acc, &pfClusteringVars.posH(), -1);
          vB = (idx > pfClusteringVars.topH()) ? pfClusteringVars[idx].wl_d() : -1;
        }
        alpaka::syncBlockThreads(acc); // all threads call sync
      }
    }
  };

  /* link all vertices to sink */
  class ECLCCFlatten {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                  reco::PFClusteringVarsDeviceCollection::View pfClusteringVars,
                                  reco::PFClusteringEdgeVarsDeviceCollection::View pfClusteringEdgeVars) const {
      const int nRH = pfRecHits.size();

      for (int v : cms::alpakatools::elements_with_stride(acc, nRH)) {
        int next, vstat = pfClusteringVars[v].pfrh_topoId();
        const int old = vstat;
        while (vstat > (next = pfClusteringVars[vstat].pfrh_topoId())) {
          vstat = next;
        }
        if (old != vstat)
          pfClusteringVars[v].pfrh_topoId() = vstat;
      }
    }
  };

  // ECL-CC ends

} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
