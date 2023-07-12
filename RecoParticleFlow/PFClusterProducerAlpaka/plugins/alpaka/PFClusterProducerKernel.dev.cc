#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterProducerKernel.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/plugins/AlpakaPFCommon.h"

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
  static const int ThreadsPerBlock = 256;
  static const int threadsPerBlockForClustering = 512;
  static const int warpsize = 32;

  typedef struct float3 {
      float x;
      float y;
      float z;
  } float3;

  ALPAKA_FN_ACC float3 make_float3(float x, float y, float z) {
      float3 tmp;
      tmp.x = x;
      tmp.y = y;
      tmp.z = z;
      return tmp;
  }

  typedef struct float4 {
      float x;
      float y;
      float z;
      float w;
  } float4;

  ALPAKA_FN_ACC float4 make_float4(float x, float y, float z, float w) {
      float4 tmp;
      tmp.x = x;
      tmp.y = y;
      tmp.z = z;
      tmp.w = w;
      return tmp;
  }


  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void ECLCC_init(const TAcc& acc,
                             const int nodes,
                             tmpPFDeviceCollection::View tmpPF) {
    //const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int from = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] + alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] * ThreadsPerBlock;
    //const int incr = gridDim.x * ThreadsPerBlock;
    const int incr = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u] * ThreadsPerBlock;

    for (int v = from; v < nodes; v += incr) {
      const int beg = tmpPF[v].pfrh_edgeIdx();
      const int end = tmpPF[v + 1].pfrh_edgeIdx();
      int m = v;
      int i = beg;
      while ((m == v) && (i < end)) {
        m = std::min(m, tmpPF[i].pfrh_edgeList());
        i++;
      }
      tmpPF[v].pfrh_topoId() = m;
    }

    if (from == 0) {
      tmpPF.topL() = 0;
      tmpPF.posL() = 0;
      tmpPF.topH() = nodes - 1;
      tmpPF.posH() = nodes - 1;
    }
  }

  /* intermediate pointer jumping */

  int representative(const int idx, int* const __restrict__ nstat) {
    int curr = nstat[idx];
    if (curr != idx) {
      int next, prev = idx;
      while (curr > (next = nstat[curr])) {
        nstat[prev] = next;
        prev = curr;
        curr = next;
      }
    }
    return curr;
  }

  ALPAKA_FN_ACC void ECLCC_compute1(const int nodes,
                                 const int* const __restrict__ nidx,
                                 const int* const __restrict__ nlist,
                                 int* const __restrict__ nstat,
                                 int* const __restrict__ wl,
                                 int* topL,
                                 int* topH) {
    const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int incr = gridDim.x * ThreadsPerBlock;

    for (int v = from; v < nodes; v += incr) {
      const int vstat = nstat[v];
      if (v != vstat) {
        const int beg = nidx[v];
        const int end = nidx[v + 1];
        int deg = end - beg;
        if (deg > 16) {
          int idx;
          if (deg <= 352) {
            idx = atomicAdd(&*topL, 1);
          } else {
            idx = atomicAdd(&*topH, -1);
          }
          wl[idx] = v;
        } else {
          int vstat = representative(v, nstat);
          for (int i = beg; i < end; i++) {
            const int nli = nlist[i];
            if (v > nli) {
              int ostat = representative(nli, nstat);
              bool repeat;
              do {
                repeat = false;
                if (vstat != ostat) {
                  int ret;
                  if (vstat < ostat) {
                    if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                      ostat = ret;
                      repeat = true;
                    }
                  } else {
                    if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
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

  float timeResolution2Endcap(PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                         const float energy) {
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

  float timeResolution2Barrel(PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                        const float energy) {
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

  float dR2(float4 pos1, float4 pos2) {
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

  //static float atomicMaxF(const TAcc& acc, float* address, float val) {
  //  int ret = __float_as_int(*address);
  //  while (val > __int_as_float(ret)) {
  //    int old = ret;
  //    if ((ret = alpaka::atomicCas(acc, (int*)address, old, __float_as_int(val))) == old)
  //      break;
  //  }
  //  return __int_as_float(ret);
  //}
  
  auto dev_getSeedRhIdx(int* seeds, int seedNum) { return seeds[seedNum]; }

  auto dev_getRhFracIdx(int* rechits, int rhNum) {
    if (rhNum <= 0) {
      printf("Invalid rhNum (%d) for get RhFracIdx!\n", rhNum);
    }
    return rechits[rhNum - 1];
  }

  //auto dev_getRhFrac(
  //    int* topoSeedList, int topoSeedBegin, PFClusterDeviceCollection::View<1> fracView, int* seedFracOffsets, int seedNum, int rhNum) {
  //  int seedIdx = topoSeedList[topoSeedBegin + seedNum];
  //  return fracView[seedFracOffsets[seedIdx] + rhNum].pcrh_frac();

  //}

  auto dev_computeClusterPos(PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                        float4& pos4,
                                        float frac,
                                        int rhInd,
                                        bool isDebug,
                                        const float* __restrict__ pfrh_x,
                                        const float* __restrict__ pfrh_y,
                                        const float* __restrict__ pfrh_z,
                                        const float* __restrict__ pfrh_energy,
                                        float rhENormInv) {
    float4 rechitPos;
    rechitPos.x = pfrh_x[rhInd];
    rechitPos.y = pfrh_y[rhInd];
    rechitPos.z = pfrh_z[rhInd];
    rechitPos.w = 1.0;
    const auto rh_energy = pfrh_energy[rhInd] * frac;
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

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
  ALPAKA_FN_ACC void seedingTopoThreshKernel_HCAL(const TAcc& acc,
                                               PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                               tmpPFDeviceCollection::View tmpPF,
                                               reco::PFRecHitHostCollection::ConstView pfRecHits,
                                               size_t size) {
    //int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int i = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u] + alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u] * alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
    printf("Begin Kernel, %d, detid: %d\n", i, pfRecHits[i].detId());
    if (i < size) {
    //for (unsigned int i = 0; i < 1; i++) {
      // Initialize arrays
      tmpPF[i].pfrh_topoId() = i;
      tmpPF[i].pfrh_isSeed() = 0;
      tmpPF[i].rhCount() = 0;
      tmpPF[i].topoSeedCount() = 0;
      tmpPF[i].topoRHCount() = 0;
      tmpPF[i].seedFracOffsets() = -1;
      tmpPF[i].topoSeedOffsets() = -1;
      tmpPF[i].topoSeedList() = -1;
      tmpPF[i].pfc_iter() = -1;

      int layer = pfRecHits[i].layer();
      int depthOffset = pfRecHits[i].depth() - 1;
      float energy = pfRecHits[i].energy();
      float3 pos = make_float3(pfRecHits[i].x(), pfRecHits[i].y(), pfRecHits[i].z());

      // cmssdt.cern.ch/lxr/source/DataFormats/ParticleFlowReco/interface/PFRecHit.h#0108
      float pt2 = energy * energy * (pos.x * pos.x + pos.y * pos.y) / (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);

      // Seeding threshold test
      if ((layer == PFLayer::HCAL_BARREL1 && energy > pfClusParams.seedEThresholdEB_vec()[depthOffset] &&
           pt2 > pfClusParams.seedPt2ThresholdEB()) ||
          (layer == PFLayer::HCAL_ENDCAP && energy > pfClusParams.seedEThresholdEE_vec()[depthOffset] &&
           pt2 > pfClusParams.seedPt2ThresholdEE())) {
        tmpPF[i].pfrh_isSeed() = 1;
        for (unsigned int k = 0; k < pfRecHits[i].num_neighbours(); k++) {  // Does this seed candidate have a higher energy than four neighbours
          printf("index of neighbor: %d\n", pfRecHits[i].neighbours()(k));
          //if (pfRecHits[8 * i + k].neighbours() < 0)
          //if (pfRecHits[i].neighbours()(k) < 0)
            //continue;
          if (energy < pfRecHits[pfRecHits[i].neighbours()(k)].energy()) {
            tmpPF[i].pfrh_isSeed() = 0;
            //pfrh_topoId[i]=-1;
          //  break;
          }
        }
        if (tmpPF[i].pfrh_isSeed()) {
          //atomicAdd(&*nSeeds, 1);
          alpaka::atomicAdd(acc, &tmpPF.nSeeds(), 1, alpaka::hierarchy::Blocks{});
        //         for(int k=0; k<pfClusParams.nNeigh(); k++){
        //           if(neigh4_Ind[pfClusParams.nNeigh()*i+k]<0) continue;
        //           if(energy < pfrh_energy[neigh4_Ind[pfClusParams.nNeigh()*i+k]]){
        //             pfrh_isSeed[i]=0;
        //             //pfrh_topoId[i]=-1;
        //             break;
        //           }
        //       }
      } else {
        // pfrh_topoId[i]=-1;
        tmpPF[i].pfrh_isSeed() = 0;
      }

      // Topo clustering threshold test
      if ((layer == PFLayer::HCAL_ENDCAP && energy > pfClusParams.topoEThresholdEE_vec()[depthOffset]) ||
          (layer == PFLayer::HCAL_BARREL1 && energy > pfClusParams.topoEThresholdEB_vec()[depthOffset])) {
        tmpPF[i].pfrh_passTopoThresh() = true;
      }
      //else { pfrh_passTopoThresh[i] = false; }
      else {
        tmpPF[i].pfrh_passTopoThresh() = false;
        tmpPF[i].pfrh_topoId() = -1;
      }
    printf("end loop\n");
      }
    }
  }

  template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>> 
  ALPAKA_FN_ACC void dev_hcalFastCluster_optimizedSimple(const TAcc& acc,
                                                         PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                                         int topoId,
                                                         int nRHTopo,
                                                         const float* __restrict__ pfrh_x,
                                                         const float* __restrict__ pfrh_y,
                                                         const float* __restrict__ pfrh_z,
                                                         const float* __restrict__ pfrh_energy,
                                                         const int* __restrict__ pfrh_layer,
                                                         const int* __restrict__ pfrh_depth,
                                                         int* topoSeedOffsets,
                                                         int* topoSeedList,
                                                         int* seedFracOffsets,
                                                         int* pfcIter,
                                                         int* rhIdxToSeedIdx,
                                                         PFClusterDeviceCollection2::View<0> clusterView
                                                         //PFClusterDeviceCollection::View<1> fracView
                                                         ) {
    int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];  // thread index is rechit number
    int i, nRHOther = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    unsigned int iter = alpaka::declareSharedVar<int, __COUNTER__>(acc);
    float tol, clusterEnergy, rhENormInv, seedEnergy = alpaka::declareSharedVar<float, __COUNTER__>(acc);
    float4 clusterPos, prevClusterPos, seedPos = alpaka::declareSharedVar<float4, __COUNTER__>(acc);
    bool notDone, debug = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
    if (tid == 0) {
      i = topoSeedList[topoSeedOffsets[topoId]];  // i is the seed rechit index
      nRHOther = nRHTopo - 1;
      seedPos = make_float4(pfrh_x[i], pfrh_y[i], pfrh_z[i], 1.);
      clusterPos = seedPos;
      prevClusterPos = seedPos;
      seedEnergy = pfrh_energy[i];
      clusterEnergy = seedEnergy;
      tol = pfClusParams.stoppingTolerance();  // stopping tolerance * tolerance scaling

      if (pfrh_layer[i] == PFLayer::HCAL_BARREL1)
        rhENormInv = pfClusParams.recHitEnergyNormInvEB_vec()[pfrh_depth[i] - 1];
      else if (pfrh_layer[i] == PFLayer::HCAL_ENDCAP)
        rhENormInv = pfClusParams.recHitEnergyNormInvEE_vec()[pfrh_depth[i] - 1];
      else {
        rhENormInv = 0.;
        printf("Rechit %d has invalid layer %d!\n", i, pfrh_layer[i]);
      }

      iter = 0;
      notDone = true;
      debug = false;
      //debug = (topoId == 432 || topoId == 438 || topoId == 439) ? true : false;
    }
    alpaka::syncBlockThreads(acc);

    int j = -1;  // j is the rechit index for this thread
    int rhFracOffset = -1;
    float4 rhPos;
    float rhEnergy = -1., rhPosNorm = -1.;

    if (tid < nRHOther) {
      rhFracOffset = seedFracOffsets[i] + tid + 1;  // Offset for this rechit in pcrhfrac, pcrhfracidx arrays
      //j = fracView[rhFracOffset].pcrh_pfrhIdx();                // rechit index for this thread
      j = 0;
      rhPos = make_float4(pfrh_x[j], pfrh_y[j], pfrh_z[j], 1.);
      rhEnergy = pfrh_energy[j];
      rhPosNorm = fmaxf(0., logf(rhEnergy * rhENormInv));
    }
    alpaka::syncBlockThreads(acc);

    do {
      if (debug && tid == 0) {
        printf("\n--- Now on iter %d for topoId %d ---\n", iter, topoId);
      }
      float dist2 = -1., d2 = -1., fraction = -1.;
      if (tid < nRHOther) {
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
        //fracView[rhFracOffset].pcrh_frac() = fraction;
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
        //computeClusterPos(pfClusParams, clusterPos, rechitPos, rhEnergy, rhENormInv, debug);
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
    if (tid == 0) {
      int rhIdx = topoSeedList[topoSeedOffsets[topoId]];  // i is the seed rechit index
      int seedIdx = rhIdxToSeedIdx[rhIdx];
      pfcIter[topoId] = iter;
      clusterView[seedIdx].pfc_energy() = clusterEnergy;
      clusterView[seedIdx].pfc_x() = clusterPos.x;
      clusterView[seedIdx].pfc_y() = clusterPos.y;
      clusterView[seedIdx].pfc_z() = clusterPos.z;
    }
  }

  class seedingTopoThreshKernel {
      public:
        template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      tmpPFDeviceCollection::View tmpPF,
                                      const PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                      const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                      PFClusterDeviceCollection2::View<0> clusterView
                                      ) const {
            const int nRH = pfRecHits.size();
            seedingTopoThreshKernel_HCAL(acc, pfClusParams, tmpPF, pfRecHits, nRH);

        }
  };

  class eclccKernel {
      public:
        template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      const reco::PFRecHitHostCollection::ConstView pfRecHits,
                                      tmpPFDeviceCollection::View tmpPF
                                      ) const {
            const int nRH = pfRecHits.size();
            ECLCC_init(acc, nRH, tmpPF);

        }
  };


  void PFClusterProducerKernel::execute(const Device&,
                                        Queue& queue,
                                        const PFClusterParamsAlpakaESDataDevice& params,
                                        tmpPFDeviceCollection& tmp,
                                        const reco::PFRecHitHostCollection& pfRecHits,
                                        PFClusterDeviceCollection2& pfClusters) {
      
      const int nRH = pfRecHits->size();
      const int threadsPerBlock = 256;
      const int blocks = (nRH + threadsPerBlock - 1) / threadsPerBlock;
      // NEED CONDITIONAL WORKDIV FOR SERIAL
      //alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>((nRH + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock), PFClusterProducerKernelImpl{}, tmp.view(), params.view(), pfRecHits.view(), pfClusters.view());
      alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(blocks, threadsPerBlock), seedingTopoThreshKernel{}, tmp.view(), params.view(), pfRecHits.view(), pfClusters.view<0>());
      alpaka::exec<Acc1D>(queue, make_workdiv<Acc1D>(blocks, threadsPerBlock), eclccKernel{}, pfRecHits.view(), tmp.view());
  }



} // namespace
