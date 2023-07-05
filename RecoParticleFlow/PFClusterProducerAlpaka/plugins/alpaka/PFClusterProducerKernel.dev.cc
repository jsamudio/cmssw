#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterProducerKernel.h"

//using PFClustering::common::PFLayer;

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

  constexpr const float PI_F = 3.141592654f;

  /* intermediate pointer jumping */

  int representative(const int idx, int* const __restrict__ nstat) {
    int curr = nstat[idx];
    if (curr != idx) {
      int next = idx;
      int prev = idx;
      while (curr > (next = nstat[curr])) {
        nstat[prev] = next;
        prev = curr;
        curr = next;
      }
    }
    return curr;
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

  auto dev_getRhFrac(
      int* topoSeedList, int topoSeedBegin, PFClusterDeviceCollection::View<1> fracView, int* seedFracOffsets, int seedNum, int rhNum) {
    int seedIdx = topoSeedList[topoSeedBegin + seedNum];
    return fracView[seedFracOffsets[seedIdx] + rhNum].pcrh_frac();

  }

  auto dev_computeClusterPos(PFClusterParamsAlpakaESDataDevice::ConstView pfClusParams,
                                        float pos4x,
                                        float pos4y,
                                        float pos4z,
                                        float pos4w,
                                        float frac,
                                        int rhInd,
                                        bool isDebug,
                                        const float* __restrict__ pfrh_x,
                                        const float* __restrict__ pfrh_y,
                                        const float* __restrict__ pfrh_z,
                                        const float* __restrict__ pfrh_energy,
                                        float rhENormInv) {
    float rechitPosx = pfrh_x[rhInd];
    float rechitPosy = pfrh_y[rhInd];
    float rechitPosz = pfrh_z[rhInd];
    float rechitPosw = 1.0;
    const auto rh_energy = pfrh_energy[rhInd] * frac;
    const auto norm = (frac < pfClusParams.minFracInCalc() ? 0.0f : std::max(0.0f, logf(rh_energy * rhENormInv)));
    if (isDebug)
      printf("\t\t\trechit %d: norm = %f\tfrac = %f\trh_energy = %f\tpos = (%f, %f, %f)\n",
             rhInd,
             norm,
             frac,
             rh_energy,
             rechitPosx,
             rechitPosy,
             rechitPosz);

    pos4x += rechitPosx * norm;
    pos4y += rechitPosy * norm;
    pos4z += rechitPosz * norm;
    pos4w += norm;  //  position_norm
  }
} // namespace
