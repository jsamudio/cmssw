// Check that ALPAKA_HOST_ONLY is not defined during device compilation:
#ifdef ALPAKA_HOST_ONLY
#error ALPAKA_HOST_ONLY defined in device compilation
#endif

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class PrintPFClusterESDataKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
      PFClusterParamsAlpakaESDataDevice::ConstView params) const {
      // global index of the thread within the grid
      const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

      if (thread == 0) {
        auto const seedEThresholdEB = params.seedEThresholdEB_vec();
        for (uint32_t idx = 0; idx < 4; ++idx)
          printf("PFCluster Seeding Threshold EB[depth=%d] = %8.4f\n", 1+idx, seedEThresholdEB[idx]);
      }
    }
  };

  void TestAlgo::printPFClusterESData(Queue& queue,
    PFClusterParamsAlpakaESDataDevice const& esParams) const {
    auto workDiv = make_workdiv<Acc1D>(1,1);
    alpaka::exec<Acc1D>(queue, workDiv, PrintPFClusterESDataKernel{}, esParams.const_view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
