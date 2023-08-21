#include "DataFormats/PortableTestObjects/interface/alpaka/TestDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/alpaka/PFClusterParamsAlpakaESData.h"
#include "RecoParticleFlow/PFClusterProducerAlpaka/interface/JobConfigurationAlpakaRecord2.h"

#include "TestAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TestPFClusterTestProducer : public stream::EDProducer<> {
  public:
    TestPFClusterTestProducer(edm::ParameterSet const& config)
        : esParamsToken_{esConsumes(config.getParameter<edm::ESInputTag>("pfClusterParams"))} {
      devicePutToken_ = produces("");
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto const& esParams = iSetup.getData(esParamsToken_);

      auto deviceProduct = std::make_unique<portabletest::TestDeviceCollection>(256, iEvent.queue());

      algo_.printPFClusterESData(iEvent.queue(), esParams);

      iEvent.put(devicePutToken_, std::move(deviceProduct));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::ESInputTag>("pfClusterParams", edm::ESInputTag("pfClusterParamsESProducer", ""));
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    device::ESGetToken<PFClusterParamsAlpakaESDataDevice, JobConfigurationAlpakaRecord2> const esParamsToken_;
    device::EDPutToken<portabletest::TestDeviceCollection> devicePutToken_;

    TestAlgo algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestPFClusterTestProducer);
