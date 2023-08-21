#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

class TestEmptyAnalyzer : public edm::global::EDAnalyzer<> {
public:
  TestEmptyAnalyzer(edm::ParameterSet const& config) : token_{consumes(config.getParameter<edm::InputTag>("source"))} {}

  void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const override {
    portabletest::TestHostCollection const& product = event.get(token_);
    edm::LogPrint("TestEmptyAnalyzer") << "size() = " << product.view().metadata().size();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  edm::EDGetTokenT<portabletest::TestHostCollection> const token_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestEmptyAnalyzer);
