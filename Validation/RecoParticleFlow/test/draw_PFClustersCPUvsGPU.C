void draw_PFClustersCPUvsGPU() {
  TFile* f = new TFile("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root");
  TDirectoryFile* df = f->Get<TDirectoryFile>("DQMData")
                           ->Get<TDirectoryFile>("Run 1")
                           ->Get<TDirectoryFile>("ParticleFlow")
                           ->Get<TDirectoryFile>("Run summary")
                           ->Get<TDirectoryFile>("pfClusterHBHEGPUHLTv");

  TCanvas* c = new TCanvas();

  df->Get<TH2>("pfCluster_Energy_GPUvsCPU_")->Draw("COLZ");
  c->SaveAs("pfCluster_Energy_GPUvsCPU_.png");
  c->Clear();

  df->Get<TH2>("pfCluster_Multiplicity_GPUvsCPU_")->Draw("COLZ");
  c->SaveAs("pfCluster_Multiplicity_GPUvsCPU_.png");
  c->Clear();

  df->Get<TH2>("pfCluster_RecHitMultiplicity_GPUvsCPU_")->Draw("COLZ");
  c->SaveAs("pfCluster_RecHitMultiplicity_GPUvsCPU_.png");
}
