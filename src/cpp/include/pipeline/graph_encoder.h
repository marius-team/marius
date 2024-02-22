//
// Created by Jason Mohoney on 1/21/22.
//

#ifndef MARIUS_GRAPH_ENCODER_H
#define MARIUS_GRAPH_ENCODER_H

#include "data/dataloader.h"
#include "pipeline_cpu.h"
#include "pipeline_gpu.h"

class GraphEncoder {
   public:
    shared_ptr<DataLoader> dataloader_;
    shared_ptr<ProgressReporter> progress_reporter_;

    virtual ~GraphEncoder(){};
    /**
      Encodes all of the nodes in the graph
      @param seperate_layers. If true, all the nodes at each layer will be encoded before moving onto the next layer.
    */
    virtual void encode(bool separate_layers = false) = 0;
};

class PipelineGraphEncoder : public GraphEncoder {
    shared_ptr<Pipeline> pipeline_;

   public:
    PipelineGraphEncoder(shared_ptr<DataLoader> sampler, std::shared_ptr<Model> model, shared_ptr<PipelineConfig> pipeline_config, int logs_per_epoch = 10);

    void encode(bool separate_layers = false) override;
};

class SynchronousGraphEncoder : public GraphEncoder {
    std::shared_ptr<Model> model_;

   public:
    SynchronousGraphEncoder(shared_ptr<DataLoader> sampler, std::shared_ptr<Model> model, int logs_per_epoch = 10);

    void encode(bool separate_layers = false) override;
};

#endif  // MARIUS_GRAPH_ENCODER_H
