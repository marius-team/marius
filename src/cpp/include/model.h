//
// Created by Jason Mohoney on 2/11/21.
//

#ifndef MARIUS_INCLUDE_MODEL_H_
#define MARIUS_INCLUDE_MODEL_H_

#include "decoder.h"
#include "encoder.h"

class Model {
  public:
    Encoder *encoder_;
    Decoder *decoder_;

    Model(Encoder *encoder, Decoder *decoder);

    void train(Batch *batch);

    void evaluate(Batch *batch);
};

Model *initializeModel(EncoderModelType encoder_model_type, DecoderModelType decoder_model_type);

#endif //MARIUS_INCLUDE_MODEL_H_
