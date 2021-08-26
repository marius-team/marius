//
// Created by Jason Mohoney on 2/12/21.
//

#include "model.h"

#include "logger.h"

Model::Model(Encoder *encoder, Decoder* decoder) {
    encoder_ = encoder;
    decoder_ = decoder;
}

void Model::train(Batch *batch) {
    encoder_->forward(batch, true);
    decoder_->forward(batch, true);
}

void Model::evaluate(Batch *batch) {
    encoder_->forward(batch, false);
    decoder_->forward(batch, false);
}

Model *initializeModel(EncoderModelType encoder_model_type, DecoderModelType decoder_model_type){

    SPDLOG_DEBUG("Initializing Model");
    Encoder *encoder;
    if (encoder_model_type == EncoderModelType::None) {
        encoder = new EmptyEncoder();
        SPDLOG_DEBUG("Empty Encoder");
    } else {
        SPDLOG_ERROR("Encoding currently not supported.");
        exit(-1);
    }

    Decoder *decoder;
    if (decoder_model_type == DecoderModelType::NodeClassification) {
        decoder = new NodeClassificationDecoder();
    } else if (decoder_model_type == DecoderModelType::DistMult) {
        decoder = new DistMult();
        SPDLOG_DEBUG("DistMult Decoder");
    } else if (decoder_model_type == DecoderModelType::TransE) {
        decoder = new TransE();
        SPDLOG_DEBUG("TransE Decoder");
    } else if (decoder_model_type == DecoderModelType::ComplEx) {
        decoder = new ComplEx();
        SPDLOG_DEBUG("ComplEx Decoder");
    } else {
        SPDLOG_ERROR("Decoder currently not supported.");
        exit(-1);
    }

    return new Model(encoder, decoder);
}