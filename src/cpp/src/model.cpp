//
// Created by Jason Mohoney on 2/12/21.
//

#include <model.h>

Model::Model(Encoder *encoder, Decoder* decoder) {
    encoder_ = encoder;
    decoder_ = decoder;
}

void Model::train(Batch *batch) {
    Timer encoder_time = Timer(false);
    Timer decoder_time = Timer(false);

    encoder_time.start();
    encoder_->forward(batch, true);
    encoder_time.stop();
    SPDLOG_INFO("Encoder Took: {}", encoder_time.getDuration());

    decoder_time.start();
    decoder_->forward(batch, true);
    decoder_time.stop();
    SPDLOG_INFO("Decoder Took: {}", decoder_time.getDuration());
}

void Model::evaluate(Batch *batch) {
    encoder_->forward(batch, false);
    decoder_->forward(batch, false);
}

Model *initializeModel(EncoderModelType encoder_model_type, DecoderModelType decoder_model_type){
    Encoder *encoder;
    if (encoder_model_type == EncoderModelType::None) {
        encoder = new EmptyEncoder();
    } else {
        SPDLOG_ERROR("Encoding currently not supported.");
        exit(-1);
    }

    Decoder *decoder;
    if (decoder_model_type == DecoderModelType::NodeClassification) {
        decoder = new NodeClassificationDecoder();
    } else if (decoder_model_type == DecoderModelType::DistMult) {
        decoder = new DistMult();
    } else if (decoder_model_type == DecoderModelType::TransE) {
        decoder = new TransE();
    } else if (decoder_model_type == DecoderModelType::ComplEx) {
        decoder = new ComplEx();
    } else {
        SPDLOG_ERROR("Decoder currently not supported.");
        exit(-1);
    }

    return new Model(encoder, decoder);
}