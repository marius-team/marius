//
// Created by Jason Mohoney on 9/17/21.
//

#ifndef MARIUS_MODEL_HELPERS_H
#define MARIUS_MODEL_HELPERS_H

#include "model.h"
#include "nn/decoders/edge/complex.h"
#include "nn/decoders/edge/distmult.h"
#include "nn/decoders/edge/edge_decoder.h"
#include "nn/decoders/edge/transe.h"
#include "nn/decoders/node/noop_node_decoder.h"

std::shared_ptr<Decoder> decoder_clone_helper(std::shared_ptr<Decoder> decoder, torch::Device device) {
    return std::dynamic_pointer_cast<Decoder>(std::dynamic_pointer_cast<torch::nn::Module>(decoder)->clone(device));
}

std::shared_ptr<GeneralEncoder> encoder_clone_helper(std::shared_ptr<GeneralEncoder> encoder, torch::Device device) {
    return std::dynamic_pointer_cast<GeneralEncoder>(encoder->clone(device));
}

std::shared_ptr<Decoder> get_edge_decoder(DecoderType decoder_type, EdgeDecoderMethod edge_decoder_method, int num_relations, int embedding_dim,
                                          torch::TensorOptions tensor_options, bool use_inverse_relations) {
    shared_ptr<EdgeDecoder> decoder;

    if (decoder_type == DecoderType::DISTMULT) {
        decoder = std::make_shared<DistMult>(num_relations, embedding_dim, tensor_options, use_inverse_relations, edge_decoder_method);
    } else if (decoder_type == DecoderType::TRANSE) {
        decoder = std::make_shared<TransE>(num_relations, embedding_dim, tensor_options, use_inverse_relations, edge_decoder_method);
    } else if (decoder_type == DecoderType::COMPLEX) {
        decoder = std::make_shared<ComplEx>(num_relations, embedding_dim, tensor_options, use_inverse_relations, edge_decoder_method);
    } else {
        throw std::runtime_error("Decoder not supported for learning task.");
    }

    return decoder;
}

std::shared_ptr<Decoder> get_node_decoder(DecoderType decoder_type) {
    shared_ptr<NodeDecoder> decoder;

    if (decoder_type == DecoderType::NODE) {
        decoder = std::make_shared<NoOpNodeDecoder>();
    } else {
        throw std::runtime_error("Decoder not supported for learning task.");
    }

    return decoder;
}

#endif  // MARIUS_MODEL_HELPERS_H
