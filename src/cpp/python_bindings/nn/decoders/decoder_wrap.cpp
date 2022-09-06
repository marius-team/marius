//
// Created by Jason Mohoney on 9/30/21.
//

#include "common/pybind_headers.h"
#include "nn/decoders/decoder.h"

class PyDecoder : Decoder {
    using Decoder::Decoder;
};

void init_decoder(py::module &m) { py::class_<Decoder, PyDecoder, shared_ptr<Decoder>>(m, "Decoder").def_readwrite("learning_task", &Decoder::learning_task_); }