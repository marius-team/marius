//
// Created by Jason Mohoney on 2/4/22.
//

#ifndef MARIUS_EXCEPTION_H
#define MARIUS_EXCEPTION_H

#include <exception>

#include "torch/torch.h"

struct MariusRuntimeException : public std::runtime_error {
   public:
    MariusRuntimeException(const std::string &message) : runtime_error(message) {}
};

struct UndefinedTensorException : public MariusRuntimeException {
   public:
    UndefinedTensorException() : MariusRuntimeException("Tensor undefined") {}
};

struct NANTensorException : public MariusRuntimeException {
   public:
    NANTensorException() : MariusRuntimeException("Tensor contains NANs") {}
};

struct OOMTensorException : public MariusRuntimeException {
   public:
    OOMTensorException() : MariusRuntimeException("Tensor results in OOM") {}
};

struct TensorSizeMismatchException : public MariusRuntimeException {
   public:
    //    TensorSizeMismatchException(torch::Tensor input, std::string message) : MariusRuntimeException((std::stringstream("Tensor size mismatch. Size: ") <<
    //    input.sizes() << " " << message).str()) {}
    TensorSizeMismatchException(torch::Tensor input, std::string message) : MariusRuntimeException(message) {}
};

struct UnexpectedNullPtrException : public MariusRuntimeException {
   public:
    UnexpectedNullPtrException(std::string message = "") : MariusRuntimeException(message) {}
};

#endif  // MARIUS_EXCEPTION_H
