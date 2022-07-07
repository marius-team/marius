//
// Created by Jason Mohoney on 10/7/21.
//

#ifndef MARIUS_ACTIVATION_H
#define MARIUS_ACTIVATION_H

#include "common/datatypes.h"
#include "configuration/config.h"

torch::Tensor apply_activation(ActivationFunction activation_function, torch::Tensor input);

#endif  // MARIUS_ACTIVATION_H
