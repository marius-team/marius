//
// Created by Jason Mohoney on 10/7/21.
//

#ifndef MARIUS_ACTIVATION_H
#define MARIUS_ACTIVATION_H

#include "configuration/config.h"
#include "datatypes.h"

torch::Tensor apply_activation(ActivationFunction activation_function, torch::Tensor input);

#endif //MARIUS_ACTIVATION_H
