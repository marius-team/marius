//
// Created by Jason Mohoney on 9/29/21.
//

#ifndef MARIUS_DECODER_H
#define MARIUS_DECODER_H

#include <configuration/options.h>

#include "common/datatypes.h"

class Decoder {
   public:
    LearningTask learning_task_;

    virtual ~Decoder(){};
};

#endif  // MARIUS_DECODER_H
