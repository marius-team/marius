//
// Created by Jason Mohoney on 2/11/21.
//

#ifndef MARIUS_INCLUDE_ENCODER_H_
#define MARIUS_INCLUDE_ENCODER_H_
#include <batch.h>

// Encoder Models
class Encoder {
  public:
    virtual ~Encoder() { };
    virtual void forward(Batch *batch, bool train) = 0;
};

class EmptyEncoder : public Encoder {
  public:
    EmptyEncoder();

    void forward(Batch *batch, bool train) override;
};
#endif //MARIUS_INCLUDE_ENCODER_H_
