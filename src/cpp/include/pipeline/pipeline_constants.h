//
// Created by Jason Mohoney on 1/21/22.
//

#ifndef MARIUS_PIPELINE_CONSTANTS_H
#define MARIUS_PIPELINE_CONSTANTS_H

// CPU Pipeline worker IDs
#define LOAD_BATCH_ID 0
#define CPU_COMPUTE_ID 1
#define CPU_ACCUMULATE_ID 2
#define UPDATE_BATCH_ID 3

// GPU Pipeline worker IDs
#define H2D_TRANSFER_ID 4
#define GPU_COMPUTE_ID 5
#define D2H_TRANSFER_ID 6

// Encode Pipeline worker IDs
#define CPU_ENCODE_ID 7
#define GPU_ENCODE_ID 8
#define NODE_WRITE_ID 9

#define CPU_NUM_WORKER_TYPES 4
#define GPU_NUM_WORKER_TYPES 5

#define WAIT_TIME 100000 // 100 micro seconds
#define NANOSECOND 1
#define MICROSECOND 1000
#define MILLISECOND 1000000

#endif //MARIUS_PIPELINE_CONSTANTS_H
