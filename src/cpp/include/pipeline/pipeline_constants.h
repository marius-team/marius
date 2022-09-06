//
// Created by Jason Mohoney on 1/21/22.
//

#ifndef MARIUS_PIPELINE_CONSTANTS_H
#define MARIUS_PIPELINE_CONSTANTS_H

// CPU Pipeline worker IDs
#define LOAD_BATCH_ID 0
#define CPU_COMPUTE_ID 1
#define UPDATE_BATCH_ID 2

// GPU Pipeline worker IDs
#define H2D_TRANSFER_ID 3
#define GPU_COMPUTE_ID 4
#define D2H_TRANSFER_ID 5

// Encode Pipeline worker IDs
#define CPU_ENCODE_ID 6
#define GPU_ENCODE_ID 7
#define NODE_WRITE_ID 8

#define CPU_NUM_WORKER_TYPES 3
#define GPU_NUM_WORKER_TYPES 5

#define WAIT_TIME 100000  // 100 micro seconds
#define NANOSECOND 1
#define MICROSECOND 1000
#define MILLISECOND 1000000

#endif  // MARIUS_PIPELINE_CONSTANTS_H
