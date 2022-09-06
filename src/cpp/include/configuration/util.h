//
// Created by Jason Mohoney on 1/19/22.
//

#ifndef MARIUS_CONFIGURATION_UTIL_H
#define MARIUS_CONFIGURATION_UTIL_H

#include "config.h"

std::vector<torch::Device> devices_from_config(std::shared_ptr<StorageConfig> storage_config);

#endif  // MARIUS_CONFIGURATION_UTIL_H
