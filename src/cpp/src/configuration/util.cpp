//
// Created by Jason Mohoney on 1/19/22.
//

#include "configuration/util.h"

std::vector<torch::Device> devices_from_config(std::shared_ptr<StorageConfig> storage_config) {
    std::vector<torch::Device> devices;

    if (storage_config->device_type == torch::kCUDA) {
        for (int i = 0; i < storage_config->device_ids.size(); i++) {
            devices.emplace_back(torch::Device(torch::kCUDA, storage_config->device_ids[i]));
        }
        if (devices.empty()) {
            devices.emplace_back(torch::Device(torch::kCUDA, 0));
        }
    } else {
        devices.emplace_back(torch::kCPU);
    }

    return devices;
}
