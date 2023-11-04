//
// Created by Jason Mohoney on 7/17/20.
//
#ifdef MARIUS_OMP
    #include "omp.h"
#endif

#include "common/datatypes.h"
#include "data/ordering.h"
#include "reporting/logger.h"

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getEdgeBucketOrdering(EdgeBucketOrdering edge_bucket_ordering, int num_partitions, int buffer_capacity,
                                                                               int fine_to_coarse_ratio, int num_cache_partitions,
                                                                               bool randomly_assign_edge_buckets,
                                                                               shared_ptr<c10d::ProcessGroupGloo> pg_gloo,
                                                                               shared_ptr<DistributedConfig> dist_config) {
    switch (edge_bucket_ordering) {
        case EdgeBucketOrdering::OLD_BETA:
            SPDLOG_INFO("Generating Old Beta Ordering");
            return getTwoLevelBetaOrdering(num_partitions, buffer_capacity, 1, 0, false);
        case EdgeBucketOrdering::NEW_BETA:
            SPDLOG_INFO("Generating New Beta Ordering");
            return getTwoLevelBetaOrdering(num_partitions, buffer_capacity, 1, 0, true);
        case EdgeBucketOrdering::ALL_BETA:
            return getCustomEdgeBucketOrdering();
        case EdgeBucketOrdering::COMET:
            SPDLOG_INFO("Generating COMET Ordering");
            return getTwoLevelBetaOrdering(num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions, randomly_assign_edge_buckets);
        case EdgeBucketOrdering::DIAG:
            SPDLOG_INFO("Generating Diag Ordering");
            return getDiagOrdering(num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions, randomly_assign_edge_buckets,
                                   torch::Tensor(), -1, pg_gloo, dist_config);
        case EdgeBucketOrdering::CUSTOM:
            return getCustomEdgeBucketOrdering();
        default:
            SPDLOG_ERROR("Not implemented");
            std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
            return ret;
    }
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getNodePartitionOrdering(NodePartitionOrdering node_partition_ordering, Indices train_nodes,
                                                                                  int64_t total_num_nodes, int num_partitions, int buffer_capacity,
                                                                                  int fine_to_coarse_ratio, int num_cache_partitions,
                                                                                  shared_ptr<c10d::ProcessGroupGloo> pg_gloo,
                                                                                  shared_ptr<DistributedConfig> dist_config) {
    switch (node_partition_ordering) {
        case NodePartitionOrdering::DISPERSED:
            SPDLOG_INFO("Generating Dispersed Ordering");
//            return getDispersedNodePartitionOrdering(train_nodes, total_num_nodes, num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions);
            return getDiagOrdering(num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions, true,
                                   train_nodes, total_num_nodes, pg_gloo, dist_config);
        case NodePartitionOrdering::SEQUENTIAL:
            SPDLOG_INFO("Generating Sequential Ordering");
            return getSequentialNodePartitionOrdering(train_nodes, total_num_nodes, num_partitions, buffer_capacity);
        case NodePartitionOrdering::DIAG:
            SPDLOG_INFO("Generating Diag Ordering");
            return getDiagOrdering(num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions, true,
                                   train_nodes, total_num_nodes, pg_gloo, dist_config);
        case NodePartitionOrdering::DIST_SEQUENTIAL:
            SPDLOG_INFO("Generating Distributed Sequential Ordering");
            return getDiagOrdering(num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions, true,
                                   train_nodes, total_num_nodes, pg_gloo, dist_config, true, false);
        case NodePartitionOrdering::DIST_IN_MEMORY:
            SPDLOG_INFO("Generating Distributed In Memory Ordering");
            return getDiagOrdering(num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions, true,
                                   train_nodes, total_num_nodes, pg_gloo, dist_config, false, true);
        case NodePartitionOrdering::CUSTOM:
            return getCustomNodePartitionOrdering();
        default:
            SPDLOG_ERROR("Not implemented");
            std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
            return ret;
    }
}

void printBuffers(vector<vector<int>> buffer_states, bool coarse) {
    if (coarse) {
        std::cout<<"Logical Buffer:\n";
    } else {
        std::cout<<"Physical Buffer:\n";
    }
    for (auto b : buffer_states) {
        std::cout<<b<<"\n";
    }
    std::cout<<"\n";
}

void printEdgeBuckets(vector<vector<std::pair<int, int>>> edge_buckets_per_buffer) {
int total = 0;
    std::cout<<"Edge Buckets:\n";
    for (auto edge_buckets : edge_buckets_per_buffer) {
        std::cout<<"[";
        for (int i = 0; i < edge_buckets.size(); i++) {
            std::cout<<"("<<std::get<0>(edge_buckets[i])<<","<<std::get<1>(edge_buckets[i])<<") ";
            total += 1;
        }
        std::cout<<"]\n";
    }
    std::cout<<"total: "<<total<<"\n";
}

void printTrainNodes(vector<torch::Tensor> train_nodes_per_buffer) {
    int total = 0;
    std::cout<<"Train nodes:\n";
    for (auto tensor : train_nodes_per_buffer) {
        total += tensor.size(0);
        std::cout<<tensor.size(0);
        std::cout<<"\n";
    }
    std::cout<<"total: "<<total<<"\n";
}

vector<torch::Tensor> convertToVectorOfTensors(vector<vector<int>> buffer_states, torch::TensorOptions opts) {
    vector<torch::Tensor> ret_buffer_states;

    for (auto b : buffer_states) {
        ret_buffer_states.emplace_back(torch::tensor(b, opts));
    }

    return ret_buffer_states;
}

vector<vector<int>> convertToVectorOfInts(vector<torch::Tensor> buffer_states) {
    vector<vector<int>> ret_buffer_states;

    for (int i = 0; i < buffer_states.size(); i++) {
        int *data_ptr_ = (int *)buffer_states[i].data_ptr();
        int *end = (int *)data_ptr_ + buffer_states[i].size(0);
        vector<int> partitions = vector<int>(data_ptr_, end);

        ret_buffer_states.emplace_back(partitions);
    }

    return ret_buffer_states;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> convertEdgeBucketOrderToTensors(vector<vector<int>> buffer_states,
                                                                                         vector<vector<std::pair<int, int>>> edge_buckets_per_buffer) {
    vector<torch::Tensor> ret_buffer_states;
    vector<torch::Tensor> ret_edge_buckets_per_buffer;

    for (auto b : buffer_states) {
        ret_buffer_states.emplace_back(torch::tensor(b, torch::kInt64));
    }

    for (auto edge_buckets : edge_buckets_per_buffer) {
        torch::Tensor tmp = torch::zeros({(int64_t)edge_buckets.size(), 2}, torch::kInt64);

        for (int i = 0; i < edge_buckets.size(); i++) {
            tmp[i][0] = std::get<0>(edge_buckets[i]);
            tmp[i][1] = std::get<1>(edge_buckets[i]);
        }

        ret_edge_buckets_per_buffer.emplace_back(tmp);
    }

    return std::forward_as_tuple(ret_buffer_states, ret_edge_buckets_per_buffer);
}

vector<vector<int>> getBetaOrderingHelper(int num_partitions, int buffer_capacity) {
    vector<vector<int>> buffer_states;
    Indices all_partitions = torch::randperm(num_partitions, torch::kInt32);

    // get all buffer states
    Indices in_buffer = all_partitions.index_select(0, torch::arange(buffer_capacity));

    Indices combined = torch::cat({all_partitions, in_buffer});
    auto uniques = torch::_unique2(combined, true, false, true);
    auto vals = std::get<0>(uniques);
    auto counts = std::get<2>(uniques);
    Indices on_disk = vals.masked_select(counts == 1);

    int *data_ptr_ = (int *)in_buffer.data_ptr();
    buffer_states.emplace_back(vector<int>(data_ptr_, data_ptr_ + in_buffer.size(0)));

    while (on_disk.size(0) >= 1) {
        in_buffer = in_buffer.index_select(0, torch::randperm(in_buffer.size(0), torch::kInt64));
        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));

        for (int i = 0; i < on_disk.size(0); i++) {
            auto admit_id = on_disk[i].clone();

            on_disk[i] = in_buffer[-1];

            in_buffer[-1] = admit_id;

            data_ptr_ = (int *)in_buffer.data_ptr();
            buffer_states.emplace_back(vector<int>(data_ptr_, data_ptr_ + in_buffer.size(0)));
        }

        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));

        int num_replaced = 0;
        for (int i = 0; i < buffer_capacity - 1; i++) {
            if (i >= on_disk.size(0)) {
                break;
            }
            num_replaced++;
            in_buffer[i] = on_disk[i];

            data_ptr_ = (int *)in_buffer.data_ptr();
            buffer_states.emplace_back(vector<int>(data_ptr_, data_ptr_ + in_buffer.size(0)));
        }
        on_disk = on_disk.narrow(0, num_replaced, on_disk.size(0) - num_replaced);
    }

    return buffer_states;
}

vector<vector<std::pair<int, int>>> greedyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions) {
    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer(buffer_states.size());
    torch::Tensor interacted = torch::zeros({num_partitions, num_partitions}, torch::kInt32);
    auto interacted_accessor = interacted.accessor<int32_t, 2>();

    for (int i = 0; i < buffer_states.size(); i++) {
        for (int j = 0; j < buffer_states[i].size(); j++) {
            for (int k = 0; k < buffer_states[i].size(); k++) {
                int32_t src_part = buffer_states[i][j];
                int32_t dst_part = buffer_states[i][k];
                if (interacted_accessor[src_part][dst_part] == 1) {
                    continue;
                }
                interacted_accessor[src_part][dst_part] = 1;
                edge_buckets_per_buffer[i].emplace_back(std::make_pair(src_part, dst_part));
            }
        }
    }

    return edge_buckets_per_buffer;
}

vector<vector<std::pair<int, int>>> randomlyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions) {
    // get edge buckets from buffer states
    Indices all_partitions = torch::arange(num_partitions, torch::kInt32);
    torch::Tensor left_col = all_partitions.repeat_interleave(num_partitions);
    torch::Tensor right_col = all_partitions.repeat({num_partitions});
    torch::Tensor all_buckets = torch::stack({left_col, right_col}, 1);
    auto all_buckets_accessor = all_buckets.accessor<int32_t, 2>();

    int num_buffers = buffer_states.size();
    int buffer_size = buffer_states[0].size();
    int num_buckets = all_buckets.size(0);

//    torch::Tensor choices = torch::zeros({num_buckets, num_buffers}, torch::kInt32);
//    int32_t *choices_mem = choices.data_ptr<int32_t>();
    torch::Tensor choices = torch::zeros({num_buckets, num_buffers}, torch::kInt8);
    int8_t *choices_mem = choices.data_ptr<int8_t>();

    #pragma omp parallel for
    for (int i = 0; i < num_buffers; i++) {
        for (int j = 0; j < buffer_size; j++) {
            for (int k = j; k < buffer_size; k++) {
                int src_part = buffer_states[i][j];
                int dst_part = buffer_states[i][k];
                *(choices_mem + (src_part * num_partitions + dst_part) * num_buffers + i) = 1;
                *(choices_mem + (dst_part * num_partitions + src_part) * num_buffers + i) = 1;
            }
        }
    }

    torch::Tensor pick = torch::zeros({num_buckets}, torch::kInt32);
    int32_t *pick_mem = pick.data_ptr<int32_t>();
    auto pick_accessor = pick.accessor<int32_t, 1>();
//    torch::Tensor pick_one_hot = torch::zeros({num_buckets, num_buffers}, torch::kInt32);
//    int32_t *pick_one_hot_mem = pick_one_hot.data_ptr<int32_t>();

    vector<int> num_edge_buckets_per_buffer(num_buffers);
    std::vector<std::mutex *> buffer_locks(num_buffers);
    for (int i = 0; i < buffer_locks.size(); i++) {
        buffer_locks[i] = new std::mutex();
    }

    // setup seeds
    unsigned int num_threads = 1;
    #ifdef MARIUS_OMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    #endif
    std::vector<unsigned int> tid_seeds(num_threads);

    for (int i = 0; i < num_threads; i++) {
        tid_seeds[i] = rand();
    }

    #pragma omp parallel
    {
        #ifdef MARIUS_OMP
        unsigned int seed = tid_seeds[omp_get_thread_num()];
        #else
        unsigned int seed = tid_seeds[0];
        #endif

        #pragma omp for
        for (int i = 0; i < num_buckets; i++) {
            torch::Tensor buffer_choices = torch::nonzero(choices[i]);
            buffer_choices = torch::reshape(buffer_choices, {buffer_choices.size(0)});

            int32_t buffer_choice = -1;
            if (buffer_choices.size(0) > 0) {
                buffer_choice = buffer_choices[rand_r(&seed) % buffer_choices.size(0)].item<int32_t>();
            }

            int32_t src_part = all_buckets_accessor[i][0];
            int32_t dst_part = all_buckets_accessor[i][1];
            *(pick_mem + (src_part * num_partitions + dst_part)) = buffer_choice;
            if (buffer_choice != -1) {
//                *(pick_one_hot_mem + (src_part * num_partitions + dst_part) * num_buffers + buffer_choice) = 1;
                buffer_locks[buffer_choice]->lock();
                ++num_edge_buckets_per_buffer[buffer_choice];
                buffer_locks[buffer_choice]->unlock();
            }
        }
    }

    choices = torch::Tensor();

//    torch::Tensor num_edge_buckets_per_buffer = torch::sum(pick_one_hot, 0);

    int num_edge_buckets_check = 0;
    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
//        edge_buckets_per_buffer[i] = vector<std::pair<int, int>>(num_edge_buckets_per_buffer[i].item<int>());
        edge_buckets_per_buffer[i] = vector<std::pair<int, int>>(num_edge_buckets_per_buffer[i]);
        num_edge_buckets_check += num_edge_buckets_per_buffer[i];
    }
    SPDLOG_INFO("Assigned {} edge buckets for training", num_edge_buckets_check);

    vector<int> indices(num_buffers, 0);
    for (int i = 0; i < num_buckets; i++) {
        int32_t buffer_choice = pick_accessor[i];

        if (buffer_choice == -1) continue;

        int32_t src_part = all_buckets_accessor[i][0];
        int32_t dst_part = all_buckets_accessor[i][1];
        std::pair<int, int> pair = std::make_pair(src_part, dst_part);

        edge_buckets_per_buffer[buffer_choice][indices[buffer_choice]] = pair;
        indices[buffer_choice] += 1;
    }

    return edge_buckets_per_buffer;
}

vector<vector<int>> convertToFinePartitions(vector<vector<int>> coarse_buffer_states, int num_partitions, int buffer_capacity,
                                            int fine_to_coarse_ratio, int num_cache_partitions) {
    // convert to fine partitions and add in cache (num_cache_partitions coarse mapped to fine partitions 0...x)
    // TODO: maybe num_cache partitions should just directly represent physical partitions, here and everywhere (e.g., diag, two level beta)?
    //  what are the implications for the buffer if any?

    // add in cache
    int cached_fine_partitions = num_cache_partitions * fine_to_coarse_ratio;
    torch::Tensor fine_to_coarse_map = torch::arange(cached_fine_partitions, torch::kInt32);
    fine_to_coarse_map = torch::cat({fine_to_coarse_map, torch::randperm(num_partitions - cached_fine_partitions, torch::kInt32) + cached_fine_partitions});
    int *data_ptr_ = (int *)fine_to_coarse_map.data_ptr();

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        for (int j = 0; j < coarse_buffer_states[i].size(); j++) {
            coarse_buffer_states[i][j] += num_cache_partitions;
        }
        for (int j = 0; j < num_cache_partitions; j++) {
            coarse_buffer_states[i].emplace_back(j);
        }
    }

    // convert to fine buffer states
    vector<vector<int>> buffer_states;

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        vector<int> fine_buffer_state(buffer_capacity, 0);
        for (int j = 0; j < coarse_buffer_states[i].size(); j++) {
            int *start = (int *)data_ptr_ + coarse_buffer_states[i][j] * fine_to_coarse_ratio;
            int *end = (int *)data_ptr_ + (coarse_buffer_states[i][j] + 1) * fine_to_coarse_ratio;
            vector<int> fine_partitions = vector<int>(start, end);

            for (int k = j * fine_to_coarse_ratio; k < (j + 1) * fine_to_coarse_ratio; k++) {
                fine_buffer_state[k] = fine_partitions[k - j * fine_to_coarse_ratio];
            }
        }

        buffer_states.emplace_back(fine_buffer_state);
    }

    return buffer_states;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getTwoLevelBetaOrdering(int num_partitions, int buffer_capacity, int fine_to_coarse_ratio,
                                                                                 int num_cache_partitions, bool randomly_assign_edge_buckets) {
    int coarse_num_partitions = num_partitions / fine_to_coarse_ratio;
    int coarse_buffer_capacity = buffer_capacity / fine_to_coarse_ratio;

    coarse_num_partitions = coarse_num_partitions - num_cache_partitions;
    coarse_buffer_capacity = coarse_buffer_capacity - num_cache_partitions;

    vector<vector<int>> coarse_buffer_states = getBetaOrderingHelper(coarse_num_partitions, coarse_buffer_capacity);

    vector<vector<int>> buffer_states = convertToFinePartitions(coarse_buffer_states, num_partitions, buffer_capacity,
                                                                fine_to_coarse_ratio, num_cache_partitions);

    // assign edge buckets
    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer;
    if (randomly_assign_edge_buckets) {
        edge_buckets_per_buffer = randomlyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    } else {
        edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    }

    return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);
}

vector<torch::Tensor> getDispersedOrderingHelper(int num_partitions, int buffer_capacity) {
    vector<torch::Tensor> buffer_states;
    Indices all_partitions = torch::randperm(num_partitions, torch::kInt32);
    Indices in_buffer = all_partitions.narrow(0, 0, buffer_capacity);
    Indices on_disk = all_partitions.narrow(0, buffer_capacity, num_partitions - buffer_capacity);
    buffer_states.emplace_back(in_buffer);

    while (on_disk.size(0) > 0) {
        in_buffer = in_buffer.index_select(0, torch::randperm(in_buffer.size(0), torch::kInt64));
        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));

        in_buffer[-1] = on_disk[0];
        buffer_states.emplace_back(in_buffer);
        on_disk = on_disk.narrow(0, 1, on_disk.size(0) - 1);
    }

    return buffer_states;
}

vector<torch::Tensor> randomlyAssignTrainNodesToBuffers(vector<torch::Tensor> buffer_states, Indices train_nodes,
                                                        int64_t total_num_nodes, int num_partitions, int buffer_capacity) {
    // randomly assign train nodes to buffers

    int64_t partition_size = ceil((double)total_num_nodes / num_partitions);
    torch::Tensor train_nodes_partition = train_nodes.divide(partition_size, "trunc");

    std::vector<std::vector<int>> partition_buffer_states(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        for (int j = 0; j < buffer_states.size(); j++) {
            bool partition_in_buffer = false;
            auto buffer_state_accessor = buffer_states[j].accessor<int32_t, 1>();

            for (int k = 0; k < buffer_capacity; k++) {
                if (buffer_state_accessor[k] == i) {
                    partition_in_buffer = true;
                    break;
                }
            }
            if (partition_in_buffer) {
                partition_buffer_states[i].emplace_back(j);
            }
        }
    }

    torch::Tensor train_nodes_buffer_choice = torch::zeros_like(train_nodes) - 1;
    std::vector<torch::Tensor> train_nodes_per_buffer(buffer_states.size());
    auto train_nodes_partition_accessor = train_nodes_partition.accessor<int32_t, 1>();

    for (int i = 0; i < train_nodes.size(0); i++) {
        int partition_id = train_nodes_partition_accessor[i];
        if (partition_buffer_states[partition_id].size() == 0) {
            continue;
        }
        int rand_id = rand() % partition_buffer_states[partition_id].size();
        train_nodes_buffer_choice[i] = partition_buffer_states[partition_id][rand_id];
    }

    for (int i = 0; i < buffer_states.size(); i++) {
        train_nodes_per_buffer[i] = train_nodes.masked_select(train_nodes_buffer_choice == i);
    }

    return train_nodes_per_buffer;
}

//std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getDispersedNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
//                                                                                           int buffer_capacity, int fine_to_coarse_ratio,
//                                                                                           int num_cache_partitions) {
//    int coarse_num_partitions = num_partitions / fine_to_coarse_ratio;
//    int coarse_buffer_capacity = buffer_capacity / fine_to_coarse_ratio;
//
//    coarse_num_partitions = coarse_num_partitions - num_cache_partitions;
//    coarse_buffer_capacity = coarse_buffer_capacity - num_cache_partitions;
//
//    // create coarse buffer states
//    vector<torch::Tensor> coarse_buffer_states = getDispersedOrderingHelper(coarse_num_partitions, coarse_buffer_capacity);
//
//    // add in cache partitions
//    // Note: this isn't fully correct (cache coarse 0, but then randomly mapped to fine),
//    //  can probably be more like two level beta for cache/convert to fine, but Dispersed is redundant with Diag now
//    for (int i = 0; i < coarse_buffer_states.size(); i++) {
//        coarse_buffer_states[i] =
//            torch::cat({coarse_buffer_states[i] + num_cache_partitions, torch::arange(num_cache_partitions, coarse_buffer_states[i].options())});
//    }
//
//    // convert to fine buffer states
//    torch::Tensor fine_to_coarse_map = torch::randperm(num_partitions, torch::kInt32);
//    int *data_ptr_ = (int *)fine_to_coarse_map.data_ptr();
//
//    vector<torch::Tensor> buffer_states;
//
//    for (int i = 0; i < coarse_buffer_states.size(); i++) {
//        vector<int> fine_buffer_state(buffer_capacity, 0);
//        torch::Tensor coarse_buffer_state = coarse_buffer_states[i];
//        auto coarse_buffer_state_accessor = coarse_buffer_state.accessor<int32_t, 1>();
//
//        for (int j = 0; j < coarse_buffer_state.size(0); j++) {
//            int *start = (int *)data_ptr_ + coarse_buffer_state_accessor[j] * fine_to_coarse_ratio;
//            int *end = (int *)data_ptr_ + (coarse_buffer_state_accessor[j] + 1) * fine_to_coarse_ratio;
//            vector<int> fine_partitions = vector<int>(start, end);
//
//            for (int k = j * fine_to_coarse_ratio; k < (j + 1) * fine_to_coarse_ratio; k++) {
//                fine_buffer_state[k] = fine_partitions[k - j * fine_to_coarse_ratio];
//            }
//        }
//
//        buffer_states.emplace_back(torch::from_blob(fine_buffer_state.data(), {(int)fine_buffer_state.size()}, torch::kInt32).clone());
//    }
//
//    auto train_nodes_per_buffer = randomlyAssignTrainNodesToBuffers(buffer_states, train_nodes, total_num_nodes, num_partitions, buffer_capacity);
//
//    return std::forward_as_tuple(buffer_states, train_nodes_per_buffer);
//}

torch::Tensor splitTensorAcrossMachines(torch::Tensor input, shared_ptr<c10d::ProcessGroupGloo> pg_gloo, shared_ptr<DistributedConfig> dist_config) {
    torch::Tensor local_split;

    int num_batch_workers = 0;
    for (auto worker_config : dist_config->workers) {
        if (worker_config->type == WorkerType::BATCH) {
            num_batch_workers++;
        }
    }

    int machine_chunk_size = ceil((double)input.size(0) / num_batch_workers);

    for (int machine_num = 0; machine_num < num_batch_workers; machine_num++) {
        int start_index = machine_num * machine_chunk_size;
        int end_index = (machine_num + 1) * machine_chunk_size;
        if (machine_num == num_batch_workers - 1) end_index = input.size(0);

        torch::Tensor tmp_local_split = input.narrow(0, start_index, end_index-start_index);

        if (machine_num == pg_gloo->getRank()) {
            if (machine_num == 0) {
                local_split = tmp_local_split;
            } else {
                std::vector<torch::Tensor> transfer_vec;
                transfer_vec.push_back(tmp_local_split);

                auto work = pg_gloo->recv(transfer_vec, 0, 0);
                if (!work->wait()) {
                    throw work->exception();
                }

                local_split = tmp_local_split;
            }
        }

        if (machine_num > 0 and pg_gloo->getRank() == 0) {
            std::vector<torch::Tensor> transfer_vec;
            transfer_vec.push_back(tmp_local_split);
            auto work = pg_gloo->send(transfer_vec, machine_num, 0);
            if (!work->wait()) {
                throw work->exception();
            }
        }
    }

    return local_split;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getDiagOrdering(int num_partitions, int buffer_capacity, int fine_to_coarse_ratio, int num_cache_partitions,
                                                                         bool randomly_assign_edge_buckets, Indices train_nodes, int64_t total_num_nodes,
                                                                         shared_ptr<c10d::ProcessGroupGloo> pg_gloo,
                                                                         shared_ptr<DistributedConfig> dist_config,
                                                                         bool sequential, bool in_memory) {
    if ((sequential or in_memory) and pg_gloo == nullptr)
        throw MariusRuntimeException("Use single machine orderings when not running distributed training.");

    if ((sequential or in_memory) and total_num_nodes == -1)
        throw MariusRuntimeException("LP training should not use in memory or sequential distributed orderings.");

    if (sequential and in_memory)
        throw MariusRuntimeException("Training should use in memory or sequential but not both.");

    bool split_train_nodes = false;

    int local_num_partitions = num_partitions;
    torch::Tensor local_machine_map;
    torch::Tensor local_train_partitions;

    if (pg_gloo != nullptr) {
        // if distributed, coordinate the ordering
        if (num_cache_partitions > 0) {
            throw MariusRuntimeException("Distributed Diag with cached partitions not yet implemented."); // TODO
            // probably don't need to subtract cache partitions below for caching with dist diag
        }

        if (in_memory) {
            local_machine_map = torch::arange(num_partitions, torch::kInt32);
            buffer_capacity = local_num_partitions;
            split_train_nodes = true;

            if (num_cache_partitions != 0) throw MariusRuntimeException("Cache partitions unneeded for in memory.");
        } else {
            if (sequential) {
                int partition_size = ceil((double)total_num_nodes / num_partitions);
                torch::Tensor train_nodes_partition = train_nodes.divide(partition_size, "trunc");
                int max_train_partition = torch::max(train_nodes_partition).item<int32_t>();
                int num_train_partitions = max_train_partition + 1;
                SPDLOG_INFO("Num Global Train Partitions: {}", num_train_partitions);

                torch::Tensor input = torch::arange(num_train_partitions, num_partitions, torch::kInt32);
                input = input.index_select(0, torch::randperm(input.size(0), torch::kInt64));
                local_machine_map = splitTensorAcrossMachines(input, pg_gloo, dist_config);

                input = torch::arange(num_train_partitions, torch::kInt32);
                input = input.index_select(0, torch::randperm(input.size(0), torch::kInt64));
                local_train_partitions = splitTensorAcrossMachines(input, pg_gloo, dist_config);
            } else {
                SPDLOG_INFO("Running Diag in Distributed Mode");

                torch::Tensor input = torch::randperm(num_partitions, torch::kInt32);
                local_machine_map = splitTensorAcrossMachines(input, pg_gloo, dist_config);
                local_num_partitions = local_machine_map.size(0);

                std::cout<<"local_num_partitions: "<<local_num_partitions<<"\n";
                std::cout<<"local_machine_map: "<<convertToVectorOfInts({local_machine_map})<<"\n";
            }
        }
    }

    vector<vector<int>> buffer_states_int;

    if (sequential) {
        vector<torch::Tensor> buffer_states;
        Indices in_buffer = local_train_partitions;
        Indices on_disk = local_machine_map;

        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));
        on_disk = on_disk.narrow(0, 0, buffer_capacity - in_buffer.size(0));
        buffer_states.emplace_back(torch::cat({in_buffer, on_disk}));

        buffer_states_int = convertToVectorOfInts(buffer_states);

//        printBuffers(buffer_states_int, false);
    } else {
        // coarse buffers (using local partition ids)
        int coarse_num_partitions = local_num_partitions / fine_to_coarse_ratio;
        int coarse_buffer_capacity = buffer_capacity / fine_to_coarse_ratio;

        coarse_num_partitions = coarse_num_partitions - num_cache_partitions;
        coarse_buffer_capacity = coarse_buffer_capacity - num_cache_partitions;

        // create coarse buffer states
        vector<torch::Tensor> coarse_buffer_states = getDispersedOrderingHelper(coarse_num_partitions, coarse_buffer_capacity);

        //convert to fine and add cached partitions
        vector<vector<int>> coarse_buffer_states_int = convertToVectorOfInts(coarse_buffer_states);
        buffer_states_int = convertToFinePartitions(coarse_buffer_states_int, local_num_partitions,
                                                    buffer_capacity, fine_to_coarse_ratio, num_cache_partitions);

//        printBuffers(coarse_buffer_states_int, true);
//        printBuffers(buffer_states_int, false);


        if (pg_gloo != nullptr) {
            // if distributed, map back to global partitions, from here on use global num partitions
            int *data_ptr_ = (int *)local_machine_map.data_ptr();
            vector<vector<int>> global_buffer_states;

            for (int i = 0; i < buffer_states_int.size(); i++) {
                vector<int> global_buffer_state(buffer_states_int[i].size(), 0);
                for (int j = 0; j < buffer_states_int[i].size(); j++) {
                    global_buffer_state[j] = *((int *)data_ptr_ + buffer_states_int[i][j]);
                }
                global_buffer_states.emplace_back(global_buffer_state);
            }

            buffer_states_int = global_buffer_states;

//            printBuffers(buffer_states_int, false);
        }
    }

    if (total_num_nodes == -1) {
        // for link prediction, randomly assign edge buckets
        vector<vector<std::pair<int, int>>> edge_buckets_per_buffer;
        if (randomly_assign_edge_buckets) {
            edge_buckets_per_buffer = randomlyAssignEdgeBucketsToBuffers(buffer_states_int, num_partitions);
        } else {
            edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(buffer_states_int, num_partitions);
        }

//        printEdgeBuckets(edge_buckets_per_buffer);

        return convertEdgeBucketOrderToTensors(buffer_states_int, edge_buckets_per_buffer);
    } else {
        // for node classification, randomly assign train nodes to buffers
        vector<torch::Tensor> buffer_states = convertToVectorOfTensors(buffer_states_int, torch::TensorOptions().dtype(torch::kInt32));

        if (split_train_nodes) {
            torch::Tensor input = torch::randperm(train_nodes.size(0), torch::kInt32);
            torch::Tensor local_split = splitTensorAcrossMachines(input, pg_gloo, dist_config);
            train_nodes = train_nodes.index_select(0, local_split);
        }

        auto train_nodes_per_buffer = randomlyAssignTrainNodesToBuffers(buffer_states, train_nodes, total_num_nodes, num_partitions, buffer_capacity);

//        printTrainNodes(train_nodes_per_buffer);

        return std::forward_as_tuple(buffer_states, train_nodes_per_buffer);
    }

}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getSequentialNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                            int buffer_capacity) {
    int64_t partition_size = ceil((double)total_num_nodes / num_partitions);
    torch::Tensor train_nodes_partition = train_nodes.divide(partition_size, "trunc");

    int32_t max_train_partition = torch::max(train_nodes_partition).item<int32_t>();
    int32_t num_train_partitions = max_train_partition + 1;
    SPDLOG_INFO("Num Train Partitions: {}", num_train_partitions);

    vector<torch::Tensor> buffer_states;
    Indices in_buffer = torch::arange(num_train_partitions, torch::kInt32);
    Indices on_disk = torch::arange(num_train_partitions, num_partitions, torch::kInt32);
    on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));
    on_disk = on_disk.narrow(0, 0, buffer_capacity - num_train_partitions);

    buffer_states.emplace_back(torch::cat({in_buffer, on_disk}));

    std::vector<torch::Tensor> train_nodes_per_buffer;
    train_nodes_per_buffer.emplace_back(train_nodes.clone());

    return std::forward_as_tuple(buffer_states, train_nodes_per_buffer);
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomNodePartitionOrdering() {
    SPDLOG_ERROR("Not implemented");
    std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
    return ret;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomEdgeBucketOrdering() {
    SPDLOG_ERROR("Not implemented");
    std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
    return ret;
}
