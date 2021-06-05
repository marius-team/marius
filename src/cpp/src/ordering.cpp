//
// Created by Jason Mohoney on 7/17/20.
//

#include "ordering.h"

#include "config.h"
#include "datatypes.h"
#include "logger.h"

using std::pair;

vector<Batch *> applyOrdering(vector<Batch *> batches) {
    vector<Batch *> ret_batches(batches.size());
    map<pair<int64_t, int64_t>, int64_t> ordering_map;
    switch (marius_options.storage.edge_bucket_ordering) {
        case EdgeBucketOrdering::Sequential:ordering_map = getSequentialOrdering();
            break;
        case EdgeBucketOrdering::SequentialSymmetric:ordering_map = getSequentialSymmetricOrdering();
            break;
        case EdgeBucketOrdering::Random:ordering_map = getRandomOrdering();
            break;
        case EdgeBucketOrdering::RandomSymmetric:ordering_map = getRandomSymmetricOrdering();
            break;
        case EdgeBucketOrdering::Hilbert:ordering_map = getHilbertOrdering();
            break;
        case EdgeBucketOrdering::HilbertSymmetric:ordering_map = getHilbertSymmetricOrdering();
            break;
        case EdgeBucketOrdering::Elimination:ordering_map = getEliminationOrdering();
            break;
        case EdgeBucketOrdering::Custom:ordering_map = getCustomOrdering();
            break;
    }
    pair<int64_t, int64_t> key;
    for (auto itr = batches.begin(); itr != batches.end(); itr++) {
        PartitionBatch *batch = (PartitionBatch *) *itr;
        int64_t s = batch->src_partition_idx_;
        int64_t d = batch->dst_partition_idx_;
        key = std::make_pair(s, d);
        batch->batch_id_ = ordering_map.find(key)->second;
        ret_batches.at(batch->batch_id_) = (*itr);
    }

    string ordering_str = "[";
    for (auto itr = ret_batches.begin(); itr != ret_batches.end(); itr++) {
        int64_t s = ((PartitionBatch *) (*itr))->src_partition_idx_;
        int64_t d = ((PartitionBatch *) (*itr))->dst_partition_idx_;
        ordering_str += "(" + std::to_string(s) + ", " + std::to_string(d) + "), ";
    }
    ordering_str = ordering_str.substr(0, ordering_str.size() - 2);
    ordering_str += "]";
    SPDLOG_DEBUG("Ordering: {}", ordering_str);

    return ret_batches;
}

map<pair<int64_t, int64_t>, int64_t> makeOrderingSymmetric(map<pair<int64_t, int64_t>, int64_t> ordering_map) {
    map<pair<int64_t, int64_t>, int64_t> new_ordering_map;
    pair<int64_t, int64_t> key1;
    pair<int64_t, int64_t> key2;
    pair<int64_t, int64_t> key3;
    pair<int64_t, int64_t> key4;
    int64_t idx = 0;
    int64_t i = 0;
    int64_t j = 0;

    vector<pair<int64_t, int64_t>> tmp_vec(ordering_map.size());

    for (auto itr = ordering_map.begin(); itr != ordering_map.end(); itr++) {
        tmp_vec[itr->second] = itr->first;
    }

    for (auto itr = tmp_vec.begin(); itr != tmp_vec.end(); itr++) {
        key1 = *itr;
        i = key1.first;
        j = key1.second;
        key2 = std::make_pair(j, i);
        key3 = std::make_pair(i, i);
        key4 = std::make_pair(j, j);

        if (new_ordering_map.find(key1) == new_ordering_map.end()) {
            new_ordering_map[key1] = idx++;
        }

        if (new_ordering_map.find(key2) == new_ordering_map.end()) {
            new_ordering_map[key2] = idx++;
        }

        if (new_ordering_map.find(key3) == new_ordering_map.end()) {
            new_ordering_map[key3] = idx++;
        }

        if (new_ordering_map.find(key4) == new_ordering_map.end()) {
            new_ordering_map[key4] = idx++;
        }
    }
    return new_ordering_map;
}

map<pair<int64_t, int64_t>, int64_t> getSequentialOrdering() {
    map<pair<int64_t, int64_t>, int64_t> ordering_map;
    pair<int64_t, int64_t> key;
    int64_t idx = 0;
    for (int64_t i = 0; i < marius_options.storage.num_partitions; i++) {
        for (int64_t j = 0; j < marius_options.storage.num_partitions; j++) {
            key = std::make_pair(i, j);
            ordering_map[key] = idx;
            idx++;
        }
    }
    return ordering_map;
}

map<pair<int64_t, int64_t>, int64_t> getSequentialSymmetricOrdering() {
    return makeOrderingSymmetric(getSequentialOrdering());
}

map<pair<int64_t, int64_t>, int64_t> getRandomOrdering() {
    torch::Tensor rand_idx = torch::randperm(marius_options.storage.num_partitions * marius_options.storage.num_partitions, torch::kInt64);
    map<pair<int64_t, int64_t>, int64_t> ordering_map;
    pair<int64_t, int64_t> key;
    int64_t idx = 0;
    for (int64_t i = 0; i < marius_options.storage.num_partitions; i++) {
        for (int64_t j = 0; j < marius_options.storage.num_partitions; j++) {
            key = std::make_pair(i, j);
            ordering_map[key] = rand_idx[idx].item<int64_t>();
            idx++;
        }
    }
    return ordering_map;
}

map<pair<int64_t, int64_t>, int64_t> getRandomSymmetricOrdering() {
    return makeOrderingSymmetric(getRandomOrdering());
}

map<pair<int64_t, int64_t>, int64_t> getHilbertOrdering() {
    map<pair<int64_t, int64_t>, int64_t> ordering_map;
    pair<int64_t, int64_t> key;
    int64_t idx = 0;

    // shuffle
    torch::Tensor rand_map = torch::randperm(marius_options.storage.num_partitions);

    for (int64_t i = 0; i < marius_options.storage.num_partitions; i++) {
        for (int64_t j = 0; j < marius_options.storage.num_partitions; j++) {
            key = std::make_pair(rand_map[i].item<int64_t>(), rand_map[j].item<int64_t>());
            ordering_map[key] = xy2d(marius_options.storage.num_partitions, i, j);
            idx++;
        }
    }
    return ordering_map;
}

map<pair<int64_t, int64_t>, int64_t> getHilbertSymmetricOrdering() {
    return makeOrderingSymmetric(getHilbertOrdering());
}

map<pair<int64_t, int64_t>, int64_t> getEliminationOrdering() {
    map<pair<int64_t, int64_t>, int64_t> ordering_map;

    int num_elim = marius_options.storage.buffer_capacity - 1;

    vector<int64_t> curr_elim(num_elim);

    int num_left = marius_options.storage.num_partitions - (num_elim);

    std::vector<int64_t> nodes_left(num_left);

    torch::Tensor rand_init = torch::randperm(marius_options.storage.num_partitions);

    for (int i = 0; i < marius_options.storage.num_partitions; i++) {
        if (i < num_elim) {
            curr_elim[i] = rand_init[i].item<int>();
        } else {
            nodes_left[i - num_elim] = rand_init[i].item<int>();
        }
    }

    int done = 0;

    pair<int64_t, int64_t> key1;
    pair<int64_t, int64_t> key2;
    int64_t idx = 0;
    int64_t i = 0;
    int64_t j = 0;
    int64_t offset = 0;
    int last_id;

    while (done < marius_options.storage.num_partitions) {
        vector<pair<int64_t, int64_t>> shuffle_keys;

        // Find unprocessed edge partitions that are the combinations of any two node partitions.
        for (auto elim_itr = curr_elim.begin(); elim_itr != curr_elim.end(); elim_itr++) {
            for (auto elim_itr2 = curr_elim.begin(); elim_itr2 != curr_elim.end(); elim_itr2++) {
                i = *elim_itr;
                j = *elim_itr2;
                key1 = std::make_pair(i, j);
                key2 = std::make_pair(j, i);

                if (ordering_map.find(key1) == ordering_map.end()) {
                    ordering_map[key1] = idx++;
                    shuffle_keys.emplace_back(key1);
                }

                if (ordering_map.find(key2) == ordering_map.end()) {
                    ordering_map[key2] = idx++;
                    shuffle_keys.emplace_back(key2);
                }
            }
        }

        // Cycle through the node partitions that are not in the cache
        torch::Tensor rand_left = torch::randperm(nodes_left.size());
        for (int k = 0; k < (int) nodes_left.size(); k++) {
            for (auto elim_itr = curr_elim.begin(); elim_itr != curr_elim.end(); elim_itr++) {
                i = *elim_itr;
                j = nodes_left[rand_left[k].item<int>()];

                last_id = rand_left[k].item<int>();

                key1 = std::make_pair(i, j);
                key2 = std::make_pair(j, i);

                if (ordering_map.find(key1) == ordering_map.end()) {
                    ordering_map[key1] = idx++;
                    shuffle_keys.emplace_back(key1);
                }

                if (ordering_map.find(key2) == ordering_map.end()) {
                    ordering_map[key2] = idx++;
                    shuffle_keys.emplace_back(key2);
                }
            }

            //shuffle orderings
            torch::Tensor rand_idx = torch::randperm(shuffle_keys.size());
            for (auto shuffle_itr = shuffle_keys.begin(); shuffle_itr != shuffle_keys.end(); shuffle_itr++) {
                ordering_map[*shuffle_itr] = offset + rand_idx[ordering_map[*shuffle_itr] - offset].item<int64_t>();
            }
            offset += shuffle_keys.size();
            shuffle_keys.clear();
        }
        done += num_elim;

        int tmp_size = curr_elim.size();
        if (tmp_size > (int) nodes_left.size()) {
            tmp_size = nodes_left.size();
            num_elim = nodes_left.size();
        }

        vector<int> erase_idx;

        if (tmp_size > 0) {
            curr_elim[0] = nodes_left[last_id];
            nodes_left.erase(nodes_left.begin() + last_id);
        }

        torch::Tensor rand_left2 = torch::randperm(nodes_left.size());

        for (int k = 1; k < tmp_size; k++) {

            for (auto elim_itr = curr_elim.begin(); elim_itr != curr_elim.end(); elim_itr++) {
                i = *elim_itr;
                key1 = std::make_pair(i, i);

                if (ordering_map.find(key1) == ordering_map.end()) {
                    ordering_map[key1] = idx++;
                    shuffle_keys.emplace_back(key1);
                }
            }

            torch::Tensor rand_idx = torch::randperm(shuffle_keys.size());
            for (auto shuffle_itr = shuffle_keys.begin(); shuffle_itr != shuffle_keys.end(); shuffle_itr++) {
                ordering_map[*shuffle_itr] = offset + rand_idx[ordering_map[*shuffle_itr] - offset].item<int64_t>();
            }
            offset += shuffle_keys.size();
            shuffle_keys.clear();

            curr_elim[k] = nodes_left[rand_left2[k - 1].item<int>()];
            erase_idx.emplace_back(rand_left2[k - 1].item<int>());

            for (auto elim_itr = curr_elim.begin(); elim_itr != curr_elim.end(); elim_itr++) {
                for (auto elim_itr2 = curr_elim.begin(); elim_itr2 != curr_elim.end(); elim_itr2++) {
                    i = *elim_itr;
                    j = *elim_itr2;
                    key1 = std::make_pair(i, j);
                    key2 = std::make_pair(j, i);

                    if (ordering_map.find(key1) == ordering_map.end()) {
                        ordering_map[key1] = idx++;
                        shuffle_keys.emplace_back(key1);
                    }

                    if (ordering_map.find(key2) == ordering_map.end()) {
                        ordering_map[key2] = idx++;
                        shuffle_keys.emplace_back(key2);
                    }
                }
            }

            rand_idx = torch::randperm(shuffle_keys.size());
            for (auto shuffle_itr = shuffle_keys.begin(); shuffle_itr != shuffle_keys.end(); shuffle_itr++) {
                ordering_map[*shuffle_itr] = offset + rand_idx[ordering_map[*shuffle_itr] - offset].item<int64_t>();
            }
            offset += shuffle_keys.size();
            shuffle_keys.clear();
        }
        std::sort(erase_idx.begin(), erase_idx.end());
        std::reverse(erase_idx.begin(), erase_idx.end());
        for (int k = 0; k < tmp_size - 1; k++) {
            nodes_left.erase(nodes_left.begin() + erase_idx[k]);
        }
    }
    return ordering_map;
}

map<pair<int64_t, int64_t>, int64_t> getCustomOrdering() {
    map<pair<int64_t, int64_t>, int64_t> ordering_map;
    pair<int64_t, int64_t> key;
    int64_t idx = 0;

    string path = marius_options.path.base_directory + "/" + marius_options.general.experiment_name + "/" + PathConstants::custom_ordering_file;
    std::ifstream file(path);
    string line;

    while (std::getline(file, line)) {
        std::stringstream linestream(line);
        int s;
        int d;
        linestream >> s >> d;
        key = std::make_pair<int64_t, int64_t>(s, d);
        ordering_map[key] = idx;
        idx++;

    }
    return ordering_map;
}

//convert (x,y) to d
int64_t xy2d(int64_t n, int64_t x, int64_t y) {
    int64_t rx, ry, s, d = 0;
    for (s = n / 2; s > 0; s /= 2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(n, &x, &y, rx, ry);
    }
    return d;
}

//convert d to (x,y)
void d2xy(int64_t n, int64_t d, int64_t *x, int64_t *y) {
    int64_t rx, ry, s, t = d;
    *x = *y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

void rot(int64_t n, int64_t *x, int64_t *y, int64_t rx, int64_t ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        //Swap x and y
        int64_t t = *x;
        *x = *y;
        *y = t;
    }
}
