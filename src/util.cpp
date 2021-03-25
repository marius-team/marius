//
// Created by Jason Mohoney on 7/30/20.
//

#include <util.h>

void assert_no_nans(torch::Tensor values) {
    return;
//    torch::Tensor nan_tens_mask = torch::isnan(values);
//    if (nan_tens_mask.any().item<bool>()) {
//        std::ostringstream stream;
//        stream << values.masked_select(nan_tens_mask);
//        std::string str = stream.str();
//        spdlog::error("Values: {}", str);
//        stream.str("");
//        stream.clear();
//        stream << nan_tens_mask.nonzero();
//        str = stream.str();
//        spdlog::error("Nan Indices: {}", str);
//        stream.str("");
//        stream.clear();
//        stream << nan_tens_mask.nonzero().size(0);
//        str = stream.str();
//        spdlog::error("Num Nan: {}", str);
//        stream.str("");
//        stream.clear();
//        stream << values.min();
//        str = stream.str();
//        spdlog::error("Min: {}", str);
//        stream.str("");
//        stream.clear();
//        stream << values.max();
//        str = stream.str();
//        spdlog::error("Max: {}", str);
//        exit(0);
//    }
}

void process_mem_usage() {
    double vm_usage = 0.0;
    double resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;

    SPDLOG_DEBUG("VM Usage: {}GB. RSS: {}GB", vm_usage / pow(2, 20), resident_set / pow(2, 20));
}
