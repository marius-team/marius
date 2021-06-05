//
// Created by Jason Mohoney on 4/9/21.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "marius.h"

namespace py = pybind11;

void init_marius(py::module &m) {
    m.def("marius_train", [](int argc, std::vector<std::string> argv) {

            argv[0] = "marius_train";
            std::vector<char *> c_strs;
            c_strs.reserve(argv.size());
            for (auto &s : argv) c_strs.push_back(const_cast<char *>(s.c_str()));

            marius(argc, c_strs.data());

        }, py::arg("argc"), py::arg("argv"), py::return_value_policy::reference);

    m.def("marius_eval", [](int argc, std::vector<std::string> argv) {

        argv[0] = "marius_eval";
        std::vector<char *> c_strs;
        c_strs.reserve(argv.size());
        for (auto &s : argv) c_strs.push_back(const_cast<char *>(s.c_str()));

        marius(argc, c_strs.data());

    }, py::arg("argc"), py::arg("argv"), py::return_value_policy::reference);
}