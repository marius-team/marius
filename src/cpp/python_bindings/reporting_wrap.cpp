#include "torch/extension.h"

#include "reporting.h"

namespace py = pybind11;

class PyReporter : Reporter {
  public:
    using Reporter::Reporter;
    void report() override {
        PYBIND11_OVERRIDE_PURE_NAME(void, Reporter, "report", report); }
};

void init_reporting(py::module &m) {

    py::class_<Reporter, PyReporter>(m, "Reporter")
        .def_readwrite("metrics", &Reporter::metrics_)
        .def(py::init<>())
        .def("lock", &Reporter::lock)
        .def("unlock", &Reporter::unlock)
        .def("addMetric", &Reporter::addMetric, py::arg("metric"))
        .def("report", &Reporter::report);

    py::class_<LinkPredictionReporter, Reporter>(m, "LinkPredictionReporter")
        .def(py::init<>())
        .def("clear", &LinkPredictionReporter::clear)
        .def("computeRanks", &LinkPredictionReporter::computeRanks, py::arg("pos_scores"), py::arg("neg_scores"))
        .def("addResult", &LinkPredictionReporter::addResult, py::arg("pos_scores"), py::arg("neg_scores"));

    py::class_<NodeClassificationReporter, Reporter>(m, "NodeClassificationReporter")
        .def(py::init<>())
        .def("clear", &NodeClassificationReporter::clear)
        .def("addResult", &NodeClassificationReporter::addResult, py::arg("y_true"), py::arg("y_pred"));

    py::class_<ProgressReporter, Reporter>(m, "ProgressReporter")
        .def(py::init<std::string, int64_t, int>(), py::arg("item_name"), py::arg("total_items"), py::arg("total_reports"))
        .def("clear", &ProgressReporter::clear)
        .def("addResult", &ProgressReporter::addResult, py::arg("items_processed"));
}