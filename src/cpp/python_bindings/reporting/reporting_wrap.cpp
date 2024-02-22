#include "common/pybind_headers.h"
#include "reporting/reporting.h"

class PyReporter : Reporter {
   public:
    using Reporter::Reporter;
    void report() override { PYBIND11_OVERRIDE_PURE_NAME(void, Reporter, "report", report); }
};

class PyMetric : Metric {
   public:
    using Metric::Metric;
};

void init_reporting(py::module &m) {
    py::class_<Metric, PyMetric, std::shared_ptr<Metric>>(m, "Metric").def_readwrite("name", &Metric::name_).def_readwrite("unit", &Metric::unit_);

    py::class_<RankingMetric, Metric, std::shared_ptr<RankingMetric>>(m, "RankingMetric")
        .def("compute_metric", &RankingMetric::computeMetric, py::arg("ranks"));
    py::class_<HitskMetric, RankingMetric, std::shared_ptr<HitskMetric>>(m, "Hitsk")
        .def(py::init<int>(), py::arg("k"))
        .def("compute_metric", &HitskMetric::computeMetric, py::arg("ranks"));
    py::class_<MeanRankMetric, RankingMetric, std::shared_ptr<MeanRankMetric>>(m, "MeanRank")
        .def(py::init<>())
        .def("compute_metric", &MeanRankMetric::computeMetric, py::arg("ranks"));
    py::class_<MeanReciprocalRankMetric, RankingMetric, std::shared_ptr<MeanReciprocalRankMetric>>(m, "MeanReciprocalRank")
        .def(py::init<>())
        .def("compute_metric", &MeanReciprocalRankMetric::computeMetric, py::arg("ranks"));

    py::class_<ClassificationMetric, Metric, std::shared_ptr<ClassificationMetric>>(m, "ClassificationMetric")
        .def("compute_metric", &ClassificationMetric::computeMetric, py::arg("y_true"), py::arg("y_pred"));
    py::class_<CategoricalAccuracyMetric, ClassificationMetric, std::shared_ptr<CategoricalAccuracyMetric>>(m, "CategoricalAccuracy")
        .def(py::init<>())
        .def("compute_metric", &CategoricalAccuracyMetric::computeMetric, py::arg("y_true"), py::arg("y_pred"));

    py::class_<Reporter, PyReporter, std::shared_ptr<Reporter>>(m, "Reporter")
        .def_readwrite("metrics", &Reporter::metrics_)
        .def(py::init<>())
        .def("add_metric", &Reporter::addMetric, py::arg("metric"))
        .def("report", &Reporter::report);

    py::class_<LinkPredictionReporter, Reporter, std::shared_ptr<LinkPredictionReporter>>(m, "LinkPredictionReporter")
        .def(py::init<>())
        .def("clear", &LinkPredictionReporter::clear)
        .def("compute_ranks", &LinkPredictionReporter::computeRanks, py::arg("pos_scores"), py::arg("neg_scores"))
        .def("add_result", &LinkPredictionReporter::addResult, py::arg("pos_scores"), py::arg("neg_scores"), py::arg("edges") = torch::Tensor())
        .def("save", &LinkPredictionReporter::save, py::arg("directory"), py::arg("scores") = false, py::arg("ranks") = false);

    py::class_<NodeClassificationReporter, Reporter, std::shared_ptr<NodeClassificationReporter>>(m, "NodeClassificationReporter")
        .def(py::init<>())
        .def("clear", &NodeClassificationReporter::clear)
        .def("add_result", &NodeClassificationReporter::addResult, py::arg("y_true"), py::arg("y_pred"), py::arg("node_ids") = torch::Tensor())
        .def("save", &NodeClassificationReporter::save, py::arg("directory"), py::arg("labels") = false);

    py::class_<ProgressReporter, Reporter, std::shared_ptr<ProgressReporter>>(m, "ProgressReporter")
        .def(py::init<std::string, int64_t, int>(), py::arg("item_name"), py::arg("total_items"), py::arg("total_reports"))
        .def("clear", &ProgressReporter::clear)
        .def("add_result", &ProgressReporter::addResult, py::arg("items_processed"));
}