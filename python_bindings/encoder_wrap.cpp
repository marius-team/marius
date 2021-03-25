#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <encoder.h>

namespace py = pybind11;

// Trampoline class
class PyEncoder : Encoder {
  public:
    using Encoder::Encoder;
    void forward(Batch *batch, bool train) override { 
      PYBIND11_OVERRIDE_PURE(void, Encoder, forward, batch, train); }
};

void init_encoder(py::module &m) {
  py::class_<Encoder, PyEncoder>(m, "Encoder")
    .def(py::init<>())
    .def("forward", &Encoder::forward, py::arg("batch"), py::arg("train"));

  py::class_<EmptyEncoder, Encoder>(m, "EmptyEncoder")
    .def(py::init<>());
}
