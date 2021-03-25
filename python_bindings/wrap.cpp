#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

//void init_batch(py::module &);
void init_config(py::module &);
void init_dataset(py::module &);
void init_datatypes(py::module &);
void init_decoder(py::module &);
void init_encoder(py::module &);
void init_evaluator(py::module &);
void init_model(py::module &);
void init_io(py::module &);
void init_trainer(py::module &);

PYBIND11_MODULE(pymarius, m) {

	m.doc() = "pybind11 marius plugin";

//	init_batch(m);
	init_config(m);
	init_dataset(m);
	init_datatypes(m);
	init_decoder(m);
	init_encoder(m);
	init_evaluator(m);
	init_model(m);
	init_io(m);
	init_trainer(m);
}
