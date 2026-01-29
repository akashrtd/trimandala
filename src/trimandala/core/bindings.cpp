#include "integrator.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
  m.doc() = "Trimandala C++ High-Performance Physics Core";

  py::class_<trimandala::SymplecticIntegrator>(m, "SymplecticIntegrator")
      .def(py::init<const std::vector<double> &,
                    const std::vector<std::array<double, 3>> &,
                    const std::vector<std::array<double, 3>> &>(),
           py::arg("masses"), py::arg("pos"), py::arg("vel"),
           "Initialize integrator with N bodies")

      .def("step", &trimandala::SymplecticIntegrator::step, py::arg("dt"),
           "Advance simulation by time dt")

      .def(
          "get_state",
          [](const trimandala::SymplecticIntegrator &self, int n_bodies) {
            // Allocate numpy arrays for zero-copy read
            auto pos_array = py::array_t<double>({n_bodies, 3});
            auto vel_array = py::array_t<double>({n_bodies, 3});

            auto pos_ptr = static_cast<double *>(pos_array.request().ptr);
            auto vel_ptr = static_cast<double *>(vel_array.request().ptr);

            self.get_state(pos_ptr, vel_ptr);

            return std::make_pair(pos_array, vel_array);
          },
          py::arg("n_bodies"), "Get current (pos, vel) state");
}
