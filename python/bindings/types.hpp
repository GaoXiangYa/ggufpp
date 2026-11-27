#pragma once

#include <pybind11/pybind11.h>

namespace ggufpp::python {

void bind_types(pybind11::module_& module);

} // namespace ggufpp::python
