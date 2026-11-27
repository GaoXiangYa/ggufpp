#include "types.hpp"

#include <ggufpp/ggufpp.hpp>
#include <pybind11/native_enum.h>

#include <algorithm>
#include <cctype>

namespace py = pybind11;

namespace ggufpp::python {
namespace {

void translate_parse_error(std::exception_ptr exception) {
    if (!exception) {
        return;
    }
    try {
        std::rethrow_exception(exception);
    } catch (const ParseError& error) {
        if (error.code() == error_code::io_error) {
            PyErr_SetString(PyExc_OSError, error.what());
            return;
        }
        auto type = py::module_::import("ggufpp._core").attr("ParseError");
        auto instance = type(error.what());
        instance.attr("code") = error.code();
        instance.attr("offset") = error.offset();
        PyErr_SetObject(type.ptr(), instance.ptr());
    }
}

} // namespace

void bind_types(py::module_& module) {
    py::native_enum<value_type>(module, "ValueType", "enum.IntEnum")
        .value("UINT8", value_type::uint8)
        .value("INT8", value_type::int8)
        .value("UINT16", value_type::uint16)
        .value("INT16", value_type::int16)
        .value("UINT32", value_type::uint32)
        .value("INT32", value_type::int32)
        .value("FLOAT32", value_type::float32)
        .value("BOOLEAN", value_type::boolean)
        .value("STRING", value_type::string)
        .value("ARRAY", value_type::array)
        .value("UINT64", value_type::uint64)
        .value("INT64", value_type::int64)
        .value("FLOAT64", value_type::float64)
        .finalize();

    py::native_enum<tensor_type> encodings(module, "TensorType", "enum.IntEnum");
    for (std::uint32_t i = 0; i <= static_cast<std::uint32_t>(tensor_type::f8_e4m3); ++i) {
        const auto type = static_cast<tensor_type>(i);
        if (const auto layout = layout_of(type)) {
            std::string name(layout->name);
            std::transform(name.begin(), name.end(), name.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            encodings.value(name.c_str(), type);
        }
    }
    encodings.finalize();

    py::native_enum<error_code>(module, "ErrorCode", "enum.IntEnum")
        .value("INVALID_MAGIC", error_code::invalid_magic)
        .value("UNSUPPORTED_VERSION", error_code::unsupported_version)
        .value("UNEXPECTED_END", error_code::unexpected_end)
        .value("INVALID_TYPE", error_code::invalid_type)
        .value("INVALID_VALUE", error_code::invalid_value)
        .value("INVALID_ALIGNMENT", error_code::invalid_alignment)
        .value("INVALID_TENSOR", error_code::invalid_tensor)
        .value("DUPLICATE_NAME", error_code::duplicate_name)
        .value("INTEGER_OVERFLOW", error_code::integer_overflow)
        .value("RESOURCE_LIMIT", error_code::resource_limit)
        .value("IO_ERROR", error_code::io_error)
        .finalize();
    py::exception<ParseError>(module, "ParseError", PyExc_ValueError);
    py::register_local_exception_translator(&translate_parse_error);

    constexpr parse_options defaults;
    py::class_<parse_options>(module, "ParseOptions")
        .def(py::init([](std::string_view byte_order, std::uint64_t max_metadata_entries,
                         std::uint64_t max_tensors, std::uint64_t max_string_bytes,
                         std::uint64_t max_array_elements, std::uint64_t max_total_array_elements,
                         std::uint64_t max_allocation_bytes, std::uint32_t max_array_depth) {
                 if (byte_order != "little" && byte_order != "big") {
                     throw py::value_error("byte_order must be 'little' or 'big'");
                 }
                 return parse_options{byte_order == "little" ? std::endian::little
                                                             : std::endian::big,
                                      max_metadata_entries,
                                      max_tensors,
                                      max_string_bytes,
                                      max_array_elements,
                                      max_total_array_elements,
                                      max_allocation_bytes,
                                      max_array_depth};
             }),
             py::kw_only(), py::arg("byte_order") = "little",
             py::arg("max_metadata_entries") = defaults.max_metadata_entries,
             py::arg("max_tensors") = defaults.max_tensors,
             py::arg("max_string_bytes") = defaults.max_string_bytes,
             py::arg("max_array_elements") = defaults.max_array_elements,
             py::arg("max_total_array_elements") = defaults.max_total_array_elements,
             py::arg("max_allocation_bytes") = defaults.max_allocation_bytes,
             py::arg("max_array_depth") = defaults.max_array_depth)
        .def_property_readonly("byte_order",
                               [](const parse_options& o) {
                                   return o.byte_order == std::endian::little ? "little" : "big";
                               })
        .def_readonly("max_metadata_entries", &parse_options::max_metadata_entries)
        .def_readonly("max_tensors", &parse_options::max_tensors)
        .def_readonly("max_string_bytes", &parse_options::max_string_bytes)
        .def_readonly("max_array_elements", &parse_options::max_array_elements)
        .def_readonly("max_total_array_elements", &parse_options::max_total_array_elements)
        .def_readonly("max_allocation_bytes", &parse_options::max_allocation_bytes)
        .def_readonly("max_array_depth", &parse_options::max_array_depth);

    py::class_<tensor_info>(module, "TensorInfo")
        .def_readonly("name", &tensor_info::name)
        .def_property_readonly("dimensions",
                               [](const tensor_info& info) {
                                   py::tuple result(info.dimensions.size());
                                   for (std::size_t i = 0; i < info.dimensions.size(); ++i) {
                                       result[i] = info.dimensions[i];
                                   }
                                   return result;
                               })
        .def_readonly("encoding", &tensor_info::type)
        .def_readonly("element_count", &tensor_info::element_count)
        .def_readonly("nbytes", &tensor_info::byte_size)
        .def_readonly("file_offset", &tensor_info::file_offset);
}

} // namespace ggufpp::python
