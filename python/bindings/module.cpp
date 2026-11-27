#include "types.hpp"

#include <ggufpp/ggufpp.hpp>
#include <pybind11/stl/filesystem.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace py = pybind11;

namespace ggufpp::python {
namespace {

// A file and its metadata views share this slot. Closing it invalidates every
// metadata view, including nested arrays, without copying the native document.
struct Directory {
    std::shared_ptr<const Document> document;

    auto get() const -> std::shared_ptr<const Document> {
        if (!document) {
            throw py::value_error("I/O operation on a closed GGUF file");
        }
        return document;
    }
};

template <class Owner>
auto directory_for(const std::shared_ptr<Owner>& owner, const Document& document) {
    return std::make_shared<Directory>(std::shared_ptr<const Document>{owner, &document});
}

struct ArrayView {
    std::shared_ptr<Directory> directory;
    const MetadataArray* array;

    auto get() const -> std::shared_ptr<const MetadataArray> { return {directory->get(), array}; }
};

template <class T>
auto to_python(const T& value, const std::shared_ptr<Directory>& directory) -> py::object {
    if constexpr (std::is_same_v<T, MetadataArray>) {
        return py::cast(ArrayView{directory, &value});
    } else {
        return py::cast(value);
    }
}

auto find_value(const Document& document, const std::string& key) -> const MetadataValue& {
    const auto* value = document.find_metadata(key);
    if (!value) {
        throw py::key_error(key);
    }
    return *value;
}

auto find_index(const Document& document, const std::string& name) -> std::size_t {
    const auto index = document.find_tensor_index(name);
    if (!index) {
        throw py::key_error(name);
    }
    return *index;
}

template <class T> auto at(std::span<const T> entries, std::size_t index) -> const T& {
    if (index >= entries.size()) {
        throw py::index_error();
    }
    return entries[index];
}

// A simple Py_buffer requests C-contiguous storage and pins it through the I/O.
class Buffer {
  public:
    Buffer(py::handle object, int flags) {
        if (PyObject_GetBuffer(object.ptr(), &view_, flags) < 0) {
            throw py::error_already_set();
        }
    }
    Buffer(const Buffer&) = delete;
    auto operator=(const Buffer&) -> Buffer& = delete;
    ~Buffer() { PyBuffer_Release(&view_); }

    auto bytes() const -> std::span<std::byte> {
        return {static_cast<std::byte*>(view_.buf), static_cast<std::size_t>(view_.len)};
    }

  private:
    Py_buffer view_{};
};

struct Stream {
    explicit Stream(File source) : file{std::move(source)} {}

    File file;
    std::mutex mutex;
};

auto read_into(Stream& stream, const std::string& name, py::handle destination,
               std::uint64_t offset) -> std::size_t {
    const auto index = find_index(stream.file.document(), name);
    const auto size = stream.file.document().tensors()[index].byte_size;
    if (offset > size) {
        throw py::value_error("offset exceeds tensor size");
    }
    const Buffer buffer(destination, PyBUF_WRITABLE);
    const auto bytes = buffer.bytes();
    const auto count =
        static_cast<std::size_t>(std::min<std::uint64_t>(bytes.size(), size - offset));
    {
        py::gil_scoped_release release;
        std::lock_guard lock(stream.mutex);
        stream.file.read_tensor(index, offset, bytes.first(count));
    }
    return count;
}

struct Payload {
    std::shared_ptr<MappedFile> mapping;
    std::span<const std::byte> bytes;
};

void bind_directory(py::module_& module) {
    py::class_<Directory, std::shared_ptr<Directory>>(module, "_Directory")
        .def("close", [](Directory& self) { self.document.reset(); })
        .def_property_readonly("closed", [](const Directory& self) { return !self.document; })
        .def_property_readonly("version",
                               [](const Directory& self) { return self.get()->version(); })
        .def_property_readonly("alignment",
                               [](const Directory& self) { return self.get()->alignment(); })
        .def_property_readonly("file_size",
                               [](const Directory& self) { return self.get()->file_size(); })
        .def_property_readonly("data_offset",
                               [](const Directory& self) { return self.get()->data_offset(); })
        .def_property_readonly("byte_order",
                               [](const Directory& self) {
                                   return self.get()->byte_order() == std::endian::little ? "little"
                                                                                          : "big";
                               })
        .def("metadata_count", [](const Directory& self) { return self.get()->metadata().size(); })
        .def("metadata_key",
             [](const Directory& self, std::size_t i) { return at(self.get()->metadata(), i).key; })
        .def("metadata_type",
             [](const Directory& self, const std::string& key) {
                 return find_value(*self.get(), key).type();
             })
        .def("metadata_value",
             [](const std::shared_ptr<Directory>& self, const std::string& key) {
                 const auto document = self->get();
                 return std::visit([&](const auto& v) { return to_python(v, self); },
                                   find_value(*document, key).storage());
             })
        .def("tensor_count", [](const Directory& self) { return self.get()->tensors().size(); })
        .def("tensor_name",
             [](const Directory& self, std::size_t i) { return at(self.get()->tensors(), i).name; })
        .def("tensor", [](const Directory& self, const std::string& name) {
            const auto document = self.get();
            return document->tensors()[find_index(*document, name)];
        });

    py::class_<ArrayView>(module, "_Array")
        .def("__len__", [](const ArrayView& self) { return self.get()->size(); })
        .def_property_readonly("element_type",
                               [](const ArrayView& self) { return self.get()->element_type(); })
        .def("item", [](const ArrayView& self, std::size_t index) {
            const auto array = self.get();
            return std::visit(
                [&](const auto& values) -> py::object {
                    if (index >= values.size()) {
                        throw py::index_error();
                    }
                    return to_python(values[index], self.directory);
                },
                array->storage());
        });
}

void bind_io(py::module_& module) {
    module.def("parse", [](py::handle source, const parse_options& options) {
        const Buffer buffer(source, PyBUF_SIMPLE);
        const auto parse_buffer = [&] {
            return std::make_shared<Document>(parse(buffer.bytes(), options));
        };
        std::shared_ptr<Document> document;
        if (PyBytes_CheckExact(source.ptr())) {
            py::gil_scoped_release release;
            document = parse_buffer();
        } else {
            document = parse_buffer();
        }
        return std::make_shared<Directory>(std::move(document));
    });
    py::class_<Stream, std::shared_ptr<Stream>>(module, "_File")
        .def("directory",
             [](const std::shared_ptr<Stream>& self) {
                 return directory_for(self, self->file.document());
             })
        .def("readinto", &read_into);

    module.def(
        "open",
        [](const std::filesystem::path& path, const parse_options& options) {
            return std::make_shared<Stream>(File::open(path, options));
        },
        py::call_guard<py::gil_scoped_release>());

    py::class_<Payload>(module, "_Payload", py::buffer_protocol()).def_buffer([](Payload& payload) {
        if (std::cmp_greater(payload.bytes.size(), PY_SSIZE_T_MAX)) {
            throw std::overflow_error("tensor is too large for a Python memoryview");
        }
        const auto size = static_cast<py::ssize_t>(payload.bytes.size());
        return py::buffer_info(reinterpret_cast<const unsigned char*>(payload.bytes.data()), size,
                               true);
    });
    py::class_<MappedFile, std::shared_ptr<MappedFile>>(module, "_MappedFile")
        .def("directory",
             [](const std::shared_ptr<MappedFile>& self) {
                 return directory_for(self, self->document());
             })
        .def("payload", [](const std::shared_ptr<MappedFile>& self, const std::string& name) {
            const auto bytes = self->tensor_data(name);
            if (!bytes) {
                throw py::key_error(name);
            }
            return Payload{self, *bytes};
        });
    module.def(
        "mmap",
        [](const std::filesystem::path& path, const parse_options& options) {
            return std::make_shared<MappedFile>(MappedFile::open(path, options));
        },
        py::call_guard<py::gil_scoped_release>());
}

} // namespace
} // namespace ggufpp::python

PYBIND11_MODULE(_core, module) {
    ggufpp::python::bind_types(module);
    ggufpp::python::bind_directory(module);
    ggufpp::python::bind_io(module);
}
