#include "ggufpp/mapped_file.hpp"

#include "ggufpp/error.hpp"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif GGUFPP_HAS_MMAP
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ggufpp::detail {
namespace {

// Diagnostics carry the path as UTF-8 even when the native path is wide.
auto path_text(const std::filesystem::path& path) -> std::string {
    const auto utf8 = path.u8string();
    return {reinterpret_cast<const char*>(utf8.data()), utf8.size()};
}

[[noreturn]] void fail(error_code code, const std::string& message) {
    throw ParseError(code, 0, message);
}

} // namespace

#if defined(_WIN32)

class mapped_region {
  public:
    static auto open(const std::filesystem::path& path) -> std::unique_ptr<mapped_region> {
        // Share read/write so a loader holding a writer handle can still map the file.
        const HANDLE file =
            ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            const auto error = static_cast<unsigned long>(::GetLastError());
            fail(error_code::io_error, "cannot open GGUF file '" + path_text(path) + "': " +
                                           std::system_category().message(static_cast<int>(error)));
        }
        LARGE_INTEGER size{};
        if (::GetFileSizeEx(file, &size) == 0) {
            const auto error = static_cast<unsigned long>(::GetLastError());
            ::CloseHandle(file);
            fail(error_code::io_error,
                 "cannot determine GGUF file size '" + path_text(path) +
                     "': " + std::system_category().message(static_cast<int>(error)));
        }
        if (size.QuadPart < 0 ||
            std::cmp_greater(size.QuadPart, std::numeric_limits<std::size_t>::max())) {
            ::CloseHandle(file);
            fail(error_code::resource_limit,
                 "GGUF file is larger than the address space '" + path_text(path) + "'");
        }
        const auto file_size = static_cast<std::size_t>(size.QuadPart);
        if (file_size == 0) {
            ::CloseHandle(file);
            return std::unique_ptr<mapped_region>{new mapped_region{nullptr, nullptr, 0}};
        }
        const HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping == nullptr) {
            const auto error = static_cast<unsigned long>(::GetLastError());
            ::CloseHandle(file);
            fail(error_code::io_error,
                 "cannot create a read-only mapping for '" + path_text(path) +
                     "': " + std::system_category().message(static_cast<int>(error)));
        }
        void* data = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (data == nullptr) {
            const auto error = static_cast<unsigned long>(::GetLastError());
            ::CloseHandle(mapping);
            ::CloseHandle(file);
            fail(error_code::io_error, "cannot map GGUF file '" + path_text(path) + "': " +
                                           std::system_category().message(static_cast<int>(error)));
        }
        // The section keeps the file alive; the descriptor is no longer needed.
        ::CloseHandle(file);
        return std::unique_ptr<mapped_region>{new mapped_region{data, mapping, file_size}};
    }

    mapped_region(const mapped_region&) = delete;
    auto operator=(const mapped_region&) -> mapped_region& = delete;
    mapped_region(mapped_region&&) = delete;
    auto operator=(mapped_region&&) -> mapped_region& = delete;

    ~mapped_region() {
        if (data_ != nullptr) {
            ::UnmapViewOfFile(data_);
        }
        if (mapping_ != nullptr) {
            ::CloseHandle(mapping_);
        }
    }

    [[nodiscard]] auto bytes() const noexcept -> std::span<const std::byte> {
        return {static_cast<const std::byte*>(data_), size_};
    }

  private:
    mapped_region(void* data, HANDLE mapping, std::size_t size) noexcept
        : data_{data}, mapping_{mapping}, size_{size} {}

    void* data_;
    HANDLE mapping_;
    std::size_t size_;
};

#elif GGUFPP_HAS_MMAP

class mapped_region {
  public:
    static auto open(const std::filesystem::path& path) -> std::unique_ptr<mapped_region> {
        const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            fail(error_code::io_error,
                 "cannot open GGUF file '" + path_text(path) + "': " + std::strerror(errno));
        }
        struct stat info {};
        if (::fstat(descriptor, &info) != 0) {
            const int error = errno;
            ::close(descriptor);
            fail(error_code::io_error, "cannot determine GGUF file size '" + path_text(path) +
                                           "': " + std::strerror(error));
        }
        if (!S_ISREG(info.st_mode)) {
            ::close(descriptor);
            fail(error_code::io_error, "GGUF path is not a regular file '" + path_text(path) + "'");
        }
        if (info.st_size < 0 ||
            std::cmp_greater(info.st_size, std::numeric_limits<std::size_t>::max())) {
            ::close(descriptor);
            fail(error_code::resource_limit,
                 "GGUF file is larger than the address space '" + path_text(path) + "'");
        }
        const auto size = static_cast<std::size_t>(info.st_size);
        if (size == 0) {
            ::close(descriptor);
            return std::unique_ptr<mapped_region>{new mapped_region{nullptr, 0}};
        }
        void* data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
        const int error = errno;
        ::close(descriptor);
        if (data == MAP_FAILED) {
            fail(error_code::io_error,
                 "cannot map GGUF file '" + path_text(path) + "': " + std::strerror(error));
        }
        return std::unique_ptr<mapped_region>{new mapped_region{data, size}};
    }

    mapped_region(const mapped_region&) = delete;
    auto operator=(const mapped_region&) -> mapped_region& = delete;
    mapped_region(mapped_region&&) = delete;
    auto operator=(mapped_region&&) -> mapped_region& = delete;

    ~mapped_region() {
        if (data_ != nullptr && size_ != 0) {
            ::munmap(data_, size_);
        }
    }

    [[nodiscard]] auto bytes() const noexcept -> std::span<const std::byte> {
        return {static_cast<const std::byte*>(data_), size_};
    }

  private:
    mapped_region(void* data, std::size_t size) noexcept : data_{data}, size_{size} {}

    void* data_;
    std::size_t size_;
};

#else

class mapped_region {
  public:
    static auto open(const std::filesystem::path& path) -> std::unique_ptr<mapped_region> {
        (void)path;
        fail(error_code::io_error,
             "memory-mapped GGUF access is not supported on this platform; use ggufpp::File");
    }

    mapped_region(const mapped_region&) = delete;
    auto operator=(const mapped_region&) -> mapped_region& = delete;
    mapped_region(mapped_region&&) = delete;
    auto operator=(mapped_region&&) -> mapped_region& = delete;
    ~mapped_region() = default;

    [[nodiscard]] auto bytes() const noexcept -> std::span<const std::byte> { return {}; }
};

#endif

} // namespace ggufpp::detail

namespace ggufpp {

MappedFile::MappedFile(std::unique_ptr<detail::mapped_region> region, Document document)
    : region_{std::move(region)}, document_{std::move(document)} {}

MappedFile::MappedFile(MappedFile&&) noexcept = default;
auto MappedFile::operator=(MappedFile&&) noexcept -> MappedFile& = default;
MappedFile::~MappedFile() = default;

auto MappedFile::open(const std::filesystem::path& path,
                      const parse_options& options) -> MappedFile {
    auto region = detail::mapped_region::open(path);
    auto document = parse(region->bytes(), options);
    return MappedFile{std::move(region), std::move(document)};
}

auto MappedFile::document() const noexcept -> const Document& { return document_; }

auto MappedFile::bytes() const noexcept -> std::span<const std::byte> { return region_->bytes(); }

auto MappedFile::tensor_data(std::size_t index) const -> std::span<const std::byte> {
    const auto tensors = document_.tensors();
    if (index >= tensors.size()) {
        throw std::out_of_range("tensor index out of range");
    }
    const auto& tensor = tensors[index];
    const auto data = bytes();
    // parse() validated these ranges against the file size; keep the check here so
    // the accessor cannot produce an out-of-bounds view.
    if (tensor.file_offset > data.size() ||
        tensor.byte_size > data.size() - static_cast<std::size_t>(tensor.file_offset)) {
        throw std::out_of_range("tensor payload is outside the mapped file");
    }
    return data.subspan(static_cast<std::size_t>(tensor.file_offset),
                        static_cast<std::size_t>(tensor.byte_size));
}

auto MappedFile::tensor_data(std::string_view name) const
    -> std::optional<std::span<const std::byte>> {
    const auto index = document_.find_tensor_index(name);
    if (!index) {
        return std::nullopt;
    }
    return tensor_data(*index);
}

} // namespace ggufpp
