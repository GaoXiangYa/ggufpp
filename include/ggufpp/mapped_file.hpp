#pragma once

#include "document.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

// 1 when this translation unit has a read-only memory-mapping backend.
// Supported: Windows (CreateFileMapping), Linux/macOS/BSD (mmap).
// On other platforms ggufpp::MappedFile::open throws ParseError(io_error); use
// ggufpp::File for portable stream-based access instead.
#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
#define GGUFPP_HAS_MMAP 1
#else
#define GGUFPP_HAS_MMAP 0
#endif

namespace ggufpp {

namespace detail {
class mapped_region;
}

// Read-only memory mapping of a complete GGUF file.
//
// The metadata is parsed into an owned Document exactly as ggufpp::parse does;
// the tensor payload is never copied. tensor_data() returns a view directly into
// the mapping, which is what inference-style loaders need.
//
// Lifetime: bytes()/tensor_data() views (and tensor data pointers) borrow from
// this object's mapping and stay valid until it is destroyed or moved from.
// Moving a MappedFile does not remap, so views remain valid. Concurrent const
// access from multiple threads is safe: the mapping and Document are read-only.
// The underlying file must not be modified or truncated while the mapping is
// alive; doing so can fault or expose inconsistent bytes.
//
// Unlike File, a MappedFile does not keep an open file descriptor; the mapping
// is independent of the descriptor after it has been created.
class MappedFile {
  public:
    // Maps the whole file and parses its metadata from the mapping.
    // Throws ParseError (io_error/resource_limit) on mapping or I/O failures and
    // ParseError for malformed GGUF input.
    [[nodiscard]] static auto open(const std::filesystem::path& path,
                                   const parse_options& options = {}) -> MappedFile;

    MappedFile(const MappedFile&) = delete;
    auto operator=(const MappedFile&) -> MappedFile& = delete;
    MappedFile(MappedFile&&) noexcept;
    auto operator=(MappedFile&&) noexcept -> MappedFile&;
    ~MappedFile();

    [[nodiscard]] auto document() const noexcept -> const Document&;
    // The complete file, including metadata and payload.
    [[nodiscard]] auto bytes() const noexcept -> std::span<const std::byte>;
    // Encoded payload bytes of one tensor, as a view into the mapping.
    // Throws std::out_of_range for an invalid index.
    [[nodiscard]] auto tensor_data(std::size_t index) const -> std::span<const std::byte>;
    // Same as tensor_data(index) looked up by tensor name; nullopt if absent.
    [[nodiscard]] auto
    tensor_data(std::string_view name) const -> std::optional<std::span<const std::byte>>;

  private:
    MappedFile(std::unique_ptr<detail::mapped_region> region, Document document);
    std::unique_ptr<detail::mapped_region> region_;
    Document document_;
};

} // namespace ggufpp
