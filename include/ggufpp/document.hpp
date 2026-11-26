#pragma once

#include "metadata.hpp"
#include "tensor.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ggufpp {

struct parse_options {
    std::endian byte_order = std::endian::little;
    std::uint64_t max_metadata_entries = 1'000'000;
    std::uint64_t max_tensors = 1'000'000;
    std::uint64_t max_string_bytes = 64ULL * 1024 * 1024;
    std::uint64_t max_array_elements = 16ULL * 1024 * 1024;
    std::uint64_t max_total_array_elements = 32ULL * 1024 * 1024;
    std::uint64_t max_allocation_bytes = 256ULL * 1024 * 1024;
    std::uint32_t max_array_depth = 16;
};

namespace detail {
class Parser;
}

class Document {
  public:
    [[nodiscard]] auto version() const noexcept -> std::uint32_t { return version_; }
    [[nodiscard]] auto byte_order() const noexcept -> std::endian { return byte_order_; }
    [[nodiscard]] auto alignment() const noexcept -> std::uint32_t { return alignment_; }
    [[nodiscard]] auto data_offset() const noexcept -> std::uint64_t { return data_offset_; }
    [[nodiscard]] auto file_size() const noexcept -> std::uint64_t { return file_size_; }
    [[nodiscard]] auto metadata() const noexcept -> std::span<const metadata_entry> {
        return metadata_;
    }
    [[nodiscard]] auto tensors() const noexcept -> std::span<const tensor_info> { return tensors_; }
    [[nodiscard]] auto find_metadata(std::string_view key) const noexcept -> const MetadataValue*;
    [[nodiscard]] auto find_tensor(std::string_view name) const noexcept -> const tensor_info*;

  private:
    friend class detail::Parser;
    Document() = default;
    std::uint32_t version_{};
    std::endian byte_order_ = std::endian::little;
    std::uint32_t alignment_ = 32;
    std::uint64_t data_offset_{};
    std::uint64_t file_size_{};
    std::vector<metadata_entry> metadata_;
    std::vector<tensor_info> tensors_;
    std::vector<std::size_t> metadata_index_;
    std::vector<std::size_t> tensor_index_;
};

// Owns all parsed metadata. Does not retain or copy the tensor payload.
[[nodiscard]] auto parse(std::span<const std::byte> bytes,
                         const parse_options& options = {}) -> Document;

} // namespace ggufpp
