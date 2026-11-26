#pragma once

#include "document.hpp"

#include <filesystem>
#include <fstream>
#include <optional>

namespace ggufpp {

class File {
  public:
    [[nodiscard]] static auto open(const std::filesystem::path& path,
                                   const parse_options& options = {}) -> File;

    File(const File&) = delete;
    auto operator=(const File&) -> File& = delete;
    // Standard file-stream moves may throw; do not promise unconditional noexcept.
    // NOLINTNEXTLINE(performance-noexcept-move-constructor)
    File(File&&) = default;
    // NOLINTNEXTLINE(performance-noexcept-move-constructor)
    auto operator=(File&&) -> File& = default;
    ~File() = default;

    [[nodiscard]] auto document() const noexcept -> const Document& { return document_; }

    // Copies raw encoded bytes without endian conversion or dequantization.
    // Concurrent calls on the same File require external synchronization.
    void read_tensor(std::size_t tensor_index, std::uint64_t byte_offset,
                     std::span<std::byte> destination);

  private:
    File(std::ifstream stream, Document document);
    std::ifstream stream_;
    Document document_;
    std::optional<std::uint64_t> next_read_offset_;
};

} // namespace ggufpp
