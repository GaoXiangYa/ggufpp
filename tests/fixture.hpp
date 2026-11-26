#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace test_support {

// An independent byte-level encoder used only to construct synthetic fixtures.
class Writer {
  public:
    explicit Writer(std::endian order = std::endian::little) : order_{order} {}

    template <class T> void number(T value) {
        auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
        if (order_ != std::endian::native) {
            std::reverse(bytes.begin(), bytes.end());
        }
        append(bytes);
    }
    void string(std::string_view value) {
        number<std::uint64_t>(value.size());
        append(std::as_bytes(std::span{value.data(), value.size()}));
    }
    void append(std::span<const std::byte> bytes) {
        bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
    }
    void resize(std::size_t size) { bytes_.resize(size); }
    [[nodiscard]] auto bytes() const -> const std::vector<std::byte>& { return bytes_; }
    [[nodiscard]] auto release() -> std::vector<std::byte> { return std::move(bytes_); }

  private:
    std::endian order_;
    std::vector<std::byte> bytes_;
};

class Fixture {
  public:
    explicit Fixture(std::endian order = std::endian::little, std::uint32_t version = 3)
        : order_{order}, version_{version}, metadata_{order}, tensors_{order} {}

    auto metadata(std::string_view key, std::uint32_t type) -> Writer& {
        metadata_.string(key);
        metadata_.number(type);
        ++metadata_count_;
        return metadata_;
    }
    void tensor(std::string_view name, std::span<const std::uint64_t> dimensions,
                std::uint32_t type, std::uint64_t offset, std::size_t bytes) {
        tensors_.string(name);
        tensors_.number<std::uint32_t>(static_cast<std::uint32_t>(dimensions.size()));
        for (const auto dimension : dimensions) {
            tensors_.number(dimension);
        }
        tensors_.number(type);
        tensors_.number(offset);
        ++tensor_count_;
        if (offset + bytes > payload_.size()) {
            payload_.resize(static_cast<std::size_t>(offset) + bytes);
        }
        for (std::size_t i = 0; i < bytes; ++i) {
            payload_[static_cast<std::size_t>(offset) + i] = static_cast<std::byte>(i % 251);
        }
    }
    [[nodiscard]] auto finish(std::size_t alignment = 32) const -> std::vector<std::byte> {
        Writer writer{order_};
        constexpr std::array magic{std::byte{0x47}, std::byte{0x47}, std::byte{0x55},
                                   std::byte{0x46}};
        writer.append(magic);
        writer.number(version_);
        writer.number(tensor_count_);
        writer.number(metadata_count_);
        writer.append(metadata_.bytes());
        writer.append(tensors_.bytes());
        if (tensor_count_ != 0) {
            const auto size = writer.bytes().size();
            writer.resize(size + (alignment - size % alignment) % alignment);
            writer.append(payload_);
        }
        return writer.release();
    }

  private:
    std::endian order_;
    std::uint32_t version_;
    Writer metadata_;
    Writer tensors_;
    std::uint64_t metadata_count_{};
    std::uint64_t tensor_count_{};
    std::vector<std::byte> payload_;
};

inline auto basic_fixture(std::endian order = std::endian::little) -> std::vector<std::byte> {
    Fixture fixture{order};
    fixture.metadata("general.architecture", 8).string("llama");
    fixture.metadata("general.name", 8).string("tiny test");
    fixture.metadata("general.alignment", 4).number<std::uint32_t>(32);
    constexpr std::array<std::uint64_t, 2> dimensions{4, 2};
    fixture.tensor("weight", dimensions, 0, 0, 32);
    return fixture.finish();
}

} // namespace test_support
