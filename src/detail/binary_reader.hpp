#pragma once

#include "ggufpp/error.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <span>
#include <type_traits>

namespace ggufpp::detail {

static_assert(CHAR_BIT == 8);
static_assert(sizeof(float) == 4 && sizeof(double) == 8);
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
static_assert(std::endian::native == std::endian::little ||
              std::endian::native == std::endian::big);

[[nodiscard]] inline auto checked_add(std::uint64_t lhs, std::uint64_t rhs,
                                      std::uint64_t offset) -> std::uint64_t {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw ParseError(error_code::integer_overflow, offset, "integer addition overflow");
    }
    return lhs + rhs;
}

[[nodiscard]] inline auto checked_multiply(std::uint64_t lhs, std::uint64_t rhs,
                                           std::uint64_t offset) -> std::uint64_t {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw ParseError(error_code::integer_overflow, offset, "integer multiplication overflow");
    }
    return lhs * rhs;
}

[[nodiscard]] inline auto to_size(std::uint64_t value, std::uint64_t offset) -> std::size_t {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw ParseError(error_code::resource_limit, offset, "length exceeds address space");
    }
    return static_cast<std::size_t>(value);
}

class BinaryReader {
  public:
    BinaryReader(std::span<const std::byte> bytes, std::endian order)
        : memory_{bytes}, size_{bytes.size()}, order_{order} {}
    BinaryReader(std::istream& stream, std::uint64_t size, std::endian order)
        : stream_{&stream}, size_{size}, order_{order} {}

    [[nodiscard]] auto position() const noexcept -> std::uint64_t { return position_; }
    [[nodiscard]] auto size() const noexcept -> std::uint64_t { return size_; }
    [[nodiscard]] auto remaining() const noexcept -> std::uint64_t { return size_ - position_; }
    [[nodiscard]] auto native_order() const noexcept -> bool {
        return order_ == std::endian::native;
    }

    void require(std::uint64_t count) const {
        if (count > remaining()) {
            throw ParseError(error_code::unexpected_end, position_, "truncated GGUF input");
        }
    }

    void read_bytes(std::span<std::byte> destination) {
        require(destination.size());
        if (destination.empty()) {
            return;
        }
        if (stream_ == nullptr) {
            std::memcpy(destination.data(), memory_.data() + to_size(position_, position_),
                        destination.size());
            position_ += destination.size();
            return;
        }
        while (!destination.empty()) {
            if (buffer_position_ == buffer_size_) {
                const auto count =
                    static_cast<std::size_t>(std::min<std::uint64_t>(buffer_.size(), remaining()));
                stream_->read(reinterpret_cast<char*>(buffer_.data()),
                              static_cast<std::streamsize>(count));
                if (stream_->gcount() != static_cast<std::streamsize>(count)) {
                    throw ParseError(error_code::io_error, position_, "failed to read GGUF input");
                }
                buffer_position_ = 0;
                buffer_size_ = count;
            }
            const auto count = std::min(destination.size(), buffer_size_ - buffer_position_);
            std::memcpy(destination.data(), buffer_.data() + buffer_position_, count);
            destination = destination.subspan(count);
            buffer_position_ += count;
            position_ += count;
        }
    }

    template <class T> [[nodiscard]] auto read() -> T {
        static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>);
        std::array<std::byte, sizeof(T)> bytes{};
        read_bytes(bytes);
        if constexpr (sizeof(T) > 1) {
            if (!native_order()) {
                std::reverse(bytes.begin(), bytes.end());
            }
        }
        return std::bit_cast<T>(bytes);
    }

  private:
    std::span<const std::byte> memory_;
    std::istream* stream_{}; // Borrows the stream for the duration of parsing.
    std::uint64_t size_{};
    std::uint64_t position_{};
    std::endian order_;
    std::array<std::byte, std::size_t{64} * 1024> buffer_;
    std::size_t buffer_position_{};
    std::size_t buffer_size_{};
};

} // namespace ggufpp::detail
