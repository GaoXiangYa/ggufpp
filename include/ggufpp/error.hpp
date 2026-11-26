#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ggufpp {

enum class error_code : std::uint8_t {
    invalid_magic,
    unsupported_version,
    unexpected_end,
    invalid_type,
    invalid_value,
    invalid_alignment,
    invalid_tensor,
    duplicate_name,
    integer_overflow,
    resource_limit,
    io_error
};

class ParseError : public std::runtime_error {
  public:
    ParseError(error_code code, std::uint64_t offset, const std::string& message)
        : std::runtime_error(message + " at byte " + std::to_string(offset)), code_{code},
          offset_{offset} {}

    [[nodiscard]] auto code() const noexcept -> error_code { return code_; }
    [[nodiscard]] auto offset() const noexcept -> std::uint64_t { return offset_; }

  private:
    error_code code_;
    std::uint64_t offset_;
};

} // namespace ggufpp
