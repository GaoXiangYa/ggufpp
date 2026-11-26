#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ggufpp {

enum class value_type : std::uint32_t {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12
};

class MetadataArray {
  public:
    using storage_type =
        std::variant<std::vector<std::uint8_t>, std::vector<std::int8_t>,
                     std::vector<std::uint16_t>, std::vector<std::int16_t>,
                     std::vector<std::uint32_t>, std::vector<std::int32_t>, std::vector<float>,
                     std::vector<bool>, std::vector<std::string>, std::vector<MetadataArray>,
                     std::vector<std::uint64_t>, std::vector<std::int64_t>, std::vector<double>>;

    explicit MetadataArray(storage_type values) : values_{std::move(values)} {
        if (values_.valueless_by_exception()) {
            throw std::invalid_argument("metadata array storage must contain a value");
        }
    }

    [[nodiscard]] auto element_type() const noexcept -> value_type {
        return static_cast<value_type>(values_.index());
    }
    [[nodiscard]] auto size() const -> std::size_t {
        return std::visit([](const auto& values) { return values.size(); }, values_);
    }
    template <class T> [[nodiscard]] auto get_if() const noexcept -> const std::vector<T>* {
        return std::get_if<std::vector<T>>(&values_);
    }
    [[nodiscard]] auto storage() const noexcept -> const storage_type& { return values_; }

  private:
    storage_type values_;
};

class MetadataValue {
  public:
    using storage_type = std::variant<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
                                      std::uint32_t, std::int32_t, float, bool, std::string,
                                      MetadataArray, std::uint64_t, std::int64_t, double>;

    explicit MetadataValue(storage_type value) : value_{std::move(value)} {
        if (value_.valueless_by_exception()) {
            throw std::invalid_argument("metadata storage must contain a value");
        }
    }

    [[nodiscard]] auto type() const noexcept -> value_type {
        return static_cast<value_type>(value_.index());
    }
    template <class T> [[nodiscard]] auto get_if() const noexcept -> const T* {
        return std::get_if<T>(&value_);
    }
    [[nodiscard]] auto storage() const noexcept -> const storage_type& { return value_; }

  private:
    storage_type value_;
};

struct metadata_entry {
    std::string key;
    MetadataValue value;
};

} // namespace ggufpp
