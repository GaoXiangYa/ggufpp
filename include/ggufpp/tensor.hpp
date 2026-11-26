#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ggufpp {

enum class tensor_type : std::uint32_t {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    nvfp4 = 40,
    q1_0 = 41,
    q2_0 = 42,
    f8_e4m3 = 43
};

struct tensor_layout {
    std::string_view name;
    std::uint32_t block_elements;
    std::uint32_t block_bytes;
};

[[nodiscard]] auto layout_of(tensor_type type) noexcept -> std::optional<tensor_layout>;

struct tensor_info {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    tensor_type type{};
    std::uint64_t offset{};      // Relative to the tensor data section.
    std::uint64_t file_offset{}; // Absolute file offset.
    std::uint64_t byte_size{};   // Encoded bytes, excluding alignment padding.
    std::uint64_t element_count{};
};

} // namespace ggufpp
