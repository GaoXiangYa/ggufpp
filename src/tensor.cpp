#include "ggufpp/tensor.hpp"

#include <array>

namespace ggufpp {

// Format facts from GGML's public type IDs and block layouts. No GGML code is linked.
// Removed/reserved encodings intentionally have zero block sizes.
auto layout_of(tensor_type type) noexcept -> std::optional<tensor_layout> {
    static constexpr std::array<tensor_layout, 44> layouts{
        {{"f32", 1, 4},        {"f16", 1, 2},       {"q4_0", 32, 18},     {"q4_1", 32, 20},
         {"removed", 0, 0},    {"removed", 0, 0},   {"q5_0", 32, 22},     {"q5_1", 32, 24},
         {"q8_0", 32, 34},     {"q8_1", 32, 36},    {"q2_k", 256, 84},    {"q3_k", 256, 110},
         {"q4_k", 256, 144},   {"q5_k", 256, 176},  {"q6_k", 256, 210},   {"q8_k", 256, 292},
         {"iq2_xxs", 256, 66}, {"iq2_xs", 256, 74}, {"iq3_xxs", 256, 98}, {"iq1_s", 256, 50},
         {"iq4_nl", 32, 18},   {"iq3_s", 256, 110}, {"iq2_s", 256, 82},   {"iq4_xs", 256, 136},
         {"i8", 1, 1},         {"i16", 1, 2},       {"i32", 1, 4},        {"i64", 1, 8},
         {"f64", 1, 8},        {"iq1_m", 256, 56},  {"bf16", 1, 2},       {"removed", 0, 0},
         {"removed", 0, 0},    {"removed", 0, 0},   {"tq1_0", 256, 54},   {"tq2_0", 256, 66},
         {"removed", 0, 0},    {"removed", 0, 0},   {"removed", 0, 0},    {"mxfp4", 32, 17},
         {"nvfp4", 64, 36},    {"q1_0", 128, 18},   {"q2_0", 64, 18},     {"f8_e4m3", 1, 1}}};
    const auto index = static_cast<std::uint32_t>(type);
    if (index >= layouts.size() || layouts[index].block_elements == 0) {
        return std::nullopt;
    }
    return layouts[index];
}

} // namespace ggufpp
