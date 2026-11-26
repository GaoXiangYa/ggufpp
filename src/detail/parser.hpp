#pragma once

#include "binary_reader.hpp"
#include "ggufpp/document.hpp"

namespace ggufpp::detail {

[[nodiscard]] auto parse_reader(BinaryReader& reader, const parse_options& options) -> Document;

} // namespace ggufpp::detail
