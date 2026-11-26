#include "detail/parser.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>

namespace ggufpp::detail {
namespace {

auto valid_utf8(std::string_view text) noexcept -> bool {
    std::size_t i = 0;
    while (i < text.size()) {
        const auto lead = static_cast<unsigned char>(text[i++]);
        if (lead <= 0x7f) {
            continue;
        }
        unsigned count = 0;
        std::uint32_t code = 0;
        std::uint32_t minimum = 0;
        if (lead >= 0xc2 && lead <= 0xdf) {
            count = 1;
            code = lead & 0x1fU;
            minimum = 0x80;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            count = 2;
            code = lead & 0x0fU;
            minimum = 0x800;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            count = 3;
            code = lead & 0x07U;
            minimum = 0x10000;
        } else {
            return false;
        }
        if (count > text.size() - i) {
            return false;
        }
        for (unsigned j = 0; j < count; ++j) {
            const auto next = static_cast<unsigned char>(text[i++]);
            if ((next & 0xc0U) != 0x80U) {
                return false;
            }
            code = (code << 6U) | (next & 0x3fU);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) {
            return false;
        }
    }
    return true;
}

auto valid_key(std::string_view key) noexcept -> bool {
    bool segment_empty = true;
    for (const char c : key) {
        if (c == '.') {
            if (segment_empty) {
                return false;
            }
            segment_empty = true;
        } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
            segment_empty = false;
        } else {
            return false;
        }
    }
    return !segment_empty;
}

// Both call sites return one concrete result type, avoiding a visitor hierarchy.
template <class Function> auto dispatch(value_type type, Function&& function) {
    switch (type) {
    case value_type::uint8:
        return function.template operator()<std::uint8_t>();
    case value_type::int8:
        return function.template operator()<std::int8_t>();
    case value_type::uint16:
        return function.template operator()<std::uint16_t>();
    case value_type::int16:
        return function.template operator()<std::int16_t>();
    case value_type::uint32:
        return function.template operator()<std::uint32_t>();
    case value_type::int32:
        return function.template operator()<std::int32_t>();
    case value_type::float32:
        return function.template operator()<float>();
    case value_type::boolean:
        return function.template operator()<bool>();
    case value_type::string:
        return function.template operator()<std::string>();
    case value_type::array:
        return function.template operator()<MetadataArray>();
    case value_type::uint64:
        return function.template operator()<std::uint64_t>();
    case value_type::int64:
        return function.template operator()<std::int64_t>();
    case value_type::float64:
        return function.template operator()<double>();
    }
    throw std::logic_error("unvalidated metadata type");
}

} // namespace

class Parser {
  public:
    Parser(BinaryReader& reader, const parse_options& options)
        : reader_{reader}, options_{options} {}

    auto run() -> Document {
        if (options_.byte_order != std::endian::little && options_.byte_order != std::endian::big) {
            throw std::invalid_argument("byte_order must be little or big endian");
        }
        if (options_.max_array_depth > 64) {
            throw std::invalid_argument("max_array_depth cannot exceed 64");
        }
        std::array<std::byte, 4> magic{};
        reader_.read_bytes(magic);
        constexpr std::array expected{std::byte{0x47}, std::byte{0x47}, std::byte{0x55},
                                      std::byte{0x46}};
        if (magic != expected) {
            fail(error_code::invalid_magic, "expected GGUF magic");
        }
        document_.version_ = reader_.read<std::uint32_t>();
        if (document_.version_ != 2 && document_.version_ != 3) {
            fail(error_code::unsupported_version, "only GGUF versions 2 and 3 are supported");
        }
        if (document_.version_ == 2 && options_.byte_order != std::endian::little) {
            fail(error_code::unsupported_version, "GGUF v2 requires little endian");
        }
        document_.byte_order_ = options_.byte_order;
        document_.file_size_ = reader_.size();
        const auto tensor_count = reader_.read<std::uint64_t>();
        const auto metadata_count = reader_.read<std::uint64_t>();
        limit(tensor_count, options_.max_tensors, "too many tensors");
        limit(metadata_count, options_.max_metadata_entries, "too many metadata entries");
        reader_.require(checked_add(product(tensor_count, 32), product(metadata_count, 13),
                                    reader_.position()));
        charge(product(metadata_count, sizeof(metadata_entry) + sizeof(std::size_t)));
        charge(product(tensor_count, sizeof(tensor_info) + 2 * sizeof(std::size_t)));
        document_.metadata_.reserve(to_size(metadata_count, reader_.position()));
        document_.tensors_.reserve(to_size(tensor_count, reader_.position()));
        for (std::uint64_t i = 0; i < metadata_count; ++i) {
            auto key = read_string(65535);
            if (!valid_key(key)) {
                fail(error_code::invalid_value, "invalid metadata key");
            }
            const auto type = read_type();
            auto value = read_value(type);
            document_.metadata_.push_back({std::move(key), std::move(value)});
        }
        make_index(document_.metadata_, document_.metadata_index_,
                   [](const auto& entry) -> std::string_view { return entry.key; });
        if (const auto* value = document_.find_metadata("general.alignment")) {
            const auto* alignment = value->get_if<std::uint32_t>();
            if (alignment == nullptr || *alignment == 0 || *alignment % 8 != 0) {
                fail(error_code::invalid_alignment,
                     "alignment must be a positive uint32 multiple of 8");
            }
            document_.alignment_ = *alignment;
        }
        for (std::uint64_t i = 0; i < tensor_count; ++i) {
            document_.tensors_.push_back(read_tensor());
        }
        make_index(document_.tensors_, document_.tensor_index_,
                   [](const auto& tensor) -> std::string_view { return tensor.name; });
        const auto position = reader_.position();
        const auto alignment = document_.alignment_;
        const auto padding = (alignment - position % alignment) % alignment;
        document_.data_offset_ = checked_add(position, padding, position);
        // Metadata-only files need not contain an unused tensor data section.
        if (tensor_count != 0) {
            reader_.require(padding);
        }
        validate_tensors();
        return std::move(document_);
    }

  private:
    [[noreturn]] void fail(error_code code, const std::string& message) const {
        throw ParseError(code, reader_.position(), message);
    }
    void limit(std::uint64_t count, std::uint64_t maximum, const char* message) const {
        if (count > maximum) {
            fail(error_code::resource_limit, message);
        }
    }
    auto product(std::uint64_t lhs, std::uint64_t rhs) const -> std::uint64_t {
        return checked_multiply(lhs, rhs, reader_.position());
    }
    void charge(std::uint64_t bytes) {
        allocated_ = checked_add(allocated_, bytes, reader_.position());
        limit(allocated_, options_.max_allocation_bytes, "parsed data exceeds allocation budget");
    }
    auto read_type() -> value_type {
        const auto type = reader_.read<std::uint32_t>();
        if (type > static_cast<std::uint32_t>(value_type::float64)) {
            fail(error_code::invalid_type, "unknown metadata type");
        }
        return static_cast<value_type>(type);
    }
    auto read_string(std::uint64_t maximum) -> std::string {
        const auto size = reader_.read<std::uint64_t>();
        limit(size, std::min(maximum, options_.max_string_bytes), "string length exceeds limit");
        reader_.require(size);
        charge(checked_add(size, 1, reader_.position()));
        std::string result(to_size(size, reader_.position()), '\0');
        reader_.read_bytes(std::as_writable_bytes(std::span{result.data(), result.size()}));
        if (!valid_utf8(result)) {
            fail(error_code::invalid_value, "invalid UTF-8 string");
        }
        return result;
    }
    template <class T> auto read_element(std::uint32_t depth) -> T {
        if constexpr (std::is_same_v<T, std::string>) {
            return read_string(options_.max_string_bytes);
        } else if constexpr (std::is_same_v<T, MetadataArray>) {
            return read_array(depth + 1);
        } else if constexpr (std::is_same_v<T, bool>) {
            const auto value = reader_.read<std::uint8_t>();
            if (value > 1) {
                fail(error_code::invalid_value, "boolean must be 0 or 1");
            }
            return value != 0;
        } else {
            return reader_.read<T>();
        }
    }
    auto read_value(value_type type) -> MetadataValue {
        return dispatch(
            type, [this]<class T>() -> MetadataValue { return MetadataValue{read_element<T>(0)}; });
    }
    auto read_array(std::uint32_t depth) -> MetadataArray {
        limit(depth, options_.max_array_depth, "metadata array nesting exceeds limit");
        const auto type = read_type();
        const auto count = reader_.read<std::uint64_t>();
        limit(count, options_.max_array_elements, "array length exceeds limit");
        total_array_elements_ = checked_add(total_array_elements_, count, reader_.position());
        limit(total_array_elements_, options_.max_total_array_elements,
              "total array element count exceeds limit");
        return dispatch(type, [this, count, depth]<class T>() -> MetadataArray {
            constexpr auto wire_size = [] {
                if constexpr (std::is_same_v<T, std::string>) {
                    return std::size_t{8};
                } else if constexpr (std::is_same_v<T, MetadataArray>) {
                    return std::size_t{12};
                } else {
                    return sizeof(T);
                }
            }();
            reader_.require(product(count, wire_size));
            charge(product(count, sizeof(T)));
            std::vector<T> values;
            const auto size = to_size(count, reader_.position());
            if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
                values.resize(size);
                if (reader_.native_order() || sizeof(T) == 1) {
                    reader_.read_bytes(std::as_writable_bytes(std::span{values}));
                } else {
                    for (auto& value : values) {
                        value = reader_.read<T>();
                    }
                }
            } else {
                values.reserve(size);
                for (std::size_t i = 0; i < size; ++i) {
                    values.push_back(read_element<T>(depth));
                }
            }
            return MetadataArray{std::move(values)};
        });
    }
    auto read_tensor() -> tensor_info {
        tensor_info tensor;
        tensor.name = read_string(64);
        if (tensor.name.empty() || tensor.name.find('\0') != std::string::npos) {
            fail(error_code::invalid_tensor, "invalid tensor name");
        }
        const auto dimensions = reader_.read<std::uint32_t>();
        if (dimensions == 0 || dimensions > 4) {
            fail(error_code::invalid_tensor, "tensor must have 1 to 4 dimensions");
        }
        charge(product(dimensions, sizeof(std::uint64_t)));
        tensor.dimensions.reserve(dimensions);
        tensor.element_count = 1;
        for (std::uint32_t i = 0; i < dimensions; ++i) {
            const auto size = reader_.read<std::uint64_t>();
            if (size == 0) {
                fail(error_code::invalid_tensor, "tensor dimensions must be positive");
            }
            tensor.element_count = product(tensor.element_count, size);
            tensor.dimensions.push_back(size);
        }
        tensor.type = static_cast<tensor_type>(reader_.read<std::uint32_t>());
        const auto layout = layout_of(tensor.type);
        if (!layout) {
            fail(error_code::invalid_type, "unsupported tensor encoding");
        }
        if (tensor.dimensions.front() % layout->block_elements != 0) {
            fail(error_code::invalid_tensor,
                 "first tensor dimension is not a whole encoding block");
        }
        tensor.byte_size =
            product(tensor.element_count / layout->block_elements, layout->block_bytes);
        tensor.offset = reader_.read<std::uint64_t>();
        if (tensor.offset % document_.alignment_ != 0) {
            fail(error_code::invalid_alignment, "tensor offset is not aligned");
        }
        return tensor;
    }
    template <class Entries, class Name>
    void make_index(const Entries& entries, std::vector<std::size_t>& index, Name name) {
        index.resize(entries.size());
        std::iota(index.begin(), index.end(), std::size_t{0});
        std::sort(index.begin(), index.end(),
                  [&](auto lhs, auto rhs) { return name(entries[lhs]) < name(entries[rhs]); });
        for (std::size_t i = 1; i < index.size(); ++i) {
            if (name(entries[index[i - 1]]) == name(entries[index[i]])) {
                fail(error_code::duplicate_name, "duplicate metadata key or tensor name");
            }
        }
    }
    void validate_tensors() {
        for (auto& tensor : document_.tensors_) {
            tensor.file_offset =
                checked_add(document_.data_offset_, tensor.offset, reader_.position());
            if (tensor.file_offset > reader_.size() ||
                tensor.byte_size > reader_.size() - tensor.file_offset) {
                fail(error_code::unexpected_end, "tensor payload extends beyond input");
            }
        }
        std::vector<std::size_t> order(document_.tensors_.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](auto lhs, auto rhs) {
            return document_.tensors_[lhs].offset < document_.tensors_[rhs].offset;
        });
        std::uint64_t end = 0;
        for (const auto index : order) {
            const auto& tensor = document_.tensors_[index];
            if (tensor.offset < end) {
                fail(error_code::invalid_tensor, "tensor payload ranges overlap");
            }
            end = checked_add(tensor.offset, tensor.byte_size, reader_.position());
        }
    }

    BinaryReader& reader_;
    const parse_options& options_;
    Document document_;
    std::uint64_t allocated_{};
    std::uint64_t total_array_elements_{};
};

auto parse_reader(BinaryReader& reader, const parse_options& options) -> Document {
    return Parser{reader, options}.run();
}

} // namespace ggufpp::detail

namespace ggufpp {

auto parse(std::span<const std::byte> bytes, const parse_options& options) -> Document {
    detail::BinaryReader reader{bytes, options.byte_order};
    return detail::parse_reader(reader, options);
}

auto Document::find_metadata(std::string_view key) const noexcept -> const MetadataValue* {
    const auto it =
        std::lower_bound(metadata_index_.begin(), metadata_index_.end(), key,
                         [this](auto index, auto name) { return metadata_[index].key < name; });
    if (it == metadata_index_.end() || metadata_[*it].key != key) {
        return nullptr;
    }
    return &metadata_[*it].value;
}

auto Document::find_tensor(std::string_view name) const noexcept -> const tensor_info* {
    const auto it =
        std::lower_bound(tensor_index_.begin(), tensor_index_.end(), name,
                         [this](auto index, auto key) { return tensors_[index].name < key; });
    if (it == tensor_index_.end() || tensors_[*it].name != name) {
        return nullptr;
    }
    return &tensors_[*it];
}

} // namespace ggufpp
