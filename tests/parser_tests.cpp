#include <ggufpp/ggufpp.hpp>

#include "fixture.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using ggufpp::error_code;
using test_support::Fixture;
using test_support::Writer;

void check(bool condition, const std::source_location location = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(std::string{location.file_name()} + ':' +
                                 std::to_string(location.line()) + " check failed");
    }
}

template <class Function> void expect_error(error_code code, Function&& function) {
    try {
        function();
    } catch (const ggufpp::ParseError& error) {
        check(error.code() == code);
        check(std::string_view{error.what()}.find("byte") != std::string_view::npos);
        return;
    }
    throw std::runtime_error("expected ParseError");
}

template <class Exception, class Function> void expect_exception(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("expected exception");
}

template <class T> auto get(const ggufpp::Document& document, std::string_view key) -> const T& {
    const auto* value = document.find_metadata(key);
    check(value != nullptr);
    const auto* result = value->get_if<T>();
    check(result != nullptr);
    return *result;
}

template <class T>
void overwrite(std::vector<std::byte>& bytes, std::size_t offset, T value,
               std::endian order = std::endian::little) {
    Writer writer{order};
    writer.number(value);
    check(offset + sizeof(T) <= bytes.size());
    std::copy(writer.bytes().begin(), writer.bytes().end(),
              bytes.begin() + static_cast<std::ptrdiff_t>(offset));
}

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        std::random_device random;
        for (int attempt = 0; attempt < 20; ++attempt) {
            path_ = std::filesystem::temp_directory_path() /
                    ("ggufpp-test-" + std::to_string(random()) + '-' + std::to_string(attempt));
            if (std::filesystem::create_directory(path_)) {
                return;
            }
        }
        throw std::runtime_error("cannot create temporary test directory");
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    auto operator=(const TemporaryDirectory&) -> TemporaryDirectory& = delete;
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    auto file(std::string_view name) const -> std::filesystem::path { return path_ / name; }

  private:
    std::filesystem::path path_;
};

void save(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream stream{path, std::ios::binary};
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    check(static_cast<bool>(stream));
}

void basic_and_ownership() {
    auto bytes = test_support::basic_fixture();
    auto document = ggufpp::parse(bytes);
    bytes.clear();
    bytes.shrink_to_fit();
    check(document.version() == 3);
    check(document.alignment() == 32);
    check(document.byte_order() == std::endian::little);
    check(document.metadata().size() == 3);
    check(get<std::string>(document, "general.architecture") == "llama");
    check(document.find_metadata("missing") == nullptr);
    check(document.find_metadata("general.name")->get_if<double>() == nullptr);
    check(document.find_tensor("missing") == nullptr);
    const auto* tensor = document.find_tensor("weight");
    check(tensor != nullptr);
    check(tensor->byte_size == 32 && tensor->element_count == 8);
    check(tensor->dimensions == std::vector<std::uint64_t>({4, 2}));
    check(tensor->file_offset == document.data_offset());
    check(document.file_size() == tensor->file_offset + 32);
    auto copy = document;
    auto moved = std::move(document);
    check(get<std::string>(copy, "general.name") == "tiny test");
    check(get<std::string>(moved, "general.name") == "tiny test");
}

void versions_and_byte_order() {
    for (const auto version : {2U, 3U}) {
        Fixture fixture{std::endian::little, version};
        check(ggufpp::parse(fixture.finish()).version() == version);
    }
    for (const auto version : {0U, 1U, 4U, 0xffffffffU}) {
        Fixture fixture{std::endian::little, version};
        expect_error(error_code::unsupported_version,
                     [&] { (void)ggufpp::parse(fixture.finish()); });
    }
    auto bytes = test_support::basic_fixture(std::endian::big);
    ggufpp::parse_options options;
    options.byte_order = std::endian::big;
    const auto document = ggufpp::parse(bytes, options);
    check(document.byte_order() == std::endian::big);
    check(get<std::uint32_t>(document, "general.alignment") == 32);
    check(document.tensors()[0].byte_size == 32);
    expect_error(error_code::unsupported_version, [&] { (void)ggufpp::parse(bytes); });
    Fixture v2{std::endian::big, 2};
    expect_error(error_code::unsupported_version,
                 [&] { (void)ggufpp::parse(v2.finish(), options); });
    options.byte_order = static_cast<std::endian>(123);
    expect_exception<std::invalid_argument>([&] { (void)ggufpp::parse(bytes, options); });
}

void scalars() {
    for (const auto order : {std::endian::little, std::endian::big}) {
        Fixture fixture{order};
        fixture.metadata("test.u8", 0).number<std::uint8_t>(255);
        fixture.metadata("test.i8", 1).number<std::int8_t>(-128);
        fixture.metadata("test.u16", 2).number<std::uint16_t>(65535);
        fixture.metadata("test.i16", 3).number<std::int16_t>(-32768);
        fixture.metadata("test.u32", 4).number<std::uint32_t>(0xfedcba98);
        fixture.metadata("test.i32", 5).number<std::int32_t>(-12345678);
        fixture.metadata("test.f32", 6).number<float>(1.25F);
        fixture.metadata("test.bool", 7).number<std::uint8_t>(1);
        fixture.metadata("test.string", 8).string("hello \xe4\xb8\x96\xe7\x95\x8c");
        fixture.metadata("test.u64", 10).number<std::uint64_t>(0xfedcba9876543210ULL);
        fixture.metadata("test.i64", 11)
            .number<std::int64_t>(std::numeric_limits<std::int64_t>::min());
        fixture.metadata("test.f64", 12).number<double>(-3.125);
        ggufpp::parse_options options;
        options.byte_order = order;
        const auto document = ggufpp::parse(fixture.finish(), options);
        check(get<std::uint8_t>(document, "test.u8") == 255);
        check(get<std::int8_t>(document, "test.i8") == -128);
        check(get<std::uint16_t>(document, "test.u16") == 65535);
        check(get<std::int16_t>(document, "test.i16") == -32768);
        check(get<std::uint32_t>(document, "test.u32") == 0xfedcba98);
        check(get<std::int32_t>(document, "test.i32") == -12345678);
        check(get<float>(document, "test.f32") == 1.25F);
        check(get<bool>(document, "test.bool"));
        check(get<std::string>(document, "test.string") == "hello \xe4\xb8\x96\xe7\x95\x8c");
        check(get<std::uint64_t>(document, "test.u64") == 0xfedcba9876543210ULL);
        check(get<std::int64_t>(document, "test.i64") == std::numeric_limits<std::int64_t>::min());
        check(get<double>(document, "test.f64") == -3.125);
        for (const auto& entry : document.metadata()) {
            check(entry.value.storage().index() == static_cast<unsigned>(entry.value.type()));
        }
    }
}

template <class T> void numeric_array(std::uint32_t type, T first, T second) {
    for (const auto order : {std::endian::little, std::endian::big}) {
        Fixture fixture{order};
        auto& writer = fixture.metadata("test.array", 9);
        writer.number(type);
        writer.number<std::uint64_t>(2);
        writer.number(first);
        writer.number(second);
        ggufpp::parse_options options;
        options.byte_order = order;
        const auto document = ggufpp::parse(fixture.finish(), options);
        const auto& array = get<ggufpp::MetadataArray>(document, "test.array");
        check(static_cast<unsigned>(array.element_type()) == type);
        const auto* values = array.get_if<T>();
        check(values != nullptr && values->size() == 2);
        check((*values)[0] == first && (*values)[1] == second);
    }
}

void arrays() {
    numeric_array<std::uint8_t>(0, 0, 255);
    numeric_array<std::int8_t>(1, -128, 127);
    numeric_array<std::uint16_t>(2, 0x1234, 0xffff);
    numeric_array<std::int16_t>(3, -32768, 32767);
    numeric_array<std::uint32_t>(4, 0x12345678, 0xffffffff);
    numeric_array<std::int32_t>(5, -123456, 123456);
    numeric_array<float>(6, 1.25F, -2.5F);
    numeric_array<std::uint64_t>(10, 0x123456789abcdef0ULL, 0xffffffffffffffffULL);
    numeric_array<std::int64_t>(11, std::numeric_limits<std::int64_t>::min(), 123456);
    numeric_array<double>(12, -3.125, 5.75);
    Fixture fixture;
    auto& booleans = fixture.metadata("test.bools", 9);
    booleans.number<std::uint32_t>(7);
    booleans.number<std::uint64_t>(2);
    booleans.number<std::uint8_t>(0);
    booleans.number<std::uint8_t>(1);
    auto& strings = fixture.metadata("test.strings", 9);
    strings.number<std::uint32_t>(8);
    strings.number<std::uint64_t>(2);
    strings.string("");
    strings.string(std::string_view{"a\0b", 3});
    auto& nested = fixture.metadata("test.nested", 9);
    nested.number<std::uint32_t>(9);
    nested.number<std::uint64_t>(2);
    nested.number<std::uint32_t>(4);
    nested.number<std::uint64_t>(0);
    nested.number<std::uint32_t>(8);
    nested.number<std::uint64_t>(1);
    nested.string("nested");
    const auto document = ggufpp::parse(fixture.finish());
    const auto* bools = get<ggufpp::MetadataArray>(document, "test.bools").get_if<bool>();
    check(bools != nullptr && !(*bools)[0] && (*bools)[1]);
    const auto* texts = get<ggufpp::MetadataArray>(document, "test.strings").get_if<std::string>();
    check(texts != nullptr && (*texts)[0].empty() && (*texts)[1] == std::string("a\0b", 3));
    const auto& outer = get<ggufpp::MetadataArray>(document, "test.nested");
    const auto* children = outer.get_if<ggufpp::MetadataArray>();
    check(children != nullptr && children->size() == 2);
    check((*children)[0].element_type() == ggufpp::value_type::uint32);
    check((*children)[0].size() == 0);
    check((*children)[1].get_if<std::string>()->front() == "nested");
    check(outer.get_if<float>() == nullptr);
}

void all_tensor_encodings() {
    // Independent expected block element/byte counts, including reserved slots.
    constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 44> expected{
        {{1, 4},     {1, 2},     {32, 18},   {32, 20},  {0, 0},     {0, 0},     {32, 22},
         {32, 24},   {32, 34},   {32, 36},   {256, 84}, {256, 110}, {256, 144}, {256, 176},
         {256, 210}, {256, 292}, {256, 66},  {256, 74}, {256, 98},  {256, 50},  {32, 18},
         {256, 110}, {256, 82},  {256, 136}, {1, 1},    {1, 2},     {1, 4},     {1, 8},
         {1, 8},     {256, 56},  {1, 2},     {0, 0},    {0, 0},     {0, 0},     {256, 54},
         {256, 66},  {0, 0},     {0, 0},     {0, 0},    {32, 17},   {64, 36},   {128, 18},
         {64, 18},   {1, 1}}};
    for (std::uint32_t type = 0; type < expected.size(); ++type) {
        const auto [elements, bytes] = expected[type];
        const auto layout = ggufpp::layout_of(static_cast<ggufpp::tensor_type>(type));
        if (elements == 0) {
            check(!layout);
            Fixture fixture;
            constexpr std::array<std::uint64_t, 1> shape{256};
            fixture.tensor("removed", shape, type, 0, 1);
            expect_error(error_code::invalid_type, [&] { (void)ggufpp::parse(fixture.finish()); });
            continue;
        }
        check(layout && layout->block_elements == elements && layout->block_bytes == bytes);
        Fixture fixture;
        const std::array<std::uint64_t, 2> shape{elements * 2, 3};
        fixture.tensor("weight", shape, type, 0, bytes * 6);
        const auto document = ggufpp::parse(fixture.finish());
        check(document.tensors()[0].byte_size == bytes * 6);
        check(document.tensors()[0].element_count == elements * 6);
    }
    check(!ggufpp::layout_of(static_cast<ggufpp::tensor_type>(999)));
}

void alignment_and_tensor_order() {
    Fixture fixture;
    fixture.metadata("general.alignment", 4).number<std::uint32_t>(24);
    constexpr std::array<std::uint64_t, 1> shape{4};
    fixture.tensor("z", shape, 0, 48, 16);
    fixture.tensor("a", shape, 0, 0, 16);
    const auto document = ggufpp::parse(fixture.finish(24));
    check(document.alignment() == 24 && document.data_offset() % 24 == 0);
    check(document.tensors()[0].name == "z");
    check(document.find_tensor("a")->offset == 0);
    check(document.find_tensor("z")->file_offset == document.data_offset() + 48);
}

void malformed_metadata() {
    for (const std::string key :
         {"", ".bad", "bad.", "bad..key", "Bad.key", "bad-key", "bad key"}) {
        Fixture fixture;
        fixture.metadata(key, 0).number<std::uint8_t>(0);
        expect_error(error_code::invalid_value, [&] { (void)ggufpp::parse(fixture.finish()); });
    }
    for (const std::string& text : {std::string("\xc0\x80", 2), std::string("\xed\xa0\x80", 3),
                                    std::string("\xf4\x90\x80\x80", 4), std::string("\xe4\xb8", 2),
                                    std::string("\xff", 1), std::string("\xe0\x80\x80", 3)}) {
        Fixture fixture;
        fixture.metadata("test.string", 8).string(text);
        expect_error(error_code::invalid_value, [&] { (void)ggufpp::parse(fixture.finish()); });
    }
    Fixture boolean;
    boolean.metadata("test.bool", 7).number<std::uint8_t>(2);
    expect_error(error_code::invalid_value, [&] { (void)ggufpp::parse(boolean.finish()); });
    Fixture type;
    type.metadata("test.type", 99).number<std::uint8_t>(0);
    expect_error(error_code::invalid_type, [&] { (void)ggufpp::parse(type.finish()); });
    Fixture array;
    auto& writer = array.metadata("test.array", 9);
    writer.number<std::uint32_t>(99);
    writer.number<std::uint64_t>(0);
    expect_error(error_code::invalid_type, [&] { (void)ggufpp::parse(array.finish()); });
    Fixture duplicate;
    duplicate.metadata("test.key", 0).number<std::uint8_t>(0);
    duplicate.metadata("test.key", 0).number<std::uint8_t>(1);
    expect_error(error_code::duplicate_name, [&] { (void)ggufpp::parse(duplicate.finish()); });
}

void malformed_tensors() {
    constexpr std::array<std::uint64_t, 1> shape{32};
    for (const auto alignment : {0U, 1U, 7U, 12U}) {
        Fixture fixture;
        fixture.metadata("general.alignment", 4).number(alignment);
        expect_error(error_code::invalid_alignment, [&] { (void)ggufpp::parse(fixture.finish()); });
    }
    Fixture wrong_alignment_type;
    wrong_alignment_type.metadata("general.alignment", 10).number<std::uint64_t>(32);
    expect_error(error_code::invalid_alignment,
                 [&] { (void)ggufpp::parse(wrong_alignment_type.finish()); });
    Fixture unaligned;
    unaligned.tensor("weight", shape, 0, 1, 128);
    expect_error(error_code::invalid_alignment, [&] { (void)ggufpp::parse(unaligned.finish()); });
    Fixture duplicate;
    duplicate.tensor("weight", shape, 0, 0, 128);
    duplicate.tensor("weight", shape, 0, 128, 128);
    expect_error(error_code::duplicate_name, [&] { (void)ggufpp::parse(duplicate.finish()); });
    Fixture overlap;
    overlap.tensor("a", shape, 0, 0, 128);
    overlap.tensor("b", shape, 0, 32, 128);
    expect_error(error_code::invalid_tensor, [&] { (void)ggufpp::parse(overlap.finish()); });
    Fixture quantized;
    constexpr std::array<std::uint64_t, 2> invalid_block_shape{16, 2};
    quantized.tensor("weight", invalid_block_shape, 2, 0, 18);
    expect_error(error_code::invalid_tensor, [&] { (void)ggufpp::parse(quantized.finish()); });
    for (const auto& dimensions : {std::vector<std::uint64_t>{}, std::vector<std::uint64_t>{0},
                                   std::vector<std::uint64_t>{1, 1, 1, 1, 1}}) {
        Fixture fixture;
        fixture.tensor("weight", dimensions, 0, 0, 4);
        expect_error(error_code::invalid_tensor, [&] { (void)ggufpp::parse(fixture.finish()); });
    }
    Fixture name;
    name.tensor("", shape, 0, 0, 128);
    expect_error(error_code::invalid_tensor, [&] { (void)ggufpp::parse(name.finish()); });
    Fixture long_name;
    long_name.tensor(std::string(65, 'x'), shape, 0, 0, 128);
    expect_error(error_code::resource_limit, [&] { (void)ggufpp::parse(long_name.finish()); });
}

void truncation_and_overflow() {
    const auto bytes = test_support::basic_fixture();
    for (std::size_t size = 0; size < bytes.size(); ++size) {
        try {
            (void)ggufpp::parse(std::span{bytes}.first(size));
            check(false);
        } catch (const ggufpp::ParseError&) {
            // Every prefix of this tensor-bearing file is incomplete.
        }
    }
    auto bad_magic = bytes;
    bad_magic[0] = std::byte{0};
    expect_error(error_code::invalid_magic, [&] { (void)ggufpp::parse(bad_magic); });
    Fixture overflow;
    const std::array<std::uint64_t, 2> shape{std::numeric_limits<std::uint64_t>::max(), 2};
    overflow.tensor("overflow", shape, 0, 0, 1);
    expect_error(error_code::integer_overflow, [&] { (void)ggufpp::parse(overflow.finish()); });
    Fixture storage_overflow;
    const std::array<std::uint64_t, 1> huge{std::numeric_limits<std::uint64_t>::max() / 2};
    storage_overflow.tensor("huge", huge, 0, 0, 1);
    expect_error(error_code::integer_overflow,
                 [&] { (void)ggufpp::parse(storage_overflow.finish()); });
    auto offset_overflow = test_support::basic_fixture();
    const auto data_offset = ggufpp::parse(offset_overflow).data_offset();
    // Last descriptor field immediately precedes section padding; locate it by name length.
    constexpr std::size_t descriptor_start =
        24 + (8 + 20 + 4 + 8 + 5) + (8 + 12 + 4 + 8 + 9) + (8 + 17 + 4 + 4);
    constexpr std::size_t offset_field = descriptor_start + 8 + 6 + 4 + 16 + 4;
    check(offset_field + 8 <= data_offset);
    overwrite(offset_overflow, offset_field, std::uint64_t{0xffffffffffffffe0ULL});
    expect_error(error_code::integer_overflow, [&] { (void)ggufpp::parse(offset_overflow); });
}

void resource_limits() {
    const auto bytes = test_support::basic_fixture();
    ggufpp::parse_options options;
    options.max_metadata_entries = 2;
    expect_error(error_code::resource_limit, [&] { (void)ggufpp::parse(bytes, options); });
    options = {};
    options.max_tensors = 0;
    expect_error(error_code::resource_limit, [&] { (void)ggufpp::parse(bytes, options); });
    options = {};
    options.max_string_bytes = 4;
    expect_error(error_code::resource_limit, [&] { (void)ggufpp::parse(bytes, options); });
    options = {};
    options.max_allocation_bytes = 1;
    expect_error(error_code::resource_limit, [&] { (void)ggufpp::parse(bytes, options); });
    Fixture fixture;
    auto& writer = fixture.metadata("test.array", 9);
    writer.number<std::uint32_t>(0);
    writer.number<std::uint64_t>(3);
    writer.number<std::uint8_t>(1);
    writer.number<std::uint8_t>(2);
    writer.number<std::uint8_t>(3);
    options = {};
    options.max_array_elements = 2;
    expect_error(error_code::resource_limit,
                 [&] { (void)ggufpp::parse(fixture.finish(), options); });
    options = {};
    options.max_total_array_elements = 2;
    expect_error(error_code::resource_limit,
                 [&] { (void)ggufpp::parse(fixture.finish(), options); });
    options = {};
    options.max_array_depth = 0;
    expect_error(error_code::resource_limit,
                 [&] { (void)ggufpp::parse(fixture.finish(), options); });
    options.max_array_depth = 65;
    expect_exception<std::invalid_argument>([&] { (void)ggufpp::parse(bytes, options); });
    auto huge_count = Fixture{}.finish();
    overwrite(huge_count, 16, std::uint64_t{0xffffffffffffffffULL});
    expect_error(error_code::resource_limit, [&] { (void)ggufpp::parse(huge_count); });
}

void file_reads_and_moves() {
    static_assert(!std::is_copy_constructible_v<ggufpp::File>);
    static_assert(std::is_move_constructible_v<ggufpp::File>);
    TemporaryDirectory directory;
    const auto path = directory.file("model.gguf");
    const auto bytes = test_support::basic_fixture();
    save(path, bytes);
    auto file = ggufpp::File::open(path);
    std::array<std::byte, 32> payload{};
    file.read_tensor(0, 0, payload);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        check(payload[i] == static_cast<std::byte>(i));
    }
    std::array<std::byte, 3> chunk{};
    file.read_tensor(0, 12, chunk);
    check(chunk[0] == std::byte{12} && chunk[2] == std::byte{14});
    file.read_tensor(0, 15, chunk);
    check(chunk[0] == std::byte{15} && chunk[2] == std::byte{17});
    file.read_tensor(0, 32, {});
    expect_exception<std::out_of_range>([&] { file.read_tensor(1, 0, {}); });
    expect_exception<std::out_of_range>([&] { file.read_tensor(0, 33, {}); });
    expect_exception<std::out_of_range>([&] { file.read_tensor(0, 31, chunk); });
    expect_exception<std::out_of_range>([&] { file.read_tensor(0, 0xffffffffffffffffULL, chunk); });
    auto moved = std::move(file);
    moved.read_tensor(0, 0, chunk);
    check(chunk[0] == std::byte{0});
    auto assigned = ggufpp::File::open(path);
    assigned = std::move(moved);
    assigned.read_tensor(0, 1, chunk);
    check(chunk[0] == std::byte{1});
    expect_error(error_code::io_error,
                 [&] { (void)ggufpp::File::open(directory.file("missing")); });
    save(path, std::span{bytes}.first(10));
    expect_error(error_code::unexpected_end, [&] { (void)ggufpp::File::open(path); });
}

void buffered_metadata_and_files() {
    TemporaryDirectory directory;
    for (const auto order : {std::endian::little, std::endian::big}) {
        Fixture fixture{order};
        fixture.metadata("general.description", 8).string(std::string(200'000, 'x'));
        auto& array = fixture.metadata("test.values", 9);
        array.number<std::uint32_t>(4);
        array.number<std::uint64_t>(50'000);
        for (std::uint32_t i = 0; i < 50'000; ++i) {
            array.number(i);
        }
        constexpr std::array<std::uint64_t, 1> shape{100'000};
        fixture.tensor("weight", shape, 24, 0, 100'000);
        const auto path = directory.file("buffered.gguf");
        save(path, fixture.finish());
        ggufpp::parse_options options;
        options.byte_order = order;
        auto file = ggufpp::File::open(path, options);
        check(get<std::string>(file.document(), "general.description").size() == 200'000);
        const auto* values =
            get<ggufpp::MetadataArray>(file.document(), "test.values").get_if<std::uint32_t>();
        check(values != nullptr && values->back() == 49'999);
        std::vector<std::byte> payload(100'000);
        file.read_tensor(0, 0, payload);
        for (std::size_t i = 0; i < payload.size(); ++i) {
            check(payload[i] == static_cast<std::byte>(i % 251));
        }
        std::filesystem::resize_file(path, file.document().data_offset());
        expect_error(error_code::io_error,
                     [&] { file.read_tensor(0, 90'000, std::span{payload}.first(8)); });
    }
}

void mapped_files() {
#if GGUFPP_HAS_MMAP
    static_assert(!std::is_copy_constructible_v<ggufpp::MappedFile>);
    static_assert(std::is_move_constructible_v<ggufpp::MappedFile>);
    TemporaryDirectory directory;
    const auto path = directory.file("mapped.gguf");
    const auto bytes = test_support::basic_fixture();
    save(path, bytes);

    auto mapped = ggufpp::MappedFile::open(path);
    check(mapped.document().version() == 3);
    check(mapped.document().file_size() == bytes.size());
    check(mapped.bytes().size() == bytes.size());
    check(std::equal(mapped.bytes().begin(), mapped.bytes().end(), bytes.begin(), bytes.end()));
    check(mapped.document().find_tensor_index("weight").has_value());
    check(!mapped.document().find_tensor_index("missing").has_value());
    const auto* tensor = mapped.document().find_tensor("weight");
    check(tensor != nullptr);

    const auto data = mapped.tensor_data(0);
    check(data.size() == tensor->byte_size);
    // Zero copy: the view points into the file mapping, right after the metadata.
    check(data.data() == mapped.bytes().data() + tensor->file_offset);
    for (std::size_t i = 0; i < data.size(); ++i) {
        check(data[i] == static_cast<std::byte>(i));
    }

    const auto by_name = mapped.tensor_data("weight");
    check(by_name.has_value());
    check(by_name->data() == data.data() && by_name->size() == data.size());
    check(!mapped.tensor_data("missing").has_value());
    expect_exception<std::out_of_range>([&] { (void)mapped.tensor_data(1); });

    // Mapping and stream reads expose the same encoded bytes.
    std::array<std::byte, 32> copy{};
    {
        auto file = ggufpp::File::open(path);
        file.read_tensor(0, 0, copy);
    }
    check(std::equal(copy.begin(), copy.end(), data.begin(), data.end()));

    // Moving the owner keeps the mapping address and existing views valid.
    auto moved = std::move(mapped);
    const auto moved_data = moved.tensor_data(0);
    check(moved_data.data() == data.data());
    check(std::equal(moved_data.begin(), moved_data.end(), copy.begin(), copy.end()));

    // Byte order selection applies to metadata; payload stays encoded as stored.
    ggufpp::parse_options options;
    options.byte_order = std::endian::big;
    const auto big_path = directory.file("big-endian.gguf");
    save(big_path, test_support::basic_fixture(std::endian::big));
    auto big = ggufpp::MappedFile::open(big_path, options);
    check(big.document().byte_order() == std::endian::big);
    check(big.tensor_data(0).size() == 32);

    // Metadata-only files map successfully and have no tensor payload views.
    test_support::Fixture metadata_only;
    metadata_only.metadata("general.name", 8).string("mapped");
    const auto metadata_path = directory.file("metadata-only.gguf");
    save(metadata_path, metadata_only.finish());
    auto metadata_mapped = ggufpp::MappedFile::open(metadata_path);
    check(metadata_mapped.document().tensors().empty());
    check(metadata_mapped.bytes().size() == std::filesystem::file_size(metadata_path));
    expect_exception<std::out_of_range>([&] { (void)metadata_mapped.tensor_data(0); });

    expect_error(error_code::io_error,
                 [&] { (void)ggufpp::MappedFile::open(directory.file("missing.gguf")); });
    const auto empty_path = directory.file("empty.gguf");
    save(empty_path, std::span<const std::byte>{});
    expect_error(error_code::unexpected_end, [&] { (void)ggufpp::MappedFile::open(empty_path); });
    const auto truncated_path = directory.file("truncated.gguf");
    save(truncated_path, std::span{bytes}.first(10));
    expect_error(error_code::unexpected_end,
                 [&] { (void)ggufpp::MappedFile::open(truncated_path); });
#endif
}

void sparse_large_file() {
    TemporaryDirectory directory;
    const auto path = directory.file("sparse.gguf");
// 8 GiB payload, while constructing and reading only a few bytes.
#if defined(_WIN32)
    // Standard C++ cannot mark NTFS files sparse; avoid allocating 8 GiB on Windows.
    constexpr std::uint64_t payload_size = std::uint64_t{1} << 23U;
#else
    constexpr std::uint64_t payload_size = std::uint64_t{1} << 33U;
#endif
    Fixture fixture;
    constexpr std::array<std::uint64_t, 1> shape{payload_size};
    fixture.tensor("large", shape, 24, 0, 0);
    const auto prefix = fixture.finish();
    {
        std::ofstream stream{path, std::ios::binary};
        stream.write(reinterpret_cast<const char*>(prefix.data()),
                     static_cast<std::streamsize>(prefix.size()));
        stream.seekp(static_cast<std::streamoff>(prefix.size() + payload_size - 1));
        stream.put('\x5a');
        check(static_cast<bool>(stream));
    }
    auto file = ggufpp::File::open(path);
    check(file.document().tensors()[0].byte_size == payload_size);
    check(file.document().file_size() > payload_size);
    std::array<std::byte, 1> last{};
    file.read_tensor(0, payload_size - 1, last);
    check(last[0] == std::byte{0x5a});
#if GGUFPP_HAS_MMAP && !defined(_WIN32)
    // Mapping the whole file stays cheap: only the touched page is faulted in.
    if (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
        auto mapped = ggufpp::MappedFile::open(path);
        const auto data = mapped.tensor_data(0);
        check(data.size() == payload_size);
        check(data.back() == std::byte{0x5a});
        check(data.front() == std::byte{0});
    }
#endif
}

void deterministic_mutations() {
    const auto original = test_support::basic_fixture();
    std::mt19937 random{12345};
    ggufpp::parse_options options;
    options.max_allocation_bytes = 1024 * 1024;
    options.max_array_elements = 4096;
    options.max_total_array_elements = 8192;
    options.max_metadata_entries = 1024;
    options.max_tensors = 1024;
    for (unsigned iteration = 0; iteration < 5000; ++iteration) {
        auto bytes = original;
        const auto changes = 1 + random() % 4;
        for (unsigned i = 0; i < changes; ++i) {
            bytes[random() % bytes.size()] = static_cast<std::byte>(random() % 256);
        }
        try {
            const auto document = ggufpp::parse(bytes, options);
            for (const auto& tensor : document.tensors()) {
                check(tensor.file_offset <= bytes.size());
                check(tensor.byte_size <= bytes.size() - tensor.file_offset);
            }
        } catch (const ggufpp::ParseError&) {
            // Mutations may remain valid; malformed ones must fail cleanly.
        }
    }
}

void golden_bytes() {
    // Hand-authored GGUF v3: one uint32 metadata value, no tensors or padding.
    constexpr std::array<unsigned char, 48> golden{
        0x47, 0x47, 0x55, 0x46, 3,   0,   0,   0,   0, 0, 0, 0, 0,  0, 0, 0,
        1,    0,    0,    0,    0,   0,   0,   0,   8, 0, 0, 0, 0,  0, 0, 0,
        't',  'e',  's',  't',  '.', 'k', 'e', 'y', 4, 0, 0, 0, 42, 0, 0, 0};
    const auto document = ggufpp::parse(std::as_bytes(std::span{golden}));
    check(document.metadata().size() == 1 && document.tensors().empty());
    check(get<std::uint32_t>(document, "test.key") == 42);
    check(document.file_size() == golden.size());
}

void nested_limits_and_invalid_arrays() {
    Fixture nested;
    auto& writer = nested.metadata("test.nested", 9);
    for (unsigned depth = 0; depth < 4; ++depth) {
        writer.number<std::uint32_t>(9);
        writer.number<std::uint64_t>(1);
    }
    writer.number<std::uint32_t>(0);
    writer.number<std::uint64_t>(0);
    ggufpp::parse_options options;
    options.max_array_depth = 4;
    expect_error(error_code::resource_limit,
                 [&] { (void)ggufpp::parse(nested.finish(), options); });
    options.max_array_depth = 5;
    check(ggufpp::parse(nested.finish(), options).metadata().size() == 1);

    Fixture booleans;
    auto& bools = booleans.metadata("test.bools", 9);
    bools.number<std::uint32_t>(7);
    bools.number<std::uint64_t>(1);
    bools.number<std::uint8_t>(255);
    expect_error(error_code::invalid_value, [&] { (void)ggufpp::parse(booleans.finish()); });

    Fixture arrays;
    for (const auto key : {"test.first", "test.second"}) {
        auto& array = arrays.metadata(key, 9);
        array.number<std::uint32_t>(0);
        array.number<std::uint64_t>(2);
        array.number<std::uint16_t>(0);
    }
    options = {};
    options.max_total_array_elements = 3;
    expect_error(error_code::resource_limit,
                 [&] { (void)ggufpp::parse(arrays.finish(), options); });

    Fixture overflow;
    auto& count = overflow.metadata("test.huge", 9);
    count.number<std::uint32_t>(12);
    count.number<std::uint64_t>(0xffffffffffffffffULL);
    options = {};
    options.max_array_elements = 0xffffffffffffffffULL;
    options.max_total_array_elements = 0xffffffffffffffffULL;
    expect_error(error_code::integer_overflow,
                 [&] { (void)ggufpp::parse(overflow.finish(), options); });

    Fixture unknown;
    constexpr std::array<std::uint64_t, 1> shape{1};
    unknown.tensor("unknown", shape, 999, 0, 1);
    expect_error(error_code::invalid_type, [&] { (void)ggufpp::parse(unknown.finish()); });

    Fixture null_name;
    null_name.tensor(std::string_view{"a\0b", 3}, shape, 0, 0, 4);
    expect_error(error_code::invalid_tensor, [&] { (void)ggufpp::parse(null_name.finish()); });

    auto huge_counts = Fixture{}.finish();
    overwrite(huge_counts, 16, std::uint64_t{0xffffffffffffffffULL});
    options = {};
    options.max_metadata_entries = 0xffffffffffffffffULL;
    expect_error(error_code::integer_overflow, [&] { (void)ggufpp::parse(huge_counts, options); });
}

} // namespace

int main() {
    const std::array tests{
        std::pair{"golden_bytes", &golden_bytes},
        std::pair{"nested_limits_and_invalid_arrays", &nested_limits_and_invalid_arrays},
        std::pair{"basic_and_ownership", &basic_and_ownership},
        std::pair{"versions_and_byte_order", &versions_and_byte_order},
        std::pair{"scalars", &scalars},
        std::pair{"arrays", &arrays},
        std::pair{"all_tensor_encodings", &all_tensor_encodings},
        std::pair{"alignment_and_tensor_order", &alignment_and_tensor_order},
        std::pair{"malformed_metadata", &malformed_metadata},
        std::pair{"malformed_tensors", &malformed_tensors},
        std::pair{"truncation_and_overflow", &truncation_and_overflow},
        std::pair{"resource_limits", &resource_limits},
        std::pair{"file_reads_and_moves", &file_reads_and_moves},
        std::pair{"mapped_files", &mapped_files},
        std::pair{"buffered_metadata_and_files", &buffered_metadata_and_files},
        std::pair{"sparse_large_file", &sparse_large_file},
        std::pair{"deterministic_mutations", &deterministic_mutations}};
    unsigned failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << tests.size() - failures << '/' << tests.size() << " test groups passed\n";
    return failures == 0 ? 0 : 1;
}
