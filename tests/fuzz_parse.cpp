#include <ggufpp/document.hpp>
#include <ggufpp/error.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

// The entry point name is required by libFuzzer.
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    ggufpp::parse_options options;
    options.max_allocation_bytes = 4 * 1024 * 1024;
    options.max_metadata_entries = 4096;
    options.max_tensors = 4096;
    options.max_array_elements = 65536;
    options.max_total_array_elements = 131072;
    options.max_string_bytes = 1024 * 1024;
    const auto bytes = std::as_bytes(std::span{data, size});
    for (const auto order : {std::endian::little, std::endian::big}) {
        options.byte_order = order;
        try {
            (void)ggufpp::parse(bytes, options);
        } catch (const ggufpp::ParseError&) {
            // Malformed input is expected; other failures must remain visible.
        }
    }
    return 0;
}
