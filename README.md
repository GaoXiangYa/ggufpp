# ggufpp

A small GGUF reader written in standard C++20. The library, examples, tests, and
benchmark have no third-party dependencies. Building requires a C++20 compiler
and CMake 3.20 or newer. No packages are downloaded during configuration.

- GGUF v2 (little endian) and v3 (little or explicitly selected big endian).
- All 13 metadata value types, including homogeneous and nested arrays.
- Owned metadata, binary-search lookup, and contiguous tensor descriptors.
- 36 active tensor encodings, with checked block sizes, offsets, and payload ranges.
- Buffered file opening and direct, bounded tensor reads into caller-owned memory.
- Zero-copy `mmap` payload access (`MappedFile`) for inference-style loaders; `File`
  remains the portable stream fallback.
- Configurable resource limits and exceptions carrying error codes and byte offsets.

This is a structural parser. It does not execute models, dequantize weights, write
GGUF files, or require model-specific metadata such as `general.architecture`.
`MappedFile` exposes raw encoded payload views; it still does not interpret them.
Unknown metadata keys are retained. Unknown and removed tensor encodings are
rejected because their payload sizes cannot be validated safely.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For multi-configuration generators, add `--config Release` to the build command
and `-C Release` to CTest. To build just the library, use
`-DBUILD_TESTING=OFF -DGGUFPP_BUILD_EXAMPLES=OFF`.

```sh
./build/ggufpp_inspect model.gguf
./build/ggufpp_inspect big-endian-model.gguf --big-endian
```

## Read a file

```cpp
#include <ggufpp/ggufpp.hpp>
#include <algorithm>
#include <array>
#include <iostream>

int main() {
    try {
        auto file = ggufpp::File::open("model.gguf");
        const auto& document = file.document();

        if (const auto* value = document.find_metadata("general.name")) {
            if (const auto* name = value->get_if<std::string>()) {
                std::cout << *name << '\n';
            }
        }

        std::array<std::byte, 4096> buffer{};
        for (std::size_t index = 0; index < document.tensors().size(); ++index) {
            const auto& tensor = document.tensors()[index];
            std::cout << tensor.name << ": " << tensor.byte_size << " encoded bytes\n";
            for (std::uint64_t offset = 0; offset < tensor.byte_size;) {
                const auto count = static_cast<std::size_t>(
                    std::min<std::uint64_t>(buffer.size(), tensor.byte_size - offset));
                file.read_tensor(index, offset, std::span{buffer}.first(count));
                // Consume buffer[0..count) here.
                offset += count;
            }
        }
    } catch (const ggufpp::ParseError& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

`File::open` parses metadata and the tensor directory, validates payload ranges
against the file size, and retains the stream. It never loads a complete tensor.
The metadata reader uses a fixed 64 KiB buffer, so read-ahead can include a bounded
prefix of the payload. `read_tensor` copies raw encoded bytes directly into the
provided span. Consecutive reads reuse the stream buffer without repeated seeks.
It performs no endian conversion or dequantization.

## Zero-copy mmap access

For inference-style loading, `ggufpp::MappedFile` maps the complete file read-only
and exposes tensor payload views that point straight into the mapping. Nothing is
copied, an open descriptor is not retained, and the OS page cache manages the
weights.

```cpp
#include <ggufpp/ggufpp.hpp>

int main() {
    auto file = ggufpp::MappedFile::open("model.gguf");
    const auto& document = file.document();

    for (std::size_t i = 0; i < document.tensors().size(); ++i) {
        const auto bytes = file.tensor_data(i); // std::span<const std::byte>
        // Point kernels at bytes.data()/bytes.size(); no copy, no dequantization.
    }

    const auto attention = file.tensor_data("blk.0.attn_q.weight"); // optional<span>
    if (attention) {
        // attention->data() is inside the read-only mapping.
    }
}
```

`bytes()` returns the whole mapping (metadata + payload), `tensor_data(index)`
returns the encoded bytes of one tensor, and `tensor_data(name)` returns
`std::optional<std::span<const std::byte>>`. `Document::find_tensor_index()` is
available when an integer index is needed.

Lifetimes and safety:

- Views borrow from the mapping; they stay valid while the `MappedFile` is alive
  (moving it does not remap, so pointers remain valid). Do not keep them after
  destruction.
- Const access is safe from multiple threads: both the mapping and the parsed
  document are read-only. The file must not be modified or truncated while it is
  mapped; a concurrent change can fault (`SIGBUS` on POSIX) or expose inconsistent
  bytes.
- The whole file is mapped; on 32-bit platforms a file larger than the address
  space fails with `ParseError` (`resource_limit`). Use `File` there.
- Supported backends: Windows (`CreateFileMappingW`) and POSIX (`mmap`: Linux,
  macOS, BSD). Other platforms throw `ParseError` (`io_error`); `File` remains
  fully portable.
- `MappedFile` does not retain an open descriptor; the mapping keeps the file
  alive.
- `GGUFPP_HAS_MMAP` is 1 when a backend is available for the including translation
  unit.

The mapping itself does not prefetch: the first touch of each page is a demand
fault handled by the OS page cache, so cold-cache throughput is device-bound and
warm re-reads are memory-bound. Payload bytes keep the file byte order; no
conversion or dequantization is performed.

## Parse memory and inspect arrays

```cpp
// bytes is a std::span<const std::byte> covering the complete GGUF file.
auto document = ggufpp::parse(bytes);

if (const auto* value = document.find_metadata("tokenizer.ggml.tokens")) {
    if (const auto* array = value->get_if<ggufpp::MetadataArray>()) {
        if (const auto* tokens = array->get_if<std::string>()) {
            // *tokens is a const std::vector<std::string>.
        }
    }
}
```

`parse` retains no input pointers and copies no tensor payload. The input buffer
may be destroyed after parsing. Metadata preserves its exact wire type;
`get_if<std::uint32_t>()` does not convert a stored `uint64` value. Empty arrays
retain their element type. Boolean arrays use `std::vector<bool>`; numeric arrays
use a contiguous vector of their concrete numeric type. Nested arrays contain
`MetadataArray` elements.

Returned spans, strings, and pointers borrow from the `Document`. They are valid
until its destruction, assignment, or move. A `Document` can be copied when
independent ownership is needed. Read-only access to an unchanged document can be
shared between threads. A `File` is move-only; concurrent reads on the same
instance require external synchronization.

## Byte order, validation, and errors

```cpp
ggufpp::parse_options options;
options.byte_order = std::endian::big; // GGUF v3; default is little endian.
options.max_allocation_bytes = 512ULL * 1024 * 1024;
auto file = ggufpp::File::open("model.gguf", options);
```

Byte order is explicit; it is not inferred from the magic. Tensor payloads keep
the original file byte order. Both memory and file parsing require all described
payload bytes to exist, but do not inspect the payload contents. `MappedFile::open`
accepts the same `parse_options` and applies them to the mapped bytes.

The parser checks UTF-8, metadata keys, booleans, duplicate names, dimensions,
quantization block divisibility, alignment, overflow, overlapping tensor ranges,
and truncation. Dimensions must be positive and have a rank between 1 and 4.
Keys contain nonempty dot-separated segments of lowercase ASCII letters, digits,
and underscores. Tensor names are nonempty, at most 64 bytes, and contain no NUL.
Alignment is a positive uint32 multiple of 8 and defaults to 32; a power of two
is not required. Padding contents are ignored. Metadata-only files may end after
the metadata without an unused alignment region. Trailing bytes are accepted.

Default limits: one million metadata entries and tensors each, 64 MiB per string,
16 million elements per array, 32 million array elements in total, depth 16, and
256 MiB of accounted parsed storage. The allocation budget covers requested
container elements, strings, and lookup/sorting indexes; it is not an exact
process RSS bound or allocator-overhead measurement. Depth can be raised to 64.

Malformed files and file I/O failures throw `ParseError`. Invalid option values
throw `std::invalid_argument`; invalid tensor read indexes or ranges throw
`std::out_of_range`. Allocation failures propagate as standard exceptions.
The file should remain unchanged while open; later I/O failures are reported,
but concurrent file modifications are not a snapshot-consistency guarantee.

## Integrate

```cmake
add_subdirectory(path/to/ggufpp)
target_link_libraries(my_app PRIVATE ggufpp::ggufpp)
```

Or install and discover the package:

```sh
cmake --install build --prefix /your/install/prefix
```

```cmake
find_package(ggufpp 0.1 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE ggufpp::ggufpp)
```

Set `CMAKE_PREFIX_PATH` to the installation prefix when configuring the consumer.
The exported target supplies include directories and the C++20 requirement.
The initial distribution builds a static library.

## Development

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ -DGGUFPP_SANITIZERS=ON
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DGGUFPP_BUILD_BENCHMARKS=ON
cmake --build build-release --parallel
./build-release/ggufpp_benchmark
./build-release/ggufpp_benchmark model.gguf 100
```

The benchmark reports mean warm-cache parse/open latency, including destruction.
It does not claim tensor bandwidth from total model size. The default input has
10,000 string metadata entries and one 100,000-element uint32 array.

An optional Clang/libFuzzer harness is available; this is a compiler tool, not a
library dependency:

```sh
cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGUFPP_BUILD_FUZZER=ON \
  -DBUILD_TESTING=OFF -DGGUFPP_BUILD_EXAMPLES=OFF
cmake --build build-fuzz --parallel
./build-fuzz/ggufpp_fuzz -max_total_time=30 -max_len=65536
```

See [design and conventions](docs/design.md) and [test coverage](docs/testing.md).
