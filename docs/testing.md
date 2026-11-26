# Testing

Tests use standard C++20 and CTest, without external test frameworks or model
downloads. Conditions throw on failure in every build configuration.

The test executable contains 16 named groups:

- A hand-authored golden GGUF byte sequence independent of the fixture encoder.
- Metadata ownership after freeing the input, copying/moving documents, and lookup.
- GGUF versions, unsupported versions, explicit byte order, and invalid options.
- Every metadata scalar type, integer boundaries, and both byte orders.
- Every numeric array type in both byte orders, boolean/string/empty/nested arrays.
- All 36 active tensor encodings with expected block sizes and multidimensional shapes.
- Non-power-of-two alignment, gaps, and tensor descriptors in non-offset order.
- Invalid keys, UTF-8, booleans, metadata tags, and duplicate keys.
- Invalid dimensions, tensor names/encodings, quantized rows, overlaps, and alignment.
- Every truncated prefix of a valid tensor-bearing fixture and arithmetic overflow.
- Resource budgets, aggregate array element limits, and nesting-depth boundaries.
- Full/partial/empty/out-of-range file reads, missing files, and File moves.
- Metadata spanning multiple reader buffers, large arrays, and post-open truncation.
- An 8 GiB sparse payload on non-Windows systems, checking its final byte without
  allocating the tensor. Windows uses 8 MiB because portable C++ cannot request
  NTFS sparse-file semantics.
- 5,000 deterministic mutations with bounded allocations and validated successes.

Six build targets compile each public header alone. The second CTest test installs
the library into the build tree, configures and builds an independent consumer
with `find_package`, and runs it. Tests clean up their own temporary directories.

`GGUFPP_SANITIZERS=ON` enables ASan and UBSan for GCC/Clang builds. An optional
libFuzzer executable instruments the parser itself, exercises both byte orders,
and caps allocation, string, and array budgets. Supply valid GGUF files as corpus
seeds for deeper format coverage. Mutation and fuzz testing complement explicit
cases; they do not establish that every possible malformed file is handled.

GitHub Actions is configured for Linux, Windows, and macOS release builds, plus a
Clang sanitizer job. Platform coverage in configuration is distinct from platforms
actually run locally; see the execution record below.

## Local execution record

Validated on 2026-11-26, Linux x86-64, with GCC 13.3 and Clang 18.1:

| Check | Result |
| --- | --- |
| GCC Release | 16/16 functional groups; both CTest tests passed |
| Clang Debug + ASan/UBSan | Both CTest tests passed; no sanitizer findings |
| Installed consumer | Configured, compiled, linked, and ran with `find_package` |
| Public headers | All six compiled independently |
| libFuzzer | Final instrumented build: 10,000 executions, no failures |
| Deterministic mutations | 5,000 per functional test execution |
| Large file | Read the last byte of an 8 GiB sparse tensor |
| CLI | Inspected independently encoded little- and big-endian fixtures |
| Static checks | No project diagnostics from configured clang-tidy checks; format check passed |

Release benchmark: 100 parses of a 918,954-byte synthetic file containing 10,001
metadata entries averaged 1.74826 ms per parse including destruction. This is a
local synthetic measurement, not a cold-cache real-model result or tensor
throughput claim. Hardware, allocator, compiler, and input affect timing.

Commands used:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DGGUFPP_BUILD_BENCHMARKS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-release --parallel 4
ctest --test-dir build-release --output-on-failure
./build-release/ggufpp_benchmark

cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ -DGGUFPP_SANITIZERS=ON
cmake --build build-sanitize --parallel 4
ctest --test-dir build-sanitize --output-on-failure

clang-tidy -p build-release src/parser.cpp src/file.cpp src/tensor.cpp --quiet
```

The final fuzzer run used the optional target documented in the README and a
local corpus containing independently encoded valid little/big-endian GGUF
fixtures and mutations from an earlier 13,558-execution run. Neither fixtures nor
fuzz inputs require downloaded model files. The final command was:

```sh
./build-fuzz/ggufpp_fuzz build-fuzz/corpus -runs=10000 \
  -max_len=65536 -timeout=3 -rss_limit_mb=1024
```

Windows/macOS jobs are configured but were not executed locally.
