# Design and conventions

The library uses only the C++20 standard library. GGML/GGUF sources are format
references, not build inputs. The public API is deliberately small: parse memory,
open a file, inspect metadata/descriptors, and read encoded tensor byte ranges.

## Responsibilities

- `detail::BinaryReader`: exact reads, byte order, position tracking, and a fixed
  64 KiB buffer for stream input. Memory input uses direct copies from its span.
- `detail::Parser`: format decoding, resource accounting, validation, and indexes.
- `Document`: owned metadata and descriptors with const access. In-file order is
  preserved. Separate vectors of sorted integer indexes provide allocation-free
  O(log n) name lookup, and remain valid when a document is copied or moved.
- `File`: RAII ownership of an `ifstream` and a parsed document. Tensor reads seek
  to a checked absolute offset and write directly into caller memory in bounded
  chunks. Standard C++20 has no memory-mapping API; there is no platform backend.
- `layout_of`: a constexpr table behind an optional-returning function describing
  active GGML type IDs, block element counts, and block byte sizes.

Parsing costs O(metadata bytes + descriptor bytes + n log n) time, plus bounded
read-ahead for files. Storage is proportional to decoded metadata, tensor
information, and indexes, independently of payload size. Tensor validation sorts
an additional temporary vector of indexes to detect overlapping ranges.

Numeric metadata arrays are decoded in bulk when file and host byte order match.
Strings are owned and validated as UTF-8. The array representation preserves
homogeneity, avoids a variant per numeric element, and retains the element type
of empty arrays. Recursive arrays have a bounded depth and aggregate element
budget. Tensor payloads are opaque; semantic model validation is a separate task.

## C++ Core Guidelines policy

Follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines),
particularly RAII, explicit ownership, small interfaces, class invariants, and
checked conversions. No GSL dependency is required: use standard `span`,
`string_view`, containers, and smart pointers when needed.

- Class names use PascalCase: `File`, `Document`, `MetadataValue`, `BinaryReader`.
- Structs, enums, aliases, namespaces, functions, and variables use snake_case.
- Private data members use a trailing underscore: `document_`, `stream_`.
- Enum enumerators use snake_case. Public enums are scoped and wire enums have
  fixed-width underlying integer types.
- Use `class` for invariants and `struct` for records/options. Prefer the Rule of
  Zero; explicitly document move-only resource ownership when appropriate.
- Use RAII and value semantics. Raw pointers represent borrowed/nullable access.
  No manual owning `new`/`delete`, global mutable state, or packed wire structs.
- Initialize state; use `const`, `[[nodiscard]]`, and `noexcept` where accurate.
- Use fixed-width unsigned integers for wire fields and checked file arithmetic.
  Convert to container and stream sizes at explicit boundaries.
- Keep casts for byte I/O narrowly confined to the reader and file adapter.
- Invalid input is a runtime error, not an assertion. Assertions are not the test
  framework because release builds must still check every test condition.
- Use four-space indentation and the repository's `.clang-format`. Optional
  `.clang-tidy` checks supplement review; they do not prove guideline compliance.

The compiler/library platform must have 8-bit bytes, 32/64-bit IEEE754 float and
double, and uniform little or big endian scalar storage; compile-time assertions
make these assumptions explicit. Public headers are compiled independently.

## Format references

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Authoritative type IDs](https://github.com/ggml-org/ggml/blob/master/include/ggml.h)
- [Quantized block definitions](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h)
- [Type trait definitions](https://github.com/ggml-org/ggml/blob/master/src/ggml.c)

The layout table was checked against these references on 2026-11-26. It covers
active IDs through 43, including F8_E4M3. Reserved and removed IDs are rejected.
A new encoding needs a verified layout and corresponding fixture tests before
being accepted. No quantization kernels or upstream implementation code are used.
