#include "ggufpp/file.hpp"
#include "detail/parser.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace ggufpp {

File::File(std::ifstream stream, Document document)
    : stream_{std::move(stream)}, document_{std::move(document)} {}

auto File::open(const std::filesystem::path& path, const parse_options& options) -> File {
    std::ifstream stream{path, std::ios::binary | std::ios::ate};
    if (!stream) {
        throw ParseError(error_code::io_error, 0, "cannot open GGUF file");
    }
    const auto end = stream.tellg();
    if (end < std::streampos{0}) {
        throw ParseError(error_code::io_error, 0, "cannot determine GGUF file size");
    }
    const auto size = static_cast<std::uint64_t>(static_cast<std::streamoff>(end));
    stream.seekg(0);
    if (!stream) {
        throw ParseError(error_code::io_error, 0, "cannot seek GGUF file");
    }
    detail::BinaryReader reader{stream, size, options.byte_order};
    auto document = detail::parse_reader(reader, options);
    return File{std::move(stream), std::move(document)};
}

void File::read_tensor(std::size_t tensor_index, std::uint64_t byte_offset,
                       std::span<std::byte> destination) {
    const auto tensors = document_.tensors();
    if (tensor_index >= tensors.size()) {
        throw std::out_of_range("tensor index out of range");
    }
    const auto& tensor = tensors[tensor_index];
    if (byte_offset > tensor.byte_size || destination.size() > tensor.byte_size - byte_offset) {
        throw std::out_of_range("tensor read exceeds encoded payload");
    }
    if (destination.empty()) {
        return;
    }
    auto position = detail::checked_add(tensor.file_offset, byte_offset, tensor.file_offset);
    if (position > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        throw ParseError(error_code::io_error, position, "tensor offset exceeds stream range");
    }
    if (next_read_offset_ != position) {
        stream_.clear();
        stream_.seekg(static_cast<std::streamoff>(position));
        if (!stream_) {
            next_read_offset_.reset();
            throw ParseError(error_code::io_error, position, "cannot seek tensor payload");
        }
    }
    // Sequential reads reuse the stream buffer; a failure forces a fresh seek on retry.
    next_read_offset_.reset();
    while (!destination.empty()) {
        const auto count = std::min(destination.size(), std::size_t{16} * 1024 * 1024);
        stream_.read(reinterpret_cast<char*>(destination.data()),
                     static_cast<std::streamsize>(count));
        if (stream_.gcount() != static_cast<std::streamsize>(count)) {
            throw ParseError(error_code::io_error, position, "cannot read tensor payload");
        }
        position += count;
        destination = destination.subspan(count);
    }
    next_read_offset_ = position;
}

} // namespace ggufpp
