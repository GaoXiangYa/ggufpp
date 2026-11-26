#include <ggufpp/ggufpp.hpp>

#include <exception>
#include <iostream>
#include <string_view>
#include <type_traits>

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string_view{argv[2]} != "--big-endian")) {
        std::cerr << "Usage: ggufpp_inspect FILE [--big-endian]\n";
        return 2;
    }
    try {
        ggufpp::parse_options options;
        if (argc == 3) {
            options.byte_order = std::endian::big;
        }
        const auto file = ggufpp::File::open(argv[1], options);
        const auto& document = file.document();
        std::cout << "GGUF v" << document.version() << " | " << document.file_size()
                  << " bytes | alignment " << document.alignment() << '\n';
        for (const auto& entry : document.metadata()) {
            std::cout << entry.key << " = ";
            std::visit(
                [](const auto& value) {
                    using type = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<type, ggufpp::MetadataArray>) {
                        std::cout << "array(type=" << static_cast<unsigned>(value.element_type())
                                  << ", size=" << value.size() << ')';
                    } else if constexpr (std::is_same_v<type, std::string>) {
                        std::cout << '"' << value << '"';
                    } else if constexpr (std::is_same_v<type, bool>) {
                        std::cout << (value ? "true" : "false");
                    } else if constexpr (std::is_integral_v<type>) {
                        std::cout << +value;
                    } else {
                        std::cout << value;
                    }
                },
                entry.value.storage());
            std::cout << '\n';
        }
        for (const auto& tensor : document.tensors()) {
            std::cout << tensor.name << " [";
            for (std::size_t i = 0; i < tensor.dimensions.size(); ++i) {
                std::cout << (i == 0 ? "" : ", ") << tensor.dimensions[i];
            }
            std::cout << "] " << ggufpp::layout_of(tensor.type)->name << " | " << tensor.byte_size
                      << " bytes @ " << tensor.file_offset << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
