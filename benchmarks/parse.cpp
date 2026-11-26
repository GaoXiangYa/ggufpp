#include <ggufpp/ggufpp.hpp>

#include "fixture.hpp"

#include <charconv>
#include <chrono>
#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char* argv[]) {
    if (argc > 3) {
        std::cerr << "Usage: ggufpp_benchmark [FILE [ITERATIONS]]\n";
        return 2;
    }
    std::size_t iterations = 100;
    if (argc == 3) {
        const std::string_view input{argv[2]};
        const auto [end, error] =
            std::from_chars(input.data(), input.data() + input.size(), iterations);
        if (error != std::errc{} || end != input.data() + input.size() || iterations == 0) {
            std::cerr << "ITERATIONS must be a positive integer\n";
            return 2;
        }
    }
    try {
        test_support::Fixture fixture;
        if (argc == 1) {
            for (unsigned i = 0; i < 10'000; ++i) {
                fixture.metadata("benchmark.key_" + std::to_string(i), 8).string("metadata value");
            }
            auto& array = fixture.metadata("benchmark.values", 9);
            array.number<std::uint32_t>(4);
            array.number<std::uint64_t>(100'000);
            for (std::uint32_t i = 0; i < 100'000; ++i) {
                array.number(i);
            }
        }
        const auto bytes = fixture.finish();
        std::size_t entries = 0;
        std::uint64_t size = 0;
        // Warm caches and force successful parsing before timing.
        if (argc == 1) {
            (void)ggufpp::parse(bytes);
        } else {
            (void)ggufpp::File::open(argv[1]);
        }
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            if (argc == 1) {
                const auto document = ggufpp::parse(bytes);
                entries = document.metadata().size();
                size = document.file_size();
            } else {
                const auto file = ggufpp::File::open(argv[1]);
                entries = file.document().metadata().size();
                size = file.document().file_size();
            }
        }
        const auto elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        std::cout << (argc == 1 ? "synthetic memory parse" : "file open (warm cache)") << ": "
                  << iterations << " iterations, " << entries << " metadata entries, " << size
                  << " input bytes\nmean: " << elapsed / static_cast<double>(iterations)
                  << " ms (includes destruction; tensor payload is not scanned)\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
