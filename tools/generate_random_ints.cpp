#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

uint64_t parse_uint64(const char* text, const char* name)
{
    size_t parsed = 0;
    const std::string value(text);
    const unsigned long long result = std::stoull(value, &parsed, 0);

    if (parsed != value.size())
        throw std::invalid_argument(
            std::string(name) + " must be an integer: " + value);

    return result;
}

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2 && argc != 3 && argc != 4 && argc != 5)
        {
            std::cerr << "Usage: " << argv[0]
                      << " COUNT [MIN MAX] [OUTPUT_FILE]\n";
            return 1;
        }

        const uint64_t count = parse_uint64(argv[1], "COUNT");
        uint64_t minimum = 0;
        uint64_t maximum = std::numeric_limits<uint32_t>::max();
        std::string output_file = "random_ints.txt";

        if (argc == 3)
        {
            output_file = argv[2];
        }
        else if (argc == 4 || argc == 5)
        {
            minimum = parse_uint64(argv[2], "MIN");
            maximum = parse_uint64(argv[3], "MAX");

            if (argc == 5)
                output_file = argv[4];
        }

        if (count > std::numeric_limits<size_t>::max())
            throw std::out_of_range("COUNT is too large");

        if (minimum > maximum || maximum > std::numeric_limits<uint32_t>::max())
            throw std::out_of_range(
                "range must satisfy 0 <= MIN <= MAX <= 4294967295");

        std::ofstream output(output_file);
        if (!output)
            throw std::runtime_error(
                "could not open output file: " + output_file);

        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::uniform_int_distribution<uint32_t> distribution(
            static_cast<uint32_t>(minimum),
            static_cast<uint32_t>(maximum));

        for (uint64_t i = 0; i < count; ++i)
            output << distribution(generator) << '\n';

        std::cout << "Wrote " << count << " integers to "
                  << output_file << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
