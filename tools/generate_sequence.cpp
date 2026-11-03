#include <cstdint>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
        const bool reverse = argc > 1 && std::string(argv[1]) == "--reverse";
        const int first_argument = reverse ? 2 : 1;
        const int argument_count = argc - first_argument;

        if (argument_count < 2 || argument_count > 4)
        {
            std::cerr << "Usage: " << argv[0]
                      << " [--reverse] START END [STEP] [OUTPUT_FILE]\n";
            return 1;
        }

        const uint64_t start =
            parse_uint64(argv[first_argument], "START");
        const uint64_t end =
            parse_uint64(argv[first_argument + 1], "END");
        const uint64_t step = argument_count >= 3
            ? parse_uint64(argv[first_argument + 2], "STEP")
            : 1;
        const std::string output_file = argument_count == 4
            ? argv[first_argument + 3]
            : "sequence.txt";

        if (start > std::numeric_limits<uint32_t>::max() ||
            end > std::numeric_limits<uint32_t>::max())
        {
            throw std::out_of_range(
                "START and END must be between 0 and 4294967295");
        }

        if (step == 0)
            throw std::invalid_argument("STEP must be greater than zero");

        std::ofstream output(output_file);
        if (!output)
            throw std::runtime_error(
                "could not open output file: " + output_file);

        std::vector<uint64_t> sequence;
        if (start <= end)
        {
            for (uint64_t value = start; value <= end; )
            {
            sequence.push_back(value);

                if (end - value < step)
                    break;
                value += step;
            }
        }
        else
        {
            for (uint64_t value = start; value >= end; )
            {
                sequence.push_back(value);

                if (value - end < step)
                    break;
                value -= step;
            }
        }

        if (reverse)
            std::reverse(sequence.begin(), sequence.end());

        for (uint64_t value : sequence)
            output << value << '\n';

        std::cout << "Wrote " << sequence.size() << " integers to "
                  << output_file << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
