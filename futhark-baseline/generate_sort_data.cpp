#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
void save(const std::string& path, const std::vector<uint32_t>& xs) {
    std::ofstream f(path);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    f << '[';
    for (size_t i=0; i<xs.size(); ++i) {
        if (i) f << ',';
        f << xs[i] << "u32";
    }
    f << "]\n";
}
int main() {
    try {
        std::filesystem::create_directories("data");
        std::ofstream spec("datasets.spec");
        spec.exceptions(std::ios::failbit | std::ios::badbit);
        spec << "Unsigned 32-bit sorting: same generators as CUDA/CUB\n==\n";
        for (size_t n : {size_t(10000),size_t(100000),size_t(1000000)}) {
            for (const std::string pattern : {"random","all_equal","few_distinct","sorted","reversed"}) {
                std::mt19937 rng(2026);
                std::vector<uint32_t> xs(n);
                for (size_t i=0;i<n;++i) {
                    if (pattern=="random") xs[i]=rng();
                    else if (pattern=="all_equal") xs[i]=UINT32_MAX;
                    else if (pattern=="few_distinct") xs[i]=uint32_t(i%7);
                    else if (pattern=="sorted") xs[i]=uint32_t(i);
                    else xs[i]=uint32_t(n-i);
                }
                const auto name=std::to_string(n)+"_"+pattern;
                save("data/"+name+".in",xs);
                std::sort(xs.begin(),xs.end());
                save("data/"+name+".expected",xs);
                spec << '"' << name << "\" input @ data/" << name << ".in\n";
            }
        }
        std::cout << "Generated 15 inputs, 15 CPU-sorted references, and datasets.spec.\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
