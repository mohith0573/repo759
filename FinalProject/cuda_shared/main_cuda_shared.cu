#include "conv_cuda_shared.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> read_flat_csv(const std::string& filename, size_t expected_count) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<float> values;
    values.reserve(expected_count);

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    for (char& ch : content) {
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ' ';
        }
    }

    std::stringstream ss(content);
    float x;
    while (ss >> x) {
        values.push_back(x);
    }

    if (values.size() != expected_count) {
        std::ostringstream err;
        err << "File " << filename << " has " << values.size()
            << " values, expected " << expected_count;
        throw std::runtime_error(err.str());
    }

    return values;
}

static void write_output_matrices(
    const std::vector<float>& output,
    int H,
    int W,
    int Cout,
    const std::string& prefix
) {
    for (int co = 0; co < Cout; ++co) {
        std::ostringstream name;
        name << prefix << "_filter_" << co << ".csv";

        std::ofstream file(name.str());
        if (!file) {
            throw std::runtime_error("Could not write output file: " + name.str());
        }

        file << std::fixed << std::setprecision(8);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                if (w > 0) file << ",";
                file << output[(co * H + h) * W + w];
            }
            file << "\n";
        }
    }
}

static double checksum_output(const std::vector<float>& output) {
    double sum = 0.0;
    for (float v : output) {
        sum += static_cast<double>(v);
    }
    return sum;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " H W Cin Cout K repeats input.csv kernel.csv write_matrices\n\n"
              << "Example:\n"
              << "  " << prog << " 64 64 3 8 3 20 input.csv kernel.csv 1\n";
}

int main(int argc, char** argv) {
    if (argc != 10) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const int H = std::stoi(argv[1]);
        const int W = std::stoi(argv[2]);
        const int Cin = std::stoi(argv[3]);
        const int Cout = std::stoi(argv[4]);
        const int K = std::stoi(argv[5]);
        const int repeats = std::stoi(argv[6]);
        const std::string input_file = argv[7];
        const std::string kernel_file = argv[8];
        const int write_matrices = std::stoi(argv[9]);

        const size_t input_count = static_cast<size_t>(H) * W * Cin;
        const size_t kernel_count = static_cast<size_t>(Cout) * Cin * K * K;
        const size_t output_count = static_cast<size_t>(Cout) * H * W;

        auto total_start = std::chrono::high_resolution_clock::now();

        std::vector<float> input = read_flat_csv(input_file, input_count);
        std::vector<float> kernel = read_flat_csv(kernel_file, kernel_count);
        std::vector<float> output(output_count, 0.0f);

        float kernel_time_ms = 0.0f;
        std::string gpu_name;

        conv2d_cuda_shared_host(input, kernel, output, H, W, Cin, Cout, K, repeats, kernel_time_ms, gpu_name);

        if (write_matrices != 0) {
            write_output_matrices(output, H, W, Cout, "cuda_shared");
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        const double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        const double checksum = checksum_output(output);

        std::cout << "method,H,W,Cin,Cout,K,threads_per_block,repeats,kernel_time_ms,total_time_ms,checksum,input_file,kernel_file,gpu_name\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "cuda_shared," << H << "," << W << "," << Cin << "," << Cout << "," << K << ","
                  << 256 << "," << repeats << ","
                  << kernel_time_ms << "," << total_time_ms << "," << checksum << ","
                  << input_file << "," << kernel_file << ",\"" << gpu_name << "\"\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
