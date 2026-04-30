#include "conv_cuda_naive.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> read_csv_values(const std::string& filename, size_t expected_count) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<float> values;
    values.reserve(expected_count);

    float value = 0.0f;
    while (file >> value) {
        values.push_back(value);
        if (file.peek() == ',') {
            file.ignore();
        }
    }

    if (values.size() != expected_count) {
        throw std::runtime_error(
            "File " + filename + " contains " + std::to_string(values.size()) +
            " values, but expected " + std::to_string(expected_count)
        );
    }

    return values;
}

static double compute_checksum(const std::vector<float>& data) {
    double sum = 0.0;
    for (float v : data) {
        sum += static_cast<double>(v);
    }
    return sum;
}

static void write_output_matrices(
    const std::vector<float>& output,
    int H,
    int W,
    int Cout,
    const std::string& prefix
) {
    for (int co = 0; co < Cout; ++co) {
        const std::string filename = prefix + "_filter_" + std::to_string(co) + ".csv";
        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Could not write output matrix file: " + filename);
        }

        file << std::fixed << std::setprecision(8);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const size_t idx = static_cast<size_t>(co) * H * W + static_cast<size_t>(h) * W + w;
                file << output[idx];
                if (w + 1 < W) {
                    file << ',';
                }
            }
            file << '\n';
        }
    }
}

static void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " H W Cin Cout K repeats input.csv kernel.csv write_matrices\n\n"
              << "Example:\n"
              << "  " << prog << " 64 64 3 8 3 20 input.csv kernel.csv 1\n";
}

int main(int argc, char** argv) {
    try {
        if (argc != 10) {
            print_usage(argv[0]);
            return 1;
        }

        const int H = std::atoi(argv[1]);
        const int W = std::atoi(argv[2]);
        const int Cin = std::atoi(argv[3]);
        const int Cout = std::atoi(argv[4]);
        const int K = std::atoi(argv[5]);
        const int repeats = std::atoi(argv[6]);
        const std::string input_file = argv[7];
        const std::string kernel_file = argv[8];
        const int write_matrices = std::atoi(argv[9]);

        if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || repeats <= 0) {
            throw std::runtime_error("All dimensions and repeats must be positive.");
        }

        const size_t input_count = static_cast<size_t>(H) * W * Cin;
        const size_t kernel_count = static_cast<size_t>(Cout) * Cin * K * K;
        const size_t output_count = static_cast<size_t>(Cout) * H * W;

        std::vector<float> input = read_csv_values(input_file, input_count);
        std::vector<float> kernel = read_csv_values(kernel_file, kernel_count);
        std::vector<float> output(output_count, 0.0f);

        float avg_kernel_time_ms = 0.0f;
        float total_time_ms = 0.0f;
        std::string gpu_name;

        run_cuda_naive_convolution(
            input,
            kernel,
            output,
            H,
            W,
            Cin,
            Cout,
            K,
            repeats,
            avg_kernel_time_ms,
            total_time_ms,
            gpu_name
        );

        const double checksum = compute_checksum(output);

        if (write_matrices != 0) {
            write_output_matrices(output, H, W, Cout, "cuda_naive");
        }

        std::cout << "method,H,W,Cin,Cout,K,threads_per_block,repeats,kernel_time_ms,total_time_ms,checksum,input_file,kernel_file,gpu_name\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "cuda_naive,"
                  << H << ',' << W << ',' << Cin << ',' << Cout << ',' << K << ','
                  << 256 << ','
                  << repeats << ','
                  << avg_kernel_time_ms << ','
                  << total_time_ms << ','
                  << checksum << ','
                  << input_file << ','
                  << kernel_file << ','
                  << '"' << gpu_name << '"'
                  << '\n';

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
