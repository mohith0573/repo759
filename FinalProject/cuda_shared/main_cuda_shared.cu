#include "conv_cuda_shared.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_csv_values(const std::string& filename, std::size_t expected_count) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<float> values;
    values.reserve(expected_count);

    std::string token;
    while (std::getline(file, token, ',')) {
        // Also handle accidental newlines inside the comma-separated stream.
        std::stringstream ss(token);
        std::string subtok;
        while (ss >> subtok) {
            if (!subtok.empty()) {
                values.push_back(std::stof(subtok));
            }
        }
    }

    if (values.size() != expected_count) {
        std::ostringstream oss;
        oss << "File " << filename << " has " << values.size()
            << " values, expected " << expected_count;
        throw std::runtime_error(oss.str());
    }

    return values;
}

double checksum_output(const std::vector<float>& output) {
    // Weighted checksum catches more mistakes than a plain sum while remaining deterministic.
    double checksum = 0.0;
    for (std::size_t i = 0; i < output.size(); ++i) {
        const double weight = static_cast<double>((i % 131) + 1) * 0.0001;
        checksum += static_cast<double>(output[i]) * weight;
    }
    return checksum;
}

void write_output_matrices(
    const std::vector<float>& output,
    int H,
    int W,
    int Cout,
    const std::string& prefix
) {
    for (int co = 0; co < Cout; ++co) {
        std::ostringstream filename;
        filename << prefix << "_filter_" << co << ".csv";

        std::ofstream out(filename.str());
        if (!out) {
            throw std::runtime_error("Could not write file: " + filename.str());
        }

        out << std::fixed << std::setprecision(8);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                if (w > 0) out << ',';
                out << output[(co * H + h) * W + w];
            }
            out << '\n';
        }
    }
}

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " H W Cin Cout K repeats input.csv kernel.csv write_matrices\n\n"
              << "Example:\n"
              << "  " << prog << " 64 64 3 8 3 20 input.csv kernel.csv 1\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 10) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
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

        if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || repeats <= 0) {
            throw std::runtime_error("All dimensions and repeats must be positive.");
        }
        if (Cout > 64) {
            throw std::runtime_error("This CUDA shared implementation supports Cout <= 64.");
        }

        const std::size_t input_count = static_cast<std::size_t>(H) * W * Cin;
        const std::size_t kernel_count = static_cast<std::size_t>(Cout) * Cin * K * K;
        const std::size_t output_count = static_cast<std::size_t>(Cout) * H * W;

        std::vector<float> input = read_csv_values(input_file, input_count);
        std::vector<float> kernel = read_csv_values(kernel_file, kernel_count);
        std::vector<float> output(output_count, 0.0f);

        int threads_per_block = 0;
        char gpu_name[256] = {0};

        const auto total_start = std::chrono::high_resolution_clock::now();

        const float kernel_time_ms = run_cuda_shared_convolution(
            input.data(),
            kernel.data(),
            output.data(),
            H,
            W,
            Cin,
            Cout,
            K,
            repeats,
            &threads_per_block,
            gpu_name,
            sizeof(gpu_name)
        );

        if (write_matrices != 0) {
            write_output_matrices(output, H, W, Cout, "cuda_shared");
        }

        const auto total_end = std::chrono::high_resolution_clock::now();
        const double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        const double checksum = checksum_output(output);

        std::cout << "method,H,W,Cin,Cout,K,threads_per_block,repeats,kernel_time_ms,total_time_ms,checksum,input_file,kernel_file,gpu_name\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "cuda_shared," << H << ',' << W << ',' << Cin << ',' << Cout << ',' << K << ','
                  << threads_per_block << ',' << repeats << ','
                  << kernel_time_ms << ',' << total_time_ms << ',' << checksum << ','
                  << input_file << ',' << kernel_file << ",\"" << gpu_name << "\"\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
