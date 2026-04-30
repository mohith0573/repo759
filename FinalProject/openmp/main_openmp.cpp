#include "conv_openmp.hpp"

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
#include <omp.h>

static std::vector<float> read_flat_csv(const std::string& filename, size_t expected_count) {
    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<float> values;
    values.reserve(expected_count);

    std::string token;
    while (std::getline(fin, token, ',')) {
        std::stringstream ss(token);
        float x;
        while (ss >> x) {
            values.push_back(x);
        }
    }

    if (values.size() != expected_count) {
        std::ostringstream oss;
        oss << "File " << filename << " has " << values.size()
            << " values, but expected " << expected_count << ".";
        throw std::runtime_error(oss.str());
    }

    return values;
}

static double checksum_output(const std::vector<float>& output) {
    double sum = 0.0;
    for (float x : output) {
        sum += static_cast<double>(x);
    }
    return sum;
}

static void write_filter_matrices(
    const std::vector<float>& output,
    int H, int W, int Cout,
    const std::string& prefix
) {
    const int HW = H * W;

    for (int co = 0; co < Cout; ++co) {
        std::ostringstream name;
        name << prefix << "_filter_" << co << ".csv";

        std::ofstream fout(name.str());
        if (!fout) {
            throw std::runtime_error("Could not write output matrix: " + name.str());
        }

        fout << std::fixed << std::setprecision(6);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                if (w > 0) fout << ",";
                fout << output[co * HW + h * W + w];
            }
            fout << "\n";
        }
    }
}

static void usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " H W Cin Cout K repeats input.csv kernel.csv write_matrices threads\n\n"
              << "Example:\n"
              << "  " << prog << " 64 64 3 8 3 5 input.csv kernel.csv 1 20\n";
}

int main(int argc, char** argv) {
    if (argc != 11) {
        usage(argv[0]);
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
        const int threads = std::stoi(argv[10]);

        if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || repeats <= 0 || threads <= 0) {
            throw std::runtime_error("H, W, Cin, Cout, K, repeats, and threads must be positive.");
        }

        const size_t input_count = static_cast<size_t>(H) * W * Cin;
        const size_t kernel_count = static_cast<size_t>(Cout) * Cin * K * K;
        const size_t output_count = static_cast<size_t>(Cout) * H * W;

        const std::vector<float> input = read_flat_csv(input_file, input_count);
        const std::vector<float> kernel = read_flat_csv(kernel_file, kernel_count);
        std::vector<float> output(output_count, 0.0f);

        // Warm-up run so timing is not affected by first-use overhead.
        conv2d_forward_openmp(input, kernel, output, H, W, Cin, Cout, K, threads);

        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repeats; ++r) {
            conv2d_forward_openmp(input, kernel, output, H, W, Cin, Cout, K, threads);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double avg_ms = total_ms / static_cast<double>(repeats);
        const double checksum = checksum_output(output);

        if (write_matrices != 0) {
            write_filter_matrices(output, H, W, Cout, "openmp");
        }

        std::cout << "method,H,W,Cin,Cout,K,threads,repeats,time_ms,total_time_ms,checksum,input_file,kernel_file\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "openmp," << H << "," << W << "," << Cin << "," << Cout << "," << K << ","
                  << threads << "," << repeats << "," << avg_ms << "," << total_ms << ","
                  << checksum << "," << input_file << "," << kernel_file << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
