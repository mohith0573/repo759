#include "conv_openmp.hpp"

#include <algorithm>
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

static void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " H W Cin Cout K repeats input.csv kernel.csv write_matrices threads\n\n"
        << "Example:\n"
        << "  " << prog << " 64 64 3 8 3 5 input.csv kernel.csv 1 20\n\n"
        << "Notes:\n"
        << "  write_matrices = 1 writes openmp_filter_0.csv ... openmp_filter_(Cout-1).csv\n"
        << "  threads controls OMP_NUM_THREADS inside the program.\n";
}

static int parse_int(const char* s, const std::string& name) {
    try {
        size_t pos = 0;
        const int value = std::stoi(s, &pos);
        if (pos != std::string(s).size()) {
            throw std::invalid_argument("trailing characters");
        }
        if (value <= 0) {
            throw std::invalid_argument("must be positive");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for " + name + ": " + s);
    }
}

static std::vector<float> read_csv_vector(const std::string& path, size_t expected_count) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::vector<float> values;
    values.reserve(expected_count);

    std::string token;
    while (std::getline(file, token, ',')) {
        // Also support accidental newlines inside comma-separated files.
        std::stringstream ss(token);
        std::string subtoken;
        while (ss >> subtoken) {
            values.push_back(std::stof(subtoken));
        }
    }

    if (values.size() != expected_count) {
        std::ostringstream oss;
        oss << "File " << path << " has " << values.size()
            << " values, but expected " << expected_count;
        throw std::runtime_error(oss.str());
    }

    return values;
}

static double checksum_output(const std::vector<float>& output) {
    // Weighted checksum catches more errors than a simple sum, while staying compact.
    double checksum = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        const double weight = static_cast<double>((i % 131) + 1) * 0.001;
        checksum += static_cast<double>(output[i]) * weight;
    }
    return checksum;
}

static void write_filter_matrix(
    const std::vector<float>& output,
    int H,
    int W,
    int co,
    const std::string& filename
) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not write matrix file: " + filename);
    }

    out << std::setprecision(8) << std::fixed;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            if (w > 0) {
                out << ',';
            }
            out << output[(static_cast<size_t>(co) * H + h) * W + w];
        }
        out << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 11) {
            usage(argv[0]);
            return 1;
        }

        const int H = parse_int(argv[1], "H");
        const int W = parse_int(argv[2], "W");
        const int Cin = parse_int(argv[3], "Cin");
        const int Cout = parse_int(argv[4], "Cout");
        const int K = parse_int(argv[5], "K");
        const int repeats = parse_int(argv[6], "repeats");
        const std::string input_file = argv[7];
        const std::string kernel_file = argv[8];
        const int write_matrices = parse_int(argv[9], "write_matrices");
        const int threads = parse_int(argv[10], "threads");

        omp_set_num_threads(threads);

        const size_t input_count = static_cast<size_t>(Cin) * H * W;
        const size_t kernel_count = static_cast<size_t>(Cout) * Cin * K * K;
        const size_t output_count = static_cast<size_t>(Cout) * H * W;

        const std::vector<float> input = read_csv_vector(input_file, input_count);
        const std::vector<float> kernel = read_csv_vector(kernel_file, kernel_count);
        std::vector<float> output(output_count, 0.0f);

        // Warm-up run: avoids including first-touch/thread startup effects in timing.
        conv2d_forward_openmp(input, kernel, output, H, W, Cin, Cout, K);

        const auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repeats; ++r) {
            conv2d_forward_openmp(input, kernel, output, H, W, Cin, Cout, K);
        }
        const auto end = std::chrono::high_resolution_clock::now();

        const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        const double avg_ms = total_ms / static_cast<double>(repeats);
        const double checksum = checksum_output(output);

        if (write_matrices != 0) {
            for (int co = 0; co < Cout; ++co) {
                const std::string fname = "openmp_filter_" + std::to_string(co) + ".csv";
                write_filter_matrix(output, H, W, co, fname);
            }
        }

        std::cout << "method,H,W,Cin,Cout,K,threads,repeats,time_ms,total_time_ms,checksum,input_file,kernel_file\n";
        std::cout << std::setprecision(6) << std::fixed;
        std::cout << "openmp,"
                  << H << ',' << W << ',' << Cin << ',' << Cout << ',' << K << ','
                  << threads << ',' << repeats << ','
                  << avg_ms << ',' << total_ms << ',' << checksum << ','
                  << input_file << ',' << kernel_file << '\n';

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
