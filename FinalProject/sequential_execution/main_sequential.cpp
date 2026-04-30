#include "conv_sequential.hpp"

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

static std::vector<float> read_csv_values(const std::string& filename, std::size_t expected_count) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<float> values;
    values.reserve(expected_count);

    std::string token;
    while (std::getline(file, token, ',')) {
        std::stringstream ss(token);
        std::string item;
        while (ss >> item) {
            values.push_back(std::stof(item));
        }
    }

    if (values.size() != expected_count) {
        std::ostringstream msg;
        msg << "File " << filename << " has " << values.size()
            << " values, but expected " << expected_count << ".";
        throw std::runtime_error(msg.str());
    }

    return values;
}

static double checksum(const std::vector<float>& data) {
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
    const std::string& prefix)
{
    for (int co = 0; co < Cout; ++co) {
        std::ostringstream filename;
        filename << prefix << "_filter_" << co << ".csv";

        std::ofstream file(filename.str());
        if (!file) {
            throw std::runtime_error("Could not write file: " + filename.str());
        }

        file << std::setprecision(8);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                if (w > 0) file << ",";
                file << output[(co * H + h) * W + w];
            }
            file << "\n";
        }
    }
}

static void print_usage(const char* program) {
    std::cerr
        << "Usage:\n"
        << "  " << program << " H W Cin Cout K repeats input.csv kernel.csv write_matrices\n\n"
        << "Example:\n"
        << "  " << program << " 64 64 3 8 3 5 input.csv kernel.csv 1\n\n"
        << "write_matrices: 1 writes sequential_filter_0.csv ... sequential_filter_Cout-1.csv\n"
        << "                0 writes only timing CSV to stdout\n";
}

int main(int argc, char** argv) {
    if (argc != 10) {
        print_usage(argv[0]);
        return 1;
    }

    try {
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
            throw std::invalid_argument("H, W, Cin, Cout, K, and repeats must be positive.");
        }

        const std::size_t input_count = static_cast<std::size_t>(H) * W * Cin;
        const std::size_t kernel_count = static_cast<std::size_t>(Cout) * Cin * K * K;
        const std::size_t output_count = static_cast<std::size_t>(Cout) * H * W;

        std::vector<float> input = read_csv_values(input_file, input_count);
        std::vector<float> kernel = read_csv_values(kernel_file, kernel_count);
        std::vector<float> output(output_count, 0.0f);

        // Warm-up run. This avoids including one-time effects in the measured average.
        conv2d_forward_sequential(input, kernel, output, H, W, Cin, Cout, K);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repeats; ++r) {
            conv2d_forward_sequential(input, kernel, output, H, W, Cin, Cout, K);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double avg_ms = total_ms / static_cast<double>(repeats);

        if (write_matrices != 0) {
            write_output_matrices(output, H, W, Cout, "sequential");
        }

        std::cout << "method,H,W,Cin,Cout,K,repeats,time_ms,total_time_ms,checksum,input_file,kernel_file\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "sequential,"
                  << H << "," << W << "," << Cin << "," << Cout << "," << K << ","
                  << repeats << ","
                  << avg_ms << ","
                  << total_ms << ","
                  << checksum(output) << ","
                  << input_file << ","
                  << kernel_file << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
