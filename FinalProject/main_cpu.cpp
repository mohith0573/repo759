#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

void conv2d_forward_sequential(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H, int W, int Cin, int Cout, int K);

void conv2d_forward_openmp(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    std::vector<float>& output,
    int H, int W, int Cin, int Cout, int K);

static void fill_deterministic(std::vector<float>& v, float scale)
{
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = scale * static_cast<float>((static_cast<int>(i) % 17) - 8);
    }
}

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return max_diff;
}

static double seconds_now()
{
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    int H = 256;
    int W = 256;
    int Cin = 8;
    int Cout = 8;
    int K = 3;
    int repeats = 5;

    if (argc >= 6) {
        H = std::atoi(argv[1]);
        W = std::atoi(argv[2]);
        Cin = std::atoi(argv[3]);
        Cout = std::atoi(argv[4]);
        K = std::atoi(argv[5]);
    }
    if (argc >= 7) {
        repeats = std::atoi(argv[6]);
    }

    if (H <= 0 || W <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || repeats <= 0) {
        std::cerr << "Usage: ./conv_cpu H W Cin Cout K [repeats]\n";
        return 1;
    }

    std::vector<float> input(static_cast<size_t>(Cin) * H * W);
    std::vector<float> kernel(static_cast<size_t>(Cout) * Cin * K * K);
    std::vector<float> out_seq(static_cast<size_t>(Cout) * H * W, 0.0f);
    std::vector<float> out_omp(static_cast<size_t>(Cout) * H * W, 0.0f);

    fill_deterministic(input, 0.01f);
    fill_deterministic(kernel, 0.001f);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "method,H,W,Cin,Cout,K,threads,repeats,time_ms,max_abs_diff\n";

    // Sequential timing.
    conv2d_forward_sequential(input, kernel, out_seq, H, W, Cin, Cout, K); // warm-up
    double seq_total = 0.0;
    for (int r = 0; r < repeats; ++r) {
        std::fill(out_seq.begin(), out_seq.end(), 0.0f);
        const double t0 = seconds_now();
        conv2d_forward_sequential(input, kernel, out_seq, H, W, Cin, Cout, K);
        const double t1 = seconds_now();
        seq_total += (t1 - t0);
    }
    const double seq_ms = 1000.0 * seq_total / repeats;

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    std::cout << "sequential," << H << "," << W << "," << Cin << "," << Cout << "," << K
              << ",1," << repeats << "," << seq_ms << ",0.000000\n";

    // OpenMP timing.
    conv2d_forward_openmp(input, kernel, out_omp, H, W, Cin, Cout, K); // warm-up
    double omp_total = 0.0;
    for (int r = 0; r < repeats; ++r) {
        std::fill(out_omp.begin(), out_omp.end(), 0.0f);
        const double t0 = seconds_now();
        conv2d_forward_openmp(input, kernel, out_omp, H, W, Cin, Cout, K);
        const double t1 = seconds_now();
        omp_total += (t1 - t0);
    }
    const double omp_ms = 1000.0 * omp_total / repeats;
    const double diff = max_abs_diff(out_seq, out_omp);

    std::cout << "openmp," << H << "," << W << "," << Cin << "," << Cout << "," << K
              << "," << threads << "," << repeats << "," << omp_ms << "," << diff << "\n";

    return 0;
}