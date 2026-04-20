#pragma once
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Timing
// ─────────────────────────────────────────────────────────────────────────────

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// Data layout (all row-major, channels last)
//   input  : [H][W][C_in]   → index = h*W*C_in  + w*C_in  + ci
//   weight : [C_out][K][K][C_in] → index = co*K*K*C_in + kh*K*C_in + kw*C_in + ci
//   output : [H][W][C_out]  → index = h*W*C_out + w*C_out + co
//
// Each PE = one output pixel (h, w).
// Each PE contains C_out MACs; MAC[co] accumulates over all (ci, kh, kw).
// ─────────────────────────────────────────────────────────────────────────────

inline long input_size (int H, int W, int C_in)          { return (long)H*W*C_in; }
inline long weight_size(int C_out, int K, int C_in)      { return (long)C_out*K*K*C_in; }
inline long output_size(int H, int W, int C_out)         { return (long)H*W*C_out; }

// ─────────────────────────────────────────────────────────────────────────────
// Random data generation (fixed seed for reproducibility across all impls)
// ─────────────────────────────────────────────────────────────────────────────

inline void fill_random(float* data, long n, unsigned int seed = 42) {
    srand(seed);
    for (long i = 0; i < n; i++)
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // uniform [-1, 1]
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary I/O  (used by validate.py to compare outputs)
// ─────────────────────────────────────────────────────────────────────────────

inline void save_output(const std::string& path, const float* data, long n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
}

inline void load_output(const std::string& path, float* data, long n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    f.read(reinterpret_cast<char*>(data), n * sizeof(float));
}
