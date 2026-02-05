#include "convolution.h"

void convolve(const float* image,
              float* output,
              std::size_t n,
              const float* mask,
              std::size_t m) {
    int r = m / 2;

    for (std::size_t x = 0; x < n; x++) {
        for (std::size_t y = 0; y < n; y++) {
            float sum = 0.0f;

            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < m; j++) {
                    int xi = int(x) + int(i) - r;
                    int yj = int(y) + int(j) - r;

                    float val;
                    if (xi < 0 || xi >= int(n) || yj < 0 || yj >= int(n)) {
                        if ((xi < 0 || xi >= int(n)) &&
                            (yj < 0 || yj >= int(n)))
                            val = 0.0f;  // corner
                        else
                            val = 1.0f;  // edge
                    } else {
                        val = image[xi * n + yj];
                    }

                    sum += mask[i * m + j] * val;
                }
            }
            output[x * n + y] = sum;
        }
    }
}
