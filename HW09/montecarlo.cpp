#include "montecarlo.h"
#include <cmath>

int montecarlo(const size_t n, const float *x, const float *y, const float radius) {

    int incircle = 0;

#ifdef USE_SIMD
#pragma omp parallel for simd reduction(+:incircle)
#else
#pragma omp parallel for reduction(+:incircle)
#endif
    for (size_t i = 0; i < n; i++) {
        float d = x[i]*x[i] + y[i]*y[i];
        if (d <= radius*radius)
            incircle++;
    }

    return incircle;
}
