#include "cluster.h"
#include <cmath>

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {

#pragma omp parallel num_threads(t)
  {
    unsigned int tid = omp_get_thread_num();
    float local_sum = 0.0f;

#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      local_sum += std::fabs(arr[i] - centers[tid]);
    }

    dists[tid] = local_sum;  // single write → avoids false sharing
  }
}
