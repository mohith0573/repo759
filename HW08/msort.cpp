#include "msort.h"
#include <algorithm>
#include <vector>

// merge function
void merge(int* arr, int l, int m, int r) {
    int n1 = m - l;
    int n2 = r - m;

    std::vector<int> L(arr + l, arr + m);
    std::vector<int> R(arr + m, arr + r);

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

// recursive merge sort with tasks
void msort_recursive(int* arr, int l, int r, int threshold) {
    if (r - l <= 1) return;

    if (r - l <= threshold) {
        std::sort(arr + l, arr + r);
        return;
    }

    int m = (l + r) / 2;

    #pragma omp task shared(arr)
    msort_recursive(arr, l, m, threshold);

    #pragma omp task shared(arr)
    msort_recursive(arr, m, r, threshold);

    #pragma omp taskwait

    merge(arr, l, m, r);
}

// main interface
void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            msort_recursive(arr, 0, n, threshold);
        }
    }
}
