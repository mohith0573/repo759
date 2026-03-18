#include "msort.h"
#include <algorithm>
#include <vector>

void merge(int* arr, int l, int m, int r) {
    std::vector<int> temp(r - l + 1);

    int i = l, j = m+1, k = 0;

    while (i <= m && j <= r)
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];

    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];

    for (int x = 0; x < k; x++)
        arr[l + x] = temp[x];
}

void msort_rec(int* arr, int l, int r, std::size_t threshold) {

    if (r - l + 1 <= (int)threshold) {
        std::sort(arr + l, arr + r + 1);
        return;
    }

    int m = (l + r) / 2;

    #pragma omp task
    msort_rec(arr, l, m, threshold);

    #pragma omp task
    msort_rec(arr, m+1, r, threshold);

    #pragma omp taskwait
    merge(arr, l, m, r);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {

    #pragma omp parallel
    {
        #pragma omp single
        msort_rec(arr, 0, n-1, threshold);
    }
}
