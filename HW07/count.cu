#include "count.cuh"
#include <thrust/sort.h>
#include <thrust/reduce.h>

void count(const thrust::device_vector<int>& d_in,
           thrust::device_vector<int>& values,
           thrust::device_vector<int>& counts)
{
    thrust::device_vector<int> temp = d_in;

    thrust::sort(temp.begin(), temp.end());

    values.resize(temp.size());
    counts.resize(temp.size());

    auto new_end = thrust::reduce_by_key(
        temp.begin(),
        temp.end(),
        thrust::constant_iterator<int>(1),
        values.begin(),
        counts.begin()
    );

    values.resize(new_end.first - values.begin());
    counts.resize(new_end.second - counts.begin());
}
