# Report Plot Notes

Recommended final report figures from this folder:

1. `execution_time_by_image_size.pdf`
   - Shows method runtime as image size increases.
   - Uses log y-axis because CUDA kernel times can be much smaller than CPU/Python times.

2. `speedup_by_image_size.pdf`
   - Shows OpenMP, CUDA naive, and CUDA shared speedup over sequential.

3. `cuda_naive_vs_shared_by_image_size.pdf`
   - Directly compares CUDA naive global-memory implementation and CUDA shared-memory tiled implementation.

4. `roofline_style_cuda.pdf`
   - Simple roofline-style CUDA analysis using estimated arithmetic intensity and achieved GFLOP/s.

5. `benchmark_summary_table.pdf`
   - PDF timing table for the benchmark results.

`total_time_by_image_size.pdf` is optional. Use it only if you want to discuss end-to-end overhead, including file I/O, memory allocation, memory copies, and output handling.
