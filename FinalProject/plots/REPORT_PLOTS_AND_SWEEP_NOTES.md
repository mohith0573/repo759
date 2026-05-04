# Report Notes: Plots and OpenMP Sweep

This file summarizes how the generated plots should be interpreted in the final report.

---

## 1. Benchmark timing table

File:

```text
benchmark_summary_table.pdf
```

This table summarizes timing results across image sizes.

For CPU methods:

```text
time_ms
```

is reported.

For CUDA methods:

```text
kernel_time_ms
```

is reported.

This table is useful as the main numerical performance summary.

---

## 2. Execution time by image size

File:

```text
execution_time_by_image_size.pdf
```

This plot shows how runtime changes as image size increases.

Methods included:

```text
Sequential
OpenMP
CUDA Naive
CUDA Shared
Python Reference, if available
```

The y-axis uses a logarithmic scale because Python/CPU runtimes and CUDA kernel runtimes may differ by large factors.

Suggested report explanation:

```text
As image size increases, the total number of PE computations grows as H × W. The sequential baseline scales with the total amount of convolution work, while OpenMP and CUDA expose parallelism across output pixels. CUDA kernel execution time is significantly lower because many PEs are executed concurrently on the GPU.
```

---

## 3. Speedup by image size

File:

```text
speedup_by_image_size.pdf
```

This plot uses:

```text
speedup = sequential_time / method_time
```

It compares:

```text
OpenMP
CUDA Naive
CUDA Shared
```

against the sequential baseline.

Suggested report explanation:

```text
Speedup improves when the amount of useful work is large enough to amortize parallel overhead. Small image sizes may show limited OpenMP benefit because CPU thread scheduling overhead can dominate. GPU methods benefit from massive thread-level parallelism, especially as the image size increases.
```

---

## 4. CUDA naive vs CUDA shared memory

File:

```text
cuda_naive_vs_shared_by_image_size.pdf
```

This plot directly compares:

```text
CUDA Naive
CUDA Shared Memory Tiled
```

using CUDA kernel time.

Suggested report explanation:

```text
The naive CUDA kernel reads each input window directly from global memory. The shared-memory version cooperatively loads a tile, including the right and bottom halo, into shared memory so neighboring PEs can reuse overlapping input pixels. For small images, shared-memory tile loading and synchronization overhead may outweigh reuse benefits. For larger image sizes or larger kernels, shared memory can become more beneficial because input reuse increases.
```

---

## 5. OpenMP execution time vs thread count

File:

```text
openmp_time_vs_threads.pdf
```

This plot shows raw OpenMP runtime for different thread counts.

Suggested report explanation:

```text
The OpenMP implementation parallelizes the output pixel loop. Increasing thread count reduces runtime when the problem size is large enough, but excessive threads can produce diminishing returns due to scheduling overhead, memory bandwidth pressure, and load imbalance.
```

---

## 6. OpenMP strong scaling

File:

```text
openmp_strong_scaling.pdf
```

Formula:

```text
speedup_N = time_1_thread / time_N_threads
```

Suggested report explanation:

```text
Strong scaling measures how much faster the same fixed-size problem runs as more CPU threads are used. Ideal scaling would produce speedup equal to the number of threads. In practice, speedup is lower because of parallel overhead and shared-memory bandwidth limits.
```

---

## 7. OpenMP parallel efficiency

File:

```text
openmp_efficiency.pdf
```

Formula:

```text
efficiency_N = speedup_N / N
```

Suggested report explanation:

```text
Parallel efficiency indicates how effectively each additional thread contributes to speedup. Efficiency usually decreases as thread count increases because overhead and contention grow.
```

---

## 8. Roofline-style CUDA plot

File:

```text
roofline_style_cuda.pdf
```

The script estimates:

```text
FLOPs = 2 × H × W × Cin × Cout × K × K
```

The factor of 2 comes from one multiply and one add.

The plot shows:

```text
x-axis: estimated arithmetic intensity, FLOPs/byte
y-axis: achieved performance, GFLOP/s
```

Suggested report explanation:

```text
The roofline-style analysis estimates whether CUDA kernels are limited more by memory bandwidth or compute throughput. The CUDA shared-memory kernel is expected to increase effective data reuse by reducing redundant global input reads, which can improve arithmetic intensity.
```

Important note:

```text
The roofline numbers are estimates based on the analytical access model, not full Nsight Compute measurements.
```

---

## 9. Total time by image size

File:

```text
total_time_by_image_size.pdf
```

This plot includes end-to-end runtime.

For CUDA, total time includes:

```text
file reading
host memory allocation
device memory allocation
host-to-device copies
kernel execution
device-to-host copies
optional output handling
cleanup
```

Suggested report explanation:

```text
CUDA kernel time isolates GPU computation, while total time captures the full execution pipeline. For small images, total time may be dominated by setup and data movement rather than kernel execution.
```

This plot is optional in the final report.

---

## 10. Correctness statement for report

Since correctness was validated separately in `results_comparison/`, the report can state:

```text
All implementations used identical input.csv and kernel.csv files. For the validation case, each output filter matrix from sequential, OpenMP, CUDA naive, and CUDA shared-memory implementations was compared element-wise against the Python reference output. The maximum absolute error was below the selected tolerance, confirming functional correctness.
```

---

## 11. Folder-level note for grader

The project separates concerns intentionally:

```text
inputs/                Generates and stores shared inputs/kernels.
sequential_execution/  Sequential C++ implementation.
openmp/                OpenMP CPU implementation.
cuda_naive/            CUDA global-memory implementation.
cuda_shared/           CUDA shared-memory tiled implementation.
python_ref/            Python reference implementation.
results_comparison/    Correctness comparison against Python reference.
exe_results/           Timing CSVs for benchmark configurations.
plots/                 Plot generation and OpenMP sweep.
```

This organization allows the grader to reproduce correctness validation and performance plotting independently.
