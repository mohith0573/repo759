# OpenMP Sweep Notes for Report

Recommended plots to include:

1. `openmp_time_vs_threads.pdf`
   - Shows raw OpenMP execution time as thread count increases.

2. `openmp_strong_scaling.pdf`
   - Shows speedup relative to 1 thread.
   - Formula:
     ```text
     speedup_N = time_1_thread / time_N_threads
     ```

3. `openmp_efficiency.pdf`
   - Shows how efficiently threads are used.
   - Formula:
     ```text
     efficiency_N = speedup_N / N
     ```

Important discussion point:

For small image sizes like 64x64, OpenMP scaling may be poor because thread scheduling overhead can dominate the actual computation.

For larger images like 256x256 or 512x512, the useful computation increases and OpenMP parallelism should become more beneficial.
