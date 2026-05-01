# Input Data Generation

This folder is used to generate the common input image data and convolution weight data for the ME759 final project.

All implementations must use the same generated files:

```text
input.csv
kernel.csv
```

These two files should be generated once in this `inputs/` folder and then copied into each implementation folder:

```text
seq/
openmp/
cuda_naive/
cuda_shared/
python/
```

Each implementation expects the files to be named exactly:

```text
input.csv
kernel.csv
```

Do not rename these files unless you also modify the corresponding run scripts and source code.

---

## What the files mean

### `input.csv`

`input.csv` stores the image/input feature-map data.

For dimensions:

```text
H = image height
W = image width
Cin = number of input channels
```

The total number of input values is:

```text
Cin × H × W
```

Example:

```text
H = 64, W = 64, Cin = 3
```

Then `input.csv` contains:

```text
3 × 64 × 64 = 12,288 values
```

This represents 3 input-channel matrices, each of size `64×64`.

---

### `kernel.csv`

`kernel.csv` stores the convolution weight/filter data.

For dimensions:

```text
Cout = number of output filters
Cin  = number of input channels
K    = kernel size
```

The total number of kernel values is:

```text
Cout × Cin × K × K
```

Example:

```text
Cin = 3, Cout = 8, K = 3
```

Then `kernel.csv` contains:

```text
8 × 3 × 3 × 3 = 216 values
```

This means there are 8 filters, and each filter has 3 separate `3×3` weight matrices, one for each input channel.

---

## Data layout

All implementations use the same flattened memory layout.

### Input layout

Conceptually:

```text
input[ci][h][w]
```

Flattened index:

```text
index = (ci * H + h) * W + w
```

---

### Kernel layout

Conceptually:

```text
kernel[co][ci][kh][kw]
```

Flattened index:

```text
index = ((co * Cin + ci) * K + kh) * K + kw
```

---

## Important rule

The same `input.csv` and `kernel.csv` must be copied into every implementation folder before running that implementation.

Example:

```bash
cp inputs/input.csv sequential_implementation/input.csv
cp inputs/kernel.csv sequential_implementation/kernel.csv

cp inputs/input.csv openmp/input.csv
cp inputs/kernel.csv openmp/kernel.csv

cp inputs/input.csv cuda_naive/input.csv
cp inputs/kernel.csv cuda_naive/kernel.csv

cp inputs/input.csv cuda_shared/input.csv
cp inputs/kernel.csv cuda_shared/kernel.csv

cp inputs/input.csv python/input.csv
cp inputs/kernel.csv python/kernel.csv
```

This ensures that sequential, OpenMP, CUDA naive, CUDA shared-memory, and Python reference implementations are all evaluated on the same image data and the same convolution weights.
