NVCC = nvcc
TARGET = conv_cuda_naive
SOURCES = main_cuda_naive.cu conv_cuda_naive.cu

NVCCFLAGS = -O3 -std=c++17 -Xcompiler -Wall -Xptxas -O3

all: $(TARGET)

$(TARGET): $(SOURCES) conv_cuda_naive.hpp
	$(NVCC) $(SOURCES) $(NVCCFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET) cuda_naive_results.csv cuda_naive_filter_*.csv cuda_naive.out cuda_naive.err
