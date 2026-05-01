NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17 -Xcompiler -Wall -Xptxas -O3

TARGET = conv_cuda_shared
SOURCES = main_cuda_shared.cu conv_cuda_shared.cu

all: $(TARGET)

$(TARGET): $(SOURCES) conv_cuda_shared.hpp
	$(NVCC) $(SOURCES) $(NVCCFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET) cuda_shared_results.csv cuda_shared_filter_*.csv cuda_shared.out cuda_shared.err
