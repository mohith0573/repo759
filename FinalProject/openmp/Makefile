CXX=g++
CXXFLAGS=-Wall -O3 -std=c++17 -fopenmp
TARGET=conv_openmp
SOURCES=main_openmp.cpp conv_openmp.cpp

all: $(TARGET)

$(TARGET): $(SOURCES) conv_openmp.hpp
	$(CXX) $(SOURCES) $(CXXFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET) openmp_results.csv openmp_filter_*.csv openmp.out openmp.err
