CXX = g++
CXXFLAGS = -Wall -O3 -std=c++17
TARGET = conv_sequential

all: $(TARGET)

$(TARGET): main_sequential.cpp conv_sequential.cpp conv_sequential.hpp
	$(CXX) main_sequential.cpp conv_sequential.cpp $(CXXFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET) sequential_results.csv sequential_filter_*.csv task_seq.out task_seq.err generate_data.out generate_data.err
