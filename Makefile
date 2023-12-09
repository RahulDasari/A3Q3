CXX=nvcc
CXXFLAGS=-std=c++11

all: a3

a3: a3.o a3_host.o
	$(CXX) $(CXXFLAGS) a3.o a3_host.o -o a3 -lcuda -lcudart
	
a3.o: a3.cu
	$(CXX) $(CXXFLAGS) -c a3.cu -o a3.o
	
a3_host.o: a3.cpp
	$(CXX) $(CXXFLAGS) -c a3.cpp -o a3_host.o