#CXX=g++
#CXXFLAGS=-std=c++11 -O3

#all: a3

#clean:
#	rm -rf a3
#nvcc a3.cu -o a3 
#CXX=nvcc
#CXXFLAGS=-std=c++11 -O3 

#all: a3

#a3: a3.cu a3.hpp a3.cpp
#	$(CXX) $(CXXFLAGS) a3.cu -o a3_cu.o
#	$(CXX) $(CXXFLAGS) a3.cpp a3_cu.o -o a3


nvcc a3.cpp -o a3