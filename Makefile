nvcc a3.cu -o a3_cuda.o
nvcc cxxflags a3.cpp a3_cuda -o a3
all: a3

clean:
	rm -rf a3
