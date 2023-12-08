#include"math.h"
/* #include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h" */
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <functional>

//#include "a3.hpp"

__global__ void compute_kernel(int n, float h, float* x, float* y){
    int i,j;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<n && j<n){
        y[j] += (1/(n*h))*(1/sqrt(2*M_PI)) * exp(-pow((x[i] - x[j])/h ,2 )/2);
    }
}

void gaussian_kernel(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    float * d_x ;
    float * d_y;
    cudaMalloc((void**)&d_x, sizeof(float) * n);
    cudaMemcpy(d_x, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_y, sizeof(float) * n);
    cudaMemcpy(d_y, y.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    dim3 num_threads = (1024,1024);
    dim3 blocksize = ((n/1024) + 1,(n/1024)+1);
    compute_kernel<<<blocksize , num_threads>>>(n ,h ,d_x, d_y);
    cudaMemcpy(y.data(), d_y , sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaFree(d_x);
    cudaFree(d_y);

    for float i : x{
        printf("%d" ,i);
    }
    
}
void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    gaussian_kernel(n, h, x, y);
} // gaussian_kde


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " n h" << std::endl;
        return -1;
    }

    int n = std::atoi(argv[1]);
    float h = std::atof(argv[2]);

    if (n < 32) {
        std::cout << "hey, n is too small even for debugging!" << std::endl;
        return -1;
    }

    if (h < 0.00001) {
        std::cout << "this bandwidth is too small" << std::endl;
        return -1;
    }

    // in and out (in is set to 1s for fun)
    std::vector<float> x(n);
    std::vector<float> y(n, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::lognormal_distribution<float> N(0.0, 1.0);
    std::generate(std::begin(x), std::end(x), std::bind(N, gen));

    // now running your awesome code from a3.hpp
    auto t0 = std::chrono::system_clock::now();

    gaussian_kde(n, h, x, y);
    
    auto t1 = std::chrono::system_clock::now();

    auto elapsed_par = std::chrono::duration<double>(t1 - t0);
    std::cout << elapsed_par.count() << std::endl;

    return 0;
} 
// main

/* 
#include"math.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <functional>

__global__ void compute_kernel(int n, float h, float* x, float* y){
    int i,j;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<n && j<n)
        y[j] += (1/(n*h))*(1/sqrt(2*M_PI)) * exp(-pow((x[i] - x[j])/h ,2 )/2);
    
}

void gaussian_kernel(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    float * d_x ;
    float * d_y;
    cudaMalloc((void**)&d_x, sizeof(float) * n);
    cudaMemcpy(d_x, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_y, sizeof(float) * n);

    dim3 num_threads = (1024,1024);
    dim3 blocksize = ((n/1024) + 1,(n/1024)+1);
    compute_kernel<<<blocksize , num_threads>>>(n ,h ,d_x, d_y);
    cudaMemcpy(y.data(), d_y , sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaFree(d_x);
    cudaFree(d_y);
} */