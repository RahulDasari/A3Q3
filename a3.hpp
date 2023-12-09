/*  RAHUL
 *  DASARI
 *  rdasari
 */
#ifndef A3_HPP
#define A3_HPP


#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <functional>
#include"math.h"

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
}

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    gaussian_kernel( n, h, x, y);
} // gaussian_kde

#endif // A3_HPP
