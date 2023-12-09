/*  RAHUL
 *  DASARI
 *  rdasari
 */
#ifndef A3_HPP
#define A3_HPP

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "a3.cuh"

//#include "a3.hpp"

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    gaussian_kernel( n, h, x, y);
} // gaussian_kde

#endif // A3_HPP
