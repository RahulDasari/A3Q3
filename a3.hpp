/*  RAHUL
 *  DASARI
 *  rdasari
 */
#include "a3.cu"
#ifndef A3_HPP
#define A3_HPP

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    gaussian_kernel( n, h, x, y);
} // gaussian_kde

#endif // A3_HPP
