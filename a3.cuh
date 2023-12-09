#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void gaussian_kernel(int n, float h, const std::vector<float>& x, std::vector<float>& y);
