#pragma once
#include "cuda_runtime.h"
#include <vector>

__global__ void addKernel(int *c, const int *a, const int *b);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void add_with_thrust(const std::vector<int>& vect_input, std::vector<int>& vect_output);
;



