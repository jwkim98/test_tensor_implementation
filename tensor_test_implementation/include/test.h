#pragma once

#include "particle.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKernel(int *c, const int *a, const int *b);

__global__ void advanceParticles(float dt, particle * pArray, int nParticles);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

