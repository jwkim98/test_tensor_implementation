#include <iostream>
#include <cuda_runtime.h>
#include "../include/CUDA_header/kernel.h"
#include <vector>

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = {1, 2, 3, 4, 5};
	const int b[arraySize] = {10, 20, 30, 40, 50};

	std::vector<int> vect_a(arraySize);
	std::vector<int> vect_b(arraySize);

	for (int count = 0; count < arraySize; count++)
	{
		vect_a.at(count) = a[count];
		vect_b.at(count) = b[count];
	}

	int c[arraySize] = {0};

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	add_with_thrust(vect_a, vect_b);

	std::cout << "output_with_trust" << std::endl;
	for (auto output : vect_b)
	{
		std::cout << output << " ";
	}
	std::cout << std::endl;

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	       c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

 	return 0;
}
