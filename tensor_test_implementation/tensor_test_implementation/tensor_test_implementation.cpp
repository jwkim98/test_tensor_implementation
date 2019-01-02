// tensor_test_implementation.cpp : Defines the entry point for the application.
//

#include "tensor_test_implementation.h"
#include "tbb/tbb.h"

using namespace std;



int main()
{
	cout << "Hello CMake." << endl;
	std::cout << "TBB version: " << TBB_VERSION_MAJOR
		<< "." << TBB_VERSION_MINOR << std::endl;
	return 0;
}
