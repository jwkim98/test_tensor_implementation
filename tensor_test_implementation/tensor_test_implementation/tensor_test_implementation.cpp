// tensor_test_implementation.cpp : Defines the entry point for the application.
//

#include "tensor_test_implementation.h"
#include "tbb/tbb.h"
#include <cmath>

using namespace std;

using namespace tbb;

void Foo(float& input)
{
	pow(input ,3);
}

class ApplyFoo {
	float *const my_a;
public:
	void operator()(const blocked_range<size_t>& r) const {
		float *a = my_a;
		for (size_t i = r.begin(); i != r.end(); ++i)
			Foo(a[i]);
	}
	ApplyFoo(float a[]) :
		my_a(a)
	{}
};

int main()
{
	cout << "Hello CMake." << endl;
	std::cout << "TBB version: " << TBB_VERSION_MAJOR
		<< "." << TBB_VERSION_MINOR << std::endl;
	float input[5] = { 1,2,3,4,5 };
	ApplyFoo foo(input);
	foo(blocked_range<size_t>(0, 10));
	return 0;
}
