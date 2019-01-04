
  add_library(using_tbb STATIC
  randomize.cpp
   "../include/randomize.h"
  )

 add_library(cuda_impl STATIC "CUDA_impl/kernel.cu" "../include/CUDA_header/kernel.h")
 # target_link_libraries(using_tbb ${TBB_LIBRARY})


# Request that particles be built with -std=c++17
# As this is a public compile feature anything that links to particles
# will also build with -std=c++17
target_compile_features(cuda_impl PUBLIC cxx_std_17)
set_target_properties(cuda_impl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(BUILD_TESTING)

  add_executable(tensor_test main.cpp)

  set_target_properties(tensor_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(tensor_test PRIVATE using_tbb cuda_impl)

  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET particle_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()
