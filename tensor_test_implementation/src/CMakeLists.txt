
add_library(particles STATIC
  test.cu
  "../include/test.h"
  particle.cu
  "../include/particle.h"
  v3.cu
  "../include/v3.h"
  )

  add_library(using_tbb STATIC
  randomize.cpp
   "../include/randomize.h"
  )

 # target_link_libraries(using_tbb ${TBB_LIBRARY})


# Request that particles be built with -std=c++17
# As this is a public compile feature anything that links to particles
# will also build with -std=c++17
target_compile_features(particles PUBLIC cxx_std_17)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( particles
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

message("TBB_LIBRARY: " ${TBB_LIBRARY})
if(BUILD_TESTING)

  add_executable(tensor_test main.cpp)

  set_target_properties(tensor_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(tensor_test PRIVATE particles using_tbb)

  if(APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET particle_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()
