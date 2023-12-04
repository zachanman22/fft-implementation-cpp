add_library(fourier_library dft.cu fft.cu fourierAlgorithms.cu ../include/dft.h ../include/fft.h ../include/fourierAlgorithms.h)

target_include_directories(fourier_library PUBLIC ../include)

target_compile_features(fourier_library PUBLIC cuda_std_11)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fourier_library PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(fourier_library PROPERTIES CUDA_ARCHITECTURES all)