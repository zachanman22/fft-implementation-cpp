add_library(fourier_library FourierAlgorithms.cu DFT.cu FFT.cu ../include/FourierAlgorithms.h ../include/DFT.h ../include/FFT.h)

target_include_directories(fourier_library PUBLIC ../include)

include_directories(${matplot_SOURCE_DIR})

target_compile_features(fourier_library PUBLIC cuda_std_17)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fourier_library PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(fourier_library PROPERTIES CUDA_ARCHITECTURES "70")