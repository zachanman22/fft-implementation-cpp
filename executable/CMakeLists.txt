add_executable(main.out main.cu)

target_compile_features(main.out PRIVATE cuda_std_11)

set_target_properties(main.out PROPERTIES CUDA_ARCHITECTURES all)

target_link_libraries(main.out PRIVATE fourier_library)