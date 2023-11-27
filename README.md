Compile on CUDA enabled PACE ICE with
Function based (development) version
nvcc fft.cu -o fft.out -Xcompiler -fopenmp -lgomp

Class based version
nvcc main.cu fft_imp.cu -o main.out -Xcompiler -fopenmp -lgomp

Note fft with OpenMP does not work for powers of 2 with a value of 2^16 = 262144 or greater due to limited stack size for spawned OpenMP threads