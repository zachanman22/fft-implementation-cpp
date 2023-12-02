Compile on CUDA enabled PACE ICE with
Function based (development) version
nvcc fft.cu -o fft.out -Xcompiler -fopenmp -lgomp

Class based version
nvcc main.cu fft.cu dft.cu fourierAlgorithms.cu -o main.out -Xcompiler -fopenmp -lgomp
