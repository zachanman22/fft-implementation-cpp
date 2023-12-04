Compile on CUDA enabled PACE ICE
Class based version
nvcc main.cu fft.cu dft.cu fourierAlgorithms.cu -o main.out -Xcompiler -fopenmp -lgomp
