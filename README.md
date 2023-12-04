Compile on CUDA enabled PACE ICE

Load needed modules
module load cuda/11.7
module load cmake
module load gcc/10.3

Command line based version
nvcc main.cu FFT.cu DFT.cu FourierAlgorithms.cu -o main.out -Xcompiler -fopenmp -lgomp

To run main.out:
./main.out

Cmake based version
In the fft-implementation-cpp directory:
mkdir build
cd build
cmake ..
make

To run main.out:
cd executable
./main.out
