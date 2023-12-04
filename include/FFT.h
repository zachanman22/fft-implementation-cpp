#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <curand_kernel.h>
#include <algorithm>
#include <chrono>
#include <cuda/std/complex>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <bits/stdc++.h>
#include <cmath>
#include "omp.h"
#include "FourierAlgorithms.h"

using namespace std;

__global__ void binEx(cuda::std::complex<double> *bitShufTimeSig, unsigned long sigLength, int roundIdx, int tasksPerThread, cuda::std::complex<double> *tempHold);

class FFT : public FourierAlgorithms
{

    public:
        complex<double>* iterative(double *timeSignal, unsigned long sigLength);

        complex<double>* cudaParallel(double *timeSignal, unsigned long sigLength);

        complex<double>* ompParallel(double *timeSignal, unsigned long sigLength);

        // Static for the device functions to be called in kernel
        static __device__ unsigned long cudaExchangeIdx(unsigned long currIdx, unsigned long roundIdx);
    
    protected:
        static unsigned int bitRev(unsigned int num, unsigned long sigLength);

        static unsigned long zeroPad2PowerLength(double *timeSignal, unsigned long sigLength);
        static double* zeroPad2PowerArray(double *timeSignal, unsigned long sigLength);
};
