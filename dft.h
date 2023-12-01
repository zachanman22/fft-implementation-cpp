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

using namespace std;

// __global__ void dftCuda(cuda::std::complex<double> *dftResult, unsigned long sigLength, double *timeSignal);

class dft
{

    public:
        complex<double>* iterDFT(double *timeSignal, unsigned long sigLength);

        complex<double>* cudaIterDFT(double *timeSignal, unsigned long sigLength);

        complex<double>* ompIterDFT(double *timeSignal, unsigned long sigLength);
    
    protected:
        static unsigned long zeroPadLength(double *timeSignal, unsigned long sigLength);
        static double* zeroPadArray(double *timeSignal, unsigned long sigLength);
};
