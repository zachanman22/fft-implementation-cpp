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

__global__ void dftCuda(cuda::std::complex<double> *dftResult, unsigned long sigLength, double *timeSignal);

class DFT : public FourierAlgorithms
{

    public:
        complex<double>* iterative(double *timeSignal, unsigned long sigLength);

        complex<double>* cudaParallel(double *timeSignal, unsigned long sigLength);

        complex<double>* ompParallel(double *timeSignal, unsigned long sigLength);
};
