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
#include <string>
#include <fstream>

using namespace std;

class fourierAlgorithms
{

    public:
        virtual complex<double>* iterative(double *timeSignal, unsigned long sigLength)=0;

        virtual complex<double>* cudaParallel(double *timeSignal, unsigned long sigLength)=0;

        virtual complex<double>* ompParallel(double *timeSignal, unsigned long sigLength)=0;

        using algoMethod = complex<double>* (fourierAlgorithms::*)(double*, unsigned long);

        complex<double>* timeAlgorithm(double *timeSignal, unsigned long sigLength, algoMethod algo);

        complex<double>* saveAlgorithmResult(string fileName, double *timeSignal, unsigned long sigLength, algoMethod algo);

        complex<double>* plotAlgorithmResult(double samplingFreq, char *plotTitle, double *timeSignal, unsigned long sigLength, algoMethod algo);

        static unsigned long zeroPadLength(double *timeSignal, unsigned long sigLength, unsigned long numZeros);
        static double* zeroPadArray(double *timeSignal, unsigned long sigLength, unsigned long numZeros);
};
