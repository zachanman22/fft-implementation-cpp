#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <curand_kernel.h>
#include <algorithm>
#include <chrono>
#include <cuda/std/complex>
#include <bits/stdc++.h>
#include <cmath>

using namespace std;


// __global__ void calcFFT(double *timeSignal, unsigned long sigLength, unsigned long fftLength, cuda::std::complex<double> *result)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     double timeSigEven[(sigLength + 1) / 2];
//     double timeSigOdd[(sigLength + 1) / 2];
//     cuda::std::complex<double> fftEven[(sigLength + 1) / 2];
//     cuda::std::complex<double> fftOdd[(sigLength + 1) / 2];
//     for (int k = 0; k < sigLength; k++)
//     {
//         if (k % 2 == 0)
//         {
//             timeSigEven[k / 2] = timeSignal[k];
//         }
//         else
//         {
//             timeSigOdd[(k - 1) / 2] = timeSignal[k];
//         }
//     }

//     calcFFT(timeSigEven, (sigLength + 1) / 2, (fftLength + 1) / 2, fftEven);
//     calcFFT(timeSigOdd, (sigLength + 1) / 2, (fftLength + 1) / 2, fftOdd);

//     cuda::std::complex<double> factors[fftLength];
//     cuda::std::complex<double> j(0.0, 1.0);
//     for (int n = 0; n < fftLength; n++)
//     {
//         factors[n] = exp(-2 * j * ((double) n / (double) fftLength));
//     }

//     result[tid] = 
// }

unsigned int bitRev(unsigned int num, unsigned long sigLength);

// https://pages.di.unipi.it/gemignani/woerner.pdf
complex<double>* iterFFT(double *timeSignal, unsigned long sigLength)
{
    // Creates bit shuffled array
    static complex<double>* bitShufTimeSig;
    bitShufTimeSig = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    for (int k = 0; k < sigLength; k++)
    {
        // Shuffles the order of the signal based on bit reversed indices
        bitShufTimeSig[k] = timeSignal[bitRev(k, sigLength)];
    }

    double pi = 2*acos(0.0);

    int numRounds = (int) log2(sigLength);
    for (int roundIdx = 0; roundIdx < numRounds; roundIdx++)
    {
        int numSampsPerBlock = (int) pow(2, roundIdx);
        complex<double> compFactors[numSampsPerBlock];
        for (int factIdx = 0; factIdx < numSampsPerBlock; factIdx++)
        {
            // Array of complex factors for FFT
            // Paper uses Matlab and has a -2 * pi * 1i factor but that does not work in this implementation
            // -pi * 1i is the correct implementation when numSampsPerBlock is 2^roundIdx and roundIdx is from 0 to log2 sigLength - 1
            compFactors[factIdx] = exp(-pi * 1i * ((double) factIdx / numSampsPerBlock));
        }

        int numBlocksPerRound = pow(2, numRounds - roundIdx - 1);
        for (int blockIndx = 0; blockIndx < numBlocksPerRound; blockIndx++)
        {
            int startIdx = (blockIndx) * 2 * numSampsPerBlock;
            int endIdx = (blockIndx + 1) * 2 * numSampsPerBlock - 1;
            int midIdx = startIdx + (endIdx - startIdx + 1) / 2;

            complex<double> top[midIdx - startIdx];
            for (int topIdx = startIdx; topIdx < midIdx; topIdx++)
            {
                top[topIdx - startIdx] = bitShufTimeSig[topIdx];
            }

            complex<double> bot[endIdx - midIdx + 1];
            for (int botIdx = midIdx; botIdx <= endIdx; botIdx++)
            {
                bot[botIdx - midIdx] = bitShufTimeSig[botIdx] * compFactors[botIdx - midIdx];
            }

            for (int resIdx = startIdx; resIdx <= endIdx; resIdx++)
            {
                bitShufTimeSig[resIdx] = resIdx < midIdx ? top[resIdx - startIdx] + bot[resIdx - startIdx] : top[resIdx - midIdx] - bot[resIdx - midIdx];
            }
        }
    }

    return bitShufTimeSig;

}

__device__ unsigned int cudaBitRev(unsigned int num, unsigned long sigLength)
{
    int maxBit = (int) log2f(sigLength) - 1;

    unsigned int rev = 0;

    for (int i = maxBit; i >= 0; i--)
    {
        rev |= (num & 1) << i;
        num >>= 1;
    }

    return rev;
}

unsigned int bitRev(unsigned int num, unsigned long sigLength)
{
    int maxBit = (int) log2(sigLength) - 1;

    unsigned int rev = 0;

    for (int i = maxBit; i >= 0; i--)
    {
        rev |= (num & 1) << i;
        num >>= 1;
    }

    return rev;
}

complex<double>* cudaIterFFT(double *timeSignal, unsigned long sigLength)
{
    // Creates bit shuffled array
    static complex<double>* bitShufTimeSig;
    bitShufTimeSig = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    for (int k = 0; k < sigLength; k++)
    {
        // Shuffles the order of the signal based on bit reversed indices
        bitShufTimeSig[k] = timeSignal[bitRev(k, sigLength)];
    }

    double pi = 2*acos(0.0);

    int numRounds = (int) log2(sigLength);
    for (int roundIdx = 0; roundIdx < numRounds; roundIdx++)
    {
        int numSampsPerBlock = (int) pow(2, roundIdx);
        complex<double> compFactors[numSampsPerBlock];
        for (int factIdx = 0; factIdx < numSampsPerBlock; factIdx++)
        {
            // Array of complex factors for FFT
            // Paper uses Matlab and has a -2 * pi * 1i factor but that does not work in this implementation
            // -pi * 1i is the correct implementation when numSampsPerBlock is 2^roundIdx and roundIdx is from 0 to log2 sigLength - 1
            compFactors[factIdx] = exp(-pi * 1i * ((double) factIdx / numSampsPerBlock));
        }

        int numBlocksPerRound = pow(2, numRounds - roundIdx - 1);
        for (int blockIndx = 0; blockIndx < numBlocksPerRound; blockIndx++)
        {
            int startIdx = (blockIndx) * 2 * numSampsPerBlock;
            int endIdx = (blockIndx + 1) * 2 * numSampsPerBlock - 1;
            int midIdx = startIdx + (endIdx - startIdx + 1) / 2;

            complex<double> top[midIdx - startIdx];
            for (int topIdx = startIdx; topIdx < midIdx; topIdx++)
            {
                top[topIdx - startIdx] = bitShufTimeSig[topIdx];
            }

            complex<double> bot[endIdx - midIdx + 1];
            for (int botIdx = midIdx; botIdx <= endIdx; botIdx++)
            {
                bot[botIdx - midIdx] = bitShufTimeSig[botIdx] * compFactors[botIdx - midIdx];
            }

            for (int resIdx = startIdx; resIdx <= endIdx; resIdx++)
            {
                bitShufTimeSig[resIdx] = resIdx < midIdx ? top[resIdx - startIdx] + bot[resIdx - startIdx] : top[resIdx - midIdx] - bot[resIdx - midIdx];
            }
        }
    }

    return bitShufTimeSig;

}

int main()
{
    bitRev(32, 32);

    complex<double>* fftResPtr;

    int signalLength = 1024;

    double pi = 2*acos(0.0);

    double timeSignal[signalLength];

    cout << "Time Signal" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        timeSignal[i] = cos(pi / 10 * i);
        cout << timeSignal[i] << " ";
    }
    cout << endl;

    fftResPtr = iterFFT(timeSignal, signalLength);

    cout << "FFT Complex" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        cout << fftResPtr[i] << " ";
    }
    cout << endl;

    cout << "FFT Mag" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        cout << abs(fftResPtr[i]) << " ";
    }
    cout << endl;
    
}