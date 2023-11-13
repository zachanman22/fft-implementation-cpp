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

__device__ unsigned int cudaExchangeIdx(unsigned int currIdx, unsigned int roundIdx)
{
    unsigned int exchangeIdx = currIdx;
    exchangeIdx ^= 1 << (roundIdx - 1);

    return exchangeIdx;
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

__global__ void binEx(cuda::std::complex<double> *bitShufTimeSig, unsigned long sigLength, int roundIdx, cuda::std::complex<double> *tempHold)
{
    auto group = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    // printf("group %d", group.thread_rank());

    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = group.thread_rank();
    // printf("tid %d\n", tid);

    // Whether the thread should process as a top or bot value in the FFT block
    // Bit shift the thread ID by the roundIdx then check LSB with & 1
    bool isBot = (tid >> (roundIdx)) & 1;

    // printf("is bot %d\n", isBot);

    int numSampsPerBlock = (int) pow(2, roundIdx);
    // printf("numSampsPerBlock %d\n", numSampsPerBlock);

    unsigned int exchangeIdx = cudaExchangeIdx(tid, roundIdx + 1);
    
    cuda::std::complex<double> i(0.0, 1.0);

    double pi = 2*acos(0.0);

    unsigned int factorIdx = (isBot == 1) ? tid % (int) numSampsPerBlock : exchangeIdx % (int) numSampsPerBlock;
    // printf("factorAngle %f\n", -((double) factorIdx / numSampsPerBlock));
    cuda::std::complex<double> compFactor = exp(-pi * i * ((double) factorIdx / numSampsPerBlock));

    // printf("Comp factor tid %d: %f, %f\n", tid, compFactor.real(), compFactor.imag());

    // printf("Prev Shuf tid %d: %f, %f\n", tid, bitShufTimeSig[tid].real(), bitShufTimeSig[tid].imag());

    printf("tid %d: %f, %f\n", tid, (bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid]).real(), (bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid]).imag());
    printf("tid %d: %f, %f\n", tid, (bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx]).real(), (bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx]).imag());
    // if (isBot)
    // {
    //     // printf("top val for tid %d: %f\n", tid, bitShufTimeSig[exchangeIdx].real());
    //     // printf("tid %d: %f, %f\n", tid, (bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid]).real(), (bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid]).imag());
    //     // printf("tid %d: %f, %f\n", tid, (bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx]).real(), (bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx]).imag());
    //     bitShufTimeSig[tid] = bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid];
        
    // }
    // else
    // {
    //     // printf("top val for tid %d: %f\n", tid, bitShufTimeSig[tid].real());
    //     // printf("tid %d: %f, %f\n", tid, (bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid]).real(), (bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid]).imag());
    //     // printf("tid %d: %f, %f\n", tid, (bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx]).real(), (bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx]).imag());
    //     bitShufTimeSig[tid] = bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx];
        
    // }

    // group.sync();

    // __syncthreads();
    tempHold[tid] = (isBot == 1) ? bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid] : bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx];

    bitShufTimeSig[tid] = tempHold[tid];
    printf("tid %d: %f, %f\n", tid, bitShufTimeSig[tid].real(), bitShufTimeSig[tid].imag());
}

// Binary Exchange algorithm for parallel FFT
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

    cuda::std::complex<double>* d_bitShufTimeSig;
    cuda::std::complex<double>* d_tempHold;

    cudaMalloc((void**) &d_bitShufTimeSig, sizeof(cuda::std::complex<double>) * (int) sigLength);
    cudaMalloc((void**) &d_tempHold, sizeof(cuda::std::complex<double>) * (int) sigLength);

    cudaMemcpy(d_bitShufTimeSig, bitShufTimeSig, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyHostToDevice);
    cudaMemset((void**) d_tempHold, 0, sizeof(cuda::std::complex<double>) * sigLength);

    int numRounds = (int) log2(sigLength);
    cout << "numRounds " << numRounds << endl;

    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    cout << "supp " << supportsCoopLaunch << endl;

    for (int roundIdx = 0; roundIdx < numRounds; roundIdx++)
    {
        cout << "roundIdx " << roundIdx << endl; 

        void *kernelArgs[] = { &d_bitShufTimeSig, &sigLength, &roundIdx, &d_tempHold };

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        // initialize, then launch

        dim3 dimBlock(sigLength, 1, 1);
        dim3 dimGrid(1, 1, 1);
        cudaLaunchCooperativeKernel((void*)binEx, dimGrid, dimBlock, kernelArgs);
        // binEx<<<1, sigLength>>>(d_bitShufTimeSig, sigLength, roundIdx);
    }

    cudaMemcpy(bitShufTimeSig, d_bitShufTimeSig, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyDeviceToHost);
    cudaFree(d_bitShufTimeSig);

    return bitShufTimeSig;

}

int main()
{

    complex<double>* fftResPtr;
    complex<double>* fftResPtr2;

    int signalLength = 16;

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
    fftResPtr2 = cudaIterFFT(timeSignal, signalLength);

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
    
    cout << "FFT Complex" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        cout << fftResPtr2[i] << " ";
    }
    cout << endl;

    cout << "FFT Mag" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        cout << abs(fftResPtr2[i]) << " ";
    }
    cout << endl;
}