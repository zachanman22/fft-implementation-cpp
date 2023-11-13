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
    
    auto start = chrono::high_resolution_clock::now();

    for (int roundIdx = 0; roundIdx < numRounds; roundIdx++)
    {
        int numSampsPerBlock = 1 << roundIdx;
        complex<double> compFactors[numSampsPerBlock];
        for (int factIdx = 0; factIdx < numSampsPerBlock; factIdx++)
        {
            // Array of complex factors for FFT
            // Paper uses Matlab and has a -2 * pi * 1i factor but that does not work in this implementation
            // -pi * 1i is the correct implementation when numSampsPerBlock is 2^roundIdx and roundIdx is from 0 to log2 sigLength - 1
            compFactors[factIdx] = exp(-pi * 1i * ((double) factIdx / numSampsPerBlock));
        }

        int numBlocksPerRound = 1 << (numRounds - roundIdx - 1);
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

    auto stop = chrono::high_resolution_clock::now();
    auto diff = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "iterative algo time (us): " << diff.count() << endl;

    return bitShufTimeSig;

}

__device__ unsigned long cudaExchangeIdx(unsigned long currIdx, unsigned long roundIdx)
{
    unsigned long exchangeIdx = currIdx;
    exchangeIdx ^= 1 << (roundIdx - 1);

    return exchangeIdx;
}

unsigned long exchangeIdx(unsigned long currIdx, unsigned long roundIdx)
{
    unsigned long exchangeIdx = currIdx;
    exchangeIdx ^= 1 << (roundIdx - 1);

    return exchangeIdx;
}

__device__ unsigned long cudaBitRev(unsigned long num, unsigned long sigLength)
{
    int maxBit = (int) log2f(sigLength) - 1;

    unsigned long rev = 0;

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

__global__ void binEx(cuda::std::complex<double> *bitShufTimeSig, unsigned long sigLength, int roundIdx, int tasksPerThread, cuda::std::complex<double> *tempHold)
{
    auto group = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    // printf("group %d", group.thread_rank());

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned long tid = group.thread_rank();
    // printf("tid %d\n", tid);
    // cuda::std::complex<double>* holding = (cuda::std::complex<double>*) malloc(tasksPerThread * sizeof(cuda::std::complex<double>));

    for (int tid = tasksPerThread * threadId; tid < tasksPerThread * threadId + tasksPerThread; tid++)
    {
        if (tid > sigLength)
        {
            continue;
        }
        // printf("numThreads: %d\n", group.size());
        // tid = taskIdx;
        // printf("taskIdx %d\n", taskIdx);

        // Whether the thread should process as a top or bot value in the FFT block
        // Bit shift the thread ID by the roundIdx then check LSB with & 1
        bool isBot = (tid >> (roundIdx)) & 1;
        // if (tid == sigLength - 1)
        //     printf("isBot %d\n", isBot);

        // printf("tid %d: is bot %d\n", tid, isBot);

        // int numSampsPerBlock = (int) pow(2, roundIdx);
        int numSampsPerBlock = 1 << roundIdx;
        // printf("numSampsPerBlock %d\n", numSampsPerBlock);

        unsigned long exchangeIdx = cudaExchangeIdx(tid, roundIdx + 1);
        
        cuda::std::complex<double> i(0.0, 1.0);

        double pi = 2*acos(0.0);

        unsigned long factorIdx = (isBot == 1) ? tid % numSampsPerBlock : exchangeIdx % numSampsPerBlock;

        // if (tid == sigLength - 1)
        //     printf("tid %d\n", tid);

        // if (tid == sigLength - 1)
        //     printf("factorIdx %d\n", factorIdx);

        // if (tid == sigLength - 1)
        //     printf("numSampsPerBlock %d\n", numSampsPerBlock);

        // if (tid == sigLength - 1)
        //     printf("factorAngle %f\n", -((double) factorIdx / numSampsPerBlock));
        // double factorAngle = -pi * ((double) factorIdx / numSampsPerBlock);
        cuda::std::complex<double> compFactor = exp(-pi * i * ((double) factorIdx / numSampsPerBlock));
        // cuda::std::complex<double> compFactor(cos(factorAngle), sin(factorAngle));

        // if (tid == sigLength - 1)
        //     printf("Comp factor tid %d: %f, %f\n", tid, compFactor.real(), compFactor.imag());

        // printf("Prev Shuf tid %d: %f, %f\n", tid, bitShufTimeSig[tid].real(), bitShufTimeSig[tid].imag());

        // printf("tid %d: %f, %f\n", tid, (bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid]).real(), (bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid]).imag());
        // printf("tid %d: %f, %f\n", tid, (bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx]).real(), (bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx]).imag());
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


        cuda::std::complex<double> newVal = (isBot == 1) ? bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid] : bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx];
        // group.sync();

        // if (tid == sigLength - 1)
        // {
        //     printf("prev, tid %d: %f, %f\n", tid, bitShufTimeSig[tid].real(), bitShufTimeSig[tid].imag());
        //     printf("exchange, tid %d: %f, %f\n", tid, bitShufTimeSig[exchangeIdx].real(), bitShufTimeSig[exchangeIdx].imag());
        // }

        // bitShufTimeSig[tid] = newVal;
        tempHold[tid] = newVal;
        // group.sync();

        // if (tid == sigLength - 1)
            // printf("tid %d: %f, %f\n", tid, holding[tid - tasksPerThread * threadId].real(), holding[tid - tasksPerThread * threadId].imag());
        // group.sync();
        
    }

    group.sync();

    for (int tid = tasksPerThread * threadId; tid < tasksPerThread * threadId + tasksPerThread; tid++)
    {
        // bitShufTimeSig[tid] = holding[tid - tasksPerThread * threadId];
        bitShufTimeSig[tid] = tempHold[tid];
    }
}

// Binary Exchange algorithm for parallel FFT
complex<double>* cudaIterFFT(double *timeSignal, unsigned long sigLength)
{
    // Creates bit shuffled array
    static complex<double>* bitShufTimeSig;
    bitShufTimeSig = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    // cudaMallocHost((void**) &bitShufTimeSig, sigLength * sizeof(complex<double>));
    // cudaMallocManaged((void**) &bitShufTimeSig, sigLength * sizeof(complex<double>));
    for (int k = 0; k < sigLength; k++)
    {
        // Shuffles the order of the signal based on bit reversed indices
        bitShufTimeSig[k] = timeSignal[bitRev(k, sigLength)];
    }

    int threadsPerBlock;
    int numBlocks;
    int tasksPerThread;
    if (sigLength <= 1024)
    {
        threadsPerBlock = sigLength;
        numBlocks = 1;
        tasksPerThread = 1;
    }
    else
    {
        threadsPerBlock = 1024;
        numBlocks = (int) ceil(sigLength / 1024.0);
        tasksPerThread = 1;
    }

    // Can not spawn more than 64 blocks (probably more but will need to be power of 2 and could not spawn 128 blocks)
    if (numBlocks > 64)
    {
        numBlocks = 64;
        tasksPerThread = (int) ceil(sigLength / (1024.0 * 64.0));

    }

    cout << numBlocks << endl;
    cout << tasksPerThread << endl;

    double pi = 2*acos(0.0);

    cuda::std::complex<double>* d_bitShufTimeSig;
    cuda::std::complex<double>* d_tempHold;

    cudaMalloc((void**) &d_bitShufTimeSig, sizeof(cuda::std::complex<double>) * (int) sigLength);
    cudaMalloc((void**) &d_tempHold, sizeof(cuda::std::complex<double>) * (int) sigLength);

    cudaMemcpy(d_bitShufTimeSig, bitShufTimeSig, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyHostToDevice);
    cudaMemset(d_tempHold, 0, sizeof(cuda::std::complex<double>) * sigLength);

    int numRounds = (int) log2(sigLength);
    // cout << "numRounds " << numRounds << endl;

    // int dev = 0;
    // int supportsCoopLaunch = 0;
    // cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    // cout << "supp " << supportsCoopLaunch << endl;

    auto start = chrono::high_resolution_clock::now();

    for (int roundIdx = 0; roundIdx < numRounds; roundIdx++)
    {
        // cout << "round " << roundIdx << endl;

        void *kernelArgs[] = { &d_bitShufTimeSig, &sigLength, &roundIdx, &tasksPerThread, &d_tempHold};
        // int dev = 0;
        // cudaDeviceProp deviceProp;
        // cudaGetDeviceProperties(&deviceProp, dev);
        // initialize, then launch

        dim3 dimBlock(threadsPerBlock, 1, 1);
        dim3 dimGrid(numBlocks, 1, 1);
        cudaLaunchCooperativeKernel((void*)binEx, dimGrid, dimBlock, kernelArgs);
        // cudaDeviceSynchronize();
        // binEx<<<1, sigLength>>>(d_bitShufTimeSig, sigLength, roundIdx, tasksPerThread, d_tempHold);
        // cudaMemcpy(bitShufTimeSig, d_bitShufTimeSig, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyDeviceToHost);
        // cudaMemcpy(d_bitShufTimeSig, d_tempHold, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyDeviceToDevice);

        // for (int sigIdx = 0; sigIdx < sigLength; sigIdx++)
        // {
        //     if (roundIdx == 2)
        //     {
        //         cout << bitShufTimeSig[sigIdx] << " " << endl;
        //     }
        // }
        // cout << endl;
    }
    auto stop = chrono::high_resolution_clock::now();
    auto diff = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "parallel algo time (us): " << diff.count() << endl;

    cudaMemcpy(bitShufTimeSig, d_bitShufTimeSig, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyDeviceToHost);
    // cudaFree(d_bitShufTimeSig);

    return bitShufTimeSig;

}

int main()
{
    // for (int i = 0; i < (int) log2(2048); i++)
    // {
    //     cout << exchangeIdx(2047, i + 1) << " ";
    // }
    // cout << endl;
    // for (int i = 0; i < (int) log2(2048); i++)
    // {
    //     cout << (((int) 2046 >> (i)) & 1) << " ";
    // }
    // cout << endl;

    complex<double>* fftResPtr;
    complex<double>* fftResPtr2;

    unsigned long signalLength = 67108864;
    // unsigned long signalLength = 1048576;

    double pi = 2*acos(0.0);

    double timeSignal[signalLength];

    cout << "Time Signal" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        timeSignal[i] = cos(pi / 10 * i);
        // cout << timeSignal[i] << " ";
    }
    cout << endl;

    auto start2 = chrono::high_resolution_clock::now();
    fftResPtr2 = cudaIterFFT(timeSignal, signalLength);
    auto stop2 = chrono::high_resolution_clock::now();
    auto diff2 = chrono::duration_cast<chrono::microseconds>(stop2 - start2);

    cout << "parallel total time (us): " << diff2.count() << endl;

    auto start1 = chrono::high_resolution_clock::now();
    fftResPtr = iterFFT(timeSignal, signalLength);
    auto stop1 = chrono::high_resolution_clock::now();
    auto diff1 = chrono::duration_cast<chrono::microseconds>(stop1 - start1);

    cout << "iterative total time (us): " << diff1.count() << endl;

    cout << abs(fftResPtr[0]) << " " << abs(fftResPtr2[0]) << endl;

    cout << "FFT Complex" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        // cout << fftResPtr[i] << " ";
    }
    cout << endl;

    cout << "FFT Mag" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        // cout << abs(fftResPtr[i]) << " ";
    }
    cout << endl;
    
    cout << "FFT Complex" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        // cout << fftResPtr2[i] << " ";
    }
    cout << endl;

    cout << "FFT Mag" << endl;
    for (int i = 0; i < signalLength; i++)
    {
        // cout << abs(fftResPtr2[i]) << " ";
    }
    cout << endl;

    int count = 0;

    double roundingFactor = 1000.0;

    for (int i = 0; i < signalLength; i++)
    {
        if (round(abs(fftResPtr[i]) * roundingFactor) / roundingFactor != round(abs(fftResPtr2[i]) * roundingFactor) / roundingFactor)
        {
            count += 1;
            cout << i << " " << abs(fftResPtr[i]) << " " << abs(fftResPtr2[i]) << endl;
        }
    }

    cout << abs(fftResPtr2[1]) << " " << abs(fftResPtr2[signalLength - 1]) << endl;

    cout << count << endl;


}