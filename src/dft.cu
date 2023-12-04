#include "dft.h"

using namespace std;

__global__ void dftKernel(cuda::std::complex<double> *dftResult, unsigned long sigLength, double *timeSignal, int tasksPerThread)
{
    auto group = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    // printf("group %d", group.thread_rank());

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned long tid = group.thread_rank();
    // printf("tid %d\n", tid);
    // cuda::std::complex<double>* holding = (cuda::std::complex<double>*) malloc(tasksPerThread * sizeof(cuda::std::complex<double>));

    cuda::std::complex<double> j(0.0, 1.0);

    double pi = 2*acos(0.0);

    for (int tid = tasksPerThread * threadId; tid < tasksPerThread * threadId + tasksPerThread; tid++)
    {
        cuda::std::complex<double> compSum = 0;
        for (int nIdx = 0; nIdx < sigLength; nIdx++)
        {
            double fractFactor = (double) (nIdx * tid) / (double) sigLength;
            compSum += timeSignal[nIdx] * exp(-2 * pi * j * fractFactor);
        }
        dftResult[tid] = compSum;
    }
}

// https://pages.di.unipi.it/gemignani/woerner.pdf
complex<double>* dft::iterative(double *timeSignal, unsigned long sigLength)
{
    cout << "Running Iterative DFT" << endl;

    double pi = 2*acos(0.0);

    int numRounds = (int) log2(sigLength);

    complex<double>* dftResult = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    
    // auto start = chrono::high_resolution_clock::now();

    complex<double> j(0.0, 1.0);

    for (int kIdx = 0; kIdx < sigLength; kIdx++)
    {
        complex<double> compSum = 0;
        for (int nIdx = 0; nIdx < sigLength; nIdx++)
        {
            double fractFactor = (double) (nIdx * kIdx) / (double) sigLength;
            compSum += timeSignal[nIdx] * exp(-2 * pi * j * fractFactor);
        }
        dftResult[kIdx] = compSum;
        // cout << compSum << endl;
    }


    // auto stop = chrono::high_resolution_clock::now();
    // auto diff = chrono::duration_cast<chrono::microseconds>(stop - start);
    // cout << "iterative algo time (us): " << diff.count() << endl;

    return dftResult;

}

// Binary Exchange algorithm for parallel FFT
complex<double>* dft::cudaParallel(double *timeSignal, unsigned long sigLength)
{
    cout << "Running CUDA Parallelized DFT" << endl;

    complex<double>* dftResult = (complex<double>*) malloc(sigLength * sizeof(complex<double>));

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

    double pi = 2*acos(0.0);

    cuda::std::complex<double>* d_dftResult;
    double* d_timeSignal;

    cudaMalloc((void**) &d_dftResult, sizeof(cuda::std::complex<double>) * (int) sigLength);
    cudaMalloc((void**) &d_timeSignal, sizeof(double) * (int) sigLength);

    cudaMemcpy(d_timeSignal, timeSignal, sizeof(double) * sigLength, cudaMemcpyHostToDevice);

    // auto start = chrono::high_resolution_clock::now();

    // cout << "round " << roundIdx << endl;

    void *kernelArgs[] = { &d_dftResult, &sigLength, &d_timeSignal, &tasksPerThread};
    // int dev = 0;
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, dev);
    // initialize, then launch

    dim3 dimBlock(threadsPerBlock, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    cudaLaunchCooperativeKernel((void*)dftKernel, dimGrid, dimBlock, kernelArgs);
    // cudaDeviceSynchronize();
    // binEx<<<1, sigLength>>>(d_bitShufTimeSig, sigLength, roundIdx, tasksPerThread, d_tempHold);
    
    // auto stop = chrono::high_resolution_clock::now();
    // auto diff = chrono::duration_cast<chrono::microseconds>(stop - start);

    // cout << "parallel algo time (us): " << diff.count() << endl;

    cudaMemcpy(dftResult, d_dftResult, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyDeviceToHost);
    cudaFree(d_dftResult);
    cudaFree(d_timeSignal);

    return dftResult;

}

complex<double>* dft::ompParallel(double *timeSignal, unsigned long sigLength)
{
    cout << "Running OMP Parallelized DFT" << endl;

    complex<double>* dftResult = (complex<double>*) malloc(sigLength * sizeof(complex<double>));

    double pi = 2*acos(0.0);

    int numThreads = omp_get_max_threads() - 1;
    
    // auto start = chrono::high_resolution_clock::now();

    complex<double> j(0.0, 1.0);
    
    #pragma omp parallel for num_threads(numThreads)
    for (int kIdx = 0; kIdx < sigLength; kIdx++)
    {
        complex<double> compSum = 0;
        for (int nIdx = 0; nIdx < sigLength; nIdx++)
        {
            double fractFactor = (double) (nIdx * kIdx) / (double) sigLength;
            compSum += timeSignal[nIdx] * exp(-2 * pi * j * fractFactor);
        }
        dftResult[kIdx] = compSum;
    }

    // auto stop = chrono::high_resolution_clock::now();
    // auto diff = chrono::duration_cast<chrono::microseconds>(stop - start);
    // cout << "omp algo time (us): " << diff.count() << endl;

    return dftResult;

}
