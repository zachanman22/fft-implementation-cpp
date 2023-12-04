#include "DFT.h"

using namespace std;

/*
Description: kernel for the parallel dft computation using CUDA
Parameters:
    dftResult - pointer array to the dft result
    sigLength - length of the time signal pointer array
    timeSignal - time signal to be analyzed
    tasksPerThread - number of round coefficients that each thread must compute
*/
__global__ void dftKernel(cuda::std::complex<double> *dftResult, unsigned long sigLength, double *timeSignal, int tasksPerThread)
{
    auto group = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Complex j
    cuda::std::complex<double> j(0.0, 1.0);

    // Pi
    double pi = 2*acos(0.0);

    // Based on threadId and number of tasks per thread, loop through the computations each thread must do
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

/*
Description: performs iterative dft using double for loop
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
Return:
    dftResult - complex double array pointer that is the result of the fourier algorithm in correct order
*/
complex<double>* DFT::iterative(double *timeSignal, unsigned long sigLength)
{
    cout << "Running Iterative DFT" << endl;

    // Pi
    double pi = 2*acos(0.0);

    // Complex j
    complex<double> j(0.0, 1.0);

    // Allocate space for the dft result
    complex<double>* dftResult = (complex<double>*) malloc(sigLength * sizeof(complex<double>));

    // Double for loop for dft computation
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

    return dftResult;

}

/*
Description: performs parallel dft using CUDA
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
Return:
    dftResult - complex double array pointer that is the result of the fourier algorithm in correct order
*/
complex<double>* DFT::cudaParallel(double *timeSignal, unsigned long sigLength)
{
    cout << "Running CUDA Parallelized DFT" << endl;

    // Allocate space for the dft result
    complex<double>* dftResult = (complex<double>*) malloc(sigLength * sizeof(complex<double>));

    // CUDA parameter preprocessing to prevent errors
    int threadsPerBlock;
    int numBlocks;
    int tasksPerThread;

    // Spawn only 1 block if the signal length is less than 1024
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

    // Pointers for CUDA arrays
    cuda::std::complex<double>* d_dftResult;
    double* d_timeSignal;

    // Allocate space on GPU for CUDA arrays
    cudaMalloc((void**) &d_dftResult, sizeof(cuda::std::complex<double>) * (int) sigLength);
    cudaMalloc((void**) &d_timeSignal, sizeof(double) * (int) sigLength);

    // Initialize the values of the CUDA arrays
    cudaMemcpy(d_timeSignal, timeSignal, sizeof(double) * sigLength, cudaMemcpyHostToDevice);

    // Kernel arguments
    void *kernelArgs[] = { &d_dftResult, &sigLength, &d_timeSignal, &tasksPerThread};

    // Set block and grid sizes
    dim3 dimBlock(threadsPerBlock, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);

    // Launch the kernel
    cudaLaunchCooperativeKernel((void*)dftKernel, dimGrid, dimBlock, kernelArgs);

    // Copy the CUDA array result to the CPU array
    cudaMemcpy(dftResult, d_dftResult, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyDeviceToHost);

    // Free the CUDA array memory
    cudaFree(d_dftResult);
    cudaFree(d_timeSignal);

    return dftResult;

}

/*
Description: performs parallel dft using OpenMP
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
Return:
    dftResult - complex double array pointer that is the result of the fourier algorithm in correct order
*/
complex<double>* DFT::ompParallel(double *timeSignal, unsigned long sigLength)
{
    cout << "Running OMP Parallelized DFT" << endl;

    // Allocate space for the dft result
    complex<double>* dftResult = (complex<double>*) malloc(sigLength * sizeof(complex<double>));

    // Pi
    double pi = 2*acos(0.0);

    // Complex j
    complex<double> j(0.0, 1.0);

    // Use the total number of threads minus 1 (for the main thread)
    int numThreads = omp_get_max_threads() - 1;
    
    // Process the outer loop of the double for loop using OpenMP threads
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

    return dftResult;

}
