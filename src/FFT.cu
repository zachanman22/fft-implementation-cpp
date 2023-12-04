#include "FFT.h"

using namespace std;

/*
Description: kernel for the binary exchange algorithm for parallel fft computation using CUDA
Parameters:
    bitShufTimeSig - pointer array to the bit shuffled time signal to be analyzed
    sigLength - length of the time signal pointer array
    roundIdx - round of the fft algorithm (there are log_2(n) rounds of the fft algorithm)
    tasksPerThread - number of round coefficients that each thread must compute
    tempHold - pointer to a temporary array to hold round results which gets copied back to the bitShufTimeSig array
*/
__global__ void binEx(cuda::std::complex<double> *bitShufTimeSig, unsigned long sigLength, int roundIdx, int tasksPerThread, cuda::std::complex<double> *tempHold)
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
        // There might be more tid's than the length of the signal, if so, continue without computations
        if (tid > sigLength)
        {
            continue;
        }

        // Whether the thread should process as a top or bot value in the FFT block
        // Bit shift the thread ID by the roundIdx then check LSB with & 1
        bool isBot = (tid >> (roundIdx)) & 1;

        int numSampsPerBlock = 1 << roundIdx;

        // Get the exchange index
        unsigned long exchangeIdx = FFT::cudaExchangeIdx(tid, roundIdx + 1);

        // Get the factor for the exchange index
        unsigned long factorIdx = (isBot == 1) ? tid % numSampsPerBlock : exchangeIdx % numSampsPerBlock;

        cuda::std::complex<double> compFactor = exp(-pi * j * ((double) factorIdx / numSampsPerBlock));

        cuda::std::complex<double> newVal = (isBot == 1) ? bitShufTimeSig[exchangeIdx] - compFactor * bitShufTimeSig[tid] : bitShufTimeSig[tid] + compFactor * bitShufTimeSig[exchangeIdx];

        // Set the value in the temporary array
        tempHold[tid] = newVal;
    }

    // Sync all threads
    group.sync();

    // Copy the temporary array values into the main array values
    for (int tid = tasksPerThread * threadId; tid < tasksPerThread * threadId + tasksPerThread; tid++)
    {
        bitShufTimeSig[tid] = tempHold[tid];
    }
}

/*
Description: gets the bit reversed integer value given an integer and signal length (must be power of 2)
Parameters:
    num - unsigned integer to be bit reversed
    sigLength - length of the time signal pointer array
Return:
    rev - bit reversed unsigned integer
*/
unsigned int FFT::bitRev(unsigned int num, unsigned long sigLength)
{
    // Most significant bit needed for the representation (based on signal length aka number of fft points)
    int maxBit = (int) log2(sigLength) - 1;

    unsigned int rev = 0;

    // Bit reversal algorithm
    for (int i = maxBit; i >= 0; i--)
    {
        rev |= (num & 1) << i;
        num >>= 1;
    }

    return rev;
}

/*
Description: gets the index to calculate values with using the binary exchange algorithm
Parameters:
    currIdx - index to calculate the exchange index for
    roundIdx - fft round index because a current index's exchange index is based on the round index too
Return:
    exchangeIdx - exchange index for the current index and round index
*/
__device__ unsigned long FFT::cudaExchangeIdx(unsigned long currIdx, unsigned long roundIdx)
{
    unsigned long exchangeIdx = currIdx;
    exchangeIdx ^= 1 << (roundIdx - 1);

    return exchangeIdx;
}

/*
Description: gets the nearest power of 2 greater than or equal to a signal's current length
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
Return:
    nearest2Power - nearest power of 2 greater than the signal's current length unless the current length is a power of 2
*/
unsigned long FFT::zeroPad2PowerLength(double *timeSignal, unsigned long sigLength)
{
    // Check if the signal length is not a power of 2
    if (ceil(log2(sigLength)) != floor(log2(sigLength)))
    {
        unsigned long nearest2Power = 1 << (int) ceil(log2(sigLength));

        return nearest2Power;
    }
    else
    {
        return sigLength;
    }

}

/*
Description: zeros pads a time signal to its nearest power of 2 greater than or equal to its current length
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
Return:
    paddedTimeSignal - zero padded time signal unless the current signal length is a power of 2
*/
double* FFT::zeroPad2PowerArray(double *timeSignal, unsigned long sigLength)
{
    if (ceil(log2(sigLength)) != floor(log2(sigLength)))
    {
        // Determine the nearest power of 2 greater than the signal length
        unsigned long nearest2Power = 1 << (int) ceil(log2(sigLength));

        // Allocate space for the zero padded signal
        static double* paddedTimeSignal = (double*) malloc(nearest2Power * sizeof(double));;

        paddedTimeSignal = zeroPadArray(timeSignal, sigLength, nearest2Power - sigLength);

        return paddedTimeSignal;
    }
    else
    {
        return timeSignal;
    }
}

/*
Description: performs iterative fft based on https://pages.di.unipi.it/gemignani/woerner.pdf
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
Return:
    bitShufTimeSig - complex double array pointer that is the result of the fourier algorithm in correct order
*/
complex<double>* FFT::iterative(double *timeSignal, unsigned long sigLength)
{
    cout << "Running Iterative FFT" << endl;

    // Zero pad the time signal to the nearest power of 2
    unsigned long tempNewSigLength = zeroPad2PowerLength(timeSignal, sigLength);
    timeSignal = zeroPad2PowerArray(timeSignal, sigLength);
    sigLength = tempNewSigLength;

    // Creates bit shuffled array using the bit reversal algorithm for each index
    static complex<double>* bitShufTimeSig;
    bitShufTimeSig = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    for (int k = 0; k < sigLength; k++)
    {
        // Shuffles the order of the signal based on bit reversed indices
        bitShufTimeSig[k] = timeSignal[bitRev(k, sigLength)];
    }

    // Pi
    double pi = 2*acos(0.0);

    // Complex j
    complex<double> j(0.0, 1.0);

    // Number of fft rounds
    int numRounds = (int) log2(sigLength);

    // Run fft over the log_2(n) rounds
    for (int roundIdx = 0; roundIdx < numRounds; roundIdx++)
    {
        int numSampsPerBlock = 1 << roundIdx;

        // Create twiddle factor array
        complex<double> compFactors[numSampsPerBlock];
        for (int factIdx = 0; factIdx < numSampsPerBlock; factIdx++)
        {
            // Array of complex factors for FFT
            // Paper uses Matlab and has a -2 * pi * i factor but that does not work in this implementation
            // -pi * i is the correct implementation when numSampsPerBlock is 2^roundIdx and roundIdx is from 0 to log2 sigLength - 1
            compFactors[factIdx] = exp(-pi * j * ((double) factIdx / numSampsPerBlock));
        }

        // Number of subblocks for the fft round
        int numBlocksPerRound = 1 << (numRounds - roundIdx - 1);

        // Process each subblock
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

/*
Description: performs parallel fft based on binary exchange algorithm
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
Return:
    bitShufTimeSig - complex double array pointer that is the result of the fourier algorithm in correct order
*/
complex<double>* FFT::cudaParallel(double *timeSignal, unsigned long sigLength)
{
    cout << "Running CUDA Parallelized FFT" << endl;

    // Zero pad the time signal to the nearest power of 2
    unsigned long tempNewSigLength = zeroPad2PowerLength(timeSignal, sigLength);
    timeSignal = zeroPad2PowerArray(timeSignal, sigLength);
    sigLength = tempNewSigLength;

    // Creates bit shuffled array using the bit reversal algorithm for each index
    static complex<double>* bitShufTimeSig;
    bitShufTimeSig = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    for (int k = 0; k < sigLength; k++)
    {
        // Shuffles the order of the signal based on bit reversed indices
        bitShufTimeSig[k] = timeSignal[bitRev(k, sigLength)];
    }

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
    cuda::std::complex<double>* d_bitShufTimeSig;
    cuda::std::complex<double>* d_tempHold;

    // Allocate space on GPU for CUDA arrays
    cudaMalloc((void**) &d_bitShufTimeSig, sizeof(cuda::std::complex<double>) * (int) sigLength);
    cudaMalloc((void**) &d_tempHold, sizeof(cuda::std::complex<double>) * (int) sigLength);

    // Initialize the values of the CUDA arrays
    cudaMemcpy(d_bitShufTimeSig, bitShufTimeSig, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyHostToDevice);
    cudaMemset(d_tempHold, 0, sizeof(cuda::std::complex<double>) * sigLength);

    // Number of fft rounds
    int numRounds = (int) log2(sigLength);

    // Run fft over the log_2(n) rounds
    for (int roundIdx = 0; roundIdx < numRounds; roundIdx++)
    {
        // Kernel arguments
        void *kernelArgs[] = { &d_bitShufTimeSig, &sigLength, &roundIdx, &tasksPerThread, &d_tempHold};

        // Set block and grid sizes
        dim3 dimBlock(threadsPerBlock, 1, 1);
        dim3 dimGrid(numBlocks, 1, 1);

        // Launch the kernel
        cudaLaunchCooperativeKernel((void*)binEx, dimGrid, dimBlock, kernelArgs);
    }

    // Copy the CUDA array result to the CPU array
    cudaMemcpy(bitShufTimeSig, d_bitShufTimeSig, sizeof(cuda::std::complex<double>) * sigLength, cudaMemcpyDeviceToHost);
    
    // Free the CUDA array memory
    cudaFree(d_bitShufTimeSig);
    cudaFree(d_tempHold);

    return bitShufTimeSig;

}

/*
Description: performs parallel fft based on iterative fft from https://pages.di.unipi.it/gemignani/woerner.pdf
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
Return:
    bitShufTimeSig - complex double array pointer that is the result of the fourier algorithm in correct order
*/
complex<double>* FFT::ompParallel(double *timeSignal, unsigned long sigLength)
{
    cout << "Running OMP Parallelized FFT" << endl;

    // Zero pad the time signal to the nearest power of 2
    unsigned long tempNewSigLength = zeroPad2PowerLength(timeSignal, sigLength);
    timeSignal = zeroPad2PowerArray(timeSignal, sigLength);
    sigLength = tempNewSigLength;

    // Creates bit shuffled array using the bit reversal algorithm for each index
    static complex<double>* bitShufTimeSig;
    bitShufTimeSig = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    for (int k = 0; k < sigLength; k++)
    {
        // Shuffles the order of the signal based on bit reversed indices
        bitShufTimeSig[k] = timeSignal[bitRev(k, sigLength)];
    }

    // Allocate space for a temporary array
    complex<double>* tempHold = (complex<double>*) malloc(sigLength * sizeof(complex<double>));

    // Pi
    double pi = 2*acos(0.0);

    // Complex j
    complex<double> j(0.0, 1.0);

    // Number of fft rounds
    int numRounds = (int) log2(sigLength);

    // Use the total number of threads minus 1 (for the main thread)
    int numThreads = omp_get_max_threads() - 1;

    // Run fft over the log_2(n) rounds
    for (int roundIdx = 0; roundIdx < numRounds; roundIdx++)
    {
        int numSampsPerBlock = 1 << roundIdx;

        // Create twiddle factor array
        complex<double> compFactors[numSampsPerBlock];
        for (int factIdx = 0; factIdx < numSampsPerBlock; factIdx++)
        {
            // Array of complex factors for FFT
            // Paper uses Matlab and has a -2 * pi * i factor but that does not work in this implementation
            // -pi * i is the correct implementation when numSampsPerBlock is 2^roundIdx and roundIdx is from 0 to log2 sigLength - 1
            compFactors[factIdx] = exp(-pi * j * ((double) factIdx / numSampsPerBlock));
        }

        // Number of subblocks for the fft round
        int numBlocksPerRound = 1 << (numRounds - roundIdx - 1);

        // Process each subblock with OpenMP threads
        #pragma omp parallel for num_threads(numThreads)
        for (int blockIndx = 0; blockIndx < numBlocksPerRound; blockIndx++)
        {
            int startIdx = (blockIndx) * 2 * numSampsPerBlock;
            int endIdx = (blockIndx + 1) * 2 * numSampsPerBlock - 1;
            int midIdx = startIdx + (endIdx - startIdx + 1) / 2;

            // Allocate space
            complex<double>* top = (complex<double>*) malloc(sizeof(complex<double>) * (midIdx - startIdx));
            for (int topIdx = startIdx; topIdx < midIdx; topIdx++)
            {
                top[topIdx - startIdx] = bitShufTimeSig[topIdx];
            }

            // Allocate space
            complex<double>* bot = (complex<double>*) malloc(sizeof(complex<double>) * (endIdx - midIdx + 1));
            for (int botIdx = midIdx; botIdx <= endIdx; botIdx++)
            {
                bot[botIdx - midIdx] = bitShufTimeSig[botIdx] * compFactors[botIdx - midIdx];
            }

            // Set the value to the temporary array
            for (int resIdx = startIdx; resIdx <= endIdx; resIdx++)
            {
                tempHold[resIdx] = resIdx < midIdx ? top[resIdx - startIdx] + bot[resIdx - startIdx] : top[resIdx - midIdx] - bot[resIdx - midIdx];
            }

            // Free the allocated space
            free(top);
            free(bot);
        }

        // Threads will finish processing

        // Copy from the temporary array to the main array
        #pragma omp parallel for num_threads(numThreads)
        for (int blockIndx = 0; blockIndx < numBlocksPerRound; blockIndx++)
        {
            int startIdx = (blockIndx) * 2 * numSampsPerBlock;
            int endIdx = (blockIndx + 1) * 2 * numSampsPerBlock - 1;

            for (int resIdx = startIdx; resIdx <= endIdx; resIdx++)
            {
                bitShufTimeSig[resIdx] = tempHold[resIdx];
            }
        }
    }

    // Free the temporary array
    free(tempHold);

    return bitShufTimeSig;

}
