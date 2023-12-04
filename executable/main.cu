#include "FFT.h"
#include "DFT.h"

int main()
{
    FFT fftObj;
    DFT dftObj;

    // unsigned long signalLength = 262144 * 256;
    unsigned long signalLength = 16384 / 2;

    double pi = 2*acos(0.0);

    double timeSignal[signalLength];

    for (int i = 0; i < signalLength; i++)
    {
        timeSignal[i] = cos(pi / 10 * i);
        // cout << timeSignal[i] << " ";
    }

    complex<double>* fftResPtr;
    complex<double>* fftResPtrCuda;
    complex<double>* fftResPtrOmp;

    // fftResPtr = fftObj.iterative(timeSignal, signalLength);
    // fftResPtrCuda = fftObj.cudaParallel(timeSignal, signalLength);
    // fftResPtrOmp = fftObj.ompParallel(timeSignal, signalLength);

    fftResPtr = fftObj.timeAlgorithm(timeSignal, signalLength, &FourierAlgorithms::iterative);
    fftResPtrCuda = fftObj.timeAlgorithm(timeSignal, signalLength, &FourierAlgorithms::cudaParallel);
    fftResPtrOmp = fftObj.timeAlgorithm(timeSignal, signalLength, &FourierAlgorithms::ompParallel);

    // fftResPtr = fftObj.saveAlgorithmResult("iterativeFFT.csv", timeSignal, signalLength, &FourierAlgorithms::iterative);
    // fftResPtrCuda = fftObj.saveAlgorithmResult("parallelCudaFFT.csv", timeSignal, signalLength, &FourierAlgorithms::cudaParallel);
    // fftResPtrOmp = fftObj.saveAlgorithmResult("parallelOmpFFT.csv", timeSignal, signalLength, &FourierAlgorithms::ompParallel);

    // for (int i = 0; i < 100; i++)
    // {
    //     cout << fftResPtr[i] << " " << fftResPtrCuda[i] << " " << fftResPtrOmp[i] << endl;
    // }

    complex<double>* dftResPtr;
    complex<double>* dftResPtrCuda;
    complex<double>* dftResPtrOmp;

    // dftResPtr = dftObj.iterative(timeSignal, signalLength);
    // dftResPtrCuda = dftObj.cudaParallel(timeSignal, signalLength);
    // dftResPtrOmp = dftObj.ompParallel(timeSignal, signalLength);

    dftResPtr = dftObj.timeAlgorithm(timeSignal, signalLength, &FourierAlgorithms::iterative);
    dftResPtrCuda = dftObj.timeAlgorithm(timeSignal, signalLength, &FourierAlgorithms::cudaParallel);
    dftResPtrOmp = dftObj.timeAlgorithm(timeSignal, signalLength, &FourierAlgorithms::ompParallel);

    // dftResPtr = dftObj.saveAlgorithmResult("iterativeDFT.csv", timeSignal, signalLength, &FourierAlgorithms::iterative);
    // dftResPtrCuda = dftObj.saveAlgorithmResult("parallelCudaDFT.csv", timeSignal, signalLength, &FourierAlgorithms::cudaParallel);
    // dftResPtrOmp = dftObj.saveAlgorithmResult("parallelOmpDFT.csv", timeSignal, signalLength, &FourierAlgorithms::ompParallel);

    // for (int i = 0; i < 100; i++)
    // {
    //     cout << dftResPtr[i] << " " << dftResPtrCuda[i] << " " << dftResPtrOmp[i] << endl;
    // }

    // for (int i = 0; i < 100; i++)
    // {
    //     cout << fftResPtr[i] << " " << fftResPtrCuda[i] << " " << fftResPtrOmp[i] << " " << dftResPtr[i] << " " << dftResPtrCuda[i] << " " << dftResPtrOmp[i] << endl;
    // }
}