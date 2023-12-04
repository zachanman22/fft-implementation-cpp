#include "fft.h"
#include "dft.h"

int main()
{
    fft fftObj;
    dft dftObj;

    // unsigned long signalLength = 262144 * 256;
    unsigned long signalLength = 16384;

    double pi = 2*acos(0.0);

    double timeSignal[signalLength];

    cout << "Time Signal" << endl;
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

    // fftResPtr = fftObj.timeAlgorithm(timeSignal, signalLength, &fourierAlgorithms::iterative);
    // fftResPtrCuda = fftObj.timeAlgorithm(timeSignal, signalLength, &fourierAlgorithms::cudaParallel);
    // fftResPtrOmp = fftObj.timeAlgorithm(timeSignal, signalLength, &fourierAlgorithms::ompParallel);

    // fftResPtr = fftObj.saveAlgorithmResult("iterativeFFT.csv", timeSignal, signalLength, &fourierAlgorithms::iterative);
    // fftResPtrCuda = fftObj.saveAlgorithmResult("parallelCudaFFT.csv", timeSignal, signalLength, &fourierAlgorithms::cudaParallel);
    // fftResPtrOmp = fftObj.saveAlgorithmResult("parallelOmpFFT.csv", timeSignal, signalLength, &fourierAlgorithms::ompParallel);

    fftResPtr = fftObj.plotAlgorithmResult(10, "iterativeFFT", timeSignal, signalLength, &fourierAlgorithms::iterative);
    fftResPtrCuda = fftObj.plotAlgorithmResult(10, "parallelCudaFFT", timeSignal, signalLength, &fourierAlgorithms::cudaParallel);
    fftResPtrOmp = fftObj.plotAlgorithmResult(10, "parallelOmpFFT", timeSignal, signalLength, &fourierAlgorithms::ompParallel);

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

    // dftResPtr = dftObj.timeAlgorithm(timeSignal, signalLength, &fourierAlgorithms::iterative);
    // dftResPtrCuda = dftObj.timeAlgorithm(timeSignal, signalLength, &fourierAlgorithms::cudaParallel);
    // dftResPtrOmp = dftObj.timeAlgorithm(timeSignal, signalLength, &fourierAlgorithms::ompParallel);

    // dftResPtr = dftObj.saveAlgorithmResult("iterativeDFT.csv", timeSignal, signalLength, &fourierAlgorithms::iterative);
    // dftResPtrCuda = dftObj.saveAlgorithmResult("parallelCudaDFT.csv", timeSignal, signalLength, &fourierAlgorithms::cudaParallel);
    // dftResPtrOmp = dftObj.saveAlgorithmResult("parallelOmpDFT.csv", timeSignal, signalLength, &fourierAlgorithms::ompParallel);

    // dftResPtr = dftObj.plotAlgorithmResult(10, "iterativeDFT", timeSignal, signalLength, &fourierAlgorithms::iterative);
    dftResPtrCuda = dftObj.plotAlgorithmResult(10, "parallelCudaDFT", timeSignal, signalLength, &fourierAlgorithms::cudaParallel);
    dftResPtrOmp = dftObj.plotAlgorithmResult(10, "parallelOmpDFT", timeSignal, signalLength, &fourierAlgorithms::ompParallel);

    // for (int i = 0; i < 100; i++)
    // {
    //     cout << dftResPtr[i] << " " << dftResPtrCuda[i] << " " << dftResPtrOmp[i] << endl;
    // }

    for (int i = 0; i < 100; i++)
    {
        cout << fftResPtr[i] << " " << fftResPtrCuda[i] << " " << fftResPtrOmp[i] << " " << dftResPtrCuda[i] << " " << dftResPtrOmp[i] << endl;
    }
}