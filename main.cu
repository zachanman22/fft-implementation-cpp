#include "fft.h"
#include "dft.h"

int main()
{
    fft fftObj;
    dft dftObj;

    // unsigned long signalLength = 262144 * 256;
    unsigned long signalLength = 16384 * 8;

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
    dftResPtrCuda = dftObj.timeAlgorithm(timeSignal, signalLength, &fourierAlgorithms::cudaParallel);
    dftResPtrOmp = dftObj.timeAlgorithm(timeSignal, signalLength, &fourierAlgorithms::ompParallel);

    // for (int i = 0; i < 100; i++)
    // {
    //     cout << dftResPtr[i] << " " << dftResPtrCuda[i] << " " << dftResPtrOmp[i] << endl;
    // }

    // for (int i = 0; i < 100; i++)
    // {
    //     cout << fftResPtr[i] << " " << fftResPtrCuda[i] << " " << fftResPtrOmp[i] << " " << dftResPtrCuda[i] << " " << dftResPtrOmp[i] << endl;
    // }
}