#include "fft.h"
#include "dft.h"

int main()
{
    fft fftObj = fft::fft();
    dft dftObj = dft::dft();

    unsigned long signalLength = 262144;
    // unsigned long signalLength = 16384;

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

    fftResPtr = fftObj.iterFFT(timeSignal, signalLength);
    fftResPtrCuda = fftObj.cudaIterFFT(timeSignal, signalLength);
    fftResPtrOmp = fftObj.ompIterFFT(timeSignal, signalLength);

    for (int i = 0; i < 100; i++)
    {
        cout << fftResPtr[i] << " " << fftResPtrCuda[i] << " " << fftResPtrOmp[i] << endl;
    }

    complex<double>* dftResPtr;
    complex<double>* dftResPtrCuda;
    complex<double>* dftResPtrOmp;

    // dftResPtr = dftObj.iterDFT(timeSignal, signalLength);
    // dftResPtrCuda = dftObj.cudaIterDFT(timeSignal, signalLength);
    // dftResPtrOmp = dftObj.ompIterDFT(timeSignal, signalLength);

    // for (int i = 0; i < 100; i++)
    // {
    //     cout << dftResPtr[i] << " " << dftResPtrCuda[i] << " " << dftResPtrOmp[i] << endl;
    // }

    // for (int i = 0; i < 100; i++)
    // {
    //     cout << fftResPtr[i] << " " << fftResPtrCuda[i] << " " << fftResPtrOmp[i] <<  " " << dftResPtrCuda[i] << " " << dftResPtrOmp[i] << endl;
    // }
}