#include "fft.h"

int main()
{
    fft fftObj = fft::fft();

    unsigned long signalLength = 262144 / 2;

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
}