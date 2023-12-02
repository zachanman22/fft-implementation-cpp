#include "fourierAlgorithms.h"

using namespace std;

// complex<double>* fourierAlgorithms::timeAlgorithm(double *timeSignal, unsigned long sigLength, function<complex<double>*(double*, unsigned long)> algo)
complex<double>* fourierAlgorithms::timeAlgorithm(double *timeSignal, unsigned long sigLength, algoMethod algo)
{
    static complex<double>* result = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    auto start = chrono::high_resolution_clock::now();

    // result = algo(timeSignal, sigLength);
    result = (this->*algo)(timeSignal, sigLength);

    auto stop = chrono::high_resolution_clock::now();

    auto diff = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "algorithm time (us): " << diff.count() << endl;

    return result;
}

unsigned long fourierAlgorithms::zeroPadLength(double *timeSignal, unsigned long sigLength, unsigned long numZeros)
{
    return sigLength + numZeros;
}

double* fourierAlgorithms::zeroPadArray(double *timeSignal, unsigned long sigLength, unsigned long numZeros)
{

    static double* paddedTimeSignal = (double*) malloc((sigLength + numZeros) * sizeof(double));;

    for (int k = 0; k < sigLength + numZeros; k++)
    {
        if (k < sigLength)
        {
            paddedTimeSignal[k] = timeSignal[k];
        }
        else
        {
            paddedTimeSignal[k] = 0;
        }
    }
    return paddedTimeSignal;
}