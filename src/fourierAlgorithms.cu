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

complex<double>* fourierAlgorithms::saveAlgorithmResult(string fileName, double *timeSignal, unsigned long sigLength, algoMethod algo)
{
    static complex<double>* result = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    result = (this->*algo)(timeSignal, sigLength);
    ofstream outputFile(fileName);
    if (outputFile.is_open()) 
    {
        for (int resIdx = 0; resIdx < sigLength; resIdx++)
        {
            outputFile << result[resIdx];
            if (resIdx < sigLength - 1)
            {
                outputFile << ",";
            }
        }
        outputFile.close();
        cout << "Result written to " << fileName << endl;
    }
    else
    {
        cout << "Error opening " << fileName << endl; 
    }

    return result;
}

complex<double>* fourierAlgorithms::plotAlgorithmResult(double samplingFreq, char *plotTitle, double *timeSignal, unsigned long sigLength, algoMethod algo)
{
    static complex<double>* result = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    result = (this->*algo)(timeSignal, sigLength);

    double* resultAmp = (double*) malloc(sigLength * sizeof(double));

    for (int resIdx = 0; resIdx < sigLength; resIdx++)
    {
        resultAmp[resIdx] = abs(result[resIdx]);
    }
    
    double* fftfreq = (double*) malloc(sigLength * sizeof(double));

    unsigned long midIdx;

    if (sigLength % 2 == 0)
    {
        midIdx = sigLength / 2;
    }
    else
    {
        midIdx = (sigLength - 1) / 2 + 1;
    }

    for (int freqIdx = 0; freqIdx < midIdx; freqIdx++)
    {
        fftfreq[freqIdx] = freqIdx * samplingFreq / sigLength;
    }

    for (int freqIdx = midIdx; freqIdx < sigLength; freqIdx++)
    {
        fftfreq[freqIdx] = -(sigLength - freqIdx) * samplingFreq / sigLength;
    }
   

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