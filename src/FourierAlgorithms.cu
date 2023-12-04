#include "FourierAlgorithms.h"

using namespace std;

/*
Description: times the fourier algorithm chosen in microseconds
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
    algo - method pointer to the algorithm to be timed
Return:
    result - complex double array pointer that is the result of the fourier algorithm
*/
complex<double>* FourierAlgorithms::timeAlgorithm(double *timeSignal, unsigned long sigLength, algoMethod algo)
{
    // Allocate space for the fourier result
    static complex<double>* result = (complex<double>*) malloc(sigLength * sizeof(complex<double>));
    
    // Start time
    auto start = chrono::high_resolution_clock::now();

    // Run specified algorithm
    result = (this->*algo)(timeSignal, sigLength);

    // Stop time
    auto stop = chrono::high_resolution_clock::now();

    auto diff = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << "algorithm time (us): " << diff.count() << endl;

    return result;
}

/*
Description: saves the fourier algorithm result on a time signal
Parameters:
    fileName - name of the file to save (must be .csv)
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
    algo - method pointer to the algorithm to be timed
Return:
    result - complex double array pointer that is the result of the fourier algorithm
*/
complex<double>* FourierAlgorithms::saveAlgorithmResult(string fileName, double *timeSignal, unsigned long sigLength, algoMethod algo)
{
    // Allocate space for the fourier result
    static complex<double>* result = (complex<double>*) malloc(sigLength * sizeof(complex<double>));

    // Check to ensure last 4 characters are .csv
    if (fileName.substr(fileName.length() - 4) != ".csv")
    {
        cout << "Filename must be a .csv file" << endl;
        return result;
    }

    result = (this->*algo)(timeSignal, sigLength);
    
    // Save the result to a file
    ofstream outputFile(fileName);
    if (outputFile.is_open()) 
    {
        // Iterate over every value in the result array
        for (int resIdx = 0; resIdx < sigLength; resIdx++)
        {
            // Result value
            outputFile << result[resIdx];
            // Comma for CSV
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

/*
Description: gets the length of the array after padding the time signal array with zeros
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
    numZeros -number of zeros to pad to the time signal
Return:
    result - length of the zero padded time signal array
*/
unsigned long FourierAlgorithms::zeroPadLength(double *timeSignal, unsigned long sigLength, unsigned long numZeros)
{
    return sigLength + numZeros;
}

/*
Description: pads the time signal array with zeros
Parameters:
    timeSignal - pointer array to the time signal to be analyzed
    sigLength - length of the time signal pointer array
    numZeros -number of zeros to pad to the time signal
Return:
    result - pointer to the zero padded time signal array
*/
double* FourierAlgorithms::zeroPadArray(double *timeSignal, unsigned long sigLength, unsigned long numZeros)
{
    // Allocate space for the zero padded time signal
    static double* paddedTimeSignal = (double*) malloc((sigLength + numZeros) * sizeof(double));;

    for (int k = 0; k < sigLength + numZeros; k++)
    {
        // Copy the current time signal
        if (k < sigLength)
        {
            paddedTimeSignal[k] = timeSignal[k];
        }
        // Add zeros
        else
        {
            paddedTimeSignal[k] = 0;
        }
    }

    return paddedTimeSignal;
}