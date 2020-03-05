//
// Created by rakshitgl
//

// This file demoes few functions supported by the NumCPP library


// Supported Element - wise operations : [+, -, *, <, <= , >, >= , == , ^]
// Supported Matrix on Scalar operations : [+, -, *, <, <= , >, >= , == , ^]
// The matrices support broadcasting (similar to NumPy).

// Requires OpenCL Library and supported platforms. 

#include "numcpp.h"

using namespace numcpp;
int main() {

    // All matrix functions can be called after calling init_parallel()
    parallel::init_parallel();

    Matrix matA(20, 20);
    MatrixStatus status = matA.initialize_matrix(true, 10);

    // Matrix objects can be directly output using cout.

    std::cout << "\nMatrix A" << std::endl;
    std::cout << matA;

    Matrix matB(1, 1);
    status = matB.initialize_matrix(true, 10);

    std::cout << "\nMatrix B" << std::endl;
    std::cout << matB;

    Matrix temp = matA <= matB;
    Matrix tempA = matA > matB;

    Matrix matC = tempA + temp;
    matC = matC - 1;

    std::cout << "\nResult" << std::endl;
    std::cout << matC;

    matA.cleap_up();
    matB.cleap_up();
    matC.cleap_up();

    parallel::finish_parallel();
    // Program should end with finish_parallel() and [clean_up() for each matrix]
    // (to clean up allocated memories) 

    return 0;
}