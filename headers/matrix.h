//
// Created by rakshitgl
//

#ifndef MATRIXPRO_MATRIX_H
#define MATRIXPRO_MATRIX_H

#include <cstring>
#include <iostream>
#include <cmath>

namespace numcpp {

    class Matrix;

    int broadcast2(const Matrix* valid_a, const Matrix* valid_b, float* output_a, float* output_b, int flag);

    int is_broadcast_possible(const Matrix* valid_a, const Matrix* valid_b, long long int* highest_cols,
        long long int* highest_rows);

    /**
     * MatrixStatus is the class that holds the status of every operation.
     * Success/Failure message and their respective codes are enclosed in this class.
     */
    class MatrixStatus {

    private:

        //Error message for user readability [Scope for improvement]
        std::string error_message;

        //Error code for cleaner relationships between different errors
        int error_code;

    public:

        MatrixStatus(std::string error, int code);

        std::string get_error_message() {
            return this->error_message;
        }

        int get_error_code() {
            return this->error_code;
        }
    };

    MatrixStatus::MatrixStatus(std::string error, int code) {

        this->error_message = std::move(error);
        this->error_code = code;
    }

    /**
     * Matrix class allows all matrix operations to be performed on its objects.
     * It only supports float type matrices.
     *
     * Supported Element-wise operations: [+, -, *, <, <=, >, >=, ==, ^]
     * Supported Matrix on Scalar operations: [+, -, *, <, <=, >, >=, ==, ^]
     */
    class Matrix {

    private:

        //Count of the no. of columns.
        long long int columns;

        //Count of the no. of rows.
        long long int rows;

        //The matrix itself, flattened to 1D array to reduce computational complexity.
        float* matrix;

    public:

        Matrix(long long int rows, long long int columns);

        //Initialize a matrix (either random values, or user defined values)
        MatrixStatus initialize_matrix(bool random, int limit);

        //Respective getters and setters
        float get_element(long long int row, long long int column) const {
            return this->matrix[row * (this->columns) + column];
        }

        void set_element(long long int row, long long int column, float value) {
            this->matrix[row * (this->columns) + column] = value;
        }

        float* get_matrix() const {
            return this->matrix;
        }

        long long int get_rows() const {

            return this->rows;
        }

        long long int get_columns() const {

            return this->columns;
        }

        void set_matrix(float* mat) {
            this->matrix = mat;
        }

        void cleap_up() {

            delete[] get_matrix();
        }
    };

    Matrix::Matrix(long long int rows, long long int columns) {

        this->rows = rows;
        this->columns = columns;
    }

    MatrixStatus Matrix::initialize_matrix(bool random = true, int limit = 10000) {

        if (random) {

            try {
                set_matrix(new float[get_rows() * get_columns()]);

                for (long long int i = 0; i < rows; i++) {

                    for (long long int j = 0; j < columns; j++) {

                        set_element(i, j, rand() % limit);
                    }
                }

                return MatrixStatus("Success", 0);
            }
            catch (errno_t e) {

                return MatrixStatus("Error Reading Values. Only float Values Supported.", 1);
            }
        }
        else {

            try {
                set_matrix(new float[get_rows() * get_columns()]);

                for (long long int i = 0; i < rows; i++) {

                    for (long long int j = 0; j < columns; j++) {

                        float element;
                        std::cin >> element;

                        set_element(i, j, element);

                        std::cout << '\t';
                    }
                }

                return MatrixStatus("Success", 0);
            }
            catch (errno_t e) {

                return MatrixStatus("Error Reading Values. Only float Values Supported.", 1);
            }
        }
    }

    std::ostream& operator<<(std::ostream& os, Matrix const& v) {
        for (long long int i = 0; i < v.get_rows(); i++) {
            for (long long int j = 0; j < v.get_columns(); j++) {
                os << v.get_element(i, j) << "\t";
            }
            os << "\n";
        }
        return os;
    }
}

#endif //MATRIXPRO_MATRIX_H
#pragma once
