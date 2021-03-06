//
// Created by rakshitgl
//

#ifndef MATRIXPRO_MATRIX_OPS_H
#define MATRIXPRO_MATRIX_OPS_H

#include "matrix.h"
#include "setup_parallel.h"

using namespace numcpp;
using namespace parallel;

#define TS 32
#define WPT 8

namespace numcpp {

	int is_broadcast_possible(const Matrix* valid_a, const Matrix* valid_b, long long int* highest_cols,
		long long int* highest_rows) {
		int flag = 0;

		if ((valid_a->get_rows() >= valid_b->get_rows() && valid_a->get_rows() % valid_b->get_rows() == 0
			&& valid_a->get_columns() >= valid_b->get_columns() &&
			valid_a->get_columns() % valid_b->get_columns() == 0)) {

			flag = 2;

			*highest_cols = valid_a->get_columns();
			*highest_rows = valid_a->get_rows();
		}
		else if ((valid_a->get_rows() <= valid_b->get_rows() && valid_b->get_rows() % valid_a->get_rows() == 0
			&& valid_a->get_columns() <= valid_b->get_columns() &&
			valid_b->get_columns() % valid_a->get_columns() == 0)) {

			flag = 1;

			*highest_cols = valid_b->get_columns();
			*highest_rows = valid_b->get_rows();
		}
		else {
			flag = 0;

			*highest_cols = -1;
			*highest_rows = -1;
		}

		return flag;
	}

	int broadcast2(const Matrix* valid_a, const Matrix* valid_b, float* output_a, float* output_b, int flag) {

		if (flag == 0)
			return flag;
		else {

			if (flag == 2) {

				for (long long int i = 0; i < valid_a->get_rows(); i++) {

					for (long long int j = 0; j < valid_a->get_columns(); j++) {

						output_a[i * valid_a->get_columns() + j] = valid_a->get_element(i, j);
					}
				}

				long long int index = 0;

				for (long long int i = 0; i < valid_b->get_rows(); i++) {

					for (long long int j = 0; j < valid_b->get_columns(); j++) {

						output_b[index] = valid_b->get_matrix()[i * valid_b->get_columns() + j];
						index++;
					}

					index -= valid_b->get_columns();
					index += valid_a->get_columns();
				}

				long long int rows_rep = valid_a->get_rows() / valid_b->get_rows();
				long long int jump = valid_b->get_rows() * valid_a->get_columns();

				for (long long int i = 0; i < rows_rep - 1; i++) {

					long long int k = 0;

					for (long long int j = i * jump; j < i * jump + jump; j++) {

						output_b[j + jump] = output_b[j];

						k++;

						if (k >= valid_b->get_columns()) {

							k = 0;
							j -= valid_b->get_columns();
							j += valid_a->get_columns();
						}
					}
				}

				for (long long int i = 0; i < valid_a->get_rows(); i++) {

					long long int curr_position = i * valid_a->get_columns();

					for (long long int j = curr_position;
						j <= curr_position + (valid_a->get_columns() - 1 - valid_b->get_columns()); j++) {

						output_b[j + valid_b->get_columns()] = output_b[j];
					}
				}

			}
			else if (flag == 1) {

				for (long long int i = 0; i < valid_b->get_rows(); i++) {

					for (long long int j = 0; j < valid_b->get_columns(); j++) {

						output_b[i * valid_b->get_columns() + j] = valid_b->get_element(i, j);
					}
				}

				long long int index = 0;

				for (long long int i = 0; i < valid_a->get_rows(); i++) {

					for (long long int j = 0; j < valid_a->get_columns(); j++) {

						output_a[index] = valid_a->get_matrix()[i * valid_a->get_columns() + j];
						index++;
					}

					index -= valid_a->get_columns();
					index += valid_b->get_columns();
				}

				long long int rows_rep = valid_b->get_rows() / valid_a->get_rows();
				long long int jump = valid_a->get_rows() * valid_b->get_columns();

				for (long long int i = 0; i < rows_rep - 1; i++) {

					long long int k = 0;

					for (long long int j = i * jump; j < i * jump + jump; j++) {

						output_a[j + jump] = output_a[j];

						k++;

						if (k >= valid_a->get_columns()) {

							k = 0;
							j -= valid_a->get_columns();
							j += valid_b->get_columns();
						}
					}
				}

				for (long long int i = 0; i < valid_b->get_rows(); i++) {

					long long int curr_position = i * valid_b->get_columns();

					for (long long int j = curr_position;
						j <= curr_position + (valid_b->get_columns() - 1 - valid_a->get_columns()); j++) {

						output_a[j + valid_a->get_columns()] = output_a[j];
					}
				}
			}

			return flag;
		}
	}

	Matrix matmul(Matrix a, Matrix b) {

		cl_int ret;
		auto* output = new float[a.get_rows() * b.get_columns()];

		Matrix result(a.get_rows(), b.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			a.get_rows() * a.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
			b.get_rows() * b.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			a.get_rows() * b.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			a.get_rows() * a.get_columns() * sizeof(float), a.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0,
			b.get_rows() * b.get_columns() * sizeof(float), b.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		int rows = a.get_rows(), cols = b.get_columns(), inter = b.get_rows();
		
		ret = clSetKernelArg(matrix_kernel_multiply, 0, sizeof(int), (void*)&rows);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		
		ret = clSetKernelArg(matrix_kernel_multiply, 1, sizeof(int), (void*)&cols);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		
		ret = clSetKernelArg(matrix_kernel_multiply, 2, sizeof(int), (void*)&inter);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		ret = clSetKernelArg(matrix_kernel_multiply, 3, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(matrix_kernel_multiply, 4, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(matrix_kernel_multiply, 5, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		const size_t local_work_size[2] = { 1, 1 };
		const size_t global_work_size[2] = { a.get_rows(), b.get_columns() };

		ret = clEnqueueNDRangeKernel(queue, matrix_kernel_multiply, 2, nullptr,
			global_work_size, local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			a.get_rows() * b.get_columns() * sizeof(float), output, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output);


		return result;
	}

	Matrix transpose(Matrix a) {

		cl_int ret;
		auto* output = new float[a.get_rows() * a.get_columns()];

		Matrix result(a.get_columns(), a.get_rows());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			a.get_rows() * a.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			a.get_rows() * a.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			a.get_rows() * a.get_columns() * sizeof(float), a.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		int cols = a.get_columns();

		ret = clSetKernelArg(matrix_kernel_transpose, 0, sizeof(int), (void*)&cols);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		ret = clSetKernelArg(matrix_kernel_transpose, 1, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		
		ret = clSetKernelArg(matrix_kernel_transpose, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		const size_t local_work_size[2] = { 1, 1 };
		const size_t global_work_size[2] = { a.get_rows(), a.get_columns() };

		ret = clEnqueueNDRangeKernel(queue, matrix_kernel_transpose, 2, nullptr,
			global_work_size, local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			a.get_rows() * a.get_columns() * sizeof(float), output, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output);

		return result;
	}
}

Matrix operator+(Matrix& first, Matrix const& second) {

	try {

		cl_int ret;

		float* output_a = nullptr, * output_b = nullptr;
		long long int columns_highest = 0, rows_highest = 0;

		int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}
		else {

			output_a = new float[rows_highest * columns_highest];
			output_b = new float[rows_highest * columns_highest];
		}

		flag = broadcast2(&first, &second, output_a, output_b, flag);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}

		Matrix result(rows_highest, columns_highest);

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_a, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_b, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(kernel_add, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_add, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_add, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = rows_highest * columns_highest;
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, kernel_add, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[rows_highest * columns_highest];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		delete[] output_a;
		delete[] output_b;

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator-(Matrix& first, Matrix const& second) {

	try {

		cl_int ret;

		float* output_a = nullptr, * output_b = nullptr;
		long long int columns_highest = 0, rows_highest = 0;

		int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}
		else {

			output_a = new float[rows_highest * columns_highest];
			output_b = new float[rows_highest * columns_highest];
		}

		flag = broadcast2(&first, &second, output_a, output_b, flag);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}

		Matrix result(rows_highest, columns_highest);

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_a, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_b, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(kernel_subtract, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_subtract, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_subtract, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = rows_highest * columns_highest;
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, kernel_subtract, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[rows_highest * columns_highest];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		delete[] output_a;
		delete[] output_b;

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator*(Matrix& first, Matrix const& second) {

	try {

		cl_int ret;

		float* output_a = nullptr, * output_b = nullptr;
		long long int columns_highest = 0, rows_highest = 0;

		int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}
		else {

			output_a = new float[rows_highest * columns_highest];
			output_b = new float[rows_highest * columns_highest];
		}

		flag = broadcast2(&first, &second, output_a, output_b, flag);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}

		Matrix result(rows_highest, columns_highest);

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_a, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_b, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(kernel_multiply, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_multiply, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_multiply, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = rows_highest * columns_highest;
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, kernel_multiply, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[rows_highest * columns_highest];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		delete[] output_a;
		delete[] output_b;

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator>(Matrix& first, Matrix const& second) {

	try {

		cl_int ret;

		float* output_a = nullptr, * output_b = nullptr;
		long long int columns_highest = 0, rows_highest = 0;

		int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}
		else {

			output_a = new float[rows_highest * columns_highest];
			output_b = new float[rows_highest * columns_highest];
		}

		flag = broadcast2(&first, &second, output_a, output_b, flag);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}

		Matrix result(rows_highest, columns_highest);

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_a, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_b, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(kernel_gt, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_gt, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_gt, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = rows_highest * columns_highest;
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, kernel_gt, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[rows_highest * columns_highest];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		delete[] output_a;
		delete[] output_b;

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator<(Matrix& first, Matrix const& second) {

	try {

		cl_int ret;

		float* output_a = nullptr, * output_b = nullptr;
		long long int columns_highest = 0, rows_highest = 0;

		int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}
		else {

			output_a = new float[rows_highest * columns_highest];
			output_b = new float[rows_highest * columns_highest];
		}

		flag = broadcast2(&first, &second, output_a, output_b, flag);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}

		Matrix result(rows_highest, columns_highest);

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_a, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_b, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(kernel_lt, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_lt, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_lt, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = rows_highest * columns_highest;
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, kernel_lt, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[rows_highest * columns_highest];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		delete[] output_a;
		delete[] output_b;

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator==(Matrix& first, Matrix const& second) {

	try {

		cl_int ret;

		float* output_a = nullptr, * output_b = nullptr;
		long long int columns_highest = 0, rows_highest = 0;

		int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}
		else {

			output_a = new float[rows_highest * columns_highest];
			output_b = new float[rows_highest * columns_highest];
		}

		flag = broadcast2(&first, &second, output_a, output_b, flag);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}

		Matrix result(rows_highest, columns_highest);

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_a, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_b, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(kernel_equals, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_equals, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_equals, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = rows_highest * columns_highest;
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, kernel_equals, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[rows_highest * columns_highest];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		delete[] output_a;
		delete[] output_b;

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator>=(Matrix& first, Matrix const& second) {

	try {

		cl_int ret;

		float* output_a = nullptr, * output_b = nullptr;
		long long int columns_highest = 0, rows_highest = 0;

		int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}
		else {

			output_a = new float[rows_highest * columns_highest];
			output_b = new float[rows_highest * columns_highest];
		}

		flag = broadcast2(&first, &second, output_a, output_b, flag);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}

		Matrix result(rows_highest, columns_highest);

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_a, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_b, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(kernel_gte, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_gte, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_gte, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = rows_highest * columns_highest;
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, kernel_gte, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[rows_highest * columns_highest];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		delete[] output_a;
		delete[] output_b;

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator<=(Matrix& first, Matrix const& second) {

	try {

		cl_int ret;

		float* output_a = nullptr, * output_b = nullptr;
		long long int columns_highest = 0, rows_highest = 0;

		int flag = is_broadcast_possible(&first, &second, &columns_highest, &rows_highest);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}
		else {

			output_a = new float[rows_highest * columns_highest];
			output_b = new float[rows_highest * columns_highest];
		}

		flag = broadcast2(&first, &second, output_a, output_b, flag);

		if (flag == 0) {

			throw MatrixStatus("Matrix Dimensions are unmatchable and could not be broad-casted.", 10);
		}

		Matrix result(rows_highest, columns_highest);

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			rows_highest * columns_highest * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_a, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_b, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(kernel_lte, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_lte, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(kernel_lte, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = rows_highest * columns_highest;
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, kernel_lte, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[rows_highest * columns_highest];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			rows_highest * columns_highest * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		delete[] output_a;
		delete[] output_b;

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

//Scalar Operations

Matrix operator*(Matrix& first, float const& second) {

	try {

		cl_int ret;

		Matrix result(first.get_rows(), first.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			first.get_rows() * first.get_columns() * sizeof(float), first.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0, sizeof(float), &second, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(scalar_kernel_multiply, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_multiply, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_multiply, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = first.get_columns() * first.get_rows();
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, scalar_kernel_multiply, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[first.get_columns() * first.get_rows()];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator>(Matrix& first, float const& second) {

	try {

		cl_int ret;

		Matrix result(first.get_rows(), first.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			first.get_rows() * first.get_columns() * sizeof(float), first.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0, sizeof(float), &second, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(scalar_kernel_gt, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_gt, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_gt, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = first.get_columns() * first.get_rows();
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, scalar_kernel_gt, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[first.get_columns() * first.get_rows()];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}


Matrix operator<(Matrix& first, float const& second) {

	try {

		cl_int ret;

		Matrix result(first.get_rows(), first.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			first.get_rows() * first.get_columns() * sizeof(float), first.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0, sizeof(float), &second, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(scalar_kernel_lt, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_lt, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_lt, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = first.get_columns() * first.get_rows();
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, scalar_kernel_lt, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[first.get_columns() * first.get_rows()];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator==(Matrix& first, float const& second) {

	try {

		cl_int ret;

		Matrix result(first.get_rows(), first.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			first.get_rows() * first.get_columns() * sizeof(float), first.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0, sizeof(float), &second, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(scalar_kernel_equals, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_equals, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_equals, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = first.get_columns() * first.get_rows();
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, scalar_kernel_equals, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[first.get_columns() * first.get_rows()];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator>=(Matrix& first, float const& second) {

	try {

		cl_int ret;

		Matrix result(first.get_rows(), first.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			first.get_rows() * first.get_columns() * sizeof(float), first.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0, sizeof(float), &second, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(scalar_kernel_gte, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_gte, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_gte, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = first.get_columns() * first.get_rows();
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, scalar_kernel_gte, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[first.get_columns() * first.get_rows()];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator<=(Matrix& first, float const& second) {

	try {

		cl_int ret;

		Matrix result(first.get_rows(), first.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			first.get_rows() * first.get_columns() * sizeof(float), first.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0, sizeof(float), &second, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(scalar_kernel_lte, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_lte, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_lte, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = first.get_columns() * first.get_rows();
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, scalar_kernel_lte, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[first.get_columns() * first.get_rows()];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator^(Matrix& first, float const& second) {

	try {

		cl_int ret;

		Matrix result(first.get_rows(), first.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			first.get_rows() * first.get_columns() * sizeof(float), first.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0, sizeof(float), &second, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(scalar_kernel_power, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_power, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_power, 2, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = first.get_columns() * first.get_rows();
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, scalar_kernel_power, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[first.get_columns() * first.get_rows()];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator+(Matrix& first, float const& second) {

	try {

		cl_int ret;
		float num_columns = first.get_columns();

		Matrix result(first.get_rows(), first.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_c = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			first.get_rows() * first.get_columns() * sizeof(float), first.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0, sizeof(float), &second, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_c, CL_TRUE, 0, sizeof(float), &num_columns, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(scalar_kernel_adder, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_adder, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_adder, 2, sizeof(cl_mem), (void*)&memory_input_c);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_adder, 3, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = first.get_columns() * first.get_rows();
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, scalar_kernel_adder, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[first.get_columns() * first.get_rows()];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

Matrix operator-(Matrix& first, float const& second) {

	try {

		cl_int ret;
		float num_columns = first.get_columns();

		Matrix result(first.get_rows(), first.get_columns());

		cl_mem memory_input_a = clCreateBuffer(context, CL_MEM_READ_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_input_c = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		cl_mem memory_output_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			first.get_rows() * first.get_columns() * sizeof(float), nullptr, &ret);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer could not be created.", 92);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_a, CL_TRUE, 0,
			first.get_rows() * first.get_columns() * sizeof(float), first.get_matrix(), 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_b, CL_TRUE, 0, sizeof(float), &second, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clEnqueueWriteBuffer(queue, memory_input_c, CL_TRUE, 0, sizeof(float), &num_columns, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Memory buffer value could not be set.", 93);
		}

		ret = clSetKernelArg(scalar_kernel_subtracter, 0, sizeof(cl_mem), (void*)&memory_input_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_subtracter, 1, sizeof(cl_mem), (void*)&memory_input_b);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_subtracter, 2, sizeof(cl_mem), (void*)&memory_input_c);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}
		ret = clSetKernelArg(scalar_kernel_subtracter, 3, sizeof(cl_mem), (void*)&memory_output_a);

		if (ret != 0) {

			throw MatrixStatus("Kernel arguments could not be set.", 94);
		}

		size_t global_work_size = first.get_columns() * first.get_rows();
		size_t local_work_size = 1;

		ret = clEnqueueNDRangeKernel(queue, scalar_kernel_subtracter, 1, nullptr,
			&global_work_size, &local_work_size, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error launching kernel.", 95);
		}

		ret = clFinish(queue);

		if (ret != 0) {

			throw MatrixStatus("Error synchronizing kernel tasks.", 96);
		}

		auto* output_final = new float[first.get_columns() * first.get_rows()];
		ret = clEnqueueReadBuffer(queue, memory_output_a, CL_TRUE, 0,
			first.get_columns() * first.get_rows() * sizeof(float), output_final, 0, nullptr, nullptr);

		if (ret != 0) {

			throw MatrixStatus("Error reading output from kernel.", 97);
		}

		cl_int retd = clReleaseMemObject(memory_input_a);
		cl_int rete = clReleaseMemObject(memory_input_b);
		cl_int retf = clReleaseMemObject(memory_output_a);

		if (retd != 0 || rete != 0 || retf != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}

		result.set_matrix(output_final);

		return result;
	}
	catch (MatrixStatus & status) {

		std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
			<< "Aborting..." << std::endl;
		exit(0);
	}
}

#endif //MATRIXPRO_MATRIX_OPS_H
