//
// Created by rakshitgl
//

#ifndef MATRIXPRO_SETUP_PARALLEL_H
#define MATRIXPRO_SETUP_PARALLEL_H

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include <CL/cl2.hpp>
#include <fstream>
#include "matrix.h"

#define DEVICE CL_DEVICE_TYPE_DEFAULT

using namespace numcpp;

namespace parallel {

	std::string readFile(const std::string& file_name);

	cl_platform_id platformId;
	cl_device_id deviceId;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_context context;
	cl_command_queue queue;
	cl_program program;

	cl_kernel kernel_add;
	cl_kernel kernel_subtract;
	cl_kernel kernel_multiply;
	cl_kernel kernel_gt;
	cl_kernel kernel_lt;
	cl_kernel kernel_equals;
	cl_kernel kernel_gte;
	cl_kernel kernel_lte;

	cl_kernel scalar_kernel_multiply;
	cl_kernel scalar_kernel_gt;
	cl_kernel scalar_kernel_lt;
	cl_kernel scalar_kernel_equals;
	cl_kernel scalar_kernel_gte;
	cl_kernel scalar_kernel_lte;
	cl_kernel scalar_kernel_power;
	cl_kernel scalar_kernel_adder;
	cl_kernel scalar_kernel_subtracter;

	cl_kernel matrix_kernel_multiply;

	void init_parallel() {

		try {
			cl_int retP, retD, retC, retQ, ret;

			retP = clGetPlatformIDs(1, &platformId, &ret_num_platforms);
			retD = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &ret_num_devices);
			context = clCreateContext(nullptr, 1, &deviceId, nullptr, nullptr, &retC);
			queue = clCreateCommandQueueWithProperties(context, deviceId, nullptr, &retQ);

			FILE* fp;
			char* source_str;
			size_t source_size;

			int err_no = fopen_s(&fp, "kernels/kernel.cl", "r");

			if (!fp || (err_no != 0)) {

				throw MatrixStatus("Error reading kernel.", 90);
			}
			if ((retC != 0) || (retP != 0) || (retQ != 0) || (retD != 0)) {

				throw MatrixStatus("Error detecting OpenCL supported platform.", 91);
			}

			source_str = (char*)malloc(5000);
			source_size = fread(source_str, 1, 5000, fp);

			fclose(fp);

			program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program from source.", 99);
			}

			ret = clBuildProgram(program, 1, &deviceId, nullptr, nullptr, nullptr);

			if (ret != 0) {

				throw MatrixStatus("Error building kernel program.", 100);
			}

			kernel_add = clCreateKernel(program, "parallel_adder", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Adder)", 101);
			}

			kernel_subtract = clCreateKernel(program, "parallel_subtracter", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Subtracter)", 101);
			}

			kernel_multiply = clCreateKernel(program, "parallel_multiplier", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Multiplier)", 101);
			}

			kernel_gt = clCreateKernel(program, "parallel_gt", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Greater Than [gt])", 101);
			}

			kernel_lt = clCreateKernel(program, "parallel_lt", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Less Than [lt])", 101);
			}

			kernel_equals = clCreateKernel(program, "parallel_equals", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Is Equal To [equals])", 101);
			}

			kernel_gte = clCreateKernel(program, "parallel_gte", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Greater Than or Equal To [gte])", 101);
			}

			kernel_lte = clCreateKernel(program, "parallel_lte", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Less Than or Equal To [lte])", 101);
			}

			scalar_kernel_multiply = clCreateKernel(program, "scalar_parallel_multiplier", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Scalar Multiplier)", 101);
			}

			scalar_kernel_gt = clCreateKernel(program, "scalar_parallel_gt", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Scalar Greater Than)", 101);
			}

			scalar_kernel_lt = clCreateKernel(program, "scalar_parallel_lt", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Scalar Less Than)", 101);
			}

			scalar_kernel_equals = clCreateKernel(program, "scalar_parallel_equals", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Scalar Is Equal To)", 101);
			}

			scalar_kernel_gte = clCreateKernel(program, "scalar_parallel_gte", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Scalar Greater Than or Equal To)", 101);
			}

			scalar_kernel_lte = clCreateKernel(program, "scalar_parallel_lte", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Scalar Less Than or Equal To)", 101);
			}

			scalar_kernel_power = clCreateKernel(program, "scalar_parallel_power", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Scalar Power)", 101);
			}

			scalar_kernel_adder = clCreateKernel(program, "scalar_parallel_adder", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Scalar Adder)", 101);
			}

			scalar_kernel_subtracter = clCreateKernel(program, "scalar_parallel_subtracter", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Scalar Subtracter)", 101);
			}

			matrix_kernel_multiply = clCreateKernel(program, "parallel_matrix_multiply", &ret);

			if (ret != 0) {

				throw MatrixStatus("Error creating kernel program. (Matrix Multiplier)", 101);
			}

		}
		catch (MatrixStatus status) {

			std::cout << std::endl << status.get_error_code() << ": " << status.get_error_message() << std::endl
				<< "Aborting..." << std::endl;
			exit(0);
		}
	}

	std::string readFile(const std::string& file_name) {

		std::ifstream stream(file_name.c_str());

		if (!stream.is_open()) {
			throw std::invalid_argument("Kernel not Found. Error due to: " + file_name);
		}

		return std::string(
			std::istreambuf_iterator<char>(stream),
			(std::istreambuf_iterator<char>()));
	}

	void finish_parallel() {
		cl_int reta = clFlush(queue);
		cl_int retb = clReleaseKernel(kernel_add);
		cl_int retc = clReleaseProgram(program);
		cl_int retg = clReleaseCommandQueue(queue);
		cl_int reth = clReleaseContext(context);

		if (reta != 0 || retb != 0 || retc != 0 || retg != 0 || reth != 0) {

			std::cerr << "98: WARNING: Error clearing kernel space. Memory leaks may happen.\n";
		}
	}
}


#endif //MATRIXPRO_SETUP_PARALLEL_H