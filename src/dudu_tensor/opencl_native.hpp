#pragma once

#include <CL/cl.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "dudu_tensor/index_native.hpp"

namespace dd_opencl {

struct Runtime {
    cl_device_id device{};
    cl_context context{};
    cl_command_queue queue{};
};

struct DeviceTensor {
    cl_mem buffer{};
    int rows{};
    int cols{};
};

inline void check(cl_int err, const char* where) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(where) + " failed: " + std::to_string(err));
    }
}

inline Runtime& runtime() {
    static Runtime rt = [] {
        cl_platform_id platform{};
        cl_uint platform_count{};
        check(clGetPlatformIDs(1, &platform, &platform_count), "clGetPlatformIDs");
        if (platform_count == 0) {
            throw std::runtime_error("OpenCL has no platforms");
        }

        cl_device_id device{};
        check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr),
              "clGetDeviceIDs");

        cl_int err{};
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        check(err, "clCreateContext");

        cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            check(err, "clCreateCommandQueueWithProperties");
        }

        return Runtime{device, context, queue};
    }();
    return rt;
}

inline DeviceTensor* make_empty(int rows, int cols) {
    Runtime& rt = runtime();
    cl_int err{};
    const std::size_t bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) *
                              sizeof(float);
    cl_mem buffer = clCreateBuffer(rt.context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    check(err, "clCreateBuffer");

    return new DeviceTensor{buffer, rows, cols};
}

inline DeviceTensor* upload(int rows, int cols, const float* data) {
    DeviceTensor* out = make_empty(rows, cols);
    Runtime& rt = runtime();
    const std::size_t bytes = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) *
                              sizeof(float);
    cl_int err = clEnqueueWriteBuffer(rt.queue, out->buffer, CL_TRUE, 0, bytes, data, 0, nullptr,
                                      nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clEnqueueWriteBuffer failed: " + std::to_string(err));
    }
    return out;
}

inline cl_program build_program(const char* source) {
    Runtime& rt = runtime();
    cl_int err{};
    cl_program program = clCreateProgramWithSource(rt.context, 1, &source, nullptr, &err);
    check(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        check(err, "clBuildProgram");
    }
    return program;
}

inline DeviceTensor* matmul(DeviceTensor* left, DeviceTensor* right) {
    if (left->cols != right->rows) {
        throw std::runtime_error("OpenCL matmul shape mismatch");
    }

    DeviceTensor* out = make_empty(left->rows, right->cols);
    const char* source =
        "__kernel void matmul(const int m, const int n, const int k, "
        "__global const float* a, __global const float* b, __global float* out) { "
        "int row = get_global_id(0); int col = get_global_id(1); "
        "if (row >= m || col >= n) return; "
        "float acc = 0.0f; "
        "for (int i = 0; i < k; ++i) { acc += a[row * k + i] * b[i * n + col]; } "
        "out[row * n + col] = acc; }";
    cl_program program = build_program(source);

    cl_int err{};
    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    check(err, "clCreateKernel");

    const int m = left->rows;
    const int n = right->cols;
    const int k = left->cols;
    check(clSetKernelArg(kernel, 0, sizeof(int), &m), "clSetKernelArg m");
    check(clSetKernelArg(kernel, 1, sizeof(int), &n), "clSetKernelArg n");
    check(clSetKernelArg(kernel, 2, sizeof(int), &k), "clSetKernelArg k");
    check(clSetKernelArg(kernel, 3, sizeof(cl_mem), &left->buffer), "clSetKernelArg a");
    check(clSetKernelArg(kernel, 4, sizeof(cl_mem), &right->buffer), "clSetKernelArg b");
    check(clSetKernelArg(kernel, 5, sizeof(cl_mem), &out->buffer), "clSetKernelArg out");

    const std::size_t global[2] = {static_cast<std::size_t>(out->rows),
                                   static_cast<std::size_t>(out->cols)};
    check(clEnqueueNDRangeKernel(runtime().queue, kernel, 2, nullptr, global, nullptr, 0, nullptr,
                                 nullptr),
          "clEnqueueNDRangeKernel");
    check(clFinish(runtime().queue), "clFinish");

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return out;
}

inline DeviceTensor* gather_view(DeviceTensor* input, const ddtensor::IndexPlan& plan) {
    if (plan.shape.empty()) {
        throw std::runtime_error("OpenCL scalar indexing is not a tensor view");
    }
    if (plan.shape.size() > 2) {
        throw std::runtime_error("OpenCL target demo supports rank-1/rank-2 gathered views");
    }

    const int out_rows = plan.shape.size() == 1 ? 1 : plan.shape[0];
    const int out_cols = plan.shape.size() == 1 ? plan.shape[0] : plan.shape[1];
    DeviceTensor* out = make_empty(out_rows, out_cols);

    const int rank = static_cast<int>(plan.shape.size());
    const int shape0 = rank >= 1 ? plan.shape[0] : 1;
    const int shape1 = rank >= 2 ? plan.shape[1] : 1;
    const int stride0 = rank >= 1 ? plan.strides[0] : 0;
    const int stride1 = rank >= 2 ? plan.strides[1] : 0;
    const int total = out_rows * out_cols;

    const char* source =
        "__kernel void gather_view(const int rank, const int shape0, const int shape1, "
        "const int stride0, const int stride1, const int offset, const int total, "
        "__global const float* input, __global float* out) { "
        "int flat = get_global_id(0); if (flat >= total) return; "
        "int source_index = offset; "
        "if (rank == 1) { source_index += flat * stride0; } "
        "else { int row = flat / shape1; int col = flat - row * shape1; "
        "source_index += row * stride0 + col * stride1; } "
        "out[flat] = input[source_index]; }";
    cl_program program = build_program(source);

    cl_int err{};
    cl_kernel kernel = clCreateKernel(program, "gather_view", &err);
    check(err, "clCreateKernel");

    check(clSetKernelArg(kernel, 0, sizeof(int), &rank), "clSetKernelArg rank");
    check(clSetKernelArg(kernel, 1, sizeof(int), &shape0), "clSetKernelArg shape0");
    check(clSetKernelArg(kernel, 2, sizeof(int), &shape1), "clSetKernelArg shape1");
    check(clSetKernelArg(kernel, 3, sizeof(int), &stride0), "clSetKernelArg stride0");
    check(clSetKernelArg(kernel, 4, sizeof(int), &stride1), "clSetKernelArg stride1");
    check(clSetKernelArg(kernel, 5, sizeof(int), &plan.offset), "clSetKernelArg offset");
    check(clSetKernelArg(kernel, 6, sizeof(int), &total), "clSetKernelArg total");
    check(clSetKernelArg(kernel, 7, sizeof(cl_mem), &input->buffer), "clSetKernelArg input");
    check(clSetKernelArg(kernel, 8, sizeof(cl_mem), &out->buffer), "clSetKernelArg out");

    const std::size_t global[1] = {static_cast<std::size_t>(total)};
    check(clEnqueueNDRangeKernel(runtime().queue, kernel, 1, nullptr, global, nullptr, 0, nullptr,
                                 nullptr),
          "clEnqueueNDRangeKernel");
    check(clFinish(runtime().queue), "clFinish");

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return out;
}

inline void download(DeviceTensor* tensor, std::vector<float>& out) {
    out.resize(static_cast<std::size_t>(tensor->rows) * static_cast<std::size_t>(tensor->cols));
    const std::size_t bytes = out.size() * sizeof(float);
    check(clEnqueueReadBuffer(runtime().queue, tensor->buffer, CL_TRUE, 0, bytes, out.data(), 0,
                              nullptr, nullptr),
          "clEnqueueReadBuffer");
}

inline void release(DeviceTensor* tensor) {
    if (tensor == nullptr) {
        return;
    }
    if (tensor->buffer != nullptr) {
        clReleaseMemObject(tensor->buffer);
    }
    delete tensor;
}

} // namespace dd_opencl
