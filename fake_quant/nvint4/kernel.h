// kernel.h
#pragma once
#include <cuda_runtime.h>

void launch_absmax(const float*, float*, int, int, cudaStream_t);
void launch_fq_col(const float*, const float*, float*, float*, int, int, cudaStream_t);
void launch_fq_row(const float*, const float*, float*, float*, int, int, cudaStream_t);
void launch_matmul(const float*, const float*, float*, int, int, int, cudaStream_t);