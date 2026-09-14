#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp8.h>

#include "kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DATA_TYPE float

// ---------------------------------------------------------------
// Quantization configuration
//   group is taken along K (the contiguous / input-channel direction)
// ---------------------------------------------------------------
#define GROUP_SIZE      16
#define QMAX            7                               // symmetric int4 : [-7, +7]
#define E4M3_MAX        448.0f

// ---------------------------------------------------------------
// Thread Layout : MM
// ---------------------------------------------------------------
#define BLOCK_SIZE 16
#define TILING_SIZE 16

// ---------------------------------------------------------------
// Thread Layout : COL Dir
// ---------------------------------------------------------------
#define BLOCK_SIZE_X    64                              // must be a multiple of GROUP_SIZE
#define BLOCK_SIZE_Y    4
#define GROUP_PER_BLK_X (BLOCK_SIZE_X / GROUP_SIZE)

// ---------------------------------------------------------------
// Thread Layout : ROW Dir
// ---------------------------------------------------------------
#define BLOCK_SIZE_AX    4                              // must be a multiple of GROUP_SIZE
#define BLOCK_SIZE_AY    64
#define GROUP_PER_BLK_Y (BLOCK_SIZE_AY / GROUP_SIZE)

__device__ float atomicMaxFloat(float* addr, float val) {
    return __int_as_float(atomicMax((int*)addr, __float_as_int(val)));
}

__global__ void TensorAbsMax(DATA_TYPE *matIn, DATA_TYPE *tMax, int m, int n){
    __shared__ DATA_TYPE g_max;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Init block-wise group max in shared mem
    if ( threadIdx.x==0 && threadIdx.y==0 ){
        g_max = 0;
    }
    __syncthreads();

    if (row < m && col < n){
        atomicMaxFloat(&g_max, fabsf(matIn[row*n+col]));
    }
    __syncthreads();

    if ( threadIdx.x==0 && threadIdx.y==0 ){
        atomicMaxFloat(tMax, g_max);
    }
}

// GPT2-Activation
__global__ void FakeQuantColDir(DATA_TYPE* matIn, DATA_TYPE* tMax, DATA_TYPE* matOut, DATA_TYPE* scale,
                          int m, int k)
{
	// 0. Index setupW
	//    row, col     : position in the whole matrix
	//    localRow     : threadIdx.y                     (0 ~ BLOCK_SIZE_Y-1)
	//    localCol     : threadIdx.x                     (0 ~ BLOCK_SIZE_X-1)
	//    localGrp     : which group inside the tile row (0 ~ GROUP_PER_BLK_X-1)
	//    lane         : position inside the group       (0 ~ GROUP_SIZE-1)
	//    globalGrp    : index into scale[]

	int row      = blockIdx.y * blockDim.y + threadIdx.y;
	int col      = blockIdx.x * blockDim.x + threadIdx.x;
	int localRow = threadIdx.y;
	int localCol = threadIdx.x;
	int localGrp = threadIdx.x / GROUP_SIZE;
	int lane     = threadIdx.x % GROUP_SIZE;
	int globalGrp= row*(k/GROUP_SIZE) + (col/GROUP_SIZE);

	// 1. Shared memory declaration
	__shared__ DATA_TYPE sData[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	__shared__ DATA_TYPE sMax [BLOCK_SIZE_Y][GROUP_PER_BLK_X];

	// 2. Load global -> shared
    if (row<m && col<k){
        sData[localRow][localCol] = matIn[row*k + col];
    }
    else{
        sData[localRow][localCol] = 0;
    }
	__syncthreads();

	// 3. Abs-max of each group
    if (lane==0){
        DATA_TYPE max_val = 0.0f;
        int base = localGrp * GROUP_SIZE;

        for (int g = 0; g < GROUP_SIZE; g++) {
            DATA_TYPE a = fabsf(sData[localRow][base + g]);
            if (a > max_val) max_val = a;
        }
        sMax[localRow][localGrp] = max_val;
    }
    __syncthreads();
	
	// 4. Compute the scale
    DATA_TYPE s_t = (*tMax/E4M3_MAX) / QMAX;
    if (s_t == 0.0f) s_t = 1.0f;

    DATA_TYPE s_g_raw = (sMax[localRow][localGrp]/QMAX)/s_t;

    __nv_fp8_e4m3 s_fp8(s_g_raw);                 
    DATA_TYPE s_g = (DATA_TYPE)s_fp8; // 일반 변수는 thread마다 하나씩 생성

    DATA_TYPE s = s_g * s_t; 
    if (s == 0.0f) s = 1.0f;

    // 대표 thread가 scale factor global mem에 전달
    if (lane==0 && row<m && col<k){
        scale[globalGrp] = s_g;
    }

	// 5. Quantize, then dequantize back into matOut
    int q = __float2int_rn(sData[localRow][localCol]/s);
    q = min(max(q, -QMAX), QMAX);
    if (row < m && col < k){
        matOut[row * k + col] = (DATA_TYPE)q * s;
    }
}

// GPT2-Weight
__global__ void FakeQuantRowDir(DATA_TYPE* matIn, DATA_TYPE* tMax, DATA_TYPE* matOut, DATA_TYPE* scale, int k, int n){
    __shared__ DATA_TYPE sData[BLOCK_SIZE_AY][BLOCK_SIZE_AX];
    __shared__ DATA_TYPE gMax[GROUP_PER_BLK_Y][BLOCK_SIZE_AX];

    int row         = blockIdx.y * blockDim.y + threadIdx.y;
    int col         = blockIdx.x * blockDim.x + threadIdx.x;
    int localRow    = threadIdx.y;
    int localCol    = threadIdx.x;
    int localGrp    = threadIdx.y / GROUP_SIZE;
    int globalGrpRow = row / GROUP_SIZE;          // 전역 group 행 번호
    int globalGrp    = globalGrpRow * n + col;
    int lane        = threadIdx.y % GROUP_SIZE;

    if (row<k && col<n){
        sData[localRow][localCol] = matIn[row*n + col];
    }
    else{
        sData[localRow][localCol] = 0;
    }
    __syncthreads();

    if (lane==0){
        gMax[localGrp][localCol] = 0.0f;
        for (int g=0; g<GROUP_SIZE; g++){
            gMax[localGrp][localCol] = max(gMax[localGrp][localCol], fabsf(sData[localGrp*GROUP_SIZE+g][localCol]));
        }
    }
    __syncthreads();

    DATA_TYPE s_t = (*tMax/E4M3_MAX)/QMAX;
    if (s_t==0) s_t = 1.0f;

    DATA_TYPE s_g_raw = (gMax[localGrp][localCol]/QMAX)/s_t;
    __nv_fp8_e4m3 s_fp8(s_g_raw);
    if (lane==0 && row<k && col<n){
        scale[globalGrp] = (DATA_TYPE)s_fp8;
    }
    
    DATA_TYPE s = (DATA_TYPE)s_fp8*s_t;
    if (s==0) s = 1.0f;

    int q = __float2int_rn(sData[localRow][localCol]/s);
    q = min(max(q, -QMAX), QMAX);
    if(row<k && col<n){
        matOut[row*n+col] = (DATA_TYPE)q * s;
    }
}

__global__ void MatMul(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
    // shared memory
    __shared__ DATA_TYPE sA[BLOCK_SIZE ][TILING_SIZE];
    __shared__ DATA_TYPE sB[TILING_SIZE][BLOCK_SIZE];
    
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;
    int Boffset;
    DATA_TYPE val = 0;

    // global thread idx
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int bID=0; bID< ceil((float)k/TILING_SIZE); bID++){
        // Load data from global to shared memory
        Boffset = bID*TILING_SIZE;

        if (row < m && (Boffset + localCol)<k){
            sA[localRow][localCol] = matA[row*k + Boffset + localCol];
        }
        else{
            sA[localRow][localCol] = 0;
        }
        
        if (col < n && (Boffset+localRow)<k){
            sB[localRow][localCol] = matB[(Boffset + localRow)*n + col];
        }
        else{
            sB[localRow][localCol] = 0;
        }
        __syncthreads();

        for(int ich=0; ich<TILING_SIZE; ich++){
            val += __fmul_rn(sA[localRow][ich], sB[ich][localCol]);
        }
        __syncthreads();
        
    }

    if (row<m && col<n){
        matC[row*n+col] = val;
    }

}

// ------------ Launch function ------------ //
void launch_absmax(const float* in, float* tmax, int m, int n, cudaStream_t s) {
    dim3 b(16, 16), g((n+15)/16, (m+15)/16);
    TensorAbsMax<<<g, b, 0, s>>>((float*)in, tmax, m, n);
}

void launch_fq_col(const float* in, const float* tmax, float* out, float* scale,
                   int m, int k, cudaStream_t s) {          // activation
    dim3 b(BLOCK_SIZE_X, BLOCK_SIZE_Y);                     // (64, 4)
    dim3 g((k+BLOCK_SIZE_X-1)/BLOCK_SIZE_X, (m+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y);
    FakeQuantColDir<<<g, b, 0, s>>>((float*)in, (float*)tmax, out, scale, m, k);
}

void launch_fq_row(const float* in, const float* tmax, float* out, float* scale,
                   int k, int n, cudaStream_t s) {          // weight
    dim3 b(BLOCK_SIZE_AX, BLOCK_SIZE_AY);                   // (4, 64)
    dim3 g((n+BLOCK_SIZE_AX-1)/BLOCK_SIZE_AX, (k+BLOCK_SIZE_AY-1)/BLOCK_SIZE_AY);
    FakeQuantRowDir<<<g, b, 0, s>>>((float*)in, (float*)tmax, out, scale, k, n);
}

void launch_matmul(const float* A, const float* B, float* C,
                   int m, int n, int k, cudaStream_t s) {
    dim3 b(BLOCK_SIZE, BLOCK_SIZE);
    dim3 g((n+BLOCK_SIZE-1)/BLOCK_SIZE, (m+BLOCK_SIZE-1)/BLOCK_SIZE);
    MatMul<<<g, b, 0, s>>>((float*)A, (float*)B, C, m, n, k);
}