// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>

#include <iostream> 
#include <vector> 
#include <cuda_runtime.h> 
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_surface_types.h>
#include "device_launch_parameters.h" //device_launch_parameters.h"
#include <comutil.h>
#include <stdint.h>
#include <stdio.h>

#include <stdlib.h>

#include "cublas_v2.h"
#pragma comment(lib, "cudart") 
#pragma comment(lib,"cublas.lib")

//typedef amp_tausworthe_hybrid_collection<2> amp_rng;
// TODO: reference additional headers your program requires here
#define DROPOUT_CONST (3e38f)

#define DLLEXP extern "C" __declspec(dllexport)
// TODO: reference additional headers your program requires here
#define DEFAULT_THREAD_PER_BLOCK    128     // default number of threads per block 
#define DEFAULT_THREAD_PER_DIM		16
#define MAX_BATCH_SIZE              256
#define MAX_THREAD_NUM			1024
#define MAX_BLOCK_NUM			65536


void cublas_Init();
void cublas_Destroy();
void cublas_Sasum(float *x, int len, int norm, float * result);
void cublas_Matrix_Multipy(float * delta, float * weight, float * delta_low, uint32_t batchsize, uint32_t m, uint32_t n, uint32_t inverse);


void cuda_Matrix_Add(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n, float weight);
void cuda_Scale_Matrix(float * gpu_floats_a, uint32_t m, uint32_t n, float mweight);
void cuda_Matrix_Add_Tanh(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n);

void cuda_Deriv_Cosine( float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps);
void cuda_Deriv_Cosine_EX( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps);


void cuda_Derive_Cosine_Linear(float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps);
void cuda_Deriv_Cosine_Linear_EX( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps);

void cuda_Derive_Cosine_Rectified(float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps);
void cuda_Deriv_Cosine_Rectified_EX( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps);

void cuda_Deriv_Rectified( float * delta, float * layer_output, uint32_t batchsize, uint32_t m);

void cuda_Deriv_Tanh( float * delta, float * layer_output, uint32_t batchsize, uint32_t m);

void cuda_Matrix_Multipy(float * delta, float * weight, float * delta_low, uint32_t batchsize, uint32_t m, uint32_t n, uint32_t inverse );

void cuda_Cosine_Similarity(float * a, float * b, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t mindex, 
									   uint32_t batchsize, uint32_t dimension, float eps);
void cuda_Cosine_Similarity_EX(float * a, float * b, uint32_t * neg_list, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t mindex, 
									   uint32_t batchsize, uint32_t dimension, float eps);

void cuda_Calculate_Alpha(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma);
void cuda_Calculate_Alpha_MXE(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma);
void cuda_Calculate_Alpha_PAIRRANK(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma);
void cuda_Calculate_Alpha_NCE(float * alpha, float * dist, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma);
void cuda_Calculate_Alpha_NCE2(float * alpha, float * dist, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma);
void cuda_FillOut_Dist_NCE(float* dist, uint32_t* neg_list, uint32_t nTrailPlus1, uint32_t BATCH_SIZE, uint32_t mindex, uint32_t batchsize);

void cuda_Matrix_Product(float * a, float * b, float * c, uint32_t batchsize, uint32_t m, uint32_t n); //, uint32_t kept, float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index);



void cuda_SEQ_Sparse_Matrix_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Seg_Index, uint32_t * Seg_Margin, float * Seg_Len, uint32_t seg_size, uint32_t * Fea_Index, 
												   float * Fea_Value, uint32_t elementsize, float * mul_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size);

void cuda_SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Seg_Index, uint32_t * Seg_Margin, float * Seg_Len, uint32_t seg_size, uint32_t * Fea_Index, 
												   float * Fea_Value, uint32_t elementsize, 
												   float * mul_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size);





void cuda_Matrix_Add_Vector(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension);
void cuda_Matrix_Rectified_Vector(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension);

void cuda_Convolution_Sparse_Matrix_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Seg_Index, uint32_t * Seg_Margin, float * Seg_Len, uint32_t seg_size, uint32_t * Fea_Index, 
												   float * Fea_Value, uint32_t elementsize, float * con_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size);

void cuda_Max_Pooling(float * pooling_feas, int * Smp_Index, int batchsize, float * output, int * maxpooling_index, int output_dimension);

void cuda_Convolution_Sparse_Matrix_Product_INTEX(float * deriv, int * maxpooling_index, int * Seg_Index, int * SegMargin_Index, int seg_size, int win_size,
										int batchsize, int output_dimension, int * Fea_Index, float * Fea_Value, float * grad, int Feature_Dimension); 
										//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index);

void cuda_Matrix_WeightAdd(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep);

void cuda_Matrix_WeightAdd_EX(float * gpu_floats_a, float * gpu_floats_b, int * inver_neg_index, int * inver_neg_value, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep);

void cuda_Sparse2Dense_Matrix(int * Smp_Idx, int * Fea_Idx, float * Fea_Value, float * matrix, int batchsize, int outputDimension);

void cuda_Matrix_Aggragate(float * a, float * b, uint32_t batchsize, uint32_t m);

void cuda_Matrix_Add_OFFSET(float * gpu_floats_a, uint32_t offset_a, float * gpu_floats_b, uint32_t offset_b, int len, float mweight);

void cuda_Cosine_Similarity_EX_Full(float * a, float * b, uint32_t * neg_list, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float eps);

void cuda_FillOut_Dist_NCE_Full(float* dist, uint32_t* neg_list, uint32_t nTrail, uint32_t BATCH_SIZE, uint32_t batchsize);

void cuda_Deriv_Cosine_EX_Full( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t m, float eps);

void cuda_Deriv_Cosine_Linear_EX_Full(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize,  uint32_t m, float eps);

void cuda_Deriv_Cosine_Rectified_EX_Full( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t m, float eps);

void cuda_Matrix_WeightAdd_Full(float * gpu_floats_a, float * gpu_floats_b, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep);

void cuda_Matrix_WeightAdd_EX_Full(float * gpu_floats_a, float * gpu_floats_b, int * inver_neg_index, int * inver_neg_value, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep);

void cuda_Cosine_Similarity_SubSpace(float * a, float * b, float * c, uint32_t labelDim, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t subspaceDim, float eps);

void cuda_SoftMax(float * a, float * b,uint32_t labelDim, uint32_t batchsize, float gamma);

void cuda_Deriv_Cosine_Subspace( float * q, float * d, float * dcq, float * dcd, float * alpha,  uint32_t act_type, uint32_t batchsize, uint32_t labelDim, uint32_t subspaceDim, float gamma, float eps);

void cuda_InnerProduct_Similarity(float * a, float * b, float * c, uint32_t batchsize, uint32_t dimension);

void cuda_Deriv_InnerProduct( float * q, float * d, float * dcq, float * dcd, float * alpha,  uint32_t act_type, uint32_t batchsize, uint32_t Dim, float gamma, float eps);