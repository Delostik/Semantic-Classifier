// dllapp.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

#include <iostream> 
#include <vector> 
#include <cuda_runtime.h> 
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_surface_types.h>
#include "device_launch_parameters.h" //device_launch_parameters.h"
//#include <comutil.h>
#include <stdint.h>
#include <stdio.h>

#include <stdlib.h>

//#pragma comment(lib, "cudart") 

#if defined(_WIN32)
#include <comutil.h>
using namespace _com_util;
#pragma comment(lib, "cudart")
#else
typedef wchar_t* BSTR;
wchar_t* SysAllocString(const wchar_t* src) {
	int len = wcslen(src);
	wchar_t* dst = (wchar_t*)malloc((len + 1) * sizeof(wchar_t));
	wcscpy(dst, src);
	return dst;
}
#endif

using namespace std; 
//using namespace _com_util;
/**************Cuda Basic Information************************/

DLLEXP uint32_t __stdcall CudaDeviceCount()
{
	int devCount = 0;
	cudaGetDeviceCount(&devCount);
	return devCount;
}

DLLEXP uint32_t __stdcall CudaSetDevice(int device)
{
	cudaSetDevice(device);
	return 1;
}

DLLEXP BSTR __stdcall CudaDeviceProperties(uint32_t i)
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, i);
	static const int bufferSize = 4096;
	wchar_t msg[bufferSize]; //= {L"fasdfsd"};
	swprintf(msg, bufferSize, L"Major revision number : %d \n", devProp.major);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Minor revision number : %d \n", devProp.minor);

	//wstring msg = L"fssagsdf";

	swprintf(&msg[wcslen(msg)], bufferSize, L"Name : %s \n", devProp.name);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Total global memory : %u \n", devProp.totalGlobalMem);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Total shared memory per block : %d \n", devProp.sharedMemPerBlock);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Total registers per block : %d \n", devProp.regsPerBlock);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Warp size : %d \n", devProp.warpSize);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Maximum memory pitch : %d \n", devProp.memPitch);

	swprintf(&msg[wcslen(msg)], bufferSize, L"Maximum threads per block : %u \n", devProp.maxThreadsPerBlock);

	for (uint32_t t = 0; t < 3; ++t)
		swprintf(&msg[wcslen(msg)], bufferSize, L"Maximum dimension %d of block:  %d\n", t, devProp.maxThreadsDim[t]);
	for (uint32_t t = 0; t < 3; ++t)
		swprintf(&msg[wcslen(msg)], bufferSize, L"Maximum dimension %d of grid:   %d\n", t, devProp.maxGridSize[t]);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Clock rate:                    %d\n", devProp.clockRate);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Total constant memory:         %u\n", devProp.totalConstMem);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Texture alignment:             %u\n", devProp.textureAlignment);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? L"Yes" : L"No"));
	swprintf(&msg[wcslen(msg)], bufferSize, L"Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	swprintf(&msg[wcslen(msg)], bufferSize, L"Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? L"Yes" : L"No"));
	//wchar_t msg[4096] ; //= {L"fasdfsd"};
	//swprintf(msg, L"Major revision number : %d \n", devProp.major);
	//swprintf(&msg[wcslen(msg)], L"Minor revision number : %d \n",devProp.minor);

	////wstring msg = L"fssagsdf";
	//
	//swprintf(&msg[wcslen(msg)], L"Name : %s \n",devProp.name);
	//swprintf(&msg[wcslen(msg)], L"Total global memory : %u \n",devProp.totalGlobalMem);
	//swprintf(&msg[wcslen(msg)], L"Total shared memory per block : %d \n",devProp.sharedMemPerBlock);
	//swprintf(&msg[wcslen(msg)], L"Total registers per block : %d \n",devProp.regsPerBlock);
	//swprintf(&msg[wcslen(msg)], L"Warp size : %d \n",devProp.warpSize);
	//swprintf(&msg[wcslen(msg)], L"Maximum memory pitch : %d \n",devProp.memPitch);

	//swprintf(&msg[wcslen(msg)], L"Maximum threads per block : %u \n",devProp.maxThreadsPerBlock);

	//for (uint32_t t = 0; t < 3; ++t)
	//	swprintf(&msg[wcslen(msg)], L"Maximum dimension %d of block:  %d\n", t, devProp.maxThreadsDim[t]);
 //   for (uint32_t t = 0; t < 3; ++t)
	//	swprintf(&msg[wcslen(msg)], L"Maximum dimension %d of grid:   %d\n", t, devProp.maxGridSize[t]);
 //   swprintf(&msg[wcslen(msg)], L"Clock rate:                    %d\n",  devProp.clockRate);
 //   swprintf(&msg[wcslen(msg)], L"Total constant memory:         %u\n",  devProp.totalConstMem);
 //   swprintf(&msg[wcslen(msg)], L"Texture alignment:             %u\n",  devProp.textureAlignment);
 //   swprintf(&msg[wcslen(msg)], L"Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? L"Yes" : L"No"));
 //   swprintf(&msg[wcslen(msg)], L"Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
 //   swprintf(&msg[wcslen(msg)], L"Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? L"Yes" : L"No"));
	
	return ::SysAllocString(msg);

	//return ::SysAllocString(devProp.name);
	//return ::SysAllocString(devProp.devic
}

/************************************************************/


/**************Cuda  Memory Operation************************/

DLLEXP uint32_t * __stdcall CudaAllocInt(uint32_t e)
{
	uint32_t * gpu_ints;
	cudaMalloc((void **)&gpu_ints, e * sizeof(uint32_t));
	cudaMemset(gpu_ints,0,e * sizeof(uint32_t));
	return gpu_ints;
}

DLLEXP void __stdcall CudaDeallocInt(uint32_t * gpu_ints)
{
	cudaFree(gpu_ints); 
}

DLLEXP void __stdcall CudaCopyInInt(uint32_t * gpu_ints, uint32_t * int_array, uint32_t len)
{
	cudaMemcpy(gpu_ints,int_array,len * sizeof(uint32_t),cudaMemcpyHostToDevice);
}


DLLEXP float * __stdcall CudaAllocFloat(uint32_t e, float init)
{
	float * gpu_floats;
	cudaMalloc((void **)&gpu_floats, e * sizeof(float));
	if (init == 0)
		cudaMemset(gpu_floats, 0, e * sizeof(float));
	else
		cuda_Init_Float_Array(gpu_floats, init, e);
	return gpu_floats;
}

DLLEXP void __stdcall CudaDeallocFloat(float * gpu_floats)
{
	cudaFree(gpu_floats); 
}

DLLEXP void __stdcall CudaCopyInFloat(float * gpu_floats, float * float_array, uint32_t len)
{
	cudaMemcpy(gpu_floats,float_array,len * sizeof(float),cudaMemcpyHostToDevice);
}

DLLEXP void __stdcall CudaCopyOutFloat(float * gpu_floats, float * float_array, uint32_t len)
{
	cudaMemcpy(float_array,gpu_floats,len * sizeof(float),cudaMemcpyDeviceToHost);
}

DLLEXP void __stdcall Zero(float * gpu_floats, uint32_t len)
{
	cudaMemset(gpu_floats,0,len * sizeof(float));
}


/************************************************************/

/**************Cuda  Matrix Operation************************/
DLLEXP void __stdcall Matrix_Ada_Grad_Decent(float * gpu_floats_a, float * gpu_floats_b, float * adaG, uint32_t m, uint32_t n, float lr, float eps)
{
	cuda_Matrix_Ada_Grad_Decent(gpu_floats_a, gpu_floats_b, adaG, m, n, lr, eps);
}

DLLEXP void __stdcall Matrix_Grad_Decent(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n, float lr)
{
	cuda_Matrix_Grad_Decent(gpu_floats_a, gpu_floats_b, m, n, lr);
}

DLLEXP void __stdcall Matrix_Add(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n, float mweight)
{
	cuda_Matrix_Add(gpu_floats_a, gpu_floats_b,  m,  n, mweight);
}

DLLEXP void __stdcall Matrix_Add_REAL(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n)
{
	cuda_Matrix_Add_REAL(gpu_floats_a, gpu_floats_b, m, n);
}

DLLEXP void __stdcall Scale_Matrix(float * gpu_floats_a, uint32_t m, uint32_t n, float mweight)
{
	cuda_Scale_Matrix(gpu_floats_a, m,  n, mweight);
}

DLLEXP void __stdcall Matrix_Add_Tanh(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n)
{
	cuda_Matrix_Add_Tanh(gpu_floats_a,gpu_floats_b,m,n);
}

DLLEXP void __stdcall Deriv_Cosine( float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	cuda_Deriv_Cosine(q,d,dcq,dcd,batchsize,m,eps);
}

DLLEXP void __stdcall Deriv_Dis(float * s1deriv, float * s2deriv, float * s3deriv, float * s1, float * s2, float * s3, float * dis, uint32_t batchsize, uint32_t m, float margin)
{
	cuda_Deriv_Dis(s1deriv, s2deriv, s3deriv, s1, s2, s3, dis, batchsize, m, margin);
}

DLLEXP void __stdcall Deriv_Dis_Linear(float * s1deriv, float * s2deriv, float * s3deriv, float * s1, float * s2, float * s3, float * dis, uint32_t batchsize, uint32_t m, float margin)
{
	cuda_Deriv_Dis_Linear(s1deriv, s2deriv, s3deriv, s1, s2, s3, dis, batchsize, m, margin);
}

DLLEXP void __stdcall Deriv_Dis_Rectified(float * s1deriv, float * s2deriv, float * s3deriv, float * s1, float * s2, float * s3, float * dis, uint32_t batchsize, uint32_t m, float margin, float eps)
{
	cuda_Deriv_Dis_Rectified(s1deriv, s2deriv, s3deriv, s1, s2, s3, dis, batchsize, m, margin, eps);
}

DLLEXP void __stdcall Calc_EuclideanDis(float * s1, float * s2, float * s3, float * res, uint32_t batchsize, uint32_t m, float eps)
{
	cuda_Calc_EuclideanDis(s1, s2, s3, res, batchsize, m, eps);
}

DLLEXP void __stdcall Deriv_Cosine_EX( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	cuda_Deriv_Cosine_EX(q,d,neg_list,dcq,dcd,batchsize,m,eps);
}


DLLEXP void __stdcall Deriv_Tanh(float * delta, float * layer_output, uint32_t batchsize, uint32_t m)
{
	cuda_Deriv_Tanh(delta, layer_output, batchsize, m);
}



DLLEXP void __stdcall Matrix_Multipy(float * delta, float * weight, float * delta_low, uint32_t batchsize, uint32_t m, uint32_t n, uint32_t inverse )
{
	cuda_Matrix_Multipy(delta, weight, delta_low,batchsize,m,n, inverse);
}


//DLLEXP void _stdcall Sparse_Matrix_Multiply_EX(uint32_t * aRow_Index, uint32_t * aCol_Index, float * aValue, uint32_t elementSize, float * b, float * c, uint32_t m, uint32_t n, uint32_t w)
//{
DLLEXP void __stdcall Cosine_Similarity(float * a, float * b, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t mindex, 
									   uint32_t batchsize, uint32_t dimension, float eps)
{
	cuda_Cosine_Similarity(a,b,c,nTrial,BATCHSIZE,mindex,batchsize, dimension, eps);
}


DLLEXP void __stdcall Cosine_Similarity_EX(float * a, float * b, uint32_t * neg_list, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t mindex, 
									   uint32_t batchsize, uint32_t dimension, float eps)
{
	cuda_Cosine_Similarity_EX(a,b,neg_list,c,nTrial,BATCHSIZE,mindex,batchsize, dimension, eps);
}

DLLEXP void __stdcall Calculate_Alpha(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	cuda_Calculate_Alpha(alpha, nTrial, BATCHSIZE, batchsize, gamma);
}

DLLEXP void __stdcall Calculate_Alpha_MXE(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	cuda_Calculate_Alpha_MXE(alpha, nTrial, BATCHSIZE, batchsize, gamma);
}

DLLEXP void __stdcall Calculate_Alpha_NCE(float * alpha, float * dist, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	cuda_Calculate_Alpha_NCE(alpha, dist, nTrial, BATCHSIZE, batchsize, gamma);
}

DLLEXP void __stdcall Calculate_Alpha_NCE2(float * alpha, float * dist, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	cuda_Calculate_Alpha_NCE2(alpha, dist, nTrial, BATCHSIZE, batchsize, gamma);
}

DLLEXP void __stdcall Calculate_Alpha_PAIRRANK(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	cuda_Calculate_Alpha_PAIRRANK(alpha, nTrial, BATCHSIZE, batchsize, gamma);
}


DLLEXP void __stdcall FillOut_Dist_NCE(float* dist, uint32_t* neg_list, uint32_t nTrailPlus1, uint32_t BATCH_SIZE, uint32_t mindex, uint32_t batchsize)
{
	cuda_FillOut_Dist_NCE(dist, neg_list, nTrailPlus1, BATCH_SIZE, mindex, batchsize);
}



DLLEXP void __stdcall Matrix_Product(float * a1, float * b1, float * a2, float * b2, float * a3, float * b3, float * c, uint32_t batchsize, uint32_t m, uint32_t n)
			//, uint32_t kept, float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	cuda_Matrix_Product(a1, b1, a2, b2, a3, b3,c, batchsize, m,n); //,kept, alpha, ntrial, BATCH_SIZE, alpha_index);
}

DLLEXP void __stdcall Matrix_Product_Sup(float * a, float * b, float * c, uint32_t batchsize, uint32_t m, uint32_t n)
//, uint32_t kept, float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	cuda_Matrix_Product_Sup(a, b, c, batchsize, m, n); //,kept, alpha, ntrial, BATCH_SIZE, alpha_index);
}


DLLEXP void __stdcall SEQ_Sparse_Matrix_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Seg_Index, uint32_t * Seg_Margin, float * Seg_Len, 
												   uint32_t seg_size, uint32_t * Fea_Index, 
												   float * Fea_Value, uint32_t elementsize, 
												   float * mul_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size)
{
	cuda_SEQ_Sparse_Matrix_Multiply_INTEX(Smp_Index,batchsize, Seg_Index, Seg_Margin, Seg_Len, seg_size, Fea_Index, Fea_Value, elementsize, mul_weight, output, Feature_dimension, output_dimension, win_size);
}

DLLEXP void SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Seg_Index, uint32_t * Seg_Margin, float * Seg_Len, uint32_t seg_size, uint32_t * Fea_Index, 
												   float * Fea_Value, uint32_t elementsize, 
												   float * mul_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size)
{
	cuda_SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(Smp_Index,batchsize, Seg_Index, Seg_Margin, Seg_Len, seg_size, Fea_Index, Fea_Value, elementsize, mul_weight, output, Feature_dimension, output_dimension, win_size);
}


DLLEXP void __stdcall Convolution_Matrix_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Word_Index, uint32_t * Word_Margin, uint32_t Word_SeqLen, float * wordLT,										  
												   float * con_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size)
{
	cuda_Convolution_Matrix_Multiply_INTEX(Smp_Index, batchsize, Word_Index, Word_Margin, Word_SeqLen, wordLT, con_weight, output, Feature_dimension, output_dimension, win_size);
}

DLLEXP void __stdcall MultiConv_Matrix_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Word_Index, uint32_t * Word_Margin, uint32_t Word_SeqLen, float * wordLT,
	float * con_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t * win_sizes, uint32_t * fm_sizes)
{
	cuda_MultiConv_Matrix_Multiply_INTEX(Smp_Index, batchsize, Word_Index, Word_Margin, Word_SeqLen, wordLT, con_weight, output, Feature_dimension, output_dimension, win_sizes, fm_sizes);
}

DLLEXP void __stdcall Matrix_Add_Vector(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension)
{
	cuda_Matrix_Add_Vector(gpu_floats_a,gpu_floats_b,batchsize,dimension);
}

DLLEXP void __stdcall Matrix_Rectified_Vector(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension)
{
	cuda_Matrix_Rectified_Vector(gpu_floats_a,gpu_floats_b,batchsize,dimension);
}

DLLEXP void __stdcall Derive_Cosine_Linear(float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	cuda_Derive_Cosine_Linear(q,d,dcq,dcd, batchsize, m, eps);
}

DLLEXP void __stdcall Derive_Cosine_Linear_EX( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	cuda_Deriv_Cosine_Linear_EX(q, d, neg_list, dcq, dcd, batchsize, m, eps);
}

DLLEXP void __stdcall Derive_Cosine_Rectified(float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	cuda_Derive_Cosine_Rectified(q, d, dcq, dcd, batchsize, m, eps);
}

DLLEXP void __stdcall Derive_Cosine_Rectified_EX(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	cuda_Deriv_Cosine_Rectified_EX(q, d, neg_list, dcq, dcd, batchsize, m, eps);
}

DLLEXP void __stdcall Deriv_Rectified( float * delta, float * layer_output, uint32_t batchsize, uint32_t m)
{
	cuda_Deriv_Rectified(delta, layer_output, batchsize, m);
}



DLLEXP void __stdcall Max_Pooling(float * pooling_feas, int * Smp_Index, int batchsize, float * output,int * maxpooling_index, int output_dimension, int win_size)
{
	cuda_Max_Pooling(pooling_feas, Smp_Index, batchsize, output, maxpooling_index, output_dimension, win_size);
}

DLLEXP void __stdcall LSTM_Max_Pooling(float * pooling_feas, int * Smp_Index, int batchsize, float * output, int * maxpooling_index, int output_dimension)
{
	cuda_LSTM_Max_Pooling(pooling_feas, Smp_Index, batchsize, output, maxpooling_index, output_dimension);
}

DLLEXP void __stdcall Multi_Max_Pooling(float * pooling_feas, int * Smp_Index, int batchsize, float * output, int * maxpooling_index, int output_dimension, int * win_sizes, int * fm_sizes)
{
	cuda_Multi_Max_Pooling(pooling_feas, Smp_Index, batchsize, output, maxpooling_index, output_dimension, win_sizes, fm_sizes);
}

DLLEXP void __stdcall Convolution_Matrix_Product_INTEX(float * deriv1, int * maxpooling_index1, float * deriv2, int * maxpooling_index2, float * deriv3, int * maxpooling_index3, float * wordLT,
	int * Word_Index1, int * Word_Index2, int * Word_Index3, int win_size, int batchsize, int output_dimension, float * grad, int Feature_Dimension)
										//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	cuda_Convolution_Matrix_Product_INTEX(deriv1, maxpooling_index1, deriv2, maxpooling_index2, deriv3, maxpooling_index3, wordLT, Word_Index1, Word_Index2, Word_Index3, win_size,
			batchsize, output_dimension, grad, Feature_Dimension);
	//,alpha, ntrial, BATCH_SIZE, alpha_index);
}


DLLEXP void __stdcall Convolution_Matrix_Product_Sup(float * deriv, int * maxpooling_index, float * wordLT,
	int * Word_Index, int win_size, int batchsize, int output_dimension, float * grad, int Feature_Dimension)
	//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	cuda_Convolution_Matrix_Product_Sup(deriv, maxpooling_index, wordLT, Word_Index, win_size,
		batchsize, output_dimension, grad, Feature_Dimension);
	//,alpha, ntrial, BATCH_SIZE, alpha_index);
}

DLLEXP void __stdcall MultiConv_Matrix_Product_INTEX(float * deriv1, int * maxpooling_index1, float * deriv2, int * maxpooling_index2, float * deriv3, int * maxpooling_index3, float * wordLT,
	int * Word_Index1, int * Word_Index2, int * Word_Index3, int batchsize, int output_dimension, float * grad, int Feature_Dimension, int winsize, int fmsize, int accu, int accu_para)
	//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	cuda_MultiConv_Matrix_Product_INTEX(deriv1, maxpooling_index1, deriv2, maxpooling_index2, deriv3, maxpooling_index3, wordLT, Word_Index1, Word_Index2, Word_Index3,
		batchsize, output_dimension, grad, Feature_Dimension, winsize, fmsize, accu, accu_para);
}

DLLEXP void __stdcall MultiConv_Matrix_Product_Sup(float * deriv, int * maxpooling_index, float * wordLT,
	int * Word_Index, int batchsize, int output_dimension, float * grad, int Feature_Dimension, int winsize, int fmsize, int accu, int accu_para)
	//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	cuda_MultiConv_Matrix_Product_Sup(deriv, maxpooling_index, wordLT, Word_Index,
		batchsize, output_dimension, grad, Feature_Dimension, winsize, fmsize, accu, accu_para);
}

DLLEXP void __stdcall MultiConv_Compute_WVDERIV(float * deriv, int * maxpooling_index, float * weight, int batchsize, int output_dimension, float * grad, int Feature_Dimension, int * winsizes, int * fmsizes)
{
	cuda_MultiConv_Compute_WVDERIV(deriv, maxpooling_index, weight, batchsize, output_dimension, grad, Feature_Dimension, winsizes, fmsizes);
}

DLLEXP void __stdcall Conv_Compute_WVDERIV(float * deriv, int * maxpooling_index, float * weight, int batchsize, int output_dimension, float * grad, int Feature_Dimension, int winsize)
{
	cuda_Conv_Compute_WVDERIV(deriv, maxpooling_index, weight, batchsize, output_dimension, grad, Feature_Dimension, winsize);
}

DLLEXP void __stdcall Matrix_WeightAdd(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	cuda_Matrix_WeightAdd(gpu_floats_a, gpu_floats_b, batchsize, dimension, mweight, start, keep);
}

DLLEXP void __stdcall Matrix_WeightAdd_EX(float * gpu_floats_a, float * gpu_floats_b, int * inver_neg_index, int * inver_neg_value, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	cuda_Matrix_WeightAdd_EX(gpu_floats_a, gpu_floats_b, inver_neg_index, inver_neg_value, batchsize, dimension, mweight, start, keep);
}

DLLEXP void __stdcall Sparse2Dense_Matrix(int * Smp_Idx, int * Fea_Idx, float * Fea_Value, float * matrix, int batchsize, int outputDimension)
{
	cuda_Sparse2Dense_Matrix(Smp_Idx, Fea_Idx, Fea_Value, matrix, batchsize, outputDimension);
}

DLLEXP void __stdcall Matrix_Aggragate(float * a1, float * a2, float * a3, float * b, uint32_t batchsize, uint32_t m)
{
	cuda_Matrix_Aggragate(a1, a2, a3, b, batchsize, m);
}

DLLEXP void __stdcall Matrix_Aggragate_Sup(float * a, float * b, uint32_t batchsize, uint32_t m)
{
	cuda_Matrix_Aggragate_Sup(a, b, batchsize, m);
}

DLLEXP void __stdcall Matrix_Add_OFFSET(float * gpu_floats_a, uint32_t offset_a, float * gpu_floats_b, uint32_t offset_b, uint32_t len, float mweight)
{
	cuda_Matrix_Add_OFFSET(gpu_floats_a, offset_a, gpu_floats_b, offset_b, len, mweight);
}

DLLEXP void __stdcall CUBLAS_Init()
{
	cublas_Init();
}

DLLEXP void __stdcall CUBLAS_Destroy()
{
	cublas_Destroy();
}

DLLEXP float __stdcall CUBLAS_Sasum(float *x, int len, int norm)
{
	float result = 0;
	cublas_Sasum(x, len, norm, &result);
	return result;
}

DLLEXP void __stdcall CUBLAS_Matrix_Multipy(float * delta, float * weight, float * delta_low, uint32_t batchsize, uint32_t m, uint32_t n, uint32_t inverse)
{
	cublas_Matrix_Multipy(delta, weight, delta_low, batchsize, m, n, inverse);
}

DLLEXP void __stdcall Cosine_Similarity_EX_Full(float * a, float * b, uint32_t * neg_list, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float eps)
{
	cuda_Cosine_Similarity_EX_Full(a, b, neg_list, c, nTrial, BATCHSIZE, batchsize, dimension, eps);
}

DLLEXP void __stdcall FillOut_Dist_NCE_Full(float* dist, uint32_t* neg_list, uint32_t nTrail, uint32_t BATCH_SIZE, uint32_t batchsize)
{
	cuda_FillOut_Dist_NCE_Full(dist, neg_list, nTrail, BATCH_SIZE, batchsize);
}

DLLEXP void __stdcall Deriv_Cosine_EX_Full( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t m, float eps)
{
	cuda_Deriv_Cosine_EX_Full(q, d, neg_list, dcq, dcd, nTrail, BATCHSIZE, batchsize, m, eps);
}

DLLEXP void __stdcall Deriv_Cosine_Linear_EX_Full(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize,  uint32_t m, float eps)
{
	cuda_Deriv_Cosine_Linear_EX_Full(q, d, neg_list, dcq, dcd, nTrail, BATCHSIZE, batchsize, m, eps);
}

DLLEXP void __stdcall Deriv_Cosine_Rectified_EX_Full( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t m, float eps)
{
	cuda_Deriv_Cosine_Rectified_EX_Full(q, d, neg_list, dcq, dcd, nTrail, BATCHSIZE, batchsize, m, eps);
}

DLLEXP void __stdcall Matrix_WeightAdd_Full(float * gpu_floats_a, float * gpu_floats_b, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	cuda_Matrix_WeightAdd_Full(gpu_floats_a, gpu_floats_b, nTrail, BATCHSIZE, batchsize, dimension, mweight, start, keep);
}

DLLEXP void __stdcall Matrix_WeightAdd_EX_Full(float * gpu_floats_a, float * gpu_floats_b, int * inver_neg_index, int * inver_neg_value, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	cuda_Matrix_WeightAdd_EX_Full(gpu_floats_a, gpu_floats_b, inver_neg_index, inver_neg_value, nTrial, BATCHSIZE, batchsize, dimension, mweight, start, keep);
}

DLLEXP void __stdcall Cosine_Similarity_SubSpace(float * a, float * b, float * c, uint32_t labelDim, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t subspaceDim, float eps)
{
	cuda_Cosine_Similarity_SubSpace(a, b, c, labelDim, BATCHSIZE, batchsize, subspaceDim, eps);
}

DLLEXP void __stdcall SoftMax(float * a, float * b,uint32_t labelDim, uint32_t batchsize, float gamma)
{
	cuda_SoftMax(a, b, labelDim, batchsize, gamma);
}

DLLEXP void __stdcall Deriv_Cosine_Subspace( float * q, float * d, float * dcq, float * dcd, float * alpha,  uint32_t act_type, uint32_t batchsize, uint32_t labelDim, uint32_t subspaceDim, float gamma, float eps)
{
	cuda_Deriv_Cosine_Subspace(q, d, dcq, dcd, alpha, act_type, batchsize, labelDim, subspaceDim, gamma, eps);
}

DLLEXP void __stdcall InnerProduct_Similarity(float * a, float * b, float * c, uint32_t batchsize, uint32_t dimension)
{
	cuda_InnerProduct_Similarity(a, b, c, batchsize, dimension);
}

DLLEXP void __stdcall Deriv_InnerProduct( float * q, float * d, float * dcq, float * dcd, float * alpha,  uint32_t act_type, uint32_t batchsize, uint32_t Dim, float gamma, float eps)
{
	cuda_Deriv_InnerProduct( q, d, dcq, dcd, alpha, act_type, batchsize, Dim, gamma, eps);
}

DLLEXP void __stdcall FillOut_Composite(float* data, uint32_t* feaIdx, float* compData, float* context, uint32_t d1, uint32_t d2, uint32_t batchsize, uint32_t direction)
{
	cuda_FillOut_Composite(data, feaIdx, compData, context, d1, d2, batchsize, direction);
}

DLLEXP void __stdcall Sparse_Update_Lookup(float * lookupt, int * Fea_ID, int * Fea_Idx, int * Seq, float * ltDeriv1, float * ltDeriv2, float * ltDeriv3, int seq1size, int seq2size, int IDnum, int Feature_Dimension, float lr)
{
	cuda_Sparse_Update_Lookup(lookupt, Fea_ID, Fea_Idx, Seq, ltDeriv1, ltDeriv2, ltDeriv3, seq1size, seq2size, IDnum, Feature_Dimension, lr);
}

DLLEXP void __stdcall Sparse_Update_Lookup_Ada(float * lookupt, int * Fea_ID, int * Fea_Idx, int * Seq, float * ltDeriv1, float * ltDeriv2, float * ltDeriv3, int seq1size, int seq2size, int IDnum, int Feature_Dimension, float lr, float * adaGrad, float eps)
{
	cuda_Sparse_Update_Lookup_Ada(lookupt, Fea_ID, Fea_Idx, Seq, ltDeriv1, ltDeriv2, ltDeriv3, seq1size, seq2size, IDnum, Feature_Dimension, lr, adaGrad, eps);
}

DLLEXP void __stdcall Sparse_Update_Lookup_Update(float * lookupt_update, int * Fea_ID, int * Fea_Idx, int * Seq, float * ltDeriv1, float * ltDeriv2, float * ltDeriv3, int seq1size, int seq2size, int IDnum, int Feature_Dimension, float lr)
{
	cuda_Sparse_Update_Lookup_Update(lookupt_update, Fea_ID, Fea_Idx, Seq, ltDeriv1, ltDeriv2, ltDeriv3, seq1size, seq2size, IDnum, Feature_Dimension, lr);
}


DLLEXP void __stdcall Sparse_Update_Lookup_Sup(float * lookupt, int * Fea_ID, int * Fea_Idx, int * Seq, float * ltDeriv, int IDnum, int Feature_Dimension, float lr)
{
	cuda_Sparse_Update_Lookup_Sup(lookupt, Fea_ID, Fea_Idx, Seq, ltDeriv, IDnum, Feature_Dimension, lr);
}

DLLEXP void __stdcall Sparse_Update_Lookup_Ada_Sup(float * lookupt, int * Fea_ID, int * Fea_Idx, int * Seq, float * ltDeriv, int IDnum, int Feature_Dimension, float lr, float * adaGrad, float eps)
{
	cuda_Sparse_Update_Lookup_Ada_Sup(lookupt, Fea_ID, Fea_Idx, Seq, ltDeriv, IDnum, Feature_Dimension, lr, adaGrad, eps);
}

DLLEXP void __stdcall Sparse_Update_Lookup_Update_Sup(float * lookupt_update, int * Fea_ID, int * Fea_Idx, int * Seq, float * ltDeriv, int IDnum, int Feature_Dimension, float lr)
{
	cuda_Sparse_Update_Lookup_Update_Sup(lookupt_update, Fea_ID, Fea_Idx, Seq, ltDeriv, IDnum, Feature_Dimension, lr);
}

DLLEXP void __stdcall LSTM_Input_Batch_Product(uint32_t * Word_Index, uint32_t Word_SeqLen, float * wordLT,
	float * weight, float * outputA, float * outputI, float * outputF, float * outputO, uint32_t Feature_dimension, uint32_t output_dimension)
{
	cuda_LSTM_Input_Batch_Product(Word_Index, Word_SeqLen, wordLT, weight, outputA, outputI, outputF, outputO, Feature_dimension, output_dimension);
}

DLLEXP void __stdcall LSTM_Sequence_Forward(int * Smp_Index, int batchsize,
	float * reweight, float * bias, float * outputA, float * outputI, float * outputF, float * outputO, float * outputC, float * output, int output_dimension)
{
	cuda_LSTM_Sequence_Forward(Smp_Index, batchsize, reweight, bias, outputA, outputI, outputF, outputO, outputC, output, output_dimension);
}

DLLEXP void __stdcall LSTM_Sequence_Backward(int * Smp_Index, int batchsize,
	float * reweight, int * maxpooling_index, float * derivup, float * outputA, float * outputI, float * outputF, float * outputO, float * outputC, float * output, int output_dimension)
{
	cuda_LSTM_Sequence_Backward(Smp_Index, batchsize, reweight, maxpooling_index, derivup, outputA, outputI, outputF, outputO, outputC, output, output_dimension);
}

DLLEXP void __stdcall LSTM_Weight_Deriv(uint32_t * Smp_Index1, uint32_t * Smp_Index2, uint32_t * Smp_Index3,
	uint32_t * Word_Index1, uint32_t * Word_Index2, uint32_t * Word_Index3, uint32_t Word_SeqLen1, uint32_t Word_SeqLen2, uint32_t Word_SeqLen3,
	float * wordLT, float * grad, float * outA1, float * outA2, float * outA3, float * outI1, float * outI2, float * outI3,
	float * outF1, float * outF2, float * outF3, float * outO1, float * outO2, float * outO3, float * h1, float * h2, float * h3,
	uint32_t fea_dimension, uint32_t output_dimension, uint32_t b_reweight)
{
	cuda_LSTM_Weight_Deriv(Smp_Index1, Smp_Index2, Smp_Index3, Word_Index1, Word_Index2, Word_Index3, Word_SeqLen1, Word_SeqLen2, Word_SeqLen3, 
		wordLT, grad, outA1, outA2, outA3, outI1, outI2, outI3, outF1, outF2, outF3, outO1, outO2, outO3, h1, h2, h3, fea_dimension, output_dimension, b_reweight);
}

DLLEXP void __stdcall LSTM_Weight_Deriv_Sup(uint32_t * Smp_Index1, uint32_t * Word_Index1, uint32_t Word_SeqLen1,
	float * wordLT, float * grad, float * outA1, float * outI1, float * outF1, float * outO1, float * h1,
	uint32_t fea_dimension, uint32_t output_dimension, uint32_t b_reweight)
{
	cuda_LSTM_Weight_Deriv_Sup(Smp_Index1, Word_Index1, Word_SeqLen1, wordLT, grad, outA1, outI1, outF1, outO1, h1, fea_dimension, output_dimension, b_reweight);
}

DLLEXP void __stdcall LSTM_Bias_Deriv(uint32_t Word_SeqLen1, uint32_t Word_SeqLen2, uint32_t Word_SeqLen3,
	float * grad, float * outA1, float * outA2, float * outA3, float * outI1, float * outI2, float * outI3,
	float * outF1, float * outF2, float * outF3, float * outO1, float * outO2, float * outO3, uint32_t output_dimension)
{
	cuda_LSTM_Bias_Deriv(Word_SeqLen1, Word_SeqLen2, Word_SeqLen3, grad, outA1, outA2, outA3, outI1, outI2, outI3, outF1, outF2, outF3, outO1, outO2, outO3, output_dimension);
}

DLLEXP void __stdcall LSTM_Bias_Deriv_Sup(uint32_t Word_SeqLen1, float * grad, float * outA1, float * outI1,
	float * outF1, float * outO1, uint32_t output_dimension)
{
	cuda_LSTM_Bias_Deriv_Sup(Word_SeqLen1, grad, outA1, outI1, outF1, outO1, output_dimension);
}

DLLEXP void __stdcall LSTM_Compute_WVDeriv(uint32_t Word_SeqLen, float * weight, float * grad, float * outA, float * outI, float * outF, float * outO, uint32_t fea_dim, uint32_t output_dim)
{
	cuda_LSTM_Compute_WVDeriv(Word_SeqLen, weight, grad, outA, outI, outF, outO, fea_dim, output_dim);
}

/************************************************************/