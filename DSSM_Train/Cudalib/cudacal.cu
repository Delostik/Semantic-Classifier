#include "stdafx.h"

#include <iostream> 
#include <vector> 
#include <cuda_runtime.h> 
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_surface_types.h>
#include "device_launch_parameters.h" //device_launch_parameters.h"
#include <comutil.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"

#pragma comment(lib, "cudart") 
#pragma comment(lib,"cublas.lib")

using namespace std; 
using namespace _com_util;

__global__ void cuda_matrix_add(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n, float mweight)
{ 
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < n && idy < m)
	{
		gpu_floats_a[idy*n+idx] = gpu_floats_a[idy*n+idx] + gpu_floats_b[idy*n+idx] * mweight;
	}
}

void cuda_Matrix_Add(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n, float mweight)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((n + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_matrix_add<<<block_tail ,thread_tail>>>(gpu_floats_a, gpu_floats_b, m, n,mweight);
}

__global__ void cuda_matrix_add_real(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n && idy < m)
	{
		gpu_floats_a[idy*n + idx] = gpu_floats_a[idy*n + idx] - gpu_floats_b[idy*n + idx];
	}
}

void cuda_Matrix_Add_REAL(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((n + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_matrix_add_real<<<block_tail, thread_tail>>>(gpu_floats_a, gpu_floats_b, m, n);
}

__global__ void cuda_scale_matrix(float * gpu_floats_a, uint32_t m, uint32_t n, float mweight)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < n && idy < m)
	{
		gpu_floats_a[idy * n + idx] = gpu_floats_a[idy * n + idx] *  mweight; //(float)log( (float)gpu_floats_a[idx]);
	}
}

void cuda_Scale_Matrix(float * gpu_floats_a, uint32_t m, uint32_t n, float mweight)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((n + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_scale_matrix<<<block_tail ,thread_tail >>>(gpu_floats_a, m, n, mweight);
}

__global__ void cuda_matrix_add_tanh(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < n && idy < m )
	{
		uint32_t col = idx ; //% n;
		float t = gpu_floats_a[idy * n + idx] + gpu_floats_b[col];
		gpu_floats_a[idy * n + idx] = tanhf(t);
	}
}

void cuda_Matrix_Add_Tanh(float * gpu_floats_a, float * gpu_floats_b, uint32_t m, uint32_t n)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;

	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((n + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_matrix_add_tanh<<<block_tail ,thread_tail >>>(gpu_floats_a,gpu_floats_b, m, n);
}

__global__ void cuda_matrix_add_vector(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < dimension && idy < batchsize )
	{
		gpu_floats_a[idy * dimension + idx] = gpu_floats_a[idy * dimension + idx] + gpu_floats_b[idx];
	}
}

void cuda_Matrix_Add_Vector(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	cuda_matrix_add_vector<<<block_tail ,thread_tail >>>(gpu_floats_a,gpu_floats_b, batchsize, dimension);
}

__global__ void cuda_matrix_rectified_vector(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < dimension && idy < batchsize )
	{
		gpu_floats_a[idy * dimension + idx] = gpu_floats_a[idy * dimension + idx] + gpu_floats_b[idx];
		if(gpu_floats_a[idy * dimension + idx] < 0)
		{
			gpu_floats_a[idy * dimension + idx] = 0;
		}
	}
}

void cuda_Matrix_Rectified_Vector(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	cuda_matrix_rectified_vector<<<block_tail ,thread_tail >>>(gpu_floats_a,gpu_floats_b, batchsize, dimension);
}


__global__ void cuda_deriv_cosine(float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float a = 0;
		float b = eps;
		float c = eps;
		for(uint32_t i=0;i<m;i++)
		{
			a += q[idx * m + i] * d[idx * m + i];
			b += q[idx * m + i] * q[idx * m + i];
			c += d[idx * m + i] * d[idx * m + i];
		}
		b = sqrtf(b);
		c = sqrtf(c);
		for(uint32_t i=0;i<m;i++)
		{
			dcq[idx * m + i] = (float)( (1 - q[idx * m + i]) * ( 1 + q[idx * m + i]) * (d[idx*m+i] * 1.0f / (b*c) - q[idx*m+i] * a * 1.0f / (b*b*b*c)) );
			dcd[idx * m + i] = (float)( (1 - d[idx * m + i]) * ( 1 + d[idx * m + i]) * (q[idx*m+i] * 1.0f / (b*c) - d[idx*m+i] * a * 1.0f / (b*c*c*c)) );
			dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;
			dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
		}
	}
}

void cuda_Deriv_Cosine( float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_deriv_cosine<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK>>>(q,d,dcq,dcd,batchsize,m,eps);
}


__global__ void cuda_deriv_dis(float * s1deriv, float * s2deriv, float * s3deriv, float * s1, float * s2, float * s3, float * dis, uint32_t batchsize, uint32_t m, float margin)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < m && idy < 3*batchsize)
	{
		uint32_t sel = idy / batchsize;
		uint32_t pos = idy % batchsize;

		if (dis[pos * 2 + 1] - dis[pos * 2] >= margin)
		{
			if (sel == 0)
				s1deriv[pos*m + idx] = 0;
			else if (sel == 1)
				s2deriv[pos*m + idx] = 0;
			else
				s3deriv[pos*m + idx] = 0;
			return;
		}

		float tem1, tem2;

		if (sel == 0)
		{
			//s1
			tem1 = s1[pos*m + idx];
			tem2 = (tem1 - s2[pos*m + idx]) / dis[pos * 2] - (tem1 - s3[pos*m + idx]) / dis[pos * 2 + 1];
			tem2 = tem2 * (1 - tem1) * (1 + tem1);
			s1deriv[pos*m + idx] = tem2 * 1.0f / batchsize;
		}
		else if (sel == 1)
		{
			//s2
			tem1 = s2[pos*m + idx];
			tem2 = (tem1 - s1[pos*m + idx]) / dis[pos * 2];
			tem2 = tem2 * (1 - tem1) * (1 + tem1);
			s2deriv[pos*m + idx] = tem2 * 1.0f / batchsize;
		}
		else
		{
			//s3
			tem1 = s3[pos*m + idx];
			tem2 = (s1[pos*m + idx] - tem1) / dis[pos * 2 + 1];
			tem2 = tem2 * (1 - tem1) * (1 + tem1);
			s3deriv[pos*m + idx] = tem2 * 1.0f / batchsize;
		}
	}
}

void cuda_Deriv_Dis(float * s1deriv, float * s2deriv, float * s3deriv, float * s1, float * s2, float * s3, float * dis, uint32_t batchsize, uint32_t m, float margin)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (batchsize*3 + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	cuda_deriv_dis<<<block_tail, thread_tail>>>(s1deriv, s2deriv, s3deriv, s1, s2, s3, dis, batchsize, m, margin);
}


__global__ void cuda_deriv_dis_linear(float * s1deriv, float * s2deriv, float * s3deriv, float * s1, float * s2, float * s3, float * dis, uint32_t batchsize, uint32_t m, float margin)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < m && idy < 3 * batchsize)
	{
		uint32_t sel = idy / batchsize;
		uint32_t pos = idy % batchsize;

		if (dis[pos * 2 + 1] - dis[pos * 2] >= margin)
		{
			if (sel == 0)
				s1deriv[pos*m + idx] = 0;
			else if (sel == 1)
				s2deriv[pos*m + idx] = 0;
			else
				s3deriv[pos*m + idx] = 0;
			return;
		}

		float tem1, tem2;

		if (sel == 0)
		{
			//s1
			tem1 = s1[pos*m + idx];
			tem2 = (tem1 - s2[pos*m + idx]) / dis[pos * 2] - (tem1 - s3[pos*m + idx]) / dis[pos * 2 + 1];
			//tem2 = tem2 * (1 - tem1) * (1 + tem1);
			s1deriv[pos*m + idx] = tem2 * 1.0f / batchsize;
		}
		else if (sel == 1)
		{
			//s2
			tem1 = s2[pos*m + idx];
			tem2 = (tem1 - s1[pos*m + idx]) / dis[pos * 2];
			//tem2 = tem2 * (1 - tem1) * (1 + tem1);
			s2deriv[pos*m + idx] = tem2 * 1.0f / batchsize;
		}
		else
		{
			//s3
			tem1 = s3[pos*m + idx];
			tem2 = (s1[pos*m + idx] - tem1) / dis[pos * 2 + 1];
			//tem2 = tem2 * (1 - tem1) * (1 + tem1);
			s3deriv[pos*m + idx] = tem2 * 1.0f / batchsize;
		}
	}
}

void cuda_Deriv_Dis_Linear(float * s1deriv, float * s2deriv, float * s3deriv, float * s1, float * s2, float * s3, float * dis, uint32_t batchsize, uint32_t m, float margin)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (batchsize * 3 + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	cuda_deriv_dis_linear<<<block_tail, thread_tail>>>(s1deriv, s2deriv, s3deriv, s1, s2, s3, dis, batchsize, m, margin);
}


__global__ void cuda_deriv_dis_rectified(float * s1deriv, float * s2deriv, float * s3deriv, float * s1, float * s2, float * s3, float * dis, uint32_t batchsize, uint32_t m, float margin, float eps)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < m && idy < 3 * batchsize)
	{
		uint32_t sel = idy / batchsize;
		uint32_t pos = idy % batchsize;

		//check if there is error
		if (dis[pos * 2 + 1] - dis[pos * 2] >= margin)
		{
			if (sel == 0)
				s1deriv[pos*m + idx] = 0;
			else if (sel == 1)
				s2deriv[pos*m + idx] = 0;
			else
				s3deriv[pos*m + idx] = 0;
			return;
		}

		float tem1, tem2;

		if (sel == 0)
		{
			//s1
			tem1 = s1[pos*m + idx];
			if (fabsf(tem1) < eps)
			{
				s1deriv[pos*m + idx] = 0;
			}
			else
			{
				tem2 = (tem1 - s2[pos*m + idx]) / dis[pos * 2] - (tem1 - s3[pos*m + idx]) / dis[pos * 2 + 1];
				//tem2 = tem2 * (1 - tem1) * (1 + tem1);
				s1deriv[pos*m + idx] = tem2 * 1.0f / batchsize;
			}

			
		}
		else if (sel == 1)
		{
			//s2
			tem1 = s2[pos*m + idx];
			if (fabsf(tem1) < eps)
			{
				s2deriv[pos*m + idx] = 0;
			}
			else
			{
				tem2 = (tem1 - s1[pos*m + idx]) / dis[pos * 2];
				//tem2 = tem2 * (1 - tem1) * (1 + tem1);
				s2deriv[pos*m + idx] = tem2 * 1.0f / batchsize;
			}
		}
		else
		{
			//s3
			tem1 = s3[pos*m + idx];
			if (fabsf(tem1) < eps)
			{
				s3deriv[pos*m + idx] = 0;
			}
			else
			{
				tem2 = (s1[pos*m + idx] - tem1) / dis[pos * 2 + 1];
				//tem2 = tem2 * (1 - tem1) * (1 + tem1);
				s3deriv[pos*m + idx] = tem2 * 1.0f / batchsize;
			}
		}
	}
}

void cuda_Deriv_Dis_Rectified(float * s1deriv, float * s2deriv, float * s3deriv, float * s1, float * s2, float * s3, float * dis, uint32_t batchsize, uint32_t m, float margin, float eps)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (batchsize * 3 + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	cuda_deriv_dis_rectified << <block_tail, thread_tail >> >(s1deriv, s2deriv, s3deriv, s1, s2, s3, dis, batchsize, m, margin, eps);
}


__global__ void cuda_calc_euclideandis(float * s1, float * s2, float * s3, float * res, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < 2*batchsize)
	{
		int row = idx / batchsize; // first row(0): distance between s1 and s2; second row(1): distance between s1 and s3
		int col = idx % batchsize;
		float * s = row > 0 ? s3 : s2;
		float tem;
		float dist = eps;

		for (uint32_t i = 0; i<m; i++)
		{
			tem = s1[col * m + i] - s[col * m + i];
			dist += tem*tem;
		}
		dist = sqrtf(dist);
		res[2 * col + row] = dist;
	}
}

void cuda_Calc_EuclideanDis(float * s1, float * s2, float * s3, float * res, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (2 * batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_calc_euclideandis<<<nBlockPerGrid, DEFAULT_THREAD_PER_BLOCK>>>(s1, s2, s3, res, batchsize, m, eps);
}


__global__ void cuda_deriv_cosine_linear(float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float a = 0;
		float b = eps;
		float c = eps;
		for(uint32_t i=0;i<m;i++)
		{
			a += q[idx * m + i] * d[idx * m + i];
			b += q[idx * m + i] * q[idx * m + i];
			c += d[idx * m + i] * d[idx * m + i];
		}
		b = sqrtf(b);
		c = sqrtf(c);
		for(uint32_t i=0;i<m;i++)
		{
			dcq[idx * m + i] = (float)( (d[idx*m+i] * 1.0f / (b*c) - q[idx*m+i] * a * 1.0f / (b*b*b*c)) );
			dcd[idx * m + i] = (float)( (q[idx*m+i] * 1.0f / (b*c) - d[idx*m+i] * a * 1.0f / (b*c*c*c)) );
			dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;
			dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
		}
	}
}

void cuda_Derive_Cosine_Linear(float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_deriv_cosine_linear<<< nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(q,d,dcq,dcd,batchsize,m,eps);
}

__global__ void cuda_deriv_cosine_rectified(float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float a = 0;
		float b = eps;
		float c = eps;
		for(uint32_t i=0;i<m;i++)
		{
			a += q[idx * m + i] * d[idx * m + i];
			b += q[idx * m + i] * q[idx * m + i];
			c += d[idx * m + i] * d[idx * m + i];
		}
		b = sqrtf(b);
		c = sqrtf(c);
		for(uint32_t i=0;i<m;i++)
		{
			if(fabsf(q[idx * m + i]) < eps)
			{
				dcq[idx * m + i]  = 0;
			}
			else
			{
				dcq[idx * m + i] = (float)( (d[idx*m+i] * 1.0f / (b*c) - q[idx*m+i] * a * 1.0f / (b*b*b*c)) );
			}
			dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;
			
			if(fabsf(d[idx * m + i]) < eps)
			{
				dcd[idx * m + i ] =0;
			}
			else
			{
				dcd[idx * m + i] = (float)( (q[idx*m+i] * 1.0f / (b*c) - d[idx*m+i] * a * 1.0f / (b*c*c*c)) );
			}
			dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
		}
	}
}
void cuda_Derive_Cosine_Rectified(float * q, float * d, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_deriv_cosine_rectified<<< nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(q,d,dcq,dcd,batchsize,m,eps);
}

//optimized version -- hxd
__global__ void cuda_deriv_cosine_ex(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float a = 0;
		float b = 0;
		float c = 0;
		float bc, a_bbbc, a_bccc, batchsizenorm;
		float * q_iter = q + idx*m;
		float * d_iter = d + neg_list[idx]*m;
		float * q_iter_end = q_iter + m;
		while(q_iter < q_iter_end)
		{
			b += (*q_iter) * (*q_iter);
			c += (*d_iter) * (*d_iter);
			a += (*q_iter++) * (*d_iter++);
		}
		b = sqrtf(b);
		c = sqrtf(c);
		bc = b*c + eps;
		a_bbbc = a/(b*b*b*c + eps);
		a_bccc = a/(b*c*c*c + eps);
		batchsizenorm = 1.0f / batchsize;

		q_iter = q + idx*m;
		d_iter = d + neg_list[idx]*m;
		q_iter_end = q_iter + m;
		float * dcq_iter = dcq + idx*m;
		float * dcd_iter = dcd + idx*m;

		while(q_iter < q_iter_end)
		{
			*dcq_iter++ = (1.0f - *q_iter) * ( 1.0f + *q_iter) * (*d_iter / bc - *q_iter * a_bbbc) * batchsizenorm;
			*dcd_iter++ = (1.0f - *d_iter) * ( 1.0f + *d_iter) * (*q_iter / bc - *d_iter * a_bccc) * batchsizenorm;
			++q_iter;
			++d_iter;
		}
	}
}

void cuda_Deriv_Cosine_EX( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_deriv_cosine_ex<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(q,d,neg_list,dcq,dcd,batchsize,m,eps);
}

__global__ void cuda_deriv_cosine_linear_ex(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float a = 0;
		float b = eps;
		float c = eps;
		uint32_t mIndex = neg_list[idx];
		for(uint32_t i=0;i<m;i++)
		{
			a += q[idx * m + i] * d[mIndex * m + i];
			b += q[idx * m + i] * q[idx * m + i];
			c += d[mIndex * m + i] * d[mIndex * m + i];
		}
		b = sqrtf(b);
		c = sqrtf(c);
		for(uint32_t i=0;i<m;i++)
		{
			dcq[idx * m + i] = (float)( (d[mIndex*m+i] * 1.0f / (b*c) - q[idx*m+i] * a * 1.0f / (b*b*b*c)) );
			dcd[idx * m + i] = (float)( (q[idx*m+i] * 1.0f / (b*c) - d[mIndex*m+i] * a * 1.0f / (b*c*c*c)) );
			dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;
			dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
		}
	}
}

void cuda_Deriv_Cosine_Linear_EX( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_deriv_cosine_linear_ex<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(q,d,neg_list,dcq,dcd,batchsize,m,eps);
}

__global__ void cuda_deriv_cosine_rectified_ex(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float a = 0;
		float b = eps;
		float c = eps;
		uint32_t mIndex = neg_list[idx];
		for(uint32_t i=0;i<m;i++)
		{
			a += q[idx * m + i] * d[mIndex * m + i];
			b += q[idx * m + i] * q[idx * m + i];
			c += d[mIndex * m + i] * d[mIndex * m + i];
		}
		b = sqrtf(b);
		c = sqrtf(c);
		for(uint32_t i=0;i<m;i++)
		{
			if(q[idx*m+i] == 0)
			{
				dcq[idx * m + i] = 0;
			}
			else
			{
				dcq[idx * m + i] = (float)( (d[mIndex*m+i] * 1.0f / (b*c) - q[idx*m+i] * a * 1.0f / (b*b*b*c)) );
			}
			dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;


			if(d[mIndex*m+i] == 0)
			{
				dcd[idx * m + i] = 0;
			}
			else
			{
				dcd[idx * m + i] = (float)( (q[idx*m+i] * 1.0f / (b*c) - d[mIndex*m+i] * a * 1.0f / (b*c*c*c)) );
			}
			dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
		}
	}
}

void cuda_Deriv_Cosine_Rectified_EX( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_deriv_cosine_rectified_ex<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(q,d,neg_list,dcq,dcd,batchsize,m,eps);
}

__global__ void cuda_deriv_tanh(float * delta, float * layer_output, uint32_t batchsize, uint32_t m)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < m && idy < batchsize )
	{
		delta[idy * m + idx] = delta[idy * m +idx] * (1 - layer_output[idy * m + idx]) * ( 1 + layer_output[idy * m + idx]);
	}
}

void cuda_Deriv_Tanh( float * delta, float * layer_output, uint32_t batchsize, uint32_t m)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize * m + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;

	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_deriv_tanh<<<block_tail ,thread_tail >>>(delta, layer_output, batchsize, m); 
}

__global__ void cuda_deriv_rectified(float * delta, float * layer_output, uint32_t batchsize, uint32_t m)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < m && idy < batchsize )
	{
		if(layer_output[idy * m + idx] == 0)
		{
			delta[idy * m + idx] = 0; // delta[idy * m +idx] ;
		}
	}
}

void cuda_Deriv_Rectified( float * delta, float * layer_output, uint32_t batchsize, uint32_t m)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize * m + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;

	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_deriv_rectified<<<block_tail ,thread_tail >>>(delta, layer_output, batchsize, m); 
}







//optimized version -- hxd
__global__ void cuda_matrix_multipy(float * delta, float * weight, float * delta_low, uint32_t batchsize, uint32_t m, uint32_t n, uint32_t inverse)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if(idx <  n && idy < batchsize)
	{
		//uint32_t row = idy; // / n;
		//uint32_t col = idx; // % n;
		float sum = 0;
		if(inverse == 1)
		{
			float * d_iter = delta + (idy * m);
			float * w_iter = weight + (idx * m);
			float * d_end_pt = d_iter + m;
			while(d_iter < d_end_pt)
			{
				sum += (*d_iter++) * (*w_iter++);
			}
		}
		else
		{
			float * d_iter = delta + (idy * m);
			float * w_iter = weight + idx;
			float * d_end_pt = d_iter + m;
			while(d_iter < d_end_pt)
			{
				sum += (*d_iter++) * (*w_iter);
				w_iter += n;
			}
		}
		delta_low[idy * n + idx] = sum;
	}
}

void cuda_Matrix_Multipy(float * delta, float * weight, float * delta_low, uint32_t batchsize, uint32_t m, uint32_t n, uint32_t inverse)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;

	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((n + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_matrix_multipy<<<block_tail ,thread_tail >>>(delta, weight, delta_low, batchsize, m, n, inverse); 
}

__global__ void cuda_cosine_similarity(float * a, float * b, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t mindex, 
									   uint32_t batchsize, uint32_t dimension, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float sumxx = 0;
		float sumyy = 0;
		float sumxy = 0;
		for(uint32_t i=0;i<dimension;i++)
		{
			sumxx += a[idx * dimension + i] * a[idx * dimension + i];
			sumyy += b[idx * dimension + i] * b[idx * dimension + i];
			sumxy += a[idx * dimension + i] * b[idx * dimension + i];
		}
		c[mindex * BATCHSIZE + idx] = (float)( sumxy * 1.0f / (sqrtf( (float)(sumxx * sumyy)) + eps) );
	}

}
void cuda_Cosine_Similarity(float * a, float * b, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t mindex, 
									   uint32_t batchsize, uint32_t dimension, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_cosine_similarity<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(a,b,c,nTrial,BATCHSIZE,mindex,batchsize, dimension, eps);
}


__global__ void cuda_innerproduct_similarity(float * a, float * b, float * c, uint32_t batchsize, uint32_t dimension)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float sumxy = 0;
		for(uint32_t i=0;i<dimension;i++)
		{
			sumxy += a[idx * dimension + i] * b[idx * dimension + i];
		}
		c[idx] = (float)(sumxy * 1.0f);
	}

}
void cuda_InnerProduct_Similarity(float * a, float * b, float * c, uint32_t batchsize, uint32_t dimension)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_innerproduct_similarity<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(a, b, c, batchsize, dimension);
}

//optimized version -- hxd
__global__ void cuda_cosine_similarity_ex(float * a, float * b,uint32_t * neg_list, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t mindex, 
									   uint32_t batchsize, uint32_t dimension, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float sumxx = 0;
		float sumyy = 0;
		float sumxy = 0;
		float * a_iter = a + (idx * dimension);
		float * b_iter = b + (neg_list[idx] * dimension);
		float * a_iter_end = a_iter + dimension;
		while(a_iter < a_iter_end)
		{
			sumxx += (*a_iter) * (*a_iter);
			sumyy += (*b_iter) * (*b_iter);
			sumxy += (*a_iter++) * (*b_iter++);
		}
		c[mindex * BATCHSIZE + idx] = (float)( sumxy / ((float)sqrtf(sumxx * sumyy) + eps) );
	}
}


void cuda_Cosine_Similarity_EX(float * a, float * b, uint32_t * neg_list, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t mindex, 
									   uint32_t batchsize, uint32_t dimension, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_cosine_similarity_ex<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(a,b,neg_list,c,nTrial,BATCHSIZE,mindex,batchsize, dimension, eps);
}

__global__ void cuda_cal_alpha(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < (nTrial-1)*batchsize)
	{
		uint32_t row = idx / batchsize;
		uint32_t col = idx % batchsize;
		alpha[row * BATCHSIZE + col + BATCHSIZE] = expf( (float)(-gamma * (alpha[col] - alpha[row * BATCHSIZE + col + BATCHSIZE]))) ; 
	}
}

__global__ void cuda_cal_alpha_sum(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma, uint32_t init)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float sum = init;
		for(uint32_t i=1;i<nTrial;i++)
		{
			sum += alpha[i * BATCHSIZE + idx]; 
		}
		alpha[idx] =  sum; 
	}
}

__global__ void cuda_cal_alpha_norm(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < (nTrial-1)*batchsize)
	{
		uint32_t row = idx / batchsize;
		uint32_t col = idx % batchsize;
		alpha[row * BATCHSIZE + col + BATCHSIZE] = (float)((gamma * alpha[row * BATCHSIZE + col + BATCHSIZE])/ alpha[col]);
			//expf( (float)(-gamma * (alpha[col] - alpha[row * BATCHSIZE + col + BATCHSIZE]))) ; 
	}
}


void  cuda_Calculate_Alpha(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;

	uint32_t nBlockPerGrid_1 = ((nTrial-1)*batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_cal_alpha<<<nBlockPerGrid_1 ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, nTrial,BATCHSIZE,batchsize,gamma);

	uint32_t nBlockPerGrid_2 = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_cal_alpha_sum<<<nBlockPerGrid_2 ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, nTrial,BATCHSIZE,batchsize,gamma, 1);

	cuda_cal_alpha_norm<<<nBlockPerGrid_1 ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, nTrial,BATCHSIZE,batchsize,gamma);
	cuda_cal_alpha_sum<<<nBlockPerGrid_2 ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, nTrial,BATCHSIZE,batchsize,gamma, 0);
}

__global__ void cuda_cal_alpha_norm_MXE(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < (nTrial-1)*batchsize)
	{
		uint32_t row = idx / batchsize;
		uint32_t col = idx % batchsize;
		alpha[row * BATCHSIZE + col + BATCHSIZE] = (float)((gamma * alpha[row * BATCHSIZE + col + BATCHSIZE])/ alpha[col]/ alpha[col]);
			//expf( (float)(-gamma * (alpha[col] - alpha[row * BATCHSIZE + col + BATCHSIZE]))) ; 
	}
}

void  cuda_Calculate_Alpha_MXE(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;

	uint32_t nBlockPerGrid_1 = ((nTrial-1)*batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_cal_alpha<<<nBlockPerGrid_1 ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, nTrial,BATCHSIZE,batchsize,gamma);

	uint32_t nBlockPerGrid_2 = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_cal_alpha_sum<<<nBlockPerGrid_2 ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, nTrial,BATCHSIZE,batchsize,gamma, 1);

	cuda_cal_alpha_norm_MXE<<<nBlockPerGrid_1 ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, nTrial,BATCHSIZE,batchsize,gamma);
	cuda_cal_alpha_sum<<<nBlockPerGrid_2 ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, nTrial,BATCHSIZE,batchsize,gamma, 0);
}

__global__ void cuda_cal_alpha_PAIRRANK(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float msum = 0;
		for(int n = 1; n < nTrial; n++)
		{
			float a = gamma * (1.0f - 1.0f / (1 + expf(- gamma * (alpha[idx] - alpha[n * BATCHSIZE + idx] ))));
			alpha[n * BATCHSIZE + idx] =  a;
			msum += a;
		}
		alpha[idx] = msum;
	}
}

void cuda_Calculate_Alpha_PAIRRANK(float * alpha, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	
	cuda_cal_alpha_PAIRRANK<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, nTrial, BATCHSIZE, batchsize, gamma);
}

__global__ void cuda_cal_alpha_nce(float * alpha, float* dist, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)	
	{
		alpha[idx] = gamma - gamma / (1.0f + (nTrial - 1) * expf(dist[idx] - gamma * alpha[idx] + gamma)); //+gamma is from hxd, sd doesn't have this
	}
	else if(idx < nTrial*batchsize)
	{
		uint32_t row = idx / batchsize;
		uint32_t col = idx % batchsize;
		alpha[row * BATCHSIZE + col] = gamma / (1.0f + (nTrial - 1) * expf(dist[row * BATCHSIZE + col] - gamma * alpha[row * BATCHSIZE + col] + gamma)); //+gamma is from hxd, sd doesn't have this
	}
}

void cuda_Calculate_Alpha_NCE(float* alpha, float* dist, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (nTrial*batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	
	cuda_cal_alpha_nce<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, dist, nTrial,BATCHSIZE,batchsize,gamma);
}

__global__ void cuda_cal_alpha_nce2(float * alpha, float* dist, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < nTrial*batchsize)
	{
		uint32_t row = idx / batchsize;
		uint32_t col = idx % batchsize;
		float s = 1.0f / (1.0f + (nTrial - 1) * expf(dist[row * BATCHSIZE + col] - gamma * alpha[row * BATCHSIZE + col] + gamma)); //+gamma is from hxd, sd doesn't have this
		alpha[row * BATCHSIZE + col] = gamma * s * (1.0f - s);
	}
}

void cuda_Calculate_Alpha_NCE2(float* alpha, float* dist, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, float gamma)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (nTrial*batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	
	cuda_cal_alpha_nce2<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(alpha, dist, nTrial,BATCHSIZE,batchsize,gamma);
}

__global__ void cuda_fillout_dist_nce(float* dist, uint32_t* neg_list, uint32_t nTrailPlus1, uint32_t BATCH_SIZE, uint32_t mindex, uint32_t batchsize)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		uint32_t mtindex = neg_list[idx];
		dist[mindex * BATCH_SIZE + idx] = dist[mtindex];
	}
}

void cuda_FillOut_Dist_NCE(float* dist, uint32_t* neg_list, uint32_t nTrailPlus1, uint32_t BATCH_SIZE, uint32_t mindex, uint32_t batchsize)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_fillout_dist_nce<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(dist, neg_list, nTrailPlus1, BATCH_SIZE, mindex, batchsize);
}

//optimized version -- hxd
__global__ void cuda_matrix_product(float * a1, float * b1, float * a2, float * b2, float * a3, float * b3, float * c, uint32_t batchsize, uint32_t m, uint32_t n)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if(idx < n && idy < m )
	{
		float sum = 0;
		for (uint32_t i = 0; i < batchsize; i++)
		{
			sum += a1[m*i + idy] * b1[n*i + idx];
			sum += a2[m*i + idy] * b2[n*i + idx];
			sum += a3[m*i + idy] * b3[n*i + idx];
		}
		//uint32_t row = idy; // / n;
		//uint32_t col = idx;// % n;
		//float *a_iter = a+row;
		//float *b_iter = b+col;
		//float *a_end_pt = a_iter + (m*batchsize);
		//while(a_iter < a_end_pt)
		//{
		//	sum += (*a_iter) * (*b_iter);
		//	a_iter += m;
		//	b_iter += n;
		//}
		c[idy * n + idx] = sum;
	}
}


void cuda_Matrix_Product(float * a1, float * b1, float * a2, float * b2, float * a3, float * b3, float * c, uint32_t batchsize, uint32_t m, uint32_t n)
			//, uint32_t kept, float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = ( m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((n + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( m + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_matrix_product<<<block_tail, thread_tail>>>(a1, b1, a2, b2, a3, b3, c, batchsize, m, n);
		//, kept, alpha, ntrial, BATCH_SIZE, alpha_index);
}

__global__ void cuda_convolution_matrix_product_INTEX(float * deriv1, int * maxpooling_index1, float * deriv2, int * maxpooling_index2, float * deriv3, int * maxpooling_index3, float * wordLT, int * Word_Index1, int * Word_Index2, int * Word_Index3, int win_size,
										int batchsize, int output_dimension, float * grad, int Feature_Dimension, int weightDim)
										//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < output_dimension && idy < weightDim)
	{
		float sum = 0;
		int target_word1, target_word2, target_word3, widx1, widx2, widx3, wordpos, offset, precompIdx;
		wordpos = idy / Feature_Dimension;
		offset = idy % Feature_Dimension;
		for(int b=0;b<batchsize;b++)
		{
			precompIdx = b * output_dimension + idx;
			target_word1 = maxpooling_index1[precompIdx];
			target_word2 = maxpooling_index2[precompIdx];
			target_word3 = maxpooling_index3[precompIdx];


			int widx1 = Word_Index1[target_word1 + wordpos];
			int widx2 = Word_Index2[target_word2 + wordpos];
			int widx3 = Word_Index3[target_word3 + wordpos];
			
			sum += deriv1[precompIdx] * wordLT[Feature_Dimension * widx1 + offset];
			sum += deriv2[precompIdx] * wordLT[Feature_Dimension * widx2 + offset];
			sum += deriv3[precompIdx] * wordLT[Feature_Dimension * widx3 + offset];
		}
		grad[idy * output_dimension + idx] = sum;
	}
}

void cuda_Convolution_Matrix_Product_INTEX(float * deriv1, int * maxpooling_index1, float * deriv2, int * maxpooling_index2, float * deriv3, int * maxpooling_index3, float * wordLT, 
				int * Word_Index1, int * Word_Index2, int * Word_Index3, int win_size, int batchsize, int output_dimension, float * grad, int Feature_Dimension)
										//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	int weightDim = Feature_Dimension * win_size;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((output_dimension + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK, (weightDim + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK);

	cuda_convolution_matrix_product_INTEX<<<block_tail, thread_tail>>>(deriv1, maxpooling_index1, deriv2, maxpooling_index2, deriv3, maxpooling_index3, wordLT, Word_Index1, Word_Index2, Word_Index3, win_size,
					batchsize, output_dimension, grad, Feature_Dimension, weightDim);
								//,alpha, ntrial, BATCH_SIZE, alpha_index); 
}


__global__ void cuda_multiconv_matrix_product_INTEX(float * deriv1, int * maxpooling_index1, float * deriv2, int * maxpooling_index2, float * deriv3, int * maxpooling_index3, float * wordLT, int * Word_Index1, int * Word_Index2, int * Word_Index3, int win_size,
	int batchsize, int output_dimension, float * grad, int Feature_Dimension, int weightDim, int currOuputDim, int pastOutdim)
	//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < weightDim && idy < currOuputDim)
	{
		float sum = 0;
		int target_word1, target_word2, target_word3, widx1, widx2, widx3, wordpos, offset, precompIdx;

		wordpos = idx / Feature_Dimension;
		offset = idx % Feature_Dimension;
		for (int b = 0; b<batchsize; b++)
		{
			precompIdx = b * output_dimension + idy + pastOutdim;
			target_word1 = maxpooling_index1[precompIdx];
			target_word2 = maxpooling_index2[precompIdx];
			target_word3 = maxpooling_index3[precompIdx];

			int widx1 = Word_Index1[target_word1 + wordpos];
			int widx2 = Word_Index2[target_word2 + wordpos];
			int widx3 = Word_Index3[target_word3 + wordpos];

			sum += deriv1[precompIdx] * wordLT[Feature_Dimension * widx1 + offset];
			sum += deriv2[precompIdx] * wordLT[Feature_Dimension * widx2 + offset];
			sum += deriv3[precompIdx] * wordLT[Feature_Dimension * widx3 + offset];
		}
		grad[idy * weightDim + idx] = sum;
	}
}

void cuda_MultiConv_Matrix_Product_INTEX(float * deriv1, int * maxpooling_index1, float * deriv2, int * maxpooling_index2, float * deriv3, int * maxpooling_index3, float * wordLT,
	int * Word_Index1, int * Word_Index2, int * Word_Index3, int batchsize, int output_dimension, float * grad, int Feature_Dimension, int winsize, int fmsize, int accu, int accu_para) // the last two pointers are on host
	//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	int weightDim = Feature_Dimension * winsize;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((weightDim + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK, (fmsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK);

	cuda_multiconv_matrix_product_INTEX<<<block_tail, thread_tail>>>(deriv1, maxpooling_index1, deriv2, maxpooling_index2, deriv3, maxpooling_index3, wordLT, Word_Index1, Word_Index2, Word_Index3, winsize,
		batchsize, output_dimension, (grad + accu_para), Feature_Dimension, weightDim, fmsize, accu);	
}



__global__ void cuda_multiconv_compute_wvderiv(float * deriv, int * maxpooling_index, float * weight, int batchsize, int output_dimension, float * grad, int Feature_Dimension, int * winsizes, int * fmsizes)
	//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < Feature_Dimension && idy < batchsize)
	{
		int currFilterset = 0, counter = 0, accuoffset = 0, currweightDim;
		float cacheDeriv;
		int wordIdx, i;
		for (int b = 0; b < output_dimension; b++)
		{
			if (counter >= fmsizes[currFilterset])
			{
				counter = 0;
				accuoffset += Feature_Dimension * winsizes[currFilterset] * fmsizes[currFilterset];
				currFilterset++;
			}
			currweightDim = Feature_Dimension * winsizes[currFilterset];
			cacheDeriv = deriv[idy*output_dimension + b];
			wordIdx = maxpooling_index[idy*output_dimension + b];
			for (i = 0; i < winsizes[currFilterset]; i++)
			{
				grad[(wordIdx + i)*Feature_Dimension + idx] += cacheDeriv * weight[accuoffset + counter*currweightDim + (i*Feature_Dimension + idx)];
			}
			counter++;
		}
	}
}

void cuda_MultiConv_Compute_WVDERIV(float * deriv, int * maxpooling_index, float * weight, int batchsize, int output_dimension, float * grad, int Feature_Dimension, int * winsizes, int * fmsizes)
	//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	
	dim3 block_tail((Feature_Dimension + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK, (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK);

	cuda_multiconv_compute_wvderiv<<<block_tail, thread_tail>>>(deriv, maxpooling_index, weight, batchsize, output_dimension, grad, Feature_Dimension, winsizes, fmsizes);

	//,alpha, ntrial, BATCH_SIZE, alpha_index); 
}


__global__ void cuda_conv_compute_wvderiv(float * deriv, int * maxpooling_index, float * weight, int batchsize, int output_dimension, float * grad, int Feature_Dimension, int winsize)
//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < Feature_Dimension && idy < batchsize)
	{
		float cacheDeriv;
		int wordIdx, i;
		for (int b = 0; b < output_dimension; b++)
		{
			cacheDeriv = deriv[idy*output_dimension + b];
			wordIdx = maxpooling_index[idy*output_dimension + b];
			for (i = 0; i < winsize; i++)
			{
				grad[(wordIdx + i)*Feature_Dimension + idx] += cacheDeriv * weight[(i*Feature_Dimension + idx)*output_dimension + b];
			}
		}
	}
}

void cuda_Conv_Compute_WVDERIV(float * deriv, int * maxpooling_index, float * weight, int batchsize, int output_dimension, float * grad, int Feature_Dimension, int winsize)
//,float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);

	dim3 block_tail((Feature_Dimension + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK, (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK);

	cuda_conv_compute_wvderiv<<<block_tail, thread_tail>>>(deriv, maxpooling_index, weight, batchsize, output_dimension, grad, Feature_Dimension, winsize);

	//,alpha, ntrial, BATCH_SIZE, alpha_index); 
}




__global__ void cuda_convolution_matrix_multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Word_Index, uint32_t * Word_Margin, uint32_t Word_SeqLen,
												   float * wordLT,
												   float * con_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < output_dimension && idy < Word_SeqLen)
	{
		uint32_t mSmp_idx = Word_Margin[idy];
		uint32_t wordEnd = Smp_Index[mSmp_idx];
		uint32_t wordBegin = 0;
		if (mSmp_idx > 0)
			wordBegin = Smp_Index[mSmp_idx - 1];
		if (idy >= wordBegin && idy <= (wordEnd - win_size))
		{
			output[idy * output_dimension + idx] = 0;
			float sum = 0;

			for (int w = 0; w < win_size; w++)
			{
				uint32_t wordIdx = Word_Index[idy + w];
				// get its vector from word lookup table
				for (uint32_t i = 0; i < Feature_dimension; i++)
				{
					sum += wordLT[wordIdx*Feature_dimension + i] * con_weight[(w * Feature_dimension + i)*output_dimension + idx];
				}
			}
			output[idy * output_dimension + idx] = sum;
		}		
	}
}

void cuda_Convolution_Matrix_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Word_Index, uint32_t * Word_Margin, uint32_t Word_SeqLen, float * wordLT,
												   float * con_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((output_dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (Word_SeqLen + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = ( m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_convolution_matrix_multiply_INTEX<<<block_tail, thread_tail>>>(Smp_Index, batchsize, Word_Index, Word_Margin, Word_SeqLen, wordLT, con_weight, output, Feature_dimension, output_dimension, win_size);
}


__global__ void cuda_multiconv_matrix_multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Word_Index, uint32_t * Word_Margin, uint32_t Word_SeqLen,
														float * wordLT, float * con_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t * win_sizes, uint32_t * fm_sizes)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idx < output_dimension && idy < Word_SeqLen)
	{
		int filterClass = 0;
		uint32_t idx_offset = idx;
		uint32_t weightOffset = 0;
		while (idx_offset >= fm_sizes[filterClass])
		{
			weightOffset += Feature_dimension * win_sizes[filterClass] * fm_sizes[filterClass];
			idx_offset = idx_offset - fm_sizes[filterClass];
			filterClass++;
		}
		
		uint32_t win_size = win_sizes[filterClass];
		uint32_t mSmp_idx = Word_Margin[idy];
		uint32_t wordEnd = Smp_Index[mSmp_idx];
		uint32_t wordBegin = 0;
		if (mSmp_idx > 0)
			wordBegin = Smp_Index[mSmp_idx - 1];
		if (idy >= wordBegin && idy <= (wordEnd - win_size))
		{
			output[idy * output_dimension + idx] = 0;
			float sum = 0;
			uint32_t woffset = weightOffset + idx_offset * (win_size * Feature_dimension);
			for (int w = 0; w < win_size; w++)
			{
				uint32_t wordIdx = Word_Index[idy + w];
				
				// get its vector from word lookup table
				for (uint32_t i = 0; i < Feature_dimension; i++)
				{
					sum += wordLT[wordIdx*Feature_dimension + i] * con_weight[woffset + w * Feature_dimension + i];
				}
			}
			output[idy * output_dimension + idx] = sum;
		}
	}
}

void cuda_MultiConv_Matrix_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Word_Index, uint32_t * Word_Margin, uint32_t Word_SeqLen, float * wordLT,
	float * con_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t * win_sizes, uint32_t * fm_sizes)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((output_dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (Word_SeqLen + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = ( m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_multiconv_matrix_multiply_INTEX<<<block_tail, thread_tail>>>(Smp_Index, batchsize, Word_Index, Word_Margin, Word_SeqLen, wordLT, con_weight, output, Feature_dimension, output_dimension, win_sizes, fm_sizes);
}

__global__ void cuda_max_pooling(float * pooling_feas, int * Smp_Index, int batchsize, float * output, int * maxpooling_index, int output_dimension, int win_size)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if(idy < batchsize && idx < output_dimension)
	{
		//output[idy * output_dimension + idx] = 0;
		uint32_t col_end = Smp_Index[idy] - win_size;
		uint32_t col_begin = 0;
		if(idy > 0)
		{
			col_begin = Smp_Index[idy-1];
		}
		float max_value = 0;
		int max_index = -1;
		for(uint32_t i=col_begin;i<=col_end; i++)
		{
			if(max_index == -1 || pooling_feas[i * output_dimension + idx] > max_value )
			{
				max_value = pooling_feas[i * output_dimension + idx];
				max_index = i;
			}
		}
		output[idy * output_dimension + idx] = max_value;
		maxpooling_index[idy * output_dimension + idx] = max_index;
	}
}

void cuda_Max_Pooling(float * pooling_feas, int * Smp_Index, int batchsize, float * output,int * maxpooling_index, int output_dimension, int win_size)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((output_dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	cuda_max_pooling<<<block_tail, thread_tail>>>(pooling_feas, Smp_Index, batchsize, output, maxpooling_index, output_dimension, win_size); 
}


__global__ void cuda_multi_max_pooling(float * pooling_feas, int * Smp_Index, int batchsize, float * output, int * maxpooling_index, int output_dimension, int * win_sizes, int * fm_sizes)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if (idy < batchsize && idx < output_dimension)
	{
		int filterClass = 0;
		uint32_t idx_offset = idx;
		while (idx_offset >= fm_sizes[filterClass])
		{
			idx_offset = idx_offset - fm_sizes[filterClass];
			filterClass++;
		}

		uint32_t win_size = win_sizes[filterClass];
		//output[idy * output_dimension + idx] = 0;
		uint32_t col_end = Smp_Index[idy] - win_size;
		uint32_t col_begin = 0;
		if (idy > 0)
		{
			col_begin = Smp_Index[idy - 1];
		}
		float max_value = 0;
		int max_index = -1;
		for (uint32_t i = col_begin; i <= col_end; i++)
		{
			if (max_index == -1 || pooling_feas[i * output_dimension + idx] > max_value)
			{
				max_value = pooling_feas[i * output_dimension + idx];
				max_index = i;
			}
		}
		output[idy * output_dimension + idx] = max_value;
		maxpooling_index[idy * output_dimension + idx] = max_index;
	}
}

void cuda_Multi_Max_Pooling(float * pooling_feas, int * Smp_Index, int batchsize, float * output, int * maxpooling_index, int output_dimension, int * win_sizes, int * fm_sizes)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((output_dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	cuda_multi_max_pooling<<<block_tail, thread_tail>>>(pooling_feas, Smp_Index, batchsize, output, maxpooling_index, output_dimension, win_sizes, fm_sizes);
}




__global__ void cuda_seq_sparse_matrix_multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Seg_Index, uint32_t * Seg_Margin, float * Seg_Len, 
															  uint32_t seg_size, uint32_t * Fea_Index, 
												   float * Fea_Value, uint32_t elementsize, 
												   float * mul_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	if(idx < output_dimension && idy < batchsize)
	{
		uint32_t seg_end = Smp_Index[idy];
		uint32_t seg_begin = 0;
		if(idy > 0)
		{
			seg_begin = Smp_Index[idy-1];
		}

		float sum = 0;
		for(uint32_t word_idx = seg_begin; word_idx < seg_end; ++word_idx)
		{
			uint32_t col_end = Seg_Index[word_idx];
			uint32_t col_begin = 0;
			if(word_idx > 0)
			{
				col_begin = Seg_Index[word_idx - 1];
			}
			for(uint32_t i=col_begin;i<col_end; ++i)
			{
				uint32_t fea_idx = Fea_Index[i];
				sum += Fea_Value[i] * mul_weight[((word_idx - seg_begin) * Feature_dimension + fea_idx) * output_dimension + idx];
			}
		}
		output[idy * output_dimension + idx] = sum;
	}
}

/*
	Added by xinson, 2/17/2014
	This version still computes sparse matrix (batch * input)  multiples a dense matrix. However, each rwo of the sparse matrix is more than just BOW; it is a sequence of BOW. Put it another way, the 
	sparse matrix has exactly the same structure as what is used in Convolutional_Sparse_Matrix_Multiply_INTEX. As a result, the dense matrix (mul_weight) is of size (Feature_dimension * win_size) * output_dimension,
	where win_size is how many words per input sequence instance. Note that all input should have exactly the same number of words. One word is represented as an instance of BOW.
*/
void cuda_SEQ_Sparse_Matrix_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Seg_Index, uint32_t * Seg_Margin, float * Seg_Len, uint32_t seg_size, uint32_t * Fea_Index, 
												   float * Fea_Value, uint32_t elementsize, 
												   float * mul_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((output_dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = ( m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_seq_sparse_matrix_multiply_INTEX<<<block_tail, thread_tail>>>(Smp_Index, batchsize, Seg_Index, Seg_Margin, Seg_Len, seg_size, Fea_Index,  Fea_Value, elementsize, mul_weight, output, Feature_dimension, output_dimension,win_size);
}

__global__ void cuda_seq_sparse_matrix_transpose_multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Seg_Index, uint32_t * Seg_Margin, float * Seg_Len, 
															  uint32_t seg_size, uint32_t * Fea_Index, 
												   float * Fea_Value, uint32_t elementsize, 
												   float * mul_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;	
	if(idx < output_dimension)
	{
		uint32_t seg_begin = 0;
		for(uint32_t sample = 0; sample < batchsize; ++sample)
		{
			uint32_t seg_end = Smp_Index[sample];
			float sum = 0;
			for(uint32_t word_idx = seg_begin; word_idx < seg_end; ++word_idx)
			{
				uint32_t col_end = Seg_Index[word_idx];
				uint32_t col_begin = 0;
				if(word_idx > 0)
				{
					col_begin = Seg_Index[word_idx - 1];
				}
				for(uint32_t i=col_begin;i<col_end; ++i)
				{
					uint32_t fea_idx = Fea_Index[i];

					mul_weight[((word_idx - seg_begin) * Feature_dimension + fea_idx) * output_dimension + idx] += Fea_Value[i] * output[sample * output_dimension + idx];
				}
			}
			seg_begin = seg_end;
		}
	}
}

/*
	Added by xinson, 2/17/2014
	Given the same two inputs of an sparse matrix A (indexed by rows, size: batch * X), and a dense matrix B (size: batch * Y), computing C = A^T * B (size: X * Y).
	Although we compute the transpose of A multiplied by B, the code does not perform sparse transpose and indexing at all. 
	Instead, it partitioned along the columns of the result C matrix.	
	float * output is B.
	float * mul_weight is C.
	Zero initialization/clear on C is required in advance.
*/
void cuda_SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(uint32_t * Smp_Index, uint32_t batchsize, uint32_t * Seg_Index, uint32_t * Seg_Margin, float * Seg_Len, uint32_t seg_size, uint32_t * Fea_Index, 
												   float * Fea_Value, uint32_t elementsize, 
												   float * mul_weight, float * output, uint32_t Feature_dimension, uint32_t output_dimension, uint32_t win_size)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((output_dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = ( m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_seq_sparse_matrix_transpose_multiply_INTEX<<<block_tail, thread_tail>>>(Smp_Index, batchsize, Seg_Index, Seg_Margin, Seg_Len, seg_size, Fea_Index,  Fea_Value, elementsize, mul_weight, output, Feature_dimension, output_dimension,win_size);
}

__global__ void cuda_matrix_weightadd(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension, float * mweight, int start,  int keep)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < dimension && idy < batchsize)
	{
		if(keep != 0)
		{
			gpu_floats_a[idy*dimension+idx] += keep * gpu_floats_b[idy*dimension+idx] * mweight[start + idy];
		}
		else
		{
			gpu_floats_a[idy*dimension+idx] = gpu_floats_b[idy*dimension+idx] * mweight[start + idy];
		}
	}
}

void cuda_Matrix_WeightAdd(float * gpu_floats_a, float * gpu_floats_b, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_matrix_weightadd<<<block_tail ,thread_tail >>>(gpu_floats_a, gpu_floats_b, batchsize, dimension, mweight,start, keep);
}

__global__ void cuda_matrix_weightadd_ex(float * gpu_floats_a, float * gpu_floats_b, int * inver_neg_index, int * inver_neg_value, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < dimension && idy < batchsize)
	{
		int col_end = inver_neg_index[idy];
		int col_begin = 0;
		if(idy > 0)
		{
			col_begin = inver_neg_index[idy - 1];
		}
		float sum = 0;
		for(int i=col_begin; i<col_end; i++)
		{
			int row = inver_neg_value[i];
			sum += gpu_floats_b[row * dimension + idx] * mweight[start + row];
		}
		if(keep != 0)
		{
			gpu_floats_a[idy*dimension+idx] += keep * sum;
		}
		else
		{
			gpu_floats_a[idy*dimension+idx] =sum;
		}
	}
}

void cuda_Matrix_WeightAdd_EX(float * gpu_floats_a, float * gpu_floats_b, int * inver_neg_index, int * inver_neg_value, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_matrix_weightadd_ex<<<block_tail ,thread_tail >>>(gpu_floats_a, gpu_floats_b, inver_neg_index, inver_neg_value, batchsize, dimension, mweight, start, keep);
}

__global__ void cuda_sparse2dense_matrix(int * Smp_Idx, int * Fea_Idx, float * Fea_Value, float * matrix, int batchsize, int outputDimension)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		int end = Smp_Idx[idx];
        int begin = idx >= 1 ? Smp_Idx[idx - 1] : 0;
        for (int k = begin; k < end; k++)
        {
			matrix[idx * outputDimension + Fea_Idx[k]] = Fea_Value[k];
        }
	}
}

void cuda_Sparse2Dense_Matrix(int * Smp_Idx, int * Fea_Idx, float * Fea_Value, float * matrix, int batchsize, int outputDimension)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_BLOCK);
	dim3 block_tail((batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK);

	cuda_sparse2dense_matrix<<<block_tail,thread_tail>>>(Smp_Idx, Fea_Idx, Fea_Value, matrix, batchsize, outputDimension); 
}

__global__ void cuda_matrix_aggragate(float * a1, float * a2, float * a3, float * b, uint32_t batchsize, uint32_t m)
						//uint32_t kept, float * alpha, uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < m)
	{
		float sum = 0;
		for(uint32_t i=0;i<batchsize;i++)
		{
			sum += a1[i * m + idx] + a2[i * m + idx] + a3[i * m + idx]; //* alpha[alpha_index * BATCH_SIZE + i];
		}
		b[idx] = sum;
	}
}

void cuda_Matrix_Aggragate(float * a1, float * a2, float * a3, float * b, uint32_t batchsize, uint32_t m)
					//, uint32_t kept, float * alpha, 
						//		  uint32_t ntrial, uint32_t BATCH_SIZE, uint32_t alpha_index)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (m + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_matrix_aggragate<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(a1,a2,a3,b,batchsize,m ); //,kept, alpha, ntrial, BATCH_SIZE, alpha_index);
}


__global__ void cuda_matrix_add_offset(float * a, uint32_t offset_a, float * b, uint32_t offset_b, int len, float mweight)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < len)
	{
		a[offset_a + idx] += b[offset_b + idx] * mweight ; //* alpha[alpha_index * BATCH_SIZE + i];
	}
}

void cuda_Matrix_Add_OFFSET(float * gpu_floats_a, uint32_t offset_a, float * gpu_floats_b, uint32_t offset_b, int len, float mweight)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (len + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	cuda_matrix_add_offset<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(gpu_floats_a, offset_a, gpu_floats_b, offset_b, len, mweight);
}

cublasHandle_t global_handle; 

void cublas_Init()
{
	cublasCreate(&global_handle);
}

void cublas_Destroy()
{
	cublasDestroy(global_handle); 
}

void cublas_Sasum(float *x, int len, int norm, float * result)
{
	cublasSasum(global_handle, len , x , norm, result);
}

void cublas_Matrix_Multipy(float * delta, float * weight, float * delta_low, uint32_t batchsize, uint32_t m, uint32_t n, uint32_t inverse)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	float al = 1.0f;
	float bet = 0;
	if(inverse == 0)
	{
		cublasSgemm(global_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, batchsize, m, &al, weight, n, delta, m, &bet, delta_low, n);     
	}
	else
	{
		cublasSgemm(global_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, batchsize, m, &al, weight, m, delta, m, &bet, delta_low, n);     
	}
}


//optimized version -- hxd & yeshen
__global__ void cuda_cosine_similarity_ex_full(float * a, float * b, uint32_t * neg_list, float * c, uint32_t nTrial, uint32_t BATCHSIZE, 
									   uint32_t batchsize, uint32_t dimension, float eps)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize && idy < nTrial)
	{
		float sumxx = 0;
		float sumyy = 0;
		float sumxy = 0;
		float * a_iter = a + (idx * dimension);
		float * b_iter = b + (neg_list[idy * BATCHSIZE + idx] * dimension);
		float * a_iter_end = a_iter + dimension;
		while(a_iter < a_iter_end)
		{
			sumxx += (*a_iter) * (*a_iter);
			sumyy += (*b_iter) * (*b_iter);
			sumxy += (*a_iter++) * (*b_iter++);
		}
		c[ (idy + 1) * BATCHSIZE + idx] = (float)( sumxy / ((float)sqrtf(sumxx * sumyy) + eps) );
	}
}


void cuda_Cosine_Similarity_EX_Full(float * a, float * b, uint32_t * neg_list, float * c, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float eps)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;

	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( nTrial + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_cosine_similarity_ex_full<<<block_tail , thread_tail >>>(a, b, neg_list, c, nTrial, BATCHSIZE, batchsize, dimension, eps);
}



__global__ void cuda_fillout_dist_nce_full(float* dist, uint32_t* neg_list, uint32_t nTrail, uint32_t BATCH_SIZE, uint32_t batchsize)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize && idy < nTrail)
	{
		uint32_t mtindex = neg_list[idy * BATCH_SIZE + idx];
		dist[BATCH_SIZE + idy * BATCH_SIZE + idx] = dist[mtindex];
	}
}

void cuda_FillOut_Dist_NCE_Full(float* dist, uint32_t* neg_list, uint32_t nTrail, uint32_t BATCH_SIZE, uint32_t batchsize)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( nTrail + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_fillout_dist_nce_full<<<block_tail , thread_tail >>>(dist, neg_list, nTrail, BATCH_SIZE, batchsize);
}

//optimized version -- hxd & yeshen.
__global__ void cuda_deriv_cosine_ex_full(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < batchsize && idy < nTrail)
	{
		float a = 0;
		float b = 0;
		float c = 0;
		float bc, a_bbbc, a_bccc, batchsizenorm;
		float * q_iter = q + idx*m;
		float * d_iter = d + neg_list[idy * BATCHSIZE + idx] * m;
		float * q_iter_end = q_iter + m;
		
		float * q_iter_P = q_iter;
		float * d_iter_P = d_iter;
		float * q_iter_end_P = q_iter_end;

		while(q_iter < q_iter_end)
		{
			b += (*q_iter) * (*q_iter);
			c += (*d_iter) * (*d_iter);
			a += (*q_iter++) * (*d_iter++);
		}
		b = sqrtf(b);
		c = sqrtf(c);
		bc = b*c + eps;
		a_bbbc = a/(b*b*b*c + eps);
		a_bccc = a/(b*c*c*c + eps);

		batchsizenorm = 1.0f / batchsize;

		q_iter = q_iter_P;
		d_iter = d_iter_P;
		q_iter_end = q_iter_end_P;

		float * dcq_iter = dcq + idy * (BATCHSIZE * m) + idx * m;
		float * dcd_iter = dcd + idy * (BATCHSIZE * m) + idx * m;

		while(q_iter < q_iter_end)
		{
			*dcq_iter++ = (1.0f - *q_iter) * ( 1.0f + *q_iter) * (*d_iter / bc - *q_iter * a_bbbc) * batchsizenorm;
			*dcd_iter++ = (1.0f - *d_iter) * ( 1.0f + *d_iter) * (*q_iter / bc - *d_iter * a_bccc) * batchsizenorm;
			++q_iter;
			++d_iter;
		}
	}
}

void cuda_Deriv_Cosine_EX_Full( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t m, float eps)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( nTrail + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_deriv_cosine_ex_full<<<block_tail ,thread_tail >>>(q, d, neg_list, dcq, dcd, nTrail, BATCHSIZE, batchsize, m, eps);
}



__global__ void cuda_deriv_cosine_linear_ex_full(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize,  uint32_t m, float eps)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < batchsize && idy < nTrail)
	{
		float a = 0;
		float b = eps;
		float c = eps;
		uint32_t mIndex = neg_list[idy * BATCHSIZE + idx];
		for(uint32_t i=0;i<m;i++)
		{
			a += q[idx * m + i] * d[mIndex * m + i];
			b += q[idx * m + i] * q[idx * m + i];
			c += d[mIndex * m + i] * d[mIndex * m + i];
		}
		b = sqrtf(b);
		c = sqrtf(c);
		for(uint32_t i=0;i<m;i++)
		{
			dcq[idy * BATCHSIZE * m  + idx * m + i] = (float)( (d[mIndex*m+i] * 1.0f / (b*c) - q[idx*m+i] * a * 1.0f / (b*b*b*c)) );
			dcd[idy * BATCHSIZE * m  + idx * m + i] = (float)( (q[idx*m+i] * 1.0f / (b*c) - d[mIndex*m+i] * a * 1.0f / (b*c*c*c)) );
			dcq[idy * BATCHSIZE * m  + idx * m + i] = dcq[idy * BATCHSIZE * m  + idx * m + i] * 1.0f / batchsize;
			dcd[idy * BATCHSIZE * m  + idx * m + i] = dcd[idy * BATCHSIZE * m  + idx * m + i] * 1.0f / batchsize;
		}
	}
}

void cuda_Deriv_Cosine_Linear_EX_Full(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize,  uint32_t m, float eps)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( nTrail + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_deriv_cosine_linear_ex_full<<<block_tail ,thread_tail >>>(q,d,neg_list,dcq,dcd, nTrail, BATCHSIZE, batchsize, m, eps);
}


__global__ void cuda_deriv_cosine_rectified_ex_full(float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t m, float eps)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < batchsize && idy < nTrail)
	{
		float a = 0;
		float b = eps;
		float c = eps;
		uint32_t mIndex = neg_list[idy * BATCHSIZE + idx];
		for(uint32_t i=0;i<m;i++)
		{
			a += q[idx * m + i] * d[mIndex * m + i];
			b += q[idx * m + i] * q[idx * m + i];
			c += d[mIndex * m + i] * d[mIndex * m + i];
		}
		b = sqrtf(b);
		c = sqrtf(c);
		for(uint32_t i=0;i<m;i++)
		{
			if(q[idx*m+i] == 0)
			{
				dcq[idy * BATCHSIZE * m + idx * m + i] = 0;
			}
			else
			{
				dcq[idy * BATCHSIZE * m + idx * m + i] = (float)( (d[mIndex*m+i] * 1.0f / (b*c) - q[idx*m+i] * a * 1.0f / (b*b*b*c)) );
			}
			dcq[idy * BATCHSIZE * m + idx * m + i] = dcq[idy * BATCHSIZE * m + idx * m + i] * 1.0f / batchsize;

			if(d[mIndex*m+i] == 0)
			{
				dcd[idy * BATCHSIZE * m + idx * m + i] = 0;
			}
			else
			{
				dcd[idy * BATCHSIZE * m + idx * m + i] = (float)( (q[idx*m+i] * 1.0f / (b*c) - d[mIndex*m+i] * a * 1.0f / (b*c*c*c)) );
			}
			dcd[idy * BATCHSIZE * m + idx * m + i] = dcd[idy * BATCHSIZE * m + idx * m + i] * 1.0f / batchsize;
		}
	}
}

void cuda_Deriv_Cosine_Rectified_EX_Full( float * q, float * d, uint32_t * neg_list, float * dcq, float * dcd, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t m, float eps)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( nTrail + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_deriv_cosine_rectified_ex_full<<<block_tail ,thread_tail >>>(q, d, neg_list, dcq, dcd, nTrail, BATCHSIZE, batchsize, m, eps);
}


__global__ void cuda_matrix_weightadd_full(float * gpu_floats_a, float * gpu_floats_b, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float * mweight, int start,  int keep)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < dimension && idy < batchsize)
	{
		for(int i=0;i<nTrail;i++)
		{
			gpu_floats_a[idy*dimension+idx] += keep * gpu_floats_b[ i * BATCHSIZE * dimension + idy * dimension + idx] * mweight[start + i * BATCHSIZE + idy];
		}
		
	}
}

/// b add to a.
void cuda_Matrix_WeightAdd_Full(float * gpu_floats_a, float * gpu_floats_b, uint32_t nTrail, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_matrix_weightadd_full<<<block_tail ,thread_tail >>>(gpu_floats_a, gpu_floats_b, nTrail, BATCHSIZE, batchsize, dimension, mweight, start, keep);
}

__global__ void cuda_matrix_weightadd_ex_full(float * gpu_floats_a, float * gpu_floats_b, int * inver_neg_index, int * inver_neg_value, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < dimension && idy < batchsize)
	{
		for(int n=0; n<nTrial; n++)
		{
			int col_end = inver_neg_index[n * BATCHSIZE + idy];
			int col_begin = 0;
			if(idy > 0)
			{
				col_begin = inver_neg_index[n * BATCHSIZE + idy - 1];
			}

			float sum = 0;
			for(int i=col_begin; i<col_end; i++)
			{
				int row = inver_neg_value[n * BATCHSIZE + i];
				sum += gpu_floats_b[n * BATCHSIZE * dimension + row * dimension + idx] * mweight[start + n * BATCHSIZE + row];
			}

			gpu_floats_a[idy*dimension+idx] += keep * sum;
		}
	}
}

void cuda_Matrix_WeightAdd_EX_Full(float * gpu_floats_a, float * gpu_floats_b, int * inver_neg_index, int * inver_neg_value, uint32_t nTrial, uint32_t BATCHSIZE, uint32_t batchsize, uint32_t dimension, float * mweight, int start, int keep)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (m * n + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_matrix_weightadd_ex_full<<<block_tail ,thread_tail >>>(gpu_floats_a, gpu_floats_b, inver_neg_index, inver_neg_value, nTrial, BATCHSIZE,  batchsize, dimension, mweight, start, keep);
}



__global__ void cuda_cosine_similarity_subspace(float * a, float * b, float * c, uint32_t labelDim, uint32_t BATCHSIZE, 
									   uint32_t batchsize, uint32_t subspaceDim, float eps)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize && idy < labelDim)
	{
		float sumxx = 0;
		float sumyy = 0;
		float sumxy = 0;
		int id_start = idx * (labelDim * subspaceDim) + idy * subspaceDim;
		for(uint32_t i=0;i<subspaceDim;i++)
		{
			sumxx += a[id_start + i] * a[id_start + i];
			sumyy += b[id_start + i] * b[id_start + i];
			sumxy += a[id_start + i] * b[id_start + i];
		}
		c[idx * labelDim + idy] = (float)( sumxy * 1.0f / (sqrtf( (float)(sumxx * sumyy)) + eps) );
	}

}
void cuda_Cosine_Similarity_SubSpace(float * a, float * b, float * c, uint32_t labelDim, uint32_t BATCHSIZE, 
									   uint32_t batchsize, uint32_t subspaceDim, float eps)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;

	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( labelDim + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_cosine_similarity_subspace<<<block_tail , thread_tail >>>(a, b, c, labelDim, BATCHSIZE, batchsize, subspaceDim, eps);
}



__global__ void cuda_softmax(float * a, float * b,uint32_t labelDim, uint32_t batchsize, float gamma)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize )
	{
		float log_sum = 0;

		for(int i = 0; i<labelDim; i++)
		{
			float tmpa = gamma * a[idx * labelDim + i];
			if( i == 0)
			{
				log_sum = tmpa;
				continue;
			}
			else
			{
				if(log_sum >= tmpa)
				{
					log_sum = log_sum + logf(1 + expf(gamma * (tmpa - log_sum)));
				}
				else
				{
					log_sum = tmpa + logf(1 + expf(gamma * (log_sum - tmpa)));
				}
			}
		}

		for(int i=0;i<labelDim; i++)
		{
			float tmpa = gamma * a[idx * labelDim + i];
			b[idx * labelDim + i] = expf( tmpa - log_sum);
		}

	}

}
void cuda_SoftMax(float * a, float * b,uint32_t labelDim, uint32_t batchsize, float gamma)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;

	cuda_softmax<<<nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(a, b, labelDim, batchsize, gamma);
}


__global__ void cuda_deriv_cosine_subspace(float * q, float * d, float * dcq, float * dcd, float * alpha, uint32_t act_type,  uint32_t batchsize, uint32_t labelDim, uint32_t subspaceDim, float gamma, float eps)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize && idy < labelDim)
	{
		float alpha_v = gamma * alpha[idx * labelDim + idy];
		int id_start = idx * labelDim * subspaceDim + idy * subspaceDim;
		float a = 0;
		float b = eps;
		float c = eps;
		for(uint32_t i=0;i<subspaceDim;i++)
		{
			a += q[id_start + i] * d[id_start + i];
			b += q[id_start + i] * q[id_start + i];
			c += d[id_start + i] * d[id_start + i];
		}
		b = sqrtf(b);
		c = sqrtf(c);

		/// tanh function.
		if(act_type == 0)
		{
			for(uint32_t i=0;i<subspaceDim;i++)
			{
				dcq[id_start + i] = (float)( (1 - q[id_start + i]) * ( 1 + q[id_start + i]) * (d[id_start + i] * 1.0f / (b*c) - q[id_start + i] * a * 1.0f / (b*b*b*c)) );
				dcd[id_start + i] = (float)( (1 - d[id_start + i]) * ( 1 + d[id_start + i]) * (q[id_start + i] * 1.0f / (b*c) - d[id_start + i] * a * 1.0f / (b*c*c*c)) );
				dcq[id_start + i] = alpha_v * dcq[id_start + i] * 1.0f / batchsize;
				dcd[id_start + i] = alpha_v * dcd[id_start + i] * 1.0f / batchsize;
			}
		}
		/// linear function.
		else if(act_type == 1)
		{
			for(uint32_t i=0;i<subspaceDim;i++)
			{
				dcq[id_start + i] = (float)( (d[id_start + i] * 1.0f / (b*c) - q[id_start + i] * a * 1.0f / (b*b*b*c)) );
				dcd[id_start + i] = (float)( (q[id_start + i] * 1.0f / (b*c) - d[id_start + i] * a * 1.0f / (b*c*c*c)) );
				dcq[id_start + i] = alpha_v * dcq[id_start + i] * 1.0f / batchsize;
				dcd[id_start + i] = alpha_v * dcd[id_start + i] * 1.0f / batchsize;
			}
		}
		/// 
		else if(act_type == 2)
		{
			for(uint32_t i=0;i<subspaceDim;i++)
			{
				if(fabsf(q[id_start + i]) < eps)
				{
					dcq[id_start + i]  = 0;
				}
				else
				{
					dcq[id_start + i] = (float)( (d[id_start + i] * 1.0f / (b*c) - q[id_start +i] * a * 1.0f / (b*b*b*c)) );
				}
				dcq[id_start + i] = alpha_v * dcq[id_start + i] * 1.0f / batchsize;
			
				if(fabsf(d[id_start + i]) < eps)
				{
					dcd[id_start + i ] =0;
				}
				else
				{
					dcd[id_start + i] = (float)( (q[ id_start + i] * 1.0f / (b*c) - d[ id_start + i ] * a * 1.0f / (b*c*c*c)) );
				}
				dcd[id_start + i] = alpha_v * dcd[id_start + i] * 1.0f / batchsize;
			}
		}
	}
}

void cuda_Deriv_Cosine_Subspace( float * q, float * d, float * dcq, float * dcd, float * alpha,  uint32_t act_type, uint32_t batchsize, uint32_t labelDim, uint32_t subspaceDim, float gamma, float eps)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( labelDim + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_deriv_cosine_subspace<<< block_tail ,thread_tail >>>(q, d, dcq, dcd, alpha, act_type, batchsize, labelDim, subspaceDim, gamma, eps);
}



__global__ void cuda_deriv_innerproduct(float * q, float * d, float * dcq, float * dcd, float * alpha, uint32_t act_type,  uint32_t batchsize, uint32_t Dim, float gamma, float eps)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < batchsize)
	{
		float alpha_v = gamma * alpha[idx];
		int id_start = idx * Dim;

		/// tanh function.
		if(act_type == 0)
		{
			for(uint32_t i=0;i<Dim;i++)
			{
				dcq[id_start + i] = (float)( (1 - q[id_start + i]) * ( 1 + q[id_start + i]) * d[id_start + i] * alpha_v * 1.0f );
				dcd[id_start + i] = (float)( (1 - d[id_start + i]) * ( 1 + d[id_start + i]) * q[id_start + i] * alpha_v * 1.0f );
				//dcq[id_start + i] = alpha_v * dcq[id_start + i] ;
				//dcd[id_start + i] = alpha_v * dcd[id_start + i] ;
			}
		}
		/// linear function.
		else if(act_type == 1)
		{
			for(uint32_t i=0;i<Dim;i++)
			{
				dcq[id_start + i] = (float)( d[id_start + i] * alpha_v * 1.0f  );
				dcd[id_start + i] = (float)( q[id_start + i] * alpha_v * 1.0f  );
				// dcq[id_start + i] = alpha_v * dcq[id_start + i] * 1.0f / batchsize;
				// dcd[id_start + i] = alpha_v * dcd[id_start + i] * 1.0f / batchsize;
			}
		}
		/// 
		else if(act_type == 2)
		{
			for(uint32_t i=0;i<Dim;i++)
			{
				if(fabsf(q[id_start + i]) < eps)
				{
					dcq[id_start + i]  = 0;
				}
				else
				{
					dcq[id_start + i] = (float)( d[id_start + i] * alpha_v * 1.0f  );
				}
				
			
				if(fabsf(d[id_start + i]) < eps)
				{
					dcd[id_start + i ] =0;
				}
				else
				{
					dcd[id_start + i] = (float)( q[id_start + i] * alpha_v * 1.0f  );
				}
				
			}
		}
	}
}

void cuda_Deriv_InnerProduct( float * q, float * d, float * dcq, float * dcd, float * alpha,  uint32_t act_type, uint32_t batchsize, uint32_t Dim, float gamma, float eps)
{
	uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	
	//dim3 thread_tail(DEFAULT_THREAD_PER_DIM,DEFAULT_THREAD_PER_DIM);
	//dim3 block_tail((batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, ( labelDim + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	cuda_deriv_innerproduct<<< nBlockPerGrid ,DEFAULT_THREAD_PER_BLOCK >>>(q, d, dcq, dcd, alpha, act_type, batchsize, Dim, gamma, eps);
}


__global__ void cuda_fillout_composite(float* data, uint32_t* feaIdx, float* compData, float* contextLT, uint32_t inputdim, uint32_t d1, uint32_t d2, uint32_t batchsize)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < inputdim && idy < batchsize)
	{
		if (idx < d1)
		{
			compData[idy * inputdim + idx] = data[idy * d1 + idx];
		}
		else
		{
			uint32_t prodctfea = feaIdx[idy];
			compData[idy * inputdim + idx] = contextLT[prodctfea * d2 + idx - d1];
		}
	}
}

__global__ void cuda_fillout_composite_rev(float* data, float* compData, float* contextDeriv, uint32_t inputdim, uint32_t d1, uint32_t d2, uint32_t batchsize)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < inputdim && idy < batchsize)
	{
		if (idx < d1)
		{
			data[idy * d1 + idx] = compData[idy * inputdim + idx];
		}
		else
		{
			contextDeriv[idy * d2 + idx - d1] = compData[idy * inputdim + idx];
		}
	}
}

void cuda_FillOut_Composite(float* data, uint32_t* feaIdx, float* compData, float* context, uint32_t d1, uint32_t d2, uint32_t batchsize, uint32_t direction)
{
	//uint32_t nThreadPerBlock = DEFAULT_THREAD_PER_BLOCK;
	//uint32_t nBlockPerGrid = (batchsize + DEFAULT_THREAD_PER_BLOCK - 1) / DEFAULT_THREAD_PER_BLOCK;
	uint32_t inputdim = d1 + d2;
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((inputdim + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (batchsize + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

	if (direction != 0)
		cuda_fillout_composite<<<block_tail, thread_tail>>>(data, feaIdx, compData, context, inputdim, d1, d2, batchsize);
	else
		cuda_fillout_composite_rev<<<block_tail, thread_tail>>>(data, compData, context, inputdim, d1, d2, batchsize);
}


__global__ void cuda_sparse_update_lookup(float * lookupt, int * Fea_ID, int * Fea_Idx, int * Seq, float * ltDeriv1, float * ltDeriv2, float * ltDeriv3, int seq1size, int sq1sq2, int IDnum, int Feature_Dimension, float lr)
{
	uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < Feature_Dimension && idy < IDnum)
	{
		int colend = Fea_Idx[idy];
		int colbegin = 0;
		if (idy > 0)
			colbegin = Fea_Idx[idy - 1];
		float accu;
		for (int t = colbegin; t < colend; t++)
		{
			int tidx = Seq[t];
			if (tidx < seq1size)
			{
				accu += ltDeriv1[tidx*Feature_Dimension + idx];
			}
			else if (tidx < sq1sq2)
			{
				accu += ltDeriv2[(tidx - seq1size)*Feature_Dimension + idx];
			}
			else
			{
				accu += ltDeriv3[(tidx - sq1sq2)*Feature_Dimension + idx];
			}
		}
		int wid = Fea_ID[idy];
		int updatepos = wid*Feature_Dimension + idx;
		lookupt[updatepos] = lookupt[updatepos] - lr * accu;
	}
}

void cuda_Sparse_Update_Lookup(float * lookupt, int * Fea_ID, int * Fea_Idx, int * Seq, float * ltDeriv1, float * ltDeriv2, float * ltDeriv3, int seq1size, int seq2size, int IDnum, int Feature_Dimension, float lr)
{
	dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
	dim3 block_tail((Feature_Dimension + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (IDnum + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
	int sq1sq2 = seq1size + seq2size;
	cuda_sparse_update_lookup<<<block_tail, thread_tail>>>(lookupt, Fea_ID, Fea_Idx, Seq, ltDeriv1, ltDeriv2, ltDeriv3, seq1size, sq1sq2, IDnum, Feature_Dimension, lr);
}