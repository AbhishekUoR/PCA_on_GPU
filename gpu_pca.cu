#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <cstdlib>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>
#include <assert.h>
int row = 0;
int col = 0;
using namespace std;

__global__
void mean(float *A, float *means, int size_row, int size_col ){
 
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size_col){
	for(int i =0; i<size_row; i++){
			means[idx] += A[idx*size_row + i];
			
	}
means[idx] = means[idx]/size_row;

}
	
}

__global__
void center_data (float*A, float *means, int size_row, int size_col){

int idx = blockIdx.x*blockDim.x + threadIdx.x;
if (idx < size_col){
	for (int i =0; i < size_row; i++){
			A[idx*size_row + i] = A[idx*size_row + i] - means[idx];
		}
	}

}
__global__ void gpu_transpose(float *dst, float *A, int col, int row) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(idx<col){
			for (int j=0; j<row; j++){
			dst[j*col+idx] = A[idx*row+j];							
			}	
		} 
}

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
	ofstream collect("hello.txt");
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
			collect << A[j * nr_rows_A + i] << " " ;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char*argv[])
{

	if(argc != 2) {
        	fprintf(stderr, "Follow this command line statement: %s input_file \n", argv[0]);
        	exit(1);
        }
	string lineA;
	float x;
	int rows = 1000;
	int colm = 1000;
	double **a = new double* [rows];
	for(int i = 0; i < rows; i++)
	a[i] = new double[colm];
	string filename;
	ifstream in(argv[1],ios::in);
	if(in.fail()){
		cerr << "file can not be found or opened" << endl;
		exit(1);
	}
	while(in.good()){
		while(getline(in,lineA)){
			istringstream streamA(lineA);
			col = 0;
			while(streamA >> x){
				a[row][col] = x;
				col++;
			}
			row++;
		}
	}
	
	cout << "rows=" << row << endl;
	cout << "cols=" << col << endl;
	float *matrix = (float *) malloc(sizeof(float) * row * col);
        for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
                matrix[j*row+i]=a[i][j];
                }
        }
	
	/*std::cout << "matrix =" << std::endl;
	print_matrix(matrix, row, col);*/

    float accum;
    float total;
    cusolverDnHandle_t cusolverH;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;
    const int m = row;
    const int lda = col;
	float *A = (float *) malloc(sizeof (float) *lda*m);
    float *B = (float*)malloc(lda*m*sizeof(float));
    float *C = (float*)malloc(lda*lda*sizeof(float));
    float *V;
    V= (float*)malloc(lda*m*sizeof(float));  
    float *W = (float*)malloc(lda*sizeof(float));
    float *Mean = (float*)malloc(lda*sizeof(float));
    float *d_A;
    float *d_B;
	float *d_C;
    float *d_W;
	float *d_D;
    float *means;
	int *devInfo;
    float *d_work;
    int lwork =0;
    int info_gpu =0;
    float* d_C_T;
    float* d_c1;
	/*int devID;
      devID = gpuGetMaxGflopsDeviceId();
      checkCudaErrors( cudaSetDevice(devID) );
      // checkCudaErrors( cudaSetDevice(0) );  // or just use the first GPU*/
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start);
	cusolver_status = cusolverDnCreate(&cusolverH);
    cudaMalloc ((void**)&d_A, sizeof(float)*lda*m);
	cudaMalloc ((void**)&d_C, sizeof(float)*lda*lda);
    cudaMalloc ((void**)&d_C_T, sizeof(float)*lda*lda);
	cudaMalloc ((void**)&d_W, sizeof(float)*m);
    cudaMalloc ((void**)&devInfo, sizeof(int));
	cudaMalloc ((void**)&d_D, sizeof(float)*lda*m);
    cudaMalloc ((void**)&means, sizeof(float)*lda);
    cudaMemcpy(d_A, matrix, sizeof(float)*lda*m, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, sizeof(float)*m*lda);
    cudaMalloc((void**) &d_c1, sizeof(float)*lda*lda);

   	int threads = m*lda;
    int blocks;
    if (threads > 256){
		threads = 256;
		 blocks = m*lda/threads;
	}
	else{
		 blocks = 1;}
	
	mean<<< 1, lda >>>(d_A, means, m, lda);
	cudaDeviceSynchronize();
     
	/*cudaMemcpy(Mean, means, sizeof(float)*lda, cudaMemcpyDeviceToHost);
    std::cout << "Mean =" << std::endl;
	print_matrix(Mean, 1, lda);*/
	
	center_data <<<1, lda>>>(d_A, means, m, lda);
    cudaDeviceSynchronize();
	
	/*cudaStat = cudaMemcpy(A, d_A, sizeof(float)*m*lda, cudaMemcpyDeviceToHost);
    std::cout << "A =" << std::endl;
	print_matrix(A, m, lda);*/

	gpu_transpose<<<1, lda>>>(d_B, d_A, col, row);
    cudaDeviceSynchronize();

   /* cudaMemcpy(B, d_B, sizeof(float)*m*lda, cudaMemcpyDeviceToHost);
    std::cout << "B =" << std::endl;
	print_matrix(B, lda, m);*/

	const float alf = 0.0067;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
    printf("this the alf value %f \n", alf);
    // Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, lda, lda, m, alpha, d_B, lda, d_A, m, beta, d_C, lda);
	cudaDeviceSynchronize();
  
	/*cudaMemcpy(C, d_C, sizeof(float)*lda*lda, cudaMemcpyDeviceToHost);
  	std::cout << "c =" << std::endl;
  	print_matrix(C, lda, lda);*/

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    cusolver_status = cusolverDnSsyevd_bufferSize(cusolverH, jobz, uplo, lda, d_C, lda, d_W, &lwork);
	cudaMalloc ((void**) &d_work, sizeof(float)*lwork);
    //clock_gettime(CLOCK_REALTIME, &start);
    cusolver_status = cusolverDnSsyevd(cusolverH, jobz, uplo, lda, d_C, lda, d_W, d_work, lwork, devInfo);
    cudaDeviceSynchronize();
	//clock_gettime(CLOCK_REALTIME, &start);
	//accum =(stop.tv_sec-start.tv_sec) + (stop.tv_nsec - start.tv_nsec);
	/*cudaMemcpy(W, d_W, sizeof(float)*m, cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_C, sizeof(float)*lda*lda, cudaMemcpyDeviceToHost);
	std::cout << "V =" << std::endl;
	print_matrix(V, lda, lda);	*/
	
    gpu_transpose<<<1, lda>>>(d_C_T, d_C, lda, lda);
    cudaDeviceSynchronize();
    
   /* for (int i =0; i< lda; i++){
		printf("W[%d] = %E\n", i+1, W[i]);	
	}  */   
	const float alf1 = 1;
	const float bet1 = 0;
	const float *alpha1 = &alf1;
	const float *beta1 = &bet1;
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, lda, lda, alpha1, d_A, m, d_C_T, lda, beta1, d_D, m);
	cudaDeviceSynchronize();	
    cublasDestroy(handle);
	cudaMemcpy(V, d_D, sizeof(float)*m*lda, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&total, start, stop);
  	cout << "GPU time :" << total << "ms."<<endl;
	std::cout << "V =" << std::endl;
	print_matrix(V, m, lda);

	return 0;
}
