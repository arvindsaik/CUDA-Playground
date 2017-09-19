/*
HPC ASSIGNMENT 1 : QUESTION 4
Name :  Arvind Sai K , Derik Clive
RollNo: 15CO207	     , 15CO213
*/

#include<stdio.h>
#include<time.h>
#include<stdlib.h>

#define ARRAY_ROWS 700
#define ARRAY_COLS 700

__global__ void mat_add(int d_a[ARRAY_ROWS][ARRAY_COLS], int d_b[ARRAY_ROWS][ARRAY_COLS], int d_c[ARRAY_ROWS][ARRAY_COLS])
{
	int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
	if(row_idx<ARRAY_ROWS && col_idx<ARRAY_COLS)
		d_c[row_idx][ col_idx] = d_a[row_idx][col_idx] + d_b[row_idx][col_idx];
}

int main()
{	
	const int ARRAY_BYTES = ARRAY_ROWS * ARRAY_COLS * sizeof(int);

	srand(time(NULL));	

	int h_a[ARRAY_ROWS][ARRAY_COLS];
	int h_b[ARRAY_ROWS][ARRAY_COLS];
	int h_c[ARRAY_ROWS][ARRAY_COLS] = {0};	

	int i, j;	
	int range = 100;
	for(i=0; i<ARRAY_ROWS; i++)
		for(j=0;j<ARRAY_COLS; j++)
		{
			h_a[i][j] = rand()%range - range/2;
			h_b[i][j] = rand()%range - range/2;
		}	


	float time;
        cudaEvent_t start, stop;	

	cudaEventCreate(&start) ;
        cudaEventCreate(&stop) ;
        cudaEventRecord(start, 0) ;
	
	int (*d_a) [ARRAY_COLS], (*d_b) [ARRAY_COLS], (*d_c) [ARRAY_COLS];
	cudaMalloc((void **)&d_a, ARRAY_BYTES);
	cudaMalloc((void **)&d_b, ARRAY_BYTES);
	cudaMalloc((void **)&d_c, ARRAY_BYTES);
	
	//Transfer the array to the GPU
	cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, ARRAY_BYTES, cudaMemcpyHostToDevice);

	//Launching the kernel on the GPU
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil((float)ARRAY_ROWS/threadsPerBlock.x), ceil((float)ARRAY_COLS/threadsPerBlock.y));
	mat_add<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

	
	

	//Copy back the result array from device to host
	cudaMemcpy(h_c, d_c, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

	printf("Time on GPU:  %f ms \n", time);


	//Computing the results on the CPU

	int result[ARRAY_ROWS][ARRAY_COLS];
	

	cudaEventCreate(&start) ;
        cudaEventCreate(&stop) ;
        cudaEventRecord(start, 0) ;

	for(i=0; i<ARRAY_ROWS; i++)
	{
		for(j=0; j<ARRAY_COLS; j++)
		{
			result[i][j] = h_a[i][j] + h_b[i][j];
		}
	}

	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
	

	printf("Time on CPU:  %f ms \n", time);


	//Verifyting the results obtained from the CPU

	int flag = 1;

	for(int i=0;i<ARRAY_ROWS;i++)
		for(int j=0;j<ARRAY_COLS;j++)
			if(h_c[i][j] != result[i][j])
			{
				flag = 0;
				break;
			}

	if(flag==1)
		printf("The results have been verified.\n");
	else
		printf("\nTHe result obtained by the GPU is not correct.\n");
	
	//Free the memory allocated on the GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;
}
