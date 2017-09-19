/*
HPC ASSIGNMENT 1 : QUESTION 2
Name :  Arvind Sai K , Derik Clive
RollNo: 15CO207	     , 15CO213
*/

#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>
#define Arrsize 500000

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
      {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}


//kernel to find the sum of an array
__global__ void find_sum(float *d_arr, float *d_sum)
{
	__shared__ float sdata[1024];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < Arrsize){
	sdata[threadIdx.x] = d_arr[idx];
        	
	__syncthreads();
	int i, index;
	for(i=1; i<blockDim.x; i*=2)
	{
		index = 2*i*threadIdx.x;
		if(index < blockDim.x && (blockIdx.x * blockDim.x + index + i) < Arrsize)
		{
			sdata[index] += sdata[index+i];
		}
		__syncthreads();
	}
	if(threadIdx.x==0)
		atomicAdd(d_sum, sdata[0]);
	}
}
int main()
{
	//Defining the size of the array
	const int ARRAY_SIZE = Arrsize;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	//Creating a float array of size SIZE on the host
	float h_arr[ARRAY_SIZE], h_sum[ARRAY_SIZE];
	int i;
	float s = 100.0;
	srand((unsigned int)time(NULL));
	
	for(i=0;i<ARRAY_SIZE;i++)
	{	
		h_arr[i] = ((float)rand()/(float)(RAND_MAX))*s - s/2;
	}

	h_sum[0] = 0;

	//Declaring variables on the GPU
	float *d_arr;
	float *d_sum;

	float time;
	cudaEvent_t start, stop;

	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;
	cudaEventRecord(start, 0) ;


	//Allocating memory on the GPU
	HANDLE_ERROR ( cudaMalloc((void **)&d_arr, ARRAY_BYTES));
	HANDLE_ERROR ( cudaMalloc((void **)&d_sum, ARRAY_BYTES));


	//Copy the h_arr from host to device
	HANDLE_ERROR ( cudaMemcpy(d_arr, h_arr, ARRAY_BYTES, cudaMemcpyHostToDevice));
	HANDLE_ERROR ( cudaMemcpy(d_sum, h_sum, ARRAY_BYTES, cudaMemcpyHostToDevice));


	//Launch the kernel on the GPU
	dim3 block_size = (1024);
	dim3 grid_size = (ceil((float)ARRAY_SIZE/block_size.x));
	

	
	find_sum<<<grid_size, block_size>>>(d_arr, d_sum);

	//Copy back the result from device to host
	HANDLE_ERROR ( cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&time, start, stop);

	//Print the output
	printf("\nResultant Sum: %f\n", h_sum[0]);

	printf("Time on GPU:  %f ms \n", time);
	
	float expected = 0.0;
	
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;
	cudaEventRecord(start, 0) ;
	
	for(int i=0;i<ARRAY_SIZE;i++)
		expected += h_arr[i];

	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&time, start, stop);

	printf("\nExpected Sum: %f\n", expected);

	printf("Time on CPU:  %f ms \n", time);
	//Freeing memory on the GPU
	cudaFree(d_arr);
	cudaFree(d_sum);
	
	return 0;
	
}
