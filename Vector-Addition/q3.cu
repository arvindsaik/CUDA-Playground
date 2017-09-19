/*
HPC ASSIGNMENT 1 : QUESTION 3
Name :  Arvind Sai K , Derik Clive
RollNo: 15CO207        , 15CO213
*/

#include <stdio.h>
#include<math.h>
#include <time.h>
#include <stdlib.h>
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

// CUDA Kernel for Vector Addition
__global__ void Vector_Addition ( const float *dev_a , const float *dev_b , float *dev_c, int *dev_N)
{
      //Get the id of thread within a block
      int tid = blockIdx.x*1024 + threadIdx.x;
      if ( tid < *dev_N ) // check the boundry condition for the threads
      {
	      dev_c [tid] = dev_a[tid] + dev_b[tid] ;
      }
}


int main (void)
{
      int N;
      //Host array
      printf("Enter number of array elements : ");
      scanf("%d",&N);
      float Host_a[N], Host_b[N], Host_c[N];
      
      //Device array
      float *dev_a , *dev_b, *dev_c ;
      int *dev_N;
 
      //fill the Host array with random elements on the CPU
      srand(time(NULL)); 
      for ( int i = 0; i <N ; i++ )
      {
	    int a = rand();
	    int b = rand();
	    int c = rand();
	    int d = rand();
            Host_a[i] = ((float)a)/(b+1);
            Host_b[i] = ((float)c)/(d+1) ; 
      }

      float timer;
      cudaEvent_t start, stop;
      cudaEventCreate(&start) ;
      cudaEventCreate(&stop) ;
      cudaEventRecord(start, 0) ;

      //Allocate the memory on the GPU
      HANDLE_ERROR ( cudaMalloc((void **)&dev_a , N*sizeof(float) ) );
      HANDLE_ERROR ( cudaMalloc((void **)&dev_b , N*sizeof(float) ) );
      HANDLE_ERROR ( cudaMalloc((void **)&dev_c , N*sizeof(float) ) );

      HANDLE_ERROR ( cudaMalloc((void **)&dev_N , sizeof(int) ) );

      //Copy Host array to Device array
      HANDLE_ERROR (cudaMemcpy (dev_a , Host_a , N*sizeof(float) , cudaMemcpyHostToDevice));
      HANDLE_ERROR (cudaMemcpy (dev_b , Host_b , N*sizeof(float) , cudaMemcpyHostToDevice));
      HANDLE_ERROR (cudaMemcpy (dev_N , &N , sizeof(int), cudaMemcpyHostToDevice));
      int blockNos = ceil((float)(N)/1024);
      
      //Make a call to GPU kernel
      Vector_Addition <<< blockNos, 1024  >>> (dev_a , dev_b , dev_c,dev_N) ;
      
      //Copy back to Host array from Device array
      HANDLE_ERROR (cudaMemcpy(Host_c , dev_c ,N * sizeof(float) , cudaMemcpyDeviceToHost));

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);	
      cudaEventElapsedTime(&timer, start, stop);
      printf("Time on GPU:  %f ms \n", timer);


      //Free the Device array memory
      cudaFree (dev_a) ;
      cudaFree (dev_b) ;
      cudaFree (dev_c) ;
      cudaFree (dev_N);
	
      float timerc;
      cudaEventCreate(&start) ;
      cudaEventCreate(&stop) ;
      cudaEventRecord(start, 0) ;
      
      int result[N];
      for(int i = 0;i<N;++i){
	result[i] = Host_a[i] + Host_b[i];
      }

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);	
      cudaEventElapsedTime(&timerc, start, stop);	
      printf("Time on CPU:  %f ms \n", timerc);
      int flag = 0;
      for(int i=0;i<N;++i){
	if(Host_a[i] + Host_b[i] != Host_c[i]){
		flag = 1;
		break;
        }
      }		
      if(flag){
		printf("Wrong result \n");
      }
      else printf("Verified to be correct\n");
      //system("pause");
      return 0 ;
}
