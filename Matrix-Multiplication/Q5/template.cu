
#include "wb.h"
#include<stdio.h>
#include<iostream>
#warning "Proper usage of the object file is : ./MatrixMultiplication input_file0 input_file1 output_file"
using namespace std;

#define TILE_WIDTH 32


__global__
void tiled_mm(float *d_A, float *d_B, float *d_C, int P, int Q,int R)
{
    __shared__ float A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.f;

    int tileIdx;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;


    for(tileIdx = 0;tileIdx < ((Q-1)/TILE_WIDTH + 1); tileIdx++)
    {
      if(row<P && tileIdx*TILE_WIDTH +threadIdx.x<Q)
        A[threadIdx.y][threadIdx.x] = d_A[row * Q + tileIdx*TILE_WIDTH + threadIdx.x];
      else
        A[threadIdx.y][threadIdx.x] = 0;

      if(col<R && tileIdx*TILE_WIDTH + threadIdx.y<Q)
       B[threadIdx.y][threadIdx.x] = d_B[(tileIdx*TILE_WIDTH + threadIdx.y) + col * Q  ];
      else
        B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

      //  #pragma unroll
        for(int k=0;k<TILE_WIDTH;k++)
                sum += A[threadIdx.y][k] * B[k][threadIdx.x];
        __syncthreads();
    }

    if (row < P && col < R)
    {
      d_C[row*R + col] = sum;
    }
}




int main(int argc, char *argv[]) {

  float *hostInput1;
  float *hostInput2;
  float *hostInput2_trans;
  float *hostOutput;
  float *expectedOutput;

  int size1, size2, size3;

  /* parse the input arguments */
  //@@ Insert code here
  char *inputFileName1;
  char *inputFileName2;
  char *outputFileName;

  //cout<<"BLAH!!";

  if(argc!=4)
  {
    cerr<<"The proper usage is: ./a.out \"input0_x.raw\" \"input1_x.raw\" \"output_x.raw\"\n";
    exit(0);
  }


  wbArg_t args = wbArg_read(argc, argv);

  inputFileName1 = wbArg_getInputFile(args, 0);
  inputFileName2 = wbArg_getInputFile(args, 1);
  outputFileName = wbArg_getInputFile(args, 2);

  hostInput1 = (float *)wbImport(inputFileName1, &size1);
  hostInput2 = (float *)wbImport(inputFileName2, &size2);
  expectedOutput = (float *)wbImport(outputFileName, &size3);


  int P = sqrt((float)((float)((float)size1 / (float)size2) * (float)size3));
  int R = sqrt((float)((float)((float)size2 / (float)size1) * (float)size3));
  int Q = sqrt((float)((float)((float)size1 * (float)size2) / (float)size3));

  hostInput2_trans = (float *)malloc(sizeof(float) * size2);
  hostOutput = (float *)malloc(sizeof(float) * size3);
  for(int i=0;i<Q;i++)
    for(int j=0;j<R;j++)
      hostInput2_trans[j * Q + i] = hostInput2[i * R + j];

  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  //cout<<"BLAH!!";

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");

  cudaMalloc((void **)&deviceInput1, sizeof(float) * size1);
  cudaMalloc((void **)&deviceInput2, sizeof(float) * size2);
  cudaMalloc((void **)&deviceOutput, sizeof(float) * size3);

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");

  cudaMemcpy(deviceInput1, hostInput1, sizeof(float) * size1, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2_trans, sizeof(float) * size2, cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");


  dim3 thread_block(TILE_WIDTH, TILE_WIDTH   );
  dim3 grid((R-1)/TILE_WIDTH + 1, (P-1)/TILE_WIDTH + 1);


  tiled_mm<<<grid, thread_block>>>(deviceInput1, deviceInput2, deviceOutput, P, Q, R);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");


  cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * size3, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");


  int flag = 1;
  for(int i=0;i<size3;i++)
      if(hostOutput[i] != expectedOutput[i])
      {
        flag = 0;
        cout<<hostOutput[i]<<expectedOutput[i]<<endl;
        cout<<i;
        break;
      }
  if(flag)
    printf("\nThe results have been verified.\n");
  else
    printf("\nThe result is wrong.\n");

  // Import host input data
  //@@ Read data from the raw files here
  //@@ Insert code here

  // Declare and allocate host output
  //@@ Insert code here


  // Declare and allocate thrust device input and output vectors
  //@@ Insert code here
  //thrust::device_ptr<float> dp1 = &hostInput1[0] ;
  //thrust::device_ptr<float> dp2 = &hostInput2[0] ;



  // Copy to device
  //@@ Insert code here

  // Execute vector addition
  //@@ Insert Code here

  /////////////////////////////////////////////////////////

  // Copy data back to host
  //@@ Insert code here

  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  return 0;
}
