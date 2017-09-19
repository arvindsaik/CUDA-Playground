#warning "Proper usage of the object file is : ./Histogram_Template Histogram-Dataset/5-input.raw Histogram-Dataset/5-output.raw"

#include "wb.h"
#include<bits/stdc++.h>
using namespace std;

#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__
void hist(unsigned int *ip,unsigned int *histo, unsigned int size)
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int index = bx*blockDim.x + tx;

  int stride = blockDim.x * gridDim.x;

  while(index<size)
  {
    atomicAdd(&(histo[ip[index]]), 1);
    index+=stride;
  }
}

__global__
void pri_hist(unsigned int *ip,unsigned int *histo, unsigned int size)
{
  __shared__ unsigned int sh[4096];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = bx*blockDim.x + tx;

  sh[tx] = 0;
  sh[tx+1024] = 0;
  sh[tx+2048] = 0;
  sh[tx+3072] = 0;

  __syncthreads();
  int stride = blockDim.x * gridDim.x;

  while(index<size)
  {
    atomicAdd(&(histo[ip[index]]), 1);
    index+=stride;
  }

  __syncthreads();

  atomicAdd(&(histo[tx]), sh[tx]);
  atomicAdd(&(histo[tx+1024]), sh[tx+1024]);
  atomicAdd(&(histo[tx+2048]), sh[tx+2048]);
  atomicAdd(&(histo[tx+3072]), sh[tx+3072]);


}

__global__
void clamp(unsigned int *histo, unsigned int size)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = bx*blockDim.x + tx;

  if(index<NUM_BINS)
    histo[index] = histo[index]>127?127:histo[index];
}

int main(int argc, char *argv[]) {

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  /* Read input arguments here */
  wbArg_t args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength);
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  for(int i=0;i<NUM_BINS;i++)
    hostBins[i] = 0;

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  CUDA_CHECK(cudaMalloc((void **)&deviceInput, sizeof(unsigned int)*inputLength));
  CUDA_CHECK(cudaMalloc((void **)&deviceBins, sizeof(unsigned int)*NUM_BINS));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int)*inputLength, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(deviceBins, hostBins, sizeof(unsigned int)*NUM_BINS, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Launch kernel
  // ----------------------------------------------------------
  dim3 dimBlock(1024, 1, 1);
  //int numgrids = (inputLength-1)/(dimBlock.x) + 1;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties( &prop, 0 ) );
  int blocks = prop.multiProcessorCount;
  dim3 dimGrid(2*blocks, 1, 1);
//  dim3 dimGrid(1  , 1, 1);


  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here
  pri_hist<<<dimGrid, dimBlock>>>(deviceInput, deviceBins, inputLength);
  clamp<<<4, 1024>>>(deviceBins, inputLength);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int)*NUM_BINS, cudaMemcpyDeviceToHost));
  //CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  CUDA_CHECK(cudaFree(deviceBins));
  CUDA_CHECK(cudaFree(deviceInput));
  wbTime_stop(GPU, "Freeing GPU Memory");

  // Verify correctness
  // -----------------------------------------------------
  wbSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  free(hostInput);

  return 0;
}
