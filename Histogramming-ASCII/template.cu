#include<bits/stdc++.h>
#include "wb.h"

#define NUM_BINS 128

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


  __shared__ unsigned int shared_bin[NUM_BINS];
  // if(threadIdx.x == 0){
  // 	for(int i = 0;i<NUM_BINS;++i) shared_bin[i] = 0;
  // }
  if(threadIdx.x < 128 ){
  	shared_bin[threadIdx.x] = 0;
  }
  __syncthreads();
  if(index < size){
	  	atomicAdd(&(shared_bin[ip[index]]), 1);
  }

  __syncthreads();

  if(threadIdx.x <128 ){
  	// for(int i = 0;i<NUM_BINS;++i){
  	// 	int val = shared_bin[i];
  	// 	atomicAdd(&(histo[i]), val);
  	// }
  	atomicAdd(&(histo[threadIdx.x]),shared_bin[threadIdx.x]);
  }
  
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 1),
                                       &inputLength);
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  for(int i=0;i<NUM_BINS;i++)
    hostBins[i] = 0;

  // for(int i =0;i<inputLength;++i) cout<<(char)hostInput[i];
  cout<<"\n";	

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
  dim3 dimBlock(128, 1, 1);
  int numgrids = (inputLength-1)/(dimBlock.x) + 1;
  dim3 dimGrid(numgrids, 1, 1);


  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Perform kernel computation here
  hist<<<dimGrid, dimBlock>>>(deviceInput, deviceBins, inputLength);
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK(cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int)*NUM_BINS, cudaMemcpyDeviceToHost));
  // for(int i=0;i<NUM_BINS;++i){
  // 	cout<<(char)i<<" : "<<i<<" : "<<hostBins[i]<<"\n";
  // }
  CUDA_CHECK(cudaDeviceSynchronize());
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
