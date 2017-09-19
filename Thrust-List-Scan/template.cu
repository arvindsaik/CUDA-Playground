#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <bits/stdc++.h>
#include "wb.h"

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput, *hostOutput; // The input 1D list
  float *devInput, *devOutput; // The input 1D list
  int num_elements;              // number of elements in the input list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &num_elements);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        num_elements);


  // Declare and allocate the host output array
  //@@ Insert code here
  hostOutput = (float *) malloc(num_elements*sizeof(float));

  // Declare and allocate thrust device input and output vectors
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Insert code here
  cudaMalloc((void**)&devInput,sizeof(float)*num_elements);
  cudaMalloc((void**)&devOutput,sizeof(float)*num_elements);
  

  // thrust::device_vector<float> dev_in(num_elements);
  // thrust::device_vector<float> dev_out(num_elements);
  cudaMemcpy(devInput,hostInput,sizeof(float)*num_elements,cudaMemcpyHostToDevice);

  thrust::device_ptr<float> dev_in(devInput);
  thrust::device_ptr<float> dev_out(devOutput);
  wbTime_stop(GPU, "Allocating GPU memory.");

  // Execute vector addition
  wbTime_start(Compute,"Doing the computation on the GPU");
  //@@ Insert Code here
  thrust::inclusive_scan(dev_in, dev_in+num_elements, dev_out);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  cudaMemcpy(hostOutput,devOutput,sizeof(float)*num_elements,cudaMemcpyDeviceToHost);
  wbSolution(args, hostOutput, num_elements);

  // Free Host Memory
  free(hostInput);
  //@@ Insert code here

  return 0;
}
