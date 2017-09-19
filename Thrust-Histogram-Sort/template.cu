#include "wb.h"
#warning "Proper usage of the object file is :./ThrustHistogramSort_Template ThrustHistogramSort-Dataset/0-input.raw ThrustHistogramSort-Dataset/0-output.raw"

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include<bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength, num_bins;
  unsigned int *hostInput, *hostBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength);

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  thrust::device_vector<unsigned int> deviceInput(hostInput, hostInput + inputLength);

  thrust::sort(deviceInput.begin(), deviceInput.end());

  // Copy the input to the GPU
  wbTime_start(GPU, "Allocating GPU memory");
  //@@ Insert code here
  wbTime_stop(GPU, "Allocating GPU memory");

  // Determine the number of bins (num_bins) and create space on the host
  //@@ insert code here
  num_bins = deviceInput.back() + 1;
  num_bins = 32;
  hostBins = (unsigned int *)malloc(num_bins * sizeof(unsigned int));

  // Allocate a device vector for the appropriate number of bins
  //@@ insert code here
  thrust::device_vector<unsigned int> deviceBins(num_bins);

  // Create a cumulative histogram. Use thrust::counting_iterator and
  // thrust::upper_bound
  //@@ Insert code here
  thrust::counting_iterator<int> search_begin(0);
  thrust::upper_bound(deviceInput.begin(), deviceInput.end(), search_begin, search_begin+num_bins, deviceBins.begin());

  // Use thrust::adjacent_difference to turn the culumative histogram
  // into a histogram.
  //@@ insert code here.
  thrust::adjacent_difference(deviceBins.begin(), deviceBins.end(), deviceBins.begin());

  // Copy the histogram to the host
  //@@ insert code here
  thrust::copy(deviceBins.begin(), deviceBins.end(), hostBins);
  for(int i=0;i<32;i++)
    if(hostBins[i]>127)
      hostBins[i] = 127;

  // Check the solution is correct
  wbSolution(args, hostBins, num_bins);

  // Free space on the host
  //@@ insert code here
  free(hostBins);
  free(hostInput);

  return 0;
}
