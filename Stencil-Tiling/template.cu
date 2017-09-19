#warning "Proper usage of the object file is: ./Stencil_Template Stencil\ Dataset/0\ input.ppm Stencil\ Dataset/0\ output.ppm"

#include "wb.h"
#include<bits/stdc++.h>
using namespace std;

#define TILE 16

#define  value(arry, i, j, k) arry[((j)*width + (i)) * depth + (k)]
#define sh_val(arry, i, j, k) arry[(j*8  + i)*8 + k]

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__
void stencil(float *output, float *input, int width, int height,
                        int depth)
{
  int tx = threadIdx.x, ty = threadIdx.y, tz =threadIdx.z;
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

  int x = bx * blockDim.x + tx;
  int y = by * blockDim.y + ty;
  int z = bz * blockDim.z + tz;


  if(x<0 || x>=width || y<0 || y>=height || z<0 || z>=depth)
    return;

  if(x==0 || y==0 || z==0 || x==width-1 || y==height-1 || z==depth-1)
  {
    value(output, x, y, z) = 0;
//    printf("%d\n", ((y)*width + (x)) * depth + (z));

    return;
  }

  float sum;
  sum = value(input, x-1, y, z) + value(input, x, y-1, z) + value(input, x, y, z-1)
        + value(input, x+1, y, z) + value(input, x, y+1, z) + value(input, x, y, z+1)
        - 6*value(input,x, y, z);
  sum = max(min(sum, 1.f), 0.f);
  value(output, x, y, z) = sum;

}
__global__
void stencil_tiled(float *output, float *input, int width, int height,
                        int depth)
{
  __shared__ float sh[8*8*8];
  int tx = threadIdx.x, ty = threadIdx.y, tz =threadIdx.z;
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

  int x = bx * 6 + tx;
  int y = by * 6 + ty;
  int z = bz * 6 + tz;

    if(x<0 || x>=width || y<0 || y>=height || z<0 || z>=depth)
    {
      sh[((ty)*8  + tx)*8 + tz] = 0.0;
    //  sh_val(sh, tx, ty, tz) = 0.f;
    }

    else if(x==0 || y==0 || z==0 || x==width-1 || y==height-1 || z==depth-1)
    {
    //  printf("%d-%d-%d->%f\n",x, y, z, value(input, x, y, z) );
      sh[((ty)*8  + tx)*8 + tz] = value(input, x, y, z);
      //sh_val(sh, tx, ty, tz) = value(input, x, y, z);
      value(output, x, y, z) = 0.0;
  //    printf("%d\n", ((y)*width + (x)) * depth + (z));
    }
    else
    {
      sh[((ty)*8  + tx)*8 + tz] = value(input, x, y, z);
//      sh_val(sh, tx, ty, tz) = value(input, x, y, z);
    }

  __syncthreads();

  float sum;

if(tx>0 && ty>0 && tz>0 && tx<7 && ty<7 && tz<7)
{

   sum = sh[((ty)*8  + tx-1)*8 + tz] + sh[((ty-1)*8  + tx)*8 + tz] + sh[((ty)*8  + tx)*8 + tz-1] + sh[((ty)*8  + tx+1)*8 + tz]
        + sh[((ty+1)*8  + tx)*8 + tz] + sh[((ty)*8  + tx)*8 + tz+1] - 6*sh[((ty)*8  + tx)*8 + tz];

  sum = max(min(sum, 1.f), 0.f);
  //  sh_val(sh, tx-1, ty, tz) + sh_val(sh, tx, ty-1, tz) + sh_val(sh, tx, ty, tz-1)
  //       + sh_val(sh, tx+1, ty, tz) + sh_val(sh, tx, ty+1, tz) + sh_val(sh, tx, ty, tz+1)
  //       - 6*sh_val(sh, tx, ty, tz);

  if(x>0 && y>0 && z>0 && x<width-1 && y<height-1 && z<depth-1)
    value(output, x, y, z) = sum;
  }
  }



static void launch_stencil(float *deviceOutputData, float *deviceInputData,
                           int width, int height, int depth) {
  //@@ INSERT CODE HERE
  dim3 blockdims(8, 8, 8);
//  dim3 griddims((width-1)/blockdims.x + 1, (height-1)/blockdims.y + 1, (depth-1)/blockdims.z + 1);
//  stencil<<<griddims, blockdims>>>(deviceOutputData, deviceInputData, width, height, depth);
  dim3 griddims((width-1)/6 + 1, (height-1)/6 + 1, (depth-1)/6 + 1);

  cout<<griddims.x<<" "<<griddims.y<<" "<<griddims.z<<endl;
  stencil_tiled<<<griddims, blockdims>>>(deviceOutputData, deviceInputData, width, height, depth);

}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  arg = wbArg_read(argc, argv);

  inputFile = wbArg_getInputFile(arg, 0);

  input = wbImport(inputFile);

  width  = wbImage_getWidth(input);
  height = wbImage_getHeight(input);
  depth  = wbImage_getChannels(input);

  output = wbImage_new(width, height, depth);

  hostInputData  = wbImage_getData(input);
  hostOutputData = wbImage_getData(output);

  //printf("%d\n", ((0)*width + (0)) * depth + (2));
  //cout<<hostInputData[32*32*2]<<endl;


  wbTime_start(GPU, "Doing GPU memory allocation");
  wbCheck(cudaMalloc((void **)&deviceInputData,
             width * height * depth * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputData,
             width * height * depth * sizeof(float)));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  wbCheck(cudaMemcpy(deviceInputData, hostInputData,
             width * height * depth * sizeof(float),
             cudaMemcpyHostToDevice));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  wbCheck(cudaMemcpy(hostOutputData, deviceOutputData,
             width * height * depth * sizeof(float),
             cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbSolution(arg, output);

  wbImage_t solnImage = wbImport(wbArg_getInputFile(arg, arg.argc - 2));
//  float *data = solnImage.data;
  // for(int i=0;i<width*height*depth;i++)
  //   printf("%f ", data[i]);

  //cout<<depth<<endl;
  float *expected = (float *)malloc(sizeof(float) * width * height* depth);


  wbCheck(cudaFree(deviceInputData));
  wbCheck(cudaFree(deviceOutputData));

  wbImage_delete(output);
  wbImage_delete(input);

  return 0;
}
