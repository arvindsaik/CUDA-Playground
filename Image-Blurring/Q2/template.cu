#include<bits/stdc++.h>
#include "wb.h"
#warning "Proper usage of the object file is : ./ImageBlur_Template inputx.ppm outputx.ppm"
using namespace std;
#define BLUR_SIZE 5
#define O_TILE_WIDTH 22
#define BLOCK_WIDTH (O_TILE_WIDTH + 2 * BLUR_SIZE)

//@@ INSERT CODE HERE

__global__
void blur(float *input, float *output, int numRows, int numCols)
{
  int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx;

	int row_i = row_o - BLUR_SIZE;
	int col_i = col_o - BLUR_SIZE;

	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];

  if(row_i<0 || row_i>=numRows || col_i<0 || col_i>=numCols)
  {
    Ns[ty][tx] = -1;
  }
  else
  {
    int offset = row_i*numCols + col_i;
    Ns[ty][tx]  =  input[offset];
  }

  //
	// row_i = min(max(row_i,0),numRows-1);
	// col_i = min(max(col_i,0),numCols-1);
	// int offset = row_i*numCols + col_i;
	// Ns[ty][tx]  =  inputChannel[offset];


	__syncthreads();

  int filterWidth = 2 * BLUR_SIZE + 1;
  int valid = 0;

	float data = 0.f;
	if( tx < O_TILE_WIDTH && ty < O_TILE_WIDTH && row_o < numRows && col_o < numCols){

		for(int i = 0; i < filterWidth; i++)
			for(int j = 0; j < filterWidth; j++)
				{
          if(Ns[i+ty][j+tx]!=-1)
          {
            //printf("%f", Ns[i+ty][j+tx]);
            valid++;
            data += Ns[i+ty][j+tx];
          }
        }

		//if(row_o < numRows && col_o < numCols)
			output[row_o*numCols+col_o] = data/valid;
	}

}

int main(int argc, char *argv[]) {

  int imageWidth;
  int imageHeight;
  int imageChannels;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  //float *deviceInputImageData;
  //float *deviceOutputImageData;

  if(argc!=3)
  {
    cerr<<"The proper usage is:  ./a.out \"inputx.ppm\" \"outputx.ppm\" \n";
    exit(0);
  }

  /* parse the input arguments */
  //@@ Insert code here
  wbArg_t args = wbArg_read(argc, argv);

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  // The input image is in grayscale, so the number of channels
  // is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);


  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 3);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //cout<<imageHeight<<"\t"<<imageWidth;
  float *red_channel, *blue_channel, *green_channel;
  red_channel = (float *)malloc(sizeof(float) * imageWidth *imageHeight);
  green_channel = (float *)malloc(sizeof(float) * imageWidth *imageHeight);
  blue_channel = (float *)malloc(sizeof(float) * imageWidth *imageHeight);

  for(int i=0;i<imageWidth*imageHeight;i++)
  {
    red_channel[i] = hostInputImageData[3 * i];
    green_channel[i] = hostInputImageData[3 * i + 1];
    blue_channel[i] = hostInputImageData[3 * i + 2];
  }

  float *red_blurred, *green_blurred, * blue_blurred;
  red_blurred = (float *)malloc(sizeof(float) * imageWidth *imageHeight);
  green_blurred = (float *)malloc(sizeof(float) * imageWidth *imageHeight);
  blue_blurred = (float *)malloc(sizeof(float) * imageWidth *imageHeight);




  float *dred_channel, *dgreen_channel, *dblue_channel, *dred_blurred, *dgreen_blurred, *dblue_blurred;

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  // cudaMalloc((void **)&deviceInputImageData,
  //            imageWidth * imageHeight * sizeof(float));
  // cudaMalloc((void **)&deviceOutputImageData,
  //            imageWidth * imageHeight * sizeof(float));


  cudaMalloc((void **)&dred_channel, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&dgreen_channel, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&dblue_channel, imageWidth * imageHeight * sizeof(float));

  cudaMalloc((void **)&dred_blurred, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&dgreen_blurred, imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&dblue_blurred, imageWidth * imageHeight * sizeof(float));


  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  // cudaMemcpy(deviceInputImageData, hostInputImageData,
  //            imageWidth * imageHeight * sizeof(float),
  //            cudaMemcpyHostToDevice);

  cudaMemcpy(dred_channel, red_channel,imageWidth * imageHeight * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dgreen_channel, green_channel,imageWidth * imageHeight * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(dblue_channel, blue_channel,imageWidth * imageHeight * sizeof(float),cudaMemcpyHostToDevice);


  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");

  dim3 block_dim(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 grid_dim((imageWidth-1)/O_TILE_WIDTH + 1, (imageHeight-1)/O_TILE_WIDTH + 1);

  //blur<<<grid_dim, block_dim>>>(deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth);
  blur<<<grid_dim, block_dim>>>(dred_channel, dred_blurred, imageHeight, imageWidth);
  blur<<<grid_dim, block_dim>>>(dgreen_channel, dgreen_blurred, imageHeight, imageWidth);
  blur<<<grid_dim, block_dim>>>(dblue_channel, dblue_blurred, imageHeight, imageWidth);


  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  // cudaMemcpy(hostOutputImageData, deviceOutputImageData,
  //            imageWidth * imageHeight * sizeof(float),
  //            cudaMemcpyDeviceToHost);

  cudaMemcpy(red_blurred, dred_blurred,imageWidth * imageHeight * sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(green_blurred, dgreen_blurred,imageWidth * imageHeight * sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(blue_blurred, dblue_blurred,imageWidth * imageHeight * sizeof(float),cudaMemcpyDeviceToHost);


  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

 for(int i=0;i<imageWidth * imageHeight; i++)
 {
   hostOutputImageData[3 * i] = red_blurred[i];
   hostOutputImageData[3 * i + 1] = green_blurred[i];
   hostOutputImageData[3 * i + 2] = blue_blurred[i];
 }

  wbSolution(args, outputImage);



  // cudaFree(deviceInputImageData);
  // cudaFree(deviceOutputImageData);
  cudaFree(dred_channel);
  cudaFree(dgreen_channel);
  cudaFree(dblue_channel);

  cudaFree(dred_blurred);
  cudaFree(dgreen_blurred);
  cudaFree(dblue_blurred);


  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
