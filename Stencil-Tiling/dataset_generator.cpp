#include <unistd.h>
#include "wb.h"

#include<bits/stdc++.h>
// #include<cstring>
// #include <iostream>
// #include <fstream>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

#define MAX_VAL 255
#define Max(a, b) ((a) < (b) ? (b) : (a))
#define Min(a, b) ((a) > (b) ? (b) : (a))
#define Clamp(a, start, end) Max(Min(a, end), start)
#define idx(i, j, k) ((i)*width + (j)) * depth + (k)
#define value(arry, i, j, k) arry[((i)*width + (j)) * depth + (k)]

 char base_dir[100];

static void compute( unsigned char *out,  unsigned char *in, int width,
                    int height, int depth) {

#define out(i, j, k) value(out, i, j, k)
#define in(i, j, k) value(in, i, j, k)

  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      for (int k = 1; k < depth - 1; ++k) {
        int res = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) +
                  in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) -
                  6 * in(i, j, k);
        res = Clamp(res, 0, MAX_VAL);
        // printf("%d\n", res);
        out(i, j, k) = res;
      }
    }
  }
#undef out
#undef in
}

static  unsigned char *generate_data(int width, int height, int depth) {
   unsigned char *data = ( unsigned char *)malloc(sizeof( unsigned char) *
                                                width * height * depth);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < depth; ++k) {
        value(data, i, j, k) =( unsigned char)( rand() % MAX_VAL);
      }
    }
  }
  return data;
}

static void write_data(char *file_name,  unsigned char *data, int width,
                       int height, int depth) {
  FILE *handle = fopen(file_name, "wb");
  fprintf(handle, "P6\n");
  fprintf(handle, "%d %d %d\n", width, height, depth);
//  fprintf(handle, "1\n");
  fwrite(data, depth * width * sizeof( unsigned char), height, handle);
  fflush(handle);
  fclose(handle);
}

void create_dataset(int datasetNum, int width, int height, int depth) {


  stringstream ss;
  ss << datasetNum;

  char dir_name[100];
  strcpy(dir_name, base_dir);
  mkdir(dir_name, 0775);

  strcat(dir_name, "/");
  strcat(dir_name, (ss.str()).c_str());

   char input_file_name[100],output_file_name[100];
  strcpy(input_file_name, dir_name);
  strcpy(output_file_name, dir_name);

  strcat(input_file_name, " input.ppm");
  strcat(output_file_name, " output.ppm");


   unsigned char *input_data  = generate_data(width, height, depth);
   unsigned char *output_data = ( unsigned char *)calloc(
      sizeof( unsigned char), width * height * depth);

if(datasetNum==9)
{
cout<<idx(2, 0, 2)<<endl;  
cout<<width<<" "<< depth<<endl;
cout<<(int)value(input_data, 2, 0, 2)<<endl;
cout<<(int)value(output_data, 2, 0, 2)<<endl;
}
	
  compute(output_data, input_data, width, height, depth);


  write_data(input_file_name, input_data, width, height, depth);
  write_data(output_file_name, output_data, width, height, depth);
}

int main() {
  char buffer[100];
  char *answer = getcwd(buffer, sizeof(buffer));

  strcpy(base_dir, answer);


  strcat(base_dir, "/Stencil Dataset");
  create_dataset(0, 1024, 1024, 4);
  create_dataset(1, 1024, 2048, 5);
  create_dataset(2, 1023, 9, 1048);
  create_dataset(3, 1023, 1022, 8);
  create_dataset(4, 10, 1012, 1023);
  create_dataset(5, 1003, 9, 1024);
  create_dataset(6, 6, 1021, 1241);
  create_dataset(7, 9, 9, 1241);
  create_dataset(8, 1921, 19, 1241);
  create_dataset(9, 28, 28, 4);
  return 0;
}
