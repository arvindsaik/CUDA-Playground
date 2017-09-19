
// #include "wb.h"
#include<bits/stdc++.h>
using namespace std;
static char *base_dir;

static void compute(float *output, float *input, int num) {
  int ii;
  float accum = 0;
  for (ii = 0; ii < num; ++ii) {
    accum += input[ii];
    output[ii] = accum;
  }
}

static float *generate_data(int n) {
  int i;
  float *data = (float *)malloc(sizeof(float) * n);
  for (i = 0; i < n; i++) {
    data[i] = rand() % 4;
  }
  return data;
}

static void write_data(string file_name, float *data, int num) {
  int ii;
  FILE *handle = fopen(file_name.c_str(), "w");
  fprintf(handle, "%d", num);
  for (ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%.2f", *data++);
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, int dim) {
  // const char *dir_name =
  //     wbDirectory_create(wbPath_join(base_dir, datasetNum));
  string input_file_name  = "input";
  input_file_name+= char(datasetNum+'0');
  input_file_name+= ".raw";
  string output_file_name = "output";
  output_file_name+= char(datasetNum+'0');
  output_file_name+= ".raw";

  float *input_data  = generate_data(dim);
  float *output_data = (float *)malloc(sizeof(float) * dim);

  compute(output_data, input_data, dim);

  write_data(input_file_name, input_data, dim);
  write_data(output_file_name, output_data, dim);

  free(input_data);
  free(output_data);
}

int main() {
  // base_dir =
  // wbPath_join(wbDirectory_current(), "ThrustListScan", "Dataset");
  create_dataset(0, 16);
  create_dataset(1, 64);
  create_dataset(2, 93);
  create_dataset(3, 112);
  create_dataset(4, 1120);
  create_dataset(5, 9921);
  create_dataset(6, 1233);
  create_dataset(7, 1033);
  create_dataset(8, 4098);
  create_dataset(9, 4018);
  return 0;
}
