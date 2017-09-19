#include <bits/stdc++.h>
#include <sstream>

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()


using namespace std;

static char *base_dir;

static void compute(float *output, float *input0, float *input1, int P, int Q, int R) {

  for(int i=0; i<P; i++)
  {
    for(int j=0; j<R; j++)
    {
      output[i * R + j] = 0;
      for(int k=0; k<Q; k++)
      {
        output[i * R + j] += input0[i * Q + k] * input1[k * R +j];
      }
    }
  }
}

static float *generate_data(int m, int n) {
  float *data = (float *)malloc(sizeof(float) * n * m);
  for (int i = 0; i < n * m; i++) {
    data[i] = ((float)(rand() % 20) - 5) ;
   // data[i] = 1.0;
  }
  return data;
}

static void write_data(char *file_name, float *data, int num) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d", num);
  for (int ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%.2f", *data++);
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, int P, int Q, int R) {

  const char *dir_name="base_dir";

  string a = "input0_", b = "input1_", c = "output_";
  string num = SSTR(datasetNum);
  a += num;
  b += num;
  c += num;
  a += ".raw";
  b += ".raw";
  c += ".raw";

  char input0_file_name[15];
  char input1_file_name[15];
  char output_file_name[15];

  strcpy(input0_file_name, a.c_str());
  strcpy(input1_file_name, b.c_str());
  strcpy(output_file_name, c.c_str());

  float *input0_data = generate_data(P, Q);
  float *input1_data = generate_data(Q, R);
  float *output_data = (float *)calloc(sizeof(float), P * R);

  compute(output_data, input0_data, input1_data, P, Q, R);

  write_data(input0_file_name, input0_data, P * Q);
  write_data(input1_file_name, input1_data, Q * R);
  write_data(output_file_name, output_data, P * R);

  free(input0_data);
  free(input1_data);
  free(output_data);
}

int main() {

  create_dataset(0, 16, 16, 16);
  create_dataset(1, 32, 16, 32);
  create_dataset(2, 64, 32, 32);
  create_dataset(3, 128, 32, 64);
  create_dataset(4, 127, 64, 127);
  create_dataset(5, 256, 128, 64);
  create_dataset(6, 324, 124, 243);
  create_dataset(7, 434, 321, 452);
  create_dataset(8, 1020, 104, 32);
  create_dataset(9, 302, 124, 35);
  return 0;
}
