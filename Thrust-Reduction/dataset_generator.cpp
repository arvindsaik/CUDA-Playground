#include <unistd.h>

#include "wb.h"

#include "/Users/derik_clive/stdc++.h"

// #include<cstring>
// #include <iostream>
// #include <fstream>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>


using namespace std;
char base_dir[100];

static float compute(float *input, int num) {
    float ret = 0;
    for (int ii = 0; ii < num; ++ii) {
        ret += input[ii];
    }
    return ret;
}

static float *generate_data(int n) {
    float *data = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) {
        data[i] = ((float)(rand() % 20) - 5) / 5.0f;
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

static void create_dataset(int datasetNum, int dim) {
    
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
    
    strcat(input_file_name, " input.raw");
    strcat(output_file_name, " output.raw");
    
    
    float *input_data = generate_data(dim);
    float output_data;
    
    output_data = compute(input_data, dim);
    
    write_data(input_file_name, input_data, dim);
    write_data(output_file_name, &output_data, 1);
    
    free(input_data);
}

int main() {
    char buffer[100];
    char *answer = getcwd(buffer, sizeof(buffer));
    
    strcpy(base_dir, answer);
    
    strcat(base_dir, "/ListScan Dataset");
    create_dataset(0, 16);
    create_dataset(1, 64);
    create_dataset(2, 93);
    create_dataset(3, 112);
    create_dataset(4, 1120);
    create_dataset(5, 1233);
    create_dataset(6, 1033);
    create_dataset(7, 4098);
    create_dataset(8, 4018);
    create_dataset(9, 1);
    return 0;
}
