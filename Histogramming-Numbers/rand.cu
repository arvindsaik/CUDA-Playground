#include "wb.h"
#include<bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[])
{
  int inputLength;
  wbArg_t args = wbArg_read(argc, argv);
  char *file = wbArg_getInputFile(args, 0);

  float *data = wbImport(file, &inputLength);
  cout<<inputLength<<endl;
  for(int i=0;i<10;i++)
    cout<<sizeof((unsigned int)data[i])<<" ";

  return 0;
}
