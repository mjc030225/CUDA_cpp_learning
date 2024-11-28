#include <stdio.h>

void setDevice(){
    // init value is 0
    int count_device = 0;
    cudaError_t error = cudaGetDeviceCount(&count_device);
    if (error != cudaSuccess || count_device == 0) {
        printf("no CUDA capable devices were detected\n");
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        exit(-1);
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", count_device);
    }
    int set_device = 0;
    error = cudaSetDevice(set_device);
    if (error != cudaSuccess) {
        printf("cudaSetDevice returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        exit(-1);
    }
    else
    {
        printf("Set device to %d\n", set_device);
    }
}