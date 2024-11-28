//a simple ex for memory copy between host and device
#include <stdio.h>
#include "../tools/common.cuh"


int main()
{
    setDevice();
    //allocate memory on host
    int nElem = 512;
    //size_t is a type that can hold the maximum size of a theoretically possible object of any type
    //sizeof(float) is the size of a float type
    //malloc is a function that allocates a block of memory
    size_t nBytes = nElem * sizeof(float);
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *h_C = (float *)malloc(nBytes);
    //initialize data on host, dafault is 0
    if (h_A != NULL && h_B != NULL && h_C != NULL)
    {
        memset(h_A, 0, nBytes);
        memset(h_B, 0, nBytes);
        memset(h_C, 0, nBytes);
        printf("Memory allocation on host is successful\n");
    }
    else
    {
        printf("Memory allocation on host is failed\n");
        exit(-1);
    }
    //allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);
    //initialize data on device, copy data from host to device
    //there are 4 types of memory copy
    //cudaMemcpyHostToHost,means copy data from host to host
    //cudaMemcpyHostToDevice,means copy data from host to device
    //cudaMemcpyDeviceToHost,means copy data from device to host
    //cudaMemcpyDeviceToDevice,means copy data from device to device
    if (d_A != NULL && d_B != NULL && d_C != NULL)
    {   
        cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);
        printf("Memory allocation on device is successful\n");
        printf("Memory copy from host to device is successful\n");
    }
    else
    {   
        printf("Memory allocation on device is failed\n");
    //failed to copy data from host to device, so free the memory on host is enough
        free(h_A);
        free(h_B);
        free(h_C);
        //exit is different from return, exit will terminate the program
        exit(-1);
    }
    //initialize data on host,via random seed
    srand(1234);
    for (int i = 0; i < nElem; i++)
    {
        h_A[i] = (float)(rand() & 0xFF) / 10.0f;
        h_B[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    //copy data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);
    printf("Memory copy from host to device is successful\n");

}