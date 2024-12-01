#pragma once
// only include once
#include <stdio.h>
// #define CUDA_Check  ErrorCheck(cudaGetLastError(),__FILE__,__LINE__)
struct DeviceMemory {
    float *_A;
    float *_B;
    float *_C;
};

struct FullMemory {
    DeviceMemory deviceMemory;
    DeviceMemory hostMemory;
};
// set device
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

// set memory
FullMemory setMemory_Init(int nElem, int random){
    //allocate memory on host
    // int nElem = 512;
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
    srand(random);
    for (int i = 0; i < nElem; i++)
    {
        h_A[i] = (float)(rand() & 0xFF) / 10.0f;
        h_B[i] = (float)(rand() & 0xFF) / 10.0f;
        h_C[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    //copy data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);
    printf("Memory copy from host to device is successful\n");
    //put them together
    DeviceMemory deviceMemory = {d_A, d_B, d_C};
    DeviceMemory hostMemory = {h_A, h_B, h_C};
    FullMemory fullMemory = {deviceMemory, hostMemory};
    return fullMemory;
}

void freeMemory(FullMemory fullMemory){
    //free memory on host
    free(fullMemory.hostMemory._A);
    free(fullMemory.hostMemory._B);
    free(fullMemory.hostMemory._C);
    //free memory on device
    cudaFree(fullMemory.deviceMemory._A);
    cudaFree(fullMemory.deviceMemory._B);
    cudaFree(fullMemory.deviceMemory._C);
    printf("Memory free is successful\n");
}

cudaError_t ErrorCheck(cudaError_t error_code, const char* file_name,int linenumber){
    if (error_code != cudaSuccess)
    {
        printf("Find CUDA error at %s:%d\r\n, Error Type %d (%s), means that %s", file_name, linenumber
        ,error_code,cudaGetErrorName(error_code), cudaGetErrorString(error_code));
        return error_code;
    }
    return error_code;
}
// calculate time without checking error
float Calculate_Time_nocheck(cudaEvent_t start, cudaEvent_t stop){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //cudaenventquery is a function that returns cudaSuccess if all operations have completed
    cudaEventRecord(start, 0);
    cudaEventQuery(start);// cannot use Errorcheck here
    cudaEventRecord(stop, 0);
    //cudaEventSynchronize is a function that waits for an event to complete
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

// calculate time with checking error
float Calculate_Time(cudaEvent_t start, cudaEvent_t stop, const char* file_name, int linenumber){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cudaenventquery is a function that returns cudaSuccess if all operations have completed
    ErrorCheck(cudaEventRecord(start, 0), file_name, linenumber);
    cudaEventQuery(start);// cannot use Errorcheck here
    ErrorCheck(cudaEventRecord(stop, 0), file_name, linenumber);
    //cudaEventSynchronize is a function that waits for an event to complete
    ErrorCheck(cudaEventSynchronize(stop), file_name, linenumber);
    float elapsedTime;
    ErrorCheck(cudaEventElapsedTime(&elapsedTime, start, stop), file_name, linenumber);
    return elapsedTime;
}