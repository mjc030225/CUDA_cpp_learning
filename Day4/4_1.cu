// setting device and init
#include <stdio.h>
#include "../tools/common.cuh"

int main(void) {
    setDevice();
    //init the device memory set
    // DeviceMemory dev_mem = setMemory_Init(512, 666);
    FullMemory ful_mem = setMemory_Init(512, 666);
    DeviceMemory dev_mem = ful_mem.deviceMemory;
    DeviceMemory host_mem = ful_mem.hostMemory;

    float *d_A = dev_mem._A;
    float *d_B = dev_mem._B;
    float *d_C = dev_mem._C;
    float *h_A = host_mem._A;
    float *h_B = host_mem._B;
    float *h_C = host_mem._C;
    
    return 0;
}