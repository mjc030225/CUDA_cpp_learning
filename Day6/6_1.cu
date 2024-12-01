// query the info of GPU device
#include <stdio.h>
#include "../tools/common.cuh"

int main()
{
    int deviceid = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceid);
    printf("Device %d: %s\n", deviceid, prop.name);
    // Print the compute capability of the device
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    // Print the total amount of global memory available on the device in bytes
    printf("Total global memory: %lu bytes\n", prop.totalGlobalMem);
    printf("Shared memory per block: %lu\n", prop.sharedMemPerBlock);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Total constant memory: %lu\n", prop.totalConstMem);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("L2 cache size: %d\n", prop.l2CacheSize);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Unified addressing: %d\n", prop.unifiedAddressing);
    printf("Async engine count: %d\n", prop.asyncEngineCount);
    printf("Memory clock rate: %d\n", prop.memoryClockRate);
    printf("Memory bus width: %d\n", prop.memoryBusWidth);
    printf("Peak memory bandwidth: %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    return 0;
}