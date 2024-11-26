// a simple example of defining kernel func
#include <stdio.h>
// must start with __global__,return void
//kernel func
__global__ void kernel_func(){
    printf("hello world!\n");
}

int main(){
    //get the gpu space
    //<<<grid,block_size>>>
    kernel_func<<<4,4>>>();
    cudaDeviceSynchronize();
    return 0;
}