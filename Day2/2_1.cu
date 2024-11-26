// a simple example of multi-thread:1-D
#include <stdio.h>
// how to distinguish the idx of the thread
// idx = threadIdx + blkdim * blk_idx
__global__ void distinguish_idx(){
    const int bdx = blockIdx.x;
    const int thx = threadIdx.x;
    const int blk_d = blockDim.x;
    // get the thread id
    const int thread_idx = thx + bdx * blk_d;
    printf("bdx:%d,thx:%d,blk_d:%d,thread_idx:%d\n",bdx,thx,blk_d,thread_idx);
 }

int main(){
    distinguish_idx<<<2,4>>>();
    // is necessary
    cudaDeviceSynchronize();
    return 0;
}
