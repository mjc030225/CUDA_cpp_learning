//// a simple example of multi-thread 2-D
#include <stdio.h>
// how to distinguish the idx of the thread
// idx = threadIdx + blkdim * blk_idx
// has 3-D : blockIdx.x,y,z
// 2-D : blockIdx.x,y
__global__ void distinguish_idx_1D(){
    const int bdx = blockIdx.x; //
    const int thx = threadIdx.x;
    const int blk_d = blockDim.x;
    // get the thread id
    const int thread_idx = thx + bdx * blk_d;
    printf("bdx:%d,thx:%d,blk_d:%d,thread_idx:%d\n",bdx,thx,blk_d,thread_idx);
 }
 //3-d block and 3-d thread
__global__ void distinguish_idx_2D(){
    const int bdx = blockIdx.x; // x
    const int bdy = blockIdx.y; // y
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int blk_d = blockDim.x;

    // get the thread id
    const int thread_idx = thx * thy + bdx * bdy * blk_d;
    printf("bdx:%d,thx:%d,blk_d:%d,thread_idx:%d\n",bdx,thx,blk_d,thread_idx);
}
__global__ void distinguish_idx_3D(){
    const int bdx = blockIdx.x; // x
    const int bdy = blockIdx.y; // y
    const int bdz = blockIdx.z; // z
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int thz = threadIdx.z;
    const int blk_dx = blockDim.x;
    const int blk_dy = blockDim.y;
    const int blk_dz = blockDim.z;

    // 计算线程ID
    const int thread_idx = thx + thy * blk_dx + thz * blk_dx * blk_dy;
    const int block_idx = bdx + bdy * gridDim.x + bdz * gridDim.x * gridDim.y;
    const int global_thread_idx = thread_idx + block_idx * (blk_dx * blk_dy * blk_dz);

    printf("bdx:%d, bdy:%d, bdz:%d, thx:%d, thy:%d, thz:%d, blk_dx:%d, blk_dy:%d, blk_dz:%d, global_thread_idx:%d\n",
           bdx, bdy, bdz, thx, thy, thz, blk_dx, blk_dy, blk_dz, global_thread_idx);
}

int main(){
    dim3 bdx(2,4);
    dim3 thx(2,4);
    distinguish_idx_2D<<<bdx,thx>>>();
    // distinguish_idx_1D<<<2,4>>>();
    // is necessary
    cudaDeviceSynchronize();
    return 0;
}
