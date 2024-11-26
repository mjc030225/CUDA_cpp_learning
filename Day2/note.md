# Notes from 2_1.cu and 2_2.cu

## 2_1.cu
- **Introduction to CUDA Programming**
    - Basics of CUDA (Compute Unified Device Architecture)
        - Overview of CUDA and its purpose
    - Understanding the GPU architecture
        - Differences between CPU and GPU architectures
    - Writing a simple CUDA kernel
        - Syntax and structure of a CUDA kernel
    - Launching a kernel from the host code
        - How to call a CUDA kernel from C++ code
    - Memory management between host and device
        - Transferring data between CPU and GPU memory

## 2_2.cu
- **Advanced CUDA Concepts**
    - Thread hierarchy (blocks and grids)
        - Organizing threads into blocks and grids
    - Synchronization and communication between threads
        - Techniques for synchronizing threads within a block
    - Shared memory usage
        - Benefits and usage of shared memory in CUDA
    - Performance optimization techniques
        - Strategies to improve CUDA application performance
    - Debugging CUDA applications
        - Tools and methods for debugging CUDA code
        ## Coding Examples

        ### Example from 2_1.cu
        ```cpp
        // Simple CUDA kernel
        __global__ void add(int *a, int *b, int *c) {
            int index = threadIdx.x;
            c[index] = a[index] + b[index];
        }

        int main() {
            const int arraySize = 5;
            int a[arraySize] = {1, 2, 3, 4, 5};
            int b[arraySize] = {10, 20, 30, 40, 50};
            int c[arraySize] = {0};

            int *d_a, *d_b, *d_c;
            cudaMalloc((void**)&d_a, arraySize * sizeof(int));
            cudaMalloc((void**)&d_b, arraySize * sizeof(int));
            cudaMalloc((void**)&d_c, arraySize * sizeof(int));

            cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

            add<<<1, arraySize>>>(d_a, d_b, d_c);

            cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);

            for (int i = 0; i < arraySize; i++) {
                printf("%d + %d = %d\n", a[i], b[i], c[i]);
            }

            return 0;
        }
        ```

        ### Example from 2_2.cu
        ```cpp
        // CUDA kernel using shared memory
        __global__ void matrixMul(int *a, int *b, int *c, int N) {
            __shared__ int shared_a[32][32];
            __shared__ int shared_b[32][32];

            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int row = blockIdx.y * blockDim.y + ty;
            int col = blockIdx.x * blockDim.x + tx;

            int sum = 0;
            for (int i = 0; i < N / 32; ++i) {
                shared_a[ty][tx] = a[row * N + (i * 32 + tx)];
                shared_b[ty][tx] = b[(i * 32 + ty) * N + col];
                __syncthreads();

                for (int j = 0; j < 32; ++j) {
                    sum += shared_a[ty][j] * shared_b[j][tx];
                }
                __syncthreads();
            }
            c[row * N + col] = sum;
        }

        int main() {
            const int N = 64;
            int a[N][N], b[N][N], c[N][N];
            // Initialize matrices a and b with some values

            int *d_a, *d_b, *d_c;
            cudaMalloc((void**)&d_a, N * N * sizeof(int));
            cudaMalloc((void**)&d_b, N * N * sizeof(int));
            cudaMalloc((void**)&d_c, N * N * sizeof(int));

            cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

            dim3 threadsPerBlock(32, 32);
            dim3 blocksPerGrid(N / 32, N / 32);
            matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

            cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);

            // Print matrix c
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    printf("%d ", c[i][j]);
                }
                printf("\n");
            }

            return 0;
        }
        ```
