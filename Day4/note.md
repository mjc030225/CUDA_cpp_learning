### 基本的 CUDA 程序框架

A basic CUDA program typically follows these steps:

1. **Initialize the device**
    - Set the device to be used for computation.
    - Example:
      ```cpp
      setDevice();
      ```

2. **Allocate memory**
    - Allocate memory on both the host and the device.
    - Example:
      ```cpp
      FullMemory ful_mem = setMemory_Init(numElements, randomSeed);
      ```

3. **Copy data to the device**
    - Transfer data from the host to the device.
    - Example:
      ```cpp
      cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
      ```

4. **Launch the kernel**
    - Define the execution configuration and launch the kernel.
    - Example:
      ```cpp
      dim3 block(1);
      dim3 grid(1);
      add_with_gpu<<<grid, block>>>(d_A, d_B, d_C, numElements);
      ```

5. **Synchronize**
    - Ensure all threads have completed execution.
    - Example:
      ```cpp
      cudaDeviceSynchronize();
      ```

6. **Copy data back to the host**
    - Transfer the results from the device back to the host.
    - Example:
      ```cpp
      cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
      ```

7. **Free memory**
    - Free the memory allocated on both the host and the device.
    - Example:
      ```cpp
      freeMemory(ful_mem);
      ```

### Details of Device Initialization and Memory Allocation

#### Device Initialization
The main idea of device initialization is to ensure that there is an available CUDA device and set it as the current device. The specific steps are as follows:

1. **Get the number of devices**
    - Use `cudaGetDeviceCount` to get the number of available CUDA devices.
    - If no devices are available or the device count retrieval fails, the program terminates.
    - Example:
      ```cpp
      int count_device = 0;
      cudaError_t error = cudaGetDeviceCount(&count_device);
      if (error != cudaSuccess || count_device == 0) {
          printf("no CUDA capable devices were detected\n");
          exit(-1);
      }
      ```

2. **Set the device**
    - Use `cudaSetDevice` to set the device to be used.
    - If setting the device fails, the program terminates.
    - Example:
      ```cpp
      int set_device = 0;
      error = cudaSetDevice(set_device);
      if (error != cudaSuccess) {
          printf("cudaSetDevice returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
          exit(-1);
      }
      ```

#### Memory Allocation
The main idea of memory allocation is to ensure that there is enough memory on both the host and the device to store data and transfer data when needed. The specific steps are as follows:

1. **Allocate memory on the host**
    - Use `malloc` to allocate memory on the host.
    - Initialize the memory to 0.
    - Example:
      ```cpp
      size_t nBytes = nElem * sizeof(float);
      float *h_A = (float *)malloc(nBytes);
      float *h_B = (float *)malloc(nBytes);
      float *h_C = (float *)malloc(nBytes);
      memset(h_A, 0, nBytes);
      memset(h_B, 0, nBytes);
      memset(h_C, 0, nBytes);
      ```

2. **Allocate memory on the device**
    - Use `cudaMalloc` to allocate memory on the device.
    - Example:
      ```cpp
      float *d_A, *d_B, *d_C;
      cudaMalloc((float **)&d_A, nBytes);
      cudaMalloc((float **)&d_B, nBytes);
      cudaMalloc((float **)&d_C, nBytes);
      ```

3. **Copy data from the host to the device**
    - Use `cudaMemcpy` to copy data from the host to the device.
    - Example:
      ```cpp
      cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
      cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice);
      ```

4. **Free memory**
    - Use `free` to free host memory and `cudaFree` to free device memory.
    - Example:
      ```cpp
      free(fullMemory.hostMemory._A);
      free(fullMemory.hostMemory._B);
      free(fullMemory.hostMemory._C);
      cudaFree(fullMemory.deviceMemory._A);
      cudaFree(fullMemory.deviceMemory._B);
      cudaFree(fullMemory.deviceMemory._C);
      ```

By following these steps, you can ensure the efficiency and correctness of CUDA programs in terms of device initialization and memory management.