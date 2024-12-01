## use nvcc compilation to check the register using
```bash
nvcc -Xptxas -v -o output_file source_file.cu
```
or 
```bash
nvcc --resource-usage xxxx.cu -o output
```

**Different arch leads to different consumption.** 
```bash
# sm_52
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z14add_by2DthreadPfS_S_ii' for 'sm_52'
ptxas info    : Function properties for _Z14add_by2DthreadPfS_S_ii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, 352 bytes cmem[0]
# sm_70
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z14add_by2DthreadPfS_S_ii' for 'sm_70'
ptxas info    : Function properties for _Z14add_by2DthreadPfS_S_ii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, 384 bytes cmem[0]
```