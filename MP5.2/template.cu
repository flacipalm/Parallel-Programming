// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 128 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add_sum(float *input, float *output, float *block_sum, int len)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = bx * BLOCK_SIZE + tx; 
  int index_start = index * BLOCK_SIZE * 2;
  if (index < len){
    float add_item = 0;
    if (index > 0){
      add_item = block_sum[index - 1]; 
    }
    for (int i = index_start; i < index_start + BLOCK_SIZE * 2; ++i){
      output[i] = input[i] + add_item; 
    }
  }
  __syncthreads();
}

__global__ void scan(float *input, float *output, int len) {
  __shared__ float T[2 * BLOCK_SIZE];
  int tx = threadIdx.x;
  int index = blockIdx.x * BLOCK_SIZE + tx;
  int index_store = blockIdx.x * BLOCK_SIZE * 2 + tx;

  T[tx] = (index_store >= len) ? 0 : input[index_store];
  T[tx + BLOCK_SIZE] = (index_store + BLOCK_SIZE >= len) ? 0 : input[index_store + BLOCK_SIZE];
  __syncthreads();

  int stride = 1;
  while (stride < 2 * BLOCK_SIZE){
    int idx = (threadIdx.x + 1) * stride * 2 - 1;
    if (idx < 2 * BLOCK_SIZE && (idx - stride) >= 0)
      T[idx] += T[idx - stride];
    stride *= 2;
    __syncthreads();
  }
  
  stride = BLOCK_SIZE / 2;
  while (stride > 0){
    int idx = (threadIdx.x + 1) * stride * 2 - 1;
    if (idx + stride < 2 * BLOCK_SIZE){
      T[idx + stride] += T[idx]; 
    }
    stride /= 2;
    __syncthreads();
  }

  if (index_store < len){
    output[index_store] = T[tx];
  }
  if (index_store + BLOCK_SIZE < len){
    output[index_store + BLOCK_SIZE] = T[tx + BLOCK_SIZE];
  } 
  return;
}




int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int grid_size = ceil((double)numElements / (2 * BLOCK_SIZE));
  dim3 gridDim(grid_size, 1, 1);
  dim3 blockDim(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  float *aux_array = (float *) malloc(grid_size * sizeof(float));
  int index_start = 0;
  for (int i = 0; i < grid_size; ++i)
  {
    index_start += (BLOCK_SIZE * 2);
    if (index_start > numElements){
      aux_array[i] = hostOutput[numElements - 1];
    } 
    else{
      aux_array[i] = hostOutput[index_start - 1];
    } 
  }
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float))); 
  cudaMemcpy(deviceInput, aux_array, grid_size * sizeof(float), cudaMemcpyHostToDevice); 

  dim3 blockDim2(BLOCK_SIZE, 1, 1); 
  dim3 gridDim2(ceil((double) grid_size / (2 * BLOCK_SIZE)), 1, 1);
  scan<<<gridDim2, blockDim2>>>(deviceInput, deviceOutput, grid_size);
  cudaDeviceSynchronize();
  cudaMemcpy(aux_array, deviceOutput, sizeof(float) * grid_size, cudaMemcpyDeviceToHost);

  float * final_sum;
  cudaMalloc((void **) & final_sum, grid_size * sizeof(float));
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float))); 
  wbCheck(cudaMemcpy(deviceInput, hostOutput, numElements * sizeof(float), cudaMemcpyHostToDevice)); 
  wbCheck(cudaMemcpy(final_sum, aux_array, grid_size * sizeof(float), cudaMemcpyHostToDevice));
  int final_block_idx = ceil((double) (grid_size-1) / BLOCK_SIZE);
  dim3 gridDim3(final_block_idx, 1, 1);
  dim3 blockDim3(BLOCK_SIZE, 1, 1); 
  add_sum<<<gridDim3, blockDim3>>>(deviceInput, deviceOutput, final_sum, grid_size); 
  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),cudaMemcpyDeviceToHost)); 
  cudaFree(final_sum);

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(aux_array);

  return 0;
}
