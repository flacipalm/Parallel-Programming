#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
# define TILE_WIDTH 3
# define MASK_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float DeviceKernel[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x; int tx = threadIdx.x;
  int by = blockIdx.y; int ty = threadIdx.y;
  int bz = blockIdx.z; int tz = threadIdx.z;
  int row_num = by * TILE_WIDTH + ty;
  int col_num = bx * TILE_WIDTH + tx;
  int height = bz * TILE_WIDTH + tz;

  int inputIdx = height * (x_size * y_size) + row_num * x_size + col_num;

  /* if ghost */
  if((col_num < x_size) && (row_num < y_size) && (height < z_size)){
    N_ds[tz][ty][tx] = input[inputIdx];
  }else{
    N_ds[tz][ty][tx] = 0;
  }
  
  __syncthreads();

  int radius = MASK_WIDTH / 2;
  int This_tile_start_point_x = bx * blockDim.x;  int Next_tile_start_point_x = (bx + 1)* blockDim.x;
  int This_tile_start_point_y = by * blockDim.y;  int Next_tile_start_point_y = (by + 1)* blockDim.y;
  int This_tile_start_point_z = bz * blockDim.z;  int Next_tile_start_point_z = (bz + 1)* blockDim.z;
  
  int start_x = col_num - radius;
  int start_y = row_num - radius;
  int start_z = height - radius;

  float Pvalue = 0;
  if((col_num < x_size) && (row_num < y_size) && (height < z_size)){
    /* loop through mask */
    for(int i = 0; i < MASK_WIDTH; i++){
      int N_id_x = start_x + i;
      for(int j = 0; j < MASK_WIDTH; j++){
        int N_id_y = start_y + j;
        for(int k = 0; k < MASK_WIDTH; k++){
          int N_id_z = start_z + k;
          int kid = k * (MASK_WIDTH * MASK_WIDTH) + j * (MASK_WIDTH) + i;
          int if_in_tile = (N_id_x >= This_tile_start_point_x) && (N_id_x < Next_tile_start_point_x) && (N_id_y >= This_tile_start_point_y) && (N_id_y < Next_tile_start_point_y) && (N_id_z >= This_tile_start_point_z) && (N_id_z < Next_tile_start_point_z);
          if(if_in_tile){
            Pvalue += N_ds[tz + k - radius][ty + j - radius][tx + i - radius] * DeviceKernel[kid];
          }else{
            int if_in_global = (N_id_x >= 0) && (N_id_x < x_size) && (N_id_y >= 0) && (N_id_y < y_size) && (N_id_z >= 0) && (N_id_z < z_size);
            if(if_in_global){
              Pvalue += input[N_id_z * (x_size * y_size) + N_id_y * x_size + N_id_x] * DeviceKernel[kid];
            }
            
          }
        }   
      }
    }
    output[inputIdx] = Pvalue;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  int sizeInput = x_size * y_size * z_size * sizeof(float);
  int sizeOutput = sizeInput;
  cudaMalloc((void **) &deviceInput, sizeInput);
  cudaMalloc((void **) &deviceOutput, sizeOutput);

  // Recall that inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, sizeInput, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(DeviceKernel, hostKernel, kernelLength * sizeof(float), 0, cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(x_size/(1.0 * TILE_WIDTH)), ceil(y_size/(1.0 * TILE_WIDTH)), ceil(z_size/(1.0 * TILE_WIDTH)));
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, sizeOutput, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
