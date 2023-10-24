
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)



__global__  void Cal_cross_sum(){}
  __shared__ float subTile1[tile_width][tile_width]
  __shared__ float subTile2[tile_width][tile_width]
 //@@ Insert code to implement matrix multiplication 
  const int TILE_WIDTH = 5;
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  int width = numAColumns;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

  for (int q = 0; q < ((width - 1) / TILE_WIDTH + 1); ++q){
    if ((Row < numARows) && ((q * TILE_WIDTH + tx) < width)){
      subTileM[ty][tx] = A[Row * width + q * TILE_WIDTH + tx];
    } else{
      subTileM[ty][tx] = 0;
    }

    if ((q * TILE_WIDTH + ty < width) && (Col < numBColumns)){
      subTileN[ty][tx] = B[(q * TILE_WIDTH + ty) * numBColumns + Col];
    } else{
      subTileN[ty][tx] = 0;
    }

    __syncthreads();
    if ((Row < numCRows) && (Col < numCColumns)){
      for (int k = 0; k < TILE_WIDTH; ++k){
        Pvalue += subTileM[ty][k] * subTileN[k][tx];
      }
    }
    __syncthreads();
  }
  if ((Row < numARows) && (Col < numCColumns)){
    C[Row * numCColumns + Col] = Pvalue;
  }

__host__ void CrossSum(int *Data, int *OutData, int size){
  cudaMalloc((void **) &deviceA, sizeA);
  cudaMalloc((void **) &deviceB, sizeB);
  cudaMalloc((void **) &deviceC, sizeC);


}






// Compute C = A * B
__host__ void CrossSum(int *Data, int *OutData, int size){

}
 
}


__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication 
  const int TILE_WIDTH = 2;
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  int width = numAColumns;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

  for (int q = 0; q < ((width - 1) / TILE_WIDTH + 1); ++q){
    if ((Row < numARows) && ((q * TILE_WIDTH + tx) < width)){
      subTileM[ty][tx] = A[Row * width + q * TILE_WIDTH + tx];
    } else{
      subTileM[ty][tx] = 0;
    }

    if ((q * TILE_WIDTH + ty < width) && (Col < numBColumns)){
      subTileN[ty][tx] = B[(q * TILE_WIDTH + ty) * numBColumns + Col];
    } else{
      subTileN[ty][tx] = 0;
    }

    __syncthreads();
    if ((Row < numCRows) && (Col < numCColumns)){
      for (int k = 0; k < TILE_WIDTH; ++k){
        Pvalue += subTileM[ty][k] * subTileN[k][tx];
      }
    }
    __syncthreads();
  }
  if ((Row < numARows) && (Col < numCColumns)){
    C[Row * numCColumns + Col] = Pvalue;
  }
}



int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int sizeA = numARows * numAColumns * sizeof(float);
  int sizeB = numBRows * numBColumns * sizeof(float);
  int sizeC = numCRows * numCColumns * sizeof(float);

  cudaMalloc((void **) &deviceA, sizeA);
  cudaMalloc((void **) &deviceB, sizeB);
  cudaMalloc((void **) &deviceC, sizeC);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int TILE_WIDTH = 2;
  dim3 DimGrid(ceil((1.0 * numCColumns) / TILE_WIDTH), ceil((1.0 * numCRows) / TILE_WIDTH), 1); // (x, y, z)
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
