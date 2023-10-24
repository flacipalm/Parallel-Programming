#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 12

__constant__ float ConstMask[8192];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    
    /***added*/
    extern __shared__ float treeM[];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;


#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) ConstMask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

/***added*/
#define treeR(i2, i1, i0) treeM[i2 * TILE_WIDTH * Channel + i1 * Channel + i0]


    const int W_grid = (Width_out - 1)/ TILE_WIDTH + 1;
    // Insert your GPU convolution kernel code here
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    if ((h < Height_out) && (w < Width_out)){
        float res = 0;
        for (int p=0; p<K; p++){
            for (int q=0; q<K; q++){
                res += in_4d(blockIdx.x, threadIdx.z, h+p, w+q) * mask_4d(blockIdx.y, threadIdx.z, p, q);
            }
        }    
        treeR(threadIdx.y, threadIdx.x, threadIdx.z) = res;
        for (int stride = 1; stride < Channel; stride *= 2)
        {
            __syncthreads();
            if ((threadIdx.z % (2 * stride) == 0) && (threadIdx.z + stride < Channel))
            {
                treeR(threadIdx.y, threadIdx.x, threadIdx.z) += treeR(threadIdx.y, threadIdx.x, threadIdx.z + stride);
            }
        }
        __syncthreads();
        if (threadIdx.z == 0)
        {
            out_4d(blockIdx.x, blockIdx.y, h, w) = treeR(threadIdx.y, threadIdx.x, 0);
        }

    }



#undef out_4d
#undef in_4d
#undef mask_4d
#undef treeR

}

__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void **) device_output_ptr, (Batch * Map_out * Height_out * Width_out) * sizeof(float));
    cudaMalloc((void **) device_input_ptr, (Batch * Channel * Height * Width) * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, (Batch * Channel * Height * Width) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ConstMask, host_mask, (Map_out * Channel * K * K) * sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    dim3 dimGrid(Batch, Map_out, ceil((float)Height_out / TILE_WIDTH) * ceil((float)Width_out / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, Channel);
    conv_forward_kernel<<<dimGrid, dimBlock, Channel * sizeof(float) * TILE_WIDTH * TILE_WIDTH >>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);    
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_input); 
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
