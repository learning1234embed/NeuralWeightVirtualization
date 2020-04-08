#include <cuda.h>
#include <stdio.h>

__global__ void GetWeightKernel(float *input, int input_len, float *addr,
		int *page_table_addr, int page_size, int start, int end)
{
	int idx, page_num, page, offset;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_len;
			i += blockDim.x * gridDim.x) {
		idx = start+i;
		page_num = idx / page_size;
		page = page_table_addr[page_num];
		offset = idx % page_size;
		input[i] = addr[page*page_size + offset];
	}
}

extern "C" {
void GetWeightKernelLauncher(float *input, int input_len, float* addr,
		int* page_table_addr, int page_size, int start, int end)
{
	GetWeightKernel<<<32, 256>>>(input, input_len, addr,
			page_table_addr, page_size, start, end);
	cudaDeviceSynchronize();
}
}
