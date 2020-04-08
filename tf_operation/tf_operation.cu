#define EIGEN_USE_GPU
#include <cmath>
#include <cuda.h>
#include <stdio.h>

void InitWeight(const float* input, const int input_len,
		long long int* output)
{
	float *weight;
	int64_t address;

	cudaMalloc(&weight, sizeof(float)*input_len);
	cudaMemcpy(weight, input, sizeof(float)*input_len, cudaMemcpyDeviceToDevice);

	address = (int64_t)weight;
	cudaMemcpy(output, &address, sizeof(address), cudaMemcpyHostToDevice);
}

void InitPageTable(const int *input, const int input_len,
		long long int *output)
{
	int *page_table;
	int64_t page_table_address;

	cudaMalloc(&page_table, sizeof(int)*input_len);
	cudaMemcpy(page_table, input, sizeof(int)*input_len, cudaMemcpyDeviceToDevice);

	page_table_address = (int64_t)page_table;
	cudaMemcpy(output, &page_table_address, sizeof(page_table_address),
			cudaMemcpyHostToDevice);
}

__global__ void GetWeightKernel(float *input, const int input_len, float *addr,
		int *page_table_addr, const int page_size, const int start, const int end)
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

void GetWeightKernelLauncher(float *input, const int input_len,
		const long long int* address, const long long int* page_table_address,
		const int page_size, const int start, const int end)
{
	int64_t addr;
	int64_t page_table_addr;

	cudaMemcpy(&addr, address, sizeof(addr), cudaMemcpyDeviceToHost);
	cudaMemcpy(&page_table_addr, page_table_address, sizeof(page_table_addr),
			cudaMemcpyDeviceToHost);

	GetWeightKernel<<<32, 256>>>(input, input_len, (float *)addr,
			(int *)page_table_addr, page_size, start, end);
	cudaDeviceSynchronize();
}

void GetWeightAddress(long long int *weight_address_list, int *weight_len_list,
		int input_len, long long int *output1, int *output2)
{
	for (int i = 0; i < input_len; i++) {
		cudaMemcpy(&output1[i], &weight_address_list[i], sizeof(long long int),
			cudaMemcpyHostToDevice);
		cudaMemcpy(&output2[i], &weight_len_list[i], sizeof(int),
			cudaMemcpyHostToDevice);
	}
}

void ReadWeight(const long long int* address, int start, int end, float *output)
{
	int64_t addr;
	cudaMemcpy(&addr, address, sizeof(addr), cudaMemcpyDeviceToHost);
	cudaMemcpy(output, (const void *)addr,
			sizeof(float)*(end-start+1), cudaMemcpyDeviceToDevice);
}

void ReadPageTable(const long long int* address, int start, int end, int *output)
{
	int64_t addr;
	cudaMemcpy(&addr, address, sizeof(addr), cudaMemcpyDeviceToHost);
	cudaMemcpy(output, (const void *)addr,
			sizeof(int)*(end-start+1), cudaMemcpyDeviceToDevice);
}

void FreeWeight(const long long int* address)
{
	cudaFree((void *)address);
}

void FreePageTable(const long long int* address)
{
	cudaFree((void *)address);
}

__device__ __forceinline__ float atomicMinFloat(float *addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
		__uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

	return old;
}

__device__ float SharingCost(const float *fisher1, const float *weight1,
		const float *fisher2, const float *weight2, int len)
{
	float cost = 0;
	for (int p = 0; p < len; p++) {
		if (fisher1[p] <= 0 || fisher2[p] <= 0) {
			continue;
		}

		float fisher_cost = fisher1[p] + fisher2[p];
		//float fisher_cost = fisher1[p] * fisher2[p];
		float weight_dist = weight1[p] - weight2[p];
		float weight_cost = weight_dist * weight_dist;
		cost += (fisher_cost * weight_cost);
	}

	return cost;
}

__global__ void SharingCostKernel(const float *fisher1, const float *weight1,
		const float *fisher2, const float *weight2, int len, float *cost)
{
	*cost = SharingCost(fisher1, weight1, fisher2, weight2, len);
}

void SharingCostKernelLauncher(const float* input1, const float* input2, const float* input3,
		const float* input4, int len, float *output)
{
	SharingCostKernel<<<32, 128>>>(input1, input2, input3, input4, len, output);
	cudaDeviceSynchronize();
}

__global__ void MinSharingCostKernel(const float *fisher1, const float *weight1,
		const float *fisher2, const float *weight2, long long int *page_list,
		int page_num, int page_size, float *min_cost, int *min_idx)
{
	float cost, old_cost;
	*min_idx = -1;
	*min_cost = 10000000000.0;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < page_num;
			i += blockDim.x * gridDim.x) {
		if (page_list[i] < 0) {
			continue;
		}

		cost = SharingCost(fisher1, weight1, &fisher2[page_size*page_list[i]],
				&weight2[page_size*page_list[i]], page_size);
		old_cost = atomicMinFloat(min_cost, cost);
		if (cost < old_cost)
			atomicExch(min_idx, i);
	}
}

__global__ void MinSharingCostMultiKernel(const float *fisher1, const float *weight1,
		const float *fisher_addr[], const float *weight_addr[],
		long long int *base_page_list, int base_page_num, float *page_cost_accum,
		long long int *page_occupation_unsorted_addr[],
		int page_size, int num_of_list, float *min_cost, int *min_idx)
{
	float cost, old_cost;
	*min_idx = -1;
	*min_cost = 10000000000.0;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < base_page_num;
			i += blockDim.x * gridDim.x) {
		if (base_page_list[i] < 0) {
			continue;
		}

		cost = page_cost_accum[base_page_list[i]];
		const float *fisher2 = fisher_addr[0] + page_size*base_page_list[i];
		const float *weight2 = weight_addr[0] + page_size*base_page_list[i];

		cost += SharingCost(fisher1, weight1, fisher2, weight2, page_size);

		for (int j = 1; j < num_of_list; j++) {
			long long int page
				= *(page_occupation_unsorted_addr[j] + base_page_list[i]);
			if (page >= 0) {
				fisher2 = fisher_addr[j] + page_size*page;
				weight2 = weight_addr[j] + page_size*page;
				cost += SharingCost(fisher1, weight1, fisher2, weight2,
					page_size);
			}
		}

		old_cost = atomicMinFloat(min_cost, cost);
		if (cost < old_cost)
			atomicExch(min_idx, i);
	}
}

__global__ void BitonicSortKernel(float *dev_values, long long int *dev_idxs, int j, int k)
{
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i^j;

	/* The threads with the lowest ids sort the array. */
	if ((ixj)>i) {
		if ((i&k)==0) {
			if (dev_values[i]<dev_values[ixj]) {
				float temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
				long long int temp2 = dev_idxs[i];
				dev_idxs[i] = dev_idxs[ixj];
				dev_idxs[ixj] = temp2;

			}
		}
		if ((i&k)!=0) {
			if (dev_values[i]>dev_values[ixj]) {
				float temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
				long long int temp2 = dev_idxs[i];
				dev_idxs[i] = dev_idxs[ixj];
				dev_idxs[ixj] = temp2;
			}
		}
	}
}

void BitonicSort(float *values, long long int *idxs, int len)
{
	int dev = 0;
	int block, thread;
	int max_thread = 0;

	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	max_thread = deviceProp.maxThreadsPerBlock;

	if (len <= max_thread) {
		block = 1;
		thread = len;
	} else {
		block = len/max_thread;
		thread = max_thread;
	}

	dim3 blocks(block, 1);
	dim3 threads(thread, 1);

	int j, k;
	/* Major step */
	for (k = 2; k <= len; k <<= 1) {
		/* Minor step */
		for (j=k>>1; j>0; j=j>>1) {
			BitonicSortKernel<<<blocks, threads>>>(values, idxs, j, k);
		}
	}
}

__global__ void PageFisherSumKernel(const float *fisher, long long int *page_list,
		int page_list_len, int page_size, float *sum)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < page_list_len;
			i += blockDim.x * gridDim.x) {
		sum[i] = 0;
		for (int j = 0; j < page_size; j++) {
			sum[i] += fisher[page_size*page_list[i]+j];
		}
	}
}

__global__ void FloatSetKernel(float *data, float set_value, int len)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
			i += blockDim.x * gridDim.x) {
		data[i] = set_value;
	}
}

void PageAlloc(const float* fisher1, const float* weight1, const float* fisher2,
		const float* weight2, int page_list1_len, int page_list2_len,
		const long long int *page_list1, const long long int *page_list2,
		int page_size, long long int *page_allocation, float *total_cost)
{
	long long int *page_list1_sorted, *page_list2_sorted;
	long long int *page_list1_dev, *page_list2_dev;
	float *page_sum1_sorted, *page_sum2_sorted;
	float *page_sum1_sorted_dev, *page_sum2_sorted_dev;
	int valid_page_num = page_list1_len < page_list2_len ? page_list1_len : page_list2_len;
	int page_list1_len_extended = 1 << (int)ceil(log2(page_list1_len));
	int page_list2_len_extended = 1 << (int)ceil(log2(page_list2_len));
	float total_min_cost = 0;
	float min_cost;
	float *min_cost_dev;
	int min_idx;
	int *min_idx_dev;
	int pos1 = 0, pos2 = 0;
	int idx = 0;

	/* copying page_list1 & page_list2 */
	cudaMalloc(&page_list1_dev, sizeof(long long int)*page_list1_len_extended);
	cudaMalloc(&page_list2_dev, sizeof(long long int)*page_list2_len_extended);
	cudaMemset(page_list1_dev, 0, sizeof(long long int)*page_list1_len_extended);
	cudaMemset(page_list2_dev, 0, sizeof(long long int)*page_list2_len_extended);
	cudaMemcpy(page_list1_dev, page_list1, sizeof(long long int)*page_list1_len,
			cudaMemcpyDeviceToDevice);
	cudaMemcpy(page_list2_dev, page_list2, sizeof(long long int)*page_list2_len,
			cudaMemcpyDeviceToDevice);

	/* fisher sum per page */
	cudaMalloc(&page_sum1_sorted_dev, sizeof(float)*page_list1_len_extended);
	cudaMalloc(&page_sum2_sorted_dev, sizeof(float)*page_list2_len_extended);
	FloatSetKernel<<<32, 128>>>(page_sum1_sorted_dev, -page_size-1.0, page_list1_len_extended);
	cudaDeviceSynchronize();
	FloatSetKernel<<<32, 128>>>(page_sum2_sorted_dev, -page_size-1.0, page_list2_len_extended);
	cudaDeviceSynchronize();

	PageFisherSumKernel<<<32, 128>>>(fisher1, page_list1_dev, page_list1_len,
			page_size, page_sum1_sorted_dev);
	cudaDeviceSynchronize();
	PageFisherSumKernel<<<32, 128>>>(fisher2, page_list2_dev, page_list2_len,
			page_size, page_sum2_sorted_dev);
	cudaDeviceSynchronize();
#if 0
	float *page_sum1_sorted2 = (float *)malloc(sizeof(float)*page_list1_len);
	float *page_sum2_sorted2 = (float *)malloc(sizeof(float)*page_list2_len);
	cudaMemcpy(page_sum1_sorted2, page_sum1_sorted_dev,
			sizeof(float)*page_list1_len, cudaMemcpyDeviceToHost);
	cudaMemcpy(page_sum2_sorted2, page_sum2_sorted_dev,
			sizeof(float)*page_list2_len, cudaMemcpyDeviceToHost);

	for (int i = 0; i < page_list1_len; i++) {
		printf("[%d] %f ", i, page_sum1_sorted2[i]);
	}
	printf("\n");
#endif
	/* sort fisehr sum page (descending order) */
	BitonicSort(page_sum1_sorted_dev, page_list1_dev, page_list1_len_extended);
	BitonicSort(page_sum2_sorted_dev, page_list2_dev, page_list2_len_extended);

	page_sum1_sorted = (float *)malloc(sizeof(float)*page_list1_len);
	page_sum2_sorted = (float *)malloc(sizeof(float)*page_list2_len);
	cudaMemcpy(page_sum1_sorted, page_sum1_sorted_dev,
			sizeof(float)*page_list1_len, cudaMemcpyDeviceToHost);
	cudaMemcpy(page_sum2_sorted, page_sum2_sorted_dev,
			sizeof(float)*page_list2_len, cudaMemcpyDeviceToHost);
	cudaFree(page_sum1_sorted_dev);
	cudaFree(page_sum2_sorted_dev);

#if 0
	for (int i = 0; i < page_list1_len; i++) {
		printf("%f ", page_sum1_sorted[i]);
	}
	printf("\n");
	for (int i = 0; i < page_list2_len; i++) {
		printf("%f ", page_sum2_sorted[i]);
	}
	printf("\n");
#endif
	page_list1_sorted = (long long int *)malloc(sizeof(long long int)*page_list1_len);
	page_list2_sorted = (long long int *)malloc(sizeof(long long int)*page_list2_len);
	cudaMemcpy(page_list1_sorted, page_list1_dev, sizeof(long long int)*page_list1_len,
			cudaMemcpyDeviceToHost);
	cudaMemcpy(page_list2_sorted, page_list2_dev, sizeof(long long int)*page_list2_len,
			cudaMemcpyDeviceToHost);
#if 0
	for (int i = 0; i < page_list1_len; i++) {
		printf("%lld ", page_list1_sorted[i]);
	}
	printf("\n");
#endif
	/* skip redundant pages */
	if (page_list1_len < page_list2_len) {
		//for (int i = 0; i < (page_list2_len - page_list1_len); i++) {
		for (int i = page_list2_len-1; i >= page_list1_len; i--) {
			page_list2_sorted[i] = -1;
			cudaMemcpy(&page_list2_dev[i], &page_list2_sorted[i],
					sizeof(long long int), cudaMemcpyHostToDevice);
		}
	} else if (page_list1_len > page_list2_len) {
		for (int i = 0; i < (page_list1_len - page_list2_len); i++) {
			page_list1_sorted[i] = -1;
			cudaMemcpy(&page_list1_dev[i], &page_list1_sorted[i],
					sizeof(long long int), cudaMemcpyHostToDevice);
		}
	}

	/* calculate total min cost */
	cudaMalloc(&min_cost_dev, sizeof(float));
	cudaMalloc(&min_idx_dev, sizeof(int));

	for (int i = 0; i < valid_page_num; i++) {
		if ((i % 10000 == 0) or (i == valid_page_num-1)) {
			printf("%8d-th page\n", i);
		}

		while (page_list1_sorted[pos1] < 0) {
			pos1++;
		}
		while (page_list2_sorted[pos2] < 0) {
			pos2++;
		}

		if (page_sum1_sorted[pos1] > page_sum2_sorted[pos2]) {
			MinSharingCostKernel<<<32, 128>>>(fisher1+page_size*page_list1_sorted[pos1],
					weight1+page_size*page_list1_sorted[pos1],
					fisher2, weight2, page_list2_dev, page_list2_len,
					page_size, min_cost_dev, min_idx_dev);
			cudaDeviceSynchronize();

			cudaMemcpy(&min_idx, min_idx_dev, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&page_allocation[idx++], &page_list1_sorted[pos1],
					sizeof(long long int), cudaMemcpyHostToDevice);
			cudaMemcpy(&page_allocation[idx++], &page_list2_sorted[min_idx],
					sizeof(long long int), cudaMemcpyHostToDevice);

			page_list1_sorted[pos1] = -1;
			page_list2_sorted[min_idx] = -1;

			cudaMemcpy(&page_list1_dev[pos1], &page_list1_sorted[pos1],
					sizeof(long long int), cudaMemcpyHostToDevice);
			cudaMemcpy(&page_list2_dev[min_idx], &page_list2_sorted[min_idx],
					sizeof(long long int), cudaMemcpyHostToDevice);
		} else {
			MinSharingCostKernel<<<32, 128>>>(fisher2+page_size*page_list2_sorted[pos2],
					weight2+page_size*page_list2_sorted[pos2],
					fisher1, weight1, page_list1_dev, page_list1_len,
					page_size, min_cost_dev, min_idx_dev);
			cudaDeviceSynchronize();

			cudaMemcpy(&min_idx, min_idx_dev, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&page_allocation[idx++], &page_list1_sorted[min_idx],
					sizeof(long long int), cudaMemcpyHostToDevice);
			cudaMemcpy(&page_allocation[idx++], &page_list2_sorted[pos2],
					sizeof(long long int), cudaMemcpyHostToDevice);

			page_list1_sorted[min_idx] = -1;
			page_list2_sorted[pos2] = -1;

			cudaMemcpy(&page_list1_dev[min_idx], &page_list1_sorted[min_idx],
					sizeof(long long int), cudaMemcpyHostToDevice);
			cudaMemcpy(&page_list2_dev[pos2], &page_list2_sorted[pos2],
					sizeof(long long int), cudaMemcpyHostToDevice);
		}

		cudaMemcpy(&min_cost, min_cost_dev, sizeof(float), cudaMemcpyDeviceToHost);
		total_min_cost += min_cost;
	}

	cudaMemcpy(total_cost, &total_min_cost, sizeof(float), cudaMemcpyHostToDevice);

	// sanity check
	for (int i = 0; i < page_list1_len; i++) {
		if (page_list1_sorted[i] != -1) {
			printf("[PROGRAM EXIT] page alloc fails 1\n");
			exit(EXIT_FAILURE);
		}
	}

	// sanity check
	for (int i = 0; i < page_list2_len; i++) {
		if (page_list2_sorted[i] != -1) {
			printf("[PROGRAM EXIT] page alloc fails 2\n");
			exit(EXIT_FAILURE);
		}
	}

	cudaFree(min_idx_dev);
	cudaFree(min_cost_dev);
	free(page_list2_sorted);
	free(page_list1_sorted);
	free(page_sum2_sorted);
	free(page_sum1_sorted);
	cudaFree(page_list2_dev);
	cudaFree(page_list1_dev);
}

void PageAllocMulti(const float *fisher_addr[], const float *weight_addr[],
		const long long int *page_list_addr[],
		int num_of_list, int *fisher_len, int *weight_len, int *page_list_len,
		int page_size, long long int *page_allocation[], float *total_cost)
{
	float *page_sum_sorted_addr[num_of_list];
	long long int *page_list_sorted_addr[num_of_list];
	long long int *page_list_sorted_addr_dev[num_of_list];
	long long int *page_occupation_unsorted_addr[num_of_list];
	long long int *page_occupation_unsorted_addr_dev[num_of_list];

	for (int i = 0; i < num_of_list; i++) {
		float *page_sum_sorted_dev;
		int page_list_len_extended = 1 << (int)ceil(log2(page_list_len[i]));

		cudaMalloc(&page_list_sorted_addr_dev[i],
				sizeof(long long int)*page_list_len_extended);
		cudaMemset(page_list_sorted_addr_dev[i], 0,
				sizeof(long long int)*page_list_len_extended);
		cudaMemcpy(page_list_sorted_addr_dev[i], page_list_addr[i],
				sizeof(long long int)*page_list_len[i],
				cudaMemcpyDeviceToDevice);

		cudaMalloc(&page_sum_sorted_dev, sizeof(float)*page_list_len_extended);
		FloatSetKernel<<<32, 128>>>(page_sum_sorted_dev, -page_size-1.0,
				page_list_len_extended);
		cudaDeviceSynchronize();
		PageFisherSumKernel<<<32, 128>>>(fisher_addr[i], page_list_sorted_addr_dev[i],
				page_list_len[i], page_size, page_sum_sorted_dev);
		cudaDeviceSynchronize();

		BitonicSort(page_sum_sorted_dev, page_list_sorted_addr_dev[i],
				page_list_len_extended);
		cudaDeviceSynchronize();
		page_sum_sorted_addr[i] = (float *)malloc(sizeof(float)*page_list_len[i]);
		cudaMemcpy(page_sum_sorted_addr[i], page_sum_sorted_dev,
				sizeof(float)*page_list_len[i], cudaMemcpyDeviceToHost);
		page_list_sorted_addr[i]
			= (long long int *)malloc(sizeof(long long int)*page_list_len[i]);
		cudaMemcpy(page_list_sorted_addr[i], page_list_sorted_addr_dev[i],
				sizeof(long long int)*page_list_len[i], cudaMemcpyDeviceToHost);

		page_occupation_unsorted_addr[i]
			= (long long int *)malloc(sizeof(long long int)*page_list_len[0]);
		for (int j = 0; j < page_list_len[0]; j++) {
			*(page_occupation_unsorted_addr[i] + j) = -1;
		}
		cudaMalloc(&page_occupation_unsorted_addr_dev[i],
				sizeof(long long int)*page_list_len[0]);
		cudaMemcpy(page_occupation_unsorted_addr_dev[i],
				page_occupation_unsorted_addr[i],
				sizeof(long long int)*page_list_len[0], cudaMemcpyHostToDevice);

		cudaFree(page_sum_sorted_dev);
	}

	int total_new_page_list_len = 0;

	for (int i = 1; i < num_of_list; i++) {
		total_new_page_list_len += page_list_len[i];
	}

	if (page_list_len[0] > total_new_page_list_len) {
		int diff = page_list_len[0] - total_new_page_list_len;
		for (int i = 0; i < diff; i++) {
			*(page_list_sorted_addr[0] + i) = -1;
		}
		cudaMemcpy(page_list_sorted_addr_dev[0], page_list_sorted_addr[0],
				sizeof(long long int)*diff, cudaMemcpyHostToDevice);
	}

	long long int *page_occupation_addr_dev[num_of_list];

	for (int i = 0; i < num_of_list; i++) {
		cudaMalloc(&page_occupation_addr_dev[i],
				sizeof(long long int)*page_list_len[0]);
		cudaMemcpy(page_occupation_addr_dev[i], page_list_sorted_addr_dev[0],
				sizeof(long long int)*page_list_len[0], cudaMemcpyDeviceToDevice);
	}

	float *min_cost_dev;
	float total_min_cost = 0;
	float total_min_cost2 = 0;
	int min_idx, *min_idx_dev;
	int pos[num_of_list] = { 0, };
	int is_end[num_of_list] = { 0, };
	long long int occupied = -1;

	cudaMalloc(&min_cost_dev, sizeof(float));
	cudaMalloc(&min_idx_dev, sizeof(int));

	float *page_cost_accum_dev;
	cudaMalloc(&page_cost_accum_dev, sizeof(float)*page_list_len[0]);
	cudaMemset(page_cost_accum_dev, 0, sizeof(float)*page_list_len[0]);

	for (int i = 0; i < total_new_page_list_len; i++) {
		if ((i % 10000 == 0) or (i == total_new_page_list_len-1)) {
			printf("%8d-th page\n", i);
		}

		float largest_sum = -1;
		int largest_set = -1;

		for (int j = 1; j < num_of_list; j++) {
			while (!is_end[j] && *(page_list_sorted_addr[j] + pos[j]) < 0) {
				pos[j] += 1;
				if (pos[j] >= page_list_len[j]) {
					is_end[j] = 1;
					break;
				}
			}

			if (is_end[j]) {
				continue;
			}

			float fisher_sum = *(page_sum_sorted_addr[j] + pos[j]);
			if (fisher_sum > largest_sum) {
				largest_sum = fisher_sum;
				largest_set = j;
			}
		}

		long long int largest_page
			= *(page_list_sorted_addr[largest_set] + pos[largest_set]);
		const float *fisher1
			= fisher_addr[largest_set] + page_size*largest_page;
		const float *weight1
			= weight_addr[largest_set] + page_size*largest_page;

		if (largest_set == 0) {
#if 0
			float min_min_cost = -1;
			int smallest_set = -1;

			for (int j = 1; j < num_of_list; j++) {
				int is_occupied
					= *(page_occupation_unsorted_addr[j] + largest_page);
				if (is_occupied == occupied) {
					continue;
				}

				MinSharingCostKernel<<<32, 128>>>(fisher1, weight1, fisher_addr[j],
						weight_addr[j], page_list_sorted_addr_dev[j],
						page_list_len[j], page_size,
						min_cost_dev, min_idx_dev);
				cudaDeviceSynchronize();
				cudaMemcpy(&min_cost, min_cost_dev, sizeof(float),
						cudaMemcpyDeviceToHost);
				cudaMemcpy(&min_idx, min_idx_dev, sizeof(int),
						cudaMemcpyDeviceToHost);

				if (smallest_set == -1) {
					min_min_cost = min_cost;
					smallest_set = j;
				} else {
					if (min_cost < min_min_cost) {
						min_min_cost = min_cost;
						smallest_set = j;
					}
				}
			}

			long long int smallest_page
				= *(page_list_sorted_addr[smallest_set] + min_idx);

			cudaMemcpy(page_allocation[smallest_set] + smallest_page,
					page_list_sorted_addr_dev[largest_set] + pos[largest_set],
					sizeof(long long int), cudaMemcpyDeviceToDevice);

			*(page_occupation_unsorted_addr[smallest_set] + largest_page)
				= (int)occupied;
#endif
		} else {
			const float **fisher_addr_dev;
			cudaMalloc((void **)&fisher_addr_dev, sizeof(float *)*num_of_list);
			cudaMemcpy(fisher_addr_dev, fisher_addr,
					sizeof(float *)*num_of_list, cudaMemcpyHostToDevice);
			const float **weight_addr_dev;
			cudaMalloc((void **)&weight_addr_dev, sizeof(float *)*num_of_list);
			cudaMemcpy(weight_addr_dev, weight_addr,
					sizeof(float *)*num_of_list, cudaMemcpyHostToDevice);
			long long int **page_occupation_unsorted_addr_dev_dev;
			cudaMalloc((void **)&page_occupation_unsorted_addr_dev_dev,
					sizeof(long long int *)*num_of_list);
			cudaMemcpy(page_occupation_unsorted_addr_dev_dev,
					page_occupation_unsorted_addr_dev,
					sizeof(long long int *)*num_of_list,
					cudaMemcpyHostToDevice);

			MinSharingCostMultiKernel<<<32, 128>>>(fisher1, weight1, fisher_addr_dev,
					weight_addr_dev, page_occupation_addr_dev[largest_set],
					page_list_len[0], page_cost_accum_dev,
					page_occupation_unsorted_addr_dev_dev,
					page_size, num_of_list, min_cost_dev, min_idx_dev);
			cudaDeviceSynchronize();

			cudaFree(fisher_addr_dev);
			cudaFree(weight_addr_dev);
			cudaFree(page_occupation_unsorted_addr_dev_dev);

			float min_cost;
			cudaMemcpy(&min_cost, min_cost_dev,
					sizeof(float), cudaMemcpyDeviceToHost);
			if (min_cost < 0) {
				printf("[EXIT] min_cost < 0\n");
				exit(EXIT_FAILURE);
			}

			total_min_cost2 += min_cost;

			cudaMemcpy(&min_idx, min_idx_dev, sizeof(int), cudaMemcpyDeviceToHost);
			if (min_idx < 0 || min_idx >= page_list_len[0]) {
				printf("[EXIT] min_idx < 0 (%d)\n", min_idx);
				exit(EXIT_FAILURE);
			}

			cudaMemcpy(page_allocation[largest_set-1] + largest_page,
					page_occupation_addr_dev[largest_set] + min_idx,
					sizeof(long long int), cudaMemcpyDeviceToDevice);

			*(page_list_sorted_addr[largest_set] + pos[largest_set]) = occupied;
			cudaMemcpy(page_list_sorted_addr_dev[largest_set] + pos[largest_set],
					&occupied, sizeof(long long int), cudaMemcpyHostToDevice);

			cudaMemcpy(page_occupation_addr_dev[largest_set] + min_idx,
					&occupied, sizeof(long long int), cudaMemcpyHostToDevice);

			long long int base_page = -1;
			cudaMemcpy(&base_page, page_list_sorted_addr_dev[0] + min_idx,
					sizeof(long long int), cudaMemcpyDeviceToHost);
			*(page_occupation_unsorted_addr[largest_set] + base_page)
				= largest_page;
			cudaMemcpy(page_occupation_unsorted_addr_dev[largest_set] + base_page,
					&largest_page, sizeof(long long int),
					cudaMemcpyHostToDevice);

			cudaMemcpy(&page_cost_accum_dev[base_page], min_cost_dev,
					sizeof(float), cudaMemcpyDeviceToDevice);
		}
	}

	float *page_cost_accum = (float *)malloc(sizeof(float)*page_list_len[0]);
	cudaMemcpy(page_cost_accum, page_cost_accum_dev,
			sizeof(float)*page_list_len[0], cudaMemcpyDeviceToHost);

	for (int i = 0; i < page_list_len[0]; i++) {
		total_min_cost += page_cost_accum[i];
	}
	free(page_cost_accum);
	cudaMemcpy(total_cost, &total_min_cost, sizeof(float), cudaMemcpyHostToDevice);
	printf("total_min_cost2 = %f\n", total_min_cost2);

	// sanity check 1
	for (int i = 1; i < num_of_list; i++) {
		for (int j = 0; j < page_list_len[i]; j++) {
			if (*(page_list_sorted_addr[i] + j) != occupied) {
				printf("[EXIT] page alloc fails 1-1 (%d)\n", i);
				exit(EXIT_FAILURE);
			}
			long long int page;
			cudaMemcpy(&page, page_list_sorted_addr_dev[i] + j,
				sizeof(long long int), cudaMemcpyDeviceToHost);
			if (page != occupied) {
				printf("[EXIT] page alloc fails 1-2 (%d)\n", i);
				exit(EXIT_FAILURE);
			}
		}
	}

	// sanity check 2
	for (int i = 1; i < num_of_list; i++) {
		int num_of_occupied = 0;
		for (int j = 0; j < page_list_len[0]; j++) {
			long long int page;
			cudaMemcpy(&page, page_occupation_addr_dev[i] + j,
				sizeof(long long int), cudaMemcpyDeviceToHost);
			if (page == occupied) {
				num_of_occupied += 1;
			}
		}

		if (num_of_occupied != page_list_len[i]) {
			printf("[EXIT] page alloc fails 2 (%d)\n", i);
			exit(EXIT_FAILURE);
		}
	}

	// sanity check 3
	for (int i = 1; i < num_of_list; i++) {
		int num_of_occupation = 0;
		int num_of_occupation_dev = 0;
		for (int j = 0; j < page_list_len[0]; j++) {
			long long int page;
			page = *(page_occupation_unsorted_addr[i] + j);
			if (page >= 0) {
				num_of_occupation +=1;
			}

			cudaMemcpy(&page, page_occupation_unsorted_addr_dev[i] + j,
				sizeof(long long int), cudaMemcpyDeviceToHost);
			if (page >= 0) {
				num_of_occupation_dev +=1;
			}
		}

		if (num_of_occupation != page_list_len[i]) {
			printf("[EXIT] page alloc fails 3-1 (%d)\n", i);
			exit(EXIT_FAILURE);
		}

		if (num_of_occupation_dev != page_list_len[i]) {
			printf("[EXIT] page alloc fails 3-2 (%d)\n", i);
			exit(EXIT_FAILURE);
		}
	}

	cudaFree(min_cost_dev);
	cudaFree(min_idx_dev);

	for (int i = 0; i < num_of_list; i++) {
		cudaFree(page_cost_accum_dev);
		cudaFree(page_occupation_addr_dev[i]);
		free(page_sum_sorted_addr[i]);
		free(page_list_sorted_addr[i]);
		cudaFree(page_list_sorted_addr_dev[i]);
		cudaFree(page_occupation_unsorted_addr_dev[i]);
		free(page_occupation_unsorted_addr[i]);
	}
}

__global__ void HarmonicMeanKernel(const float **input_data_addr, int input_size,
		int input_len, float *output)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_len;
			i += blockDim.x * gridDim.x) {
		int numerator = 0;
		float denominator = 0;

		for (int j = 0; j < input_size; j++) {
			float num = *(input_data_addr[j] + i);
			if (num > 0) {
				denominator += (1.0f / num);
				numerator += 1;
			}
		}

		if (denominator > 0) {
			float harmonic_mean = (float)numerator / denominator;
			atomicAdd(output, harmonic_mean);
		}
	}
}

void HarmonicMeanKernelLauncher(const float *input_data_addr[], int input_size,
		int input_len, float *output)
{
	const float **input_data_addr_dev;
	cudaMalloc((void **)&input_data_addr_dev, sizeof(float *)*input_size);
	cudaMemcpy(input_data_addr_dev, input_data_addr,
			sizeof(float *)*input_size, cudaMemcpyHostToDevice);
	HarmonicMeanKernel<<<32, 256>>>(input_data_addr_dev, input_size, input_len, output);
}
