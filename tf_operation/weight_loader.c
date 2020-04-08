#include <stdio.h>

void GetWeightKernelLauncher(float *input, int input_len, float* addr,
		int* page_table_addr, int page_size, int start, int end);

void get_weight(long long int *weight_address_list, int *weight_len_list, int num_of_weight,
		long long int virtual_weight_address, long long int page_table_address,
		int page_size)
{
	int start = 0;
	int end = 0;

	for (int i = 0; i < num_of_weight; i++) {
		float *input = (float *)weight_address_list[i];
		int input_len = weight_len_list[i];
		float *address = (float *)virtual_weight_address;
		int *page_table_addr = (int *)page_table_address;

		end = start + input_len - 1;
		GetWeightKernelLauncher(input, input_len, address, page_table_addr,
			page_size, start, end);
		start = end + 1;
	}
}

