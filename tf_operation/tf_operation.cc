#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

using namespace tensorflow;

REGISTER_OP("InitWeight")
	.Input("input: float")
	.Output("virtual_weight_address: int64");

REGISTER_OP("InitPageTable")
	.Input("input: int32")
	.Output("page_table_address: int64");

REGISTER_OP("ReadWeight")
	.Attr("start: int")
	.Attr("end: int")
	.Input("virtual_weight_address: int64")
	.Output("weight: float");

REGISTER_OP("ReadPageTable")
	.Attr("start: int")
	.Attr("end: int")
	.Input("page_table_address: int64")
	.Output("page: int32");

REGISTER_OP("GetWeight")
	.Attr("num_inputs: int >= 1")
	.Attr("page_size: int")
	.Input("ref: Ref(num_inputs * float)")
	.Input("virtual_weight_address: int64")
	.Input("page_table_address: int64")
	//.Output("output_ref: Ref(num_inputs * float)")
	.SetAllowsUninitializedInput();

REGISTER_OP("GetWeightAddress")
	.Attr("num_inputs: int >= 1")
	.Input("ref: Ref(num_inputs * float)")
	.Output("weight_address: int64")
	.Output("weight_len: int32")
	.SetAllowsUninitializedInput();

REGISTER_OP("FreeWeight")
	.Input("virtual_weight_address: int64");

REGISTER_OP("FreePageTable")
	.Input("page_table_address: int64");

REGISTER_OP("SharingCost")
	.Input("fisher1: float")
	.Input("weight1: float")
	.Input("fisher2: float")
	.Input("weight2: float")
	.Output("cost: float");

REGISTER_OP("PageAlloc")
	.Attr("page_size: int")
	.Input("fisher1: float")
	.Input("weight1: float")
	.Input("page_list1: int64")
	.Input("fisher2: float")
	.Input("weight2: float")
	.Input("page_list2: int64")
	.Output("page_allocation: int64")
	.Output("total_cost: float");

REGISTER_OP("PageAllocMulti")
	.Attr("num_of_list: int >= 1")
	.Attr("page_size: int")
	.Input("curent_fisher: float")
	.Input("base_weight: float")
	.Input("base_page_list: int64")
	.Input("new_fisher: num_of_list * float")
	.Input("new_weight: num_of_list * float")
	.Input("new_page_list: num_of_list * int64")
	.Output("page_allocation: num_of_list * int64")
	.Output("total_cost: float");

REGISTER_OP("HarmonicMean")
	.Attr("num_inputs: int >= 1")
	.Input("inputs: num_inputs * float")
	.Output("harmonic_mean: float");

void InitWeight(const float *weight, const int weight_len,
		long long int *address);
void InitPageTable(const int *page_table, const int page_table_len,
		long long int *page_table_address);
void ReadWeight(const long long int *address,
		int start, int end, float *output);
void ReadPageTable(const long long int *address,
		int start, int end, int *output);
void GetWeightKernelLauncher(float *input, const int input_len,
		const long long int *address, const long long int *page_table_address,
		const int page_size, const int start, const int end);
void GetWeightAddress(int64 *weight_address_list, int *weight_len_list, int input_len,
		int64 *output1, int *output2);
void FreeWeight(const long long int *address);
void FreePageTable(const long long int *address);
void SharingCostKernelLauncher(const float *input1, const float *input2, const float *input3,
		const float *input4, int len, float *output);
void PageAlloc(const float *fisher1, const float *weight1, const float *fisher2,
		const float *weight2, int page_list1_len, int page_list2_len,
		const long long int *page_list1, const long long int *page_list2,
		int page_size, long long int *page_allocation, float *total_cost);
void PageAllocMulti(const float *fisher_addr[], const float *weight_addr[],
		const long long int *page_list_addr[],
		int num_of_list, int *fisher_len, int *weight_len, int *page_list_len,
		int page_size, long long int *page_allocation[], float *total_cost);
void HarmonicMeanKernelLauncher(const float *input_data[], int input_size, int input_len,
		float *output);

class InitWeightOp : public OpKernel {
	public:
		explicit InitWeightOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& input_tensor = context->input(0);
			auto weight = input_tensor.flat<float>();
			const int weight_len = weight.size();

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
						&output_tensor));
			auto address = output_tensor->template flat<int64>();

			InitWeight(weight.data(), weight_len, address.data());
		}
};

class InitPageTableOp : public OpKernel {
	public:
		explicit InitPageTableOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& input_tensor = context->input(0);
			auto page_table = input_tensor.flat<int32>();
			const int page_table_len = page_table.size();

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
						&output_tensor));
			auto page_table_address = output_tensor->template flat<int64>();

			InitPageTable(page_table.data(), page_table_len, page_table_address.data());
		}
};

class ReadWeightOp : public OpKernel {
	public:
		int start_, end_;
		explicit ReadWeightOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("start", &start_));
			OP_REQUIRES_OK(context, context->GetAttr("end", &end_));
		}

		void Compute(OpKernelContext* context) override {
			const Tensor& address_tensor = context->input(0);
			auto address = address_tensor.flat<int64>();

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0,
						TensorShape({end_-start_+1}), &output_tensor));
			auto output = output_tensor->template flat<float>();

			ReadWeight(address.data(), start_, end_, output.data());
		}
};

class ReadPageTableOp : public OpKernel {
	public:
		int start_, end_;
		explicit ReadPageTableOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("start", &start_));
			OP_REQUIRES_OK(context, context->GetAttr("end", &end_));
		}

		void Compute(OpKernelContext* context) override {
			const Tensor& address_tensor = context->input(0);
			auto address = address_tensor.flat<int64>();

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0,
						TensorShape({end_-start_+1}), &output_tensor));
			auto output = output_tensor->template flat<int>();

			ReadPageTable(address.data(), start_, end_, output.data());
		}
};

class GetWeightOp : public OpKernel {
	public:
		int page_size_;
		explicit GetWeightOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("page_size", &page_size_));
		}

		void Compute(OpKernelContext* context) override {

			OpMutableInputList ref_inputs;
			OP_REQUIRES_OK(context,
				context->mutable_input_list("ref", &ref_inputs));

			int start_id = ref_inputs.size();

			const Tensor& address_tensor = context->input(start_id);
			auto address = address_tensor.flat<int64>();

			const Tensor& page_table_address_tensor = context->input(start_id+1);
			auto page_table_address = page_table_address_tensor.flat<int64>();

			int start = 0;
			int end = 0;

			for (int i = 0; i < ref_inputs.size(); i++) {
				Tensor ref_tensor = ref_inputs.at(i, /*lock_held=*/ true);
				auto ref_input = ref_tensor.flat<float>();
				int num_weight = ref_input.size();

				AllocatorAttributes attr;
				attr.set_gpu_compatible(true);
				attr.set_nic_compatible(true);

				PersistentTensor persistent_tensor;
				Tensor* new_tensor = nullptr;
				OP_REQUIRES_OK(context,
					context->allocate_persistent(ref_tensor.dtype(),
					ref_tensor.shape(), &persistent_tensor, &new_tensor, attr));
				context->clear_recorded_memory();
				context->replace_ref_input(i, *new_tensor, /* lock_held */ true);
				Tensor unlocked_input_tensor = context->mutable_input(i,
						/* lock_held */ false);
				auto input = unlocked_input_tensor.flat<float>();
				const int input_len = input.size();

				end = start + num_weight - 1;
				GetWeightKernelLauncher(input.data(), input_len, address.data(),
					page_table_address.data(), page_size_, start, end);
				start = end + 1;
			}
		}
};

class GetWeightAddressOp : public OpKernel {
	public:
		explicit GetWeightAddressOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {

			OpMutableInputList ref_inputs;
			OP_REQUIRES_OK(context,
				context->mutable_input_list("ref", &ref_inputs));

			int start_id = ref_inputs.size();

			Tensor* output_tensor1 = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0,
						TensorShape({ref_inputs.size()}), &output_tensor1));
			auto output1 = output_tensor1->template flat<int64>();

			Tensor* output_tensor2 = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(1,
						TensorShape({ref_inputs.size()}), &output_tensor2));
			auto output2 = output_tensor2->template flat<int>();

			int64 weight_address_list[ref_inputs.size()];
			int weight_len_list[ref_inputs.size()];

			int start = 0;
			int end = 0;

			for (int i = 0; i < ref_inputs.size(); i++) {
				Tensor ref_tensor = ref_inputs.at(i, /*lock_held=*/ true);
				auto ref_input = ref_tensor.flat<float>();

				AllocatorAttributes attr;
				attr.set_gpu_compatible(true);
				attr.set_nic_compatible(true);

				PersistentTensor persistent_tensor;
				Tensor* new_tensor = nullptr;
				OP_REQUIRES_OK(context,
					context->allocate_persistent(ref_tensor.dtype(),
					ref_tensor.shape(), &persistent_tensor, &new_tensor, attr));
				context->clear_recorded_memory();
				context->replace_ref_input(i, *new_tensor, /* lock_held */ true);
				Tensor unlocked_input_tensor = context->mutable_input(i,
						/* lock_held */ false);
				auto input = unlocked_input_tensor.flat<float>();

				weight_address_list[i] = (int64)input.data();
				weight_len_list[i] = input.size();
			}

			GetWeightAddress(weight_address_list, weight_len_list,
				ref_inputs.size(), output1.data(), output2.data());
		}
};


class FreeWeightOp : public OpKernel {
	public:
		explicit FreeWeightOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& address_tensor = context->input(0);
			auto address = address_tensor.flat<int64>();

			FreeWeight(address.data());
		}
};

class FreePageTableOp : public OpKernel {
	public:
		explicit FreePageTableOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& address_tensor = context->input(0);
			auto address = address_tensor.flat<int64>();

			FreePageTable(address.data());
		}
};

class SharingCostOp : public OpKernel {
	public:
		explicit SharingCostOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& input_tensor1 = context->input(0);
			auto input1 = input_tensor1.flat<float>();

			const Tensor& input_tensor2 = context->input(1);
			auto input2 = input_tensor2.flat<float>();

			const Tensor& input_tensor3 = context->input(2);
			auto input3 = input_tensor3.flat<float>();

			const Tensor& input_tensor4 = context->input(3);
			auto input4 = input_tensor4.flat<float>();

			OP_REQUIRES(context, input_tensor1.NumElements()
				== input_tensor2.NumElements(),
				errors::InvalidArgument("size of input tensors needs to be same"));
			OP_REQUIRES(context, input_tensor2.NumElements()
				== input_tensor3.NumElements(),
				errors::InvalidArgument("size of input tensors needs to be same"));
			OP_REQUIRES(context, input_tensor3.NumElements()
				== input_tensor4.NumElements(),
				errors::InvalidArgument("size of input tensors needs to be same"));

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0,
						TensorShape({}), &output_tensor));
			auto output = output_tensor->template flat<float>();

			SharingCostKernelLauncher(input1.data(), input2.data(), input3.data(),
				input4.data(), input_tensor1.dim_size(0), output.data());
		}
};

class PageAllocOp : public OpKernel {
	public:
		int page_size_;
		explicit PageAllocOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("page_size", &page_size_));
		}

		void Compute(OpKernelContext* context) override {
			const Tensor& input_tensor1 = context->input(0);
			auto fisher1 = input_tensor1.flat<float>();

			const Tensor& input_tensor2 = context->input(1);
			auto weight1 = input_tensor2.flat<float>();

			const Tensor& input_tensor3 = context->input(2);
			auto page_list1 = input_tensor3.flat<int64>();

			const Tensor& input_tensor4 = context->input(3);
			auto fisher2 = input_tensor4.flat<float>();

			const Tensor& input_tensor5 = context->input(4);
			auto weight2 = input_tensor5.flat<float>();

			const Tensor& input_tensor6 = context->input(5);
			auto page_list2 = input_tensor6.flat<int64>();

			OP_REQUIRES(context, input_tensor1.dims() == 1,
				errors::InvalidArgument("dims of input tensor needs to be 1"));
			OP_REQUIRES(context, input_tensor2.dims() == 1,
				errors::InvalidArgument("dims of input tensor needs to be 1"));
			OP_REQUIRES(context, input_tensor4.dims() == 1,
				errors::InvalidArgument("dims of input tensor needs to be 1"));
			OP_REQUIRES(context, input_tensor5.dims() == 1,
				errors::InvalidArgument("dims of input tensor needs to be 1"));
			OP_REQUIRES(context, input_tensor1.NumElements()
				== input_tensor2.NumElements(),
				errors::InvalidArgument("size of input tensor 1, 2 needs to be same"));
			OP_REQUIRES(context, input_tensor4.NumElements()
				== input_tensor5.NumElements(),
				errors::InvalidArgument("size of input tensor 4, 5 needs to be same"));

			Tensor* output_tensor1 = nullptr;
			int output_tensor_len = page_list1.size() < page_list2.size()
				? page_list1.size() : page_list2.size();
			OP_REQUIRES_OK(context, context->allocate_output(0,
					TensorShape({output_tensor_len,2}), &output_tensor1));
			auto page_allocation = output_tensor1->template flat<int64>();

			Tensor* output_tensor2 = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(1,
					TensorShape({}), &output_tensor2));
			auto total_cost = output_tensor2->template flat<float>();

			PageAlloc(fisher1.data(), weight1.data(), fisher2.data(), weight2.data(),
				page_list1.size(), page_list2.size(), page_list1.data(),
				page_list2.data(), page_size_, page_allocation.data(),
				total_cost.data());
		}
};

class PageAllocMultiOp : public OpKernel {
	public:
		int page_size_;
		explicit PageAllocMultiOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("page_size", &page_size_));
		}

		void Compute(OpKernelContext* context) override {
			const Tensor& input_tensor1 = context->input(0);
			auto base_fisher = input_tensor1.flat<float>();

			const Tensor& input_tensor2 = context->input(1);
			auto base_weight = input_tensor2.flat<float>();

			const Tensor& input_tensor3 = context->input(2);
			auto base_page_list = input_tensor3.flat<int64>();
			int base_page_list_len = input_tensor3.NumElements();

			OP_REQUIRES(context, input_tensor1.NumElements() == input_tensor2.NumElements(),
				errors::InvalidArgument("size of current fisher and weight needs to be same"));
			OP_REQUIRES(context, input_tensor1.NumElements()/page_size_ == input_tensor3.NumElements(),
				errors::InvalidArgument("size of current fisher and page list needs to be same"));

			OpInputList new_fisher_list;
			OP_REQUIRES_OK(context,
				context->input_list("new_fisher", &new_fisher_list));
			int new_fisher_size = new_fisher_list.size();

			OpInputList new_weight_list;
			OP_REQUIRES_OK(context,
				context->input_list("new_weight", &new_weight_list));
			int new_weight_size = new_weight_list.size();

			OpInputList new_page_list_list;
			OP_REQUIRES_OK(context,
				context->input_list("new_page_list", &new_page_list_list));
			int new_page_list_size = new_page_list_list.size();

			OP_REQUIRES(context, new_fisher_size == new_weight_size,
				errors::InvalidArgument("size of new_fisher and new_weight_list needs to be same"));
			OP_REQUIRES(context, new_fisher_size == new_page_list_size,
				errors::InvalidArgument("size of new_fisher and new_page_list_list needs to be same"));

			const float *fisher_addr[new_fisher_size+1];
			const float *weight_addr[new_weight_size+1];
			const long long int *page_list_addr[new_page_list_size+1];
			int fisher_len[new_fisher_size+1];
			int weight_len[new_weight_size+1];
			int page_list_len[new_page_list_size+1];
			long long int *output_page_list_addr[new_page_list_size];

			fisher_addr[0] = base_fisher.data();
			weight_addr[0] = base_weight.data();
			page_list_addr[0] = base_page_list.data();
			fisher_len[0] = input_tensor1.NumElements();
			weight_len[0] = input_tensor2.NumElements();
			page_list_len[0] = input_tensor3.NumElements();

			OpOutputList page_allocation_list;
			OP_REQUIRES_OK(context,	context->output_list("page_allocation",
					&page_allocation_list));

			for (int i = 0; i < new_fisher_size; i++) {
				fisher_len[i+1] = new_fisher_list[i].NumElements();
				weight_len[i+1] = new_weight_list[i].NumElements();
				page_list_len[i+1] = new_page_list_list[i].NumElements();

				OP_REQUIRES(context, fisher_len[i] == weight_len[i],
					errors::InvalidArgument("size of fisher_len and weight_len needs to be same"));
				OP_REQUIRES(context, fisher_len[i]/page_size_ == page_list_len[i],
					errors::InvalidArgument("size of fisher_len and page_list_len needs to be same"));

				auto new_fisher = new_fisher_list[i].flat<float>();
				fisher_addr[i+1] = new_fisher.data();
				auto new_weight = new_weight_list[i].flat<float>();
				weight_addr[i+1] = new_weight.data();
				auto new_page_list = new_page_list_list[i].flat<int64>();
				page_list_addr[i+1] = new_page_list.data();

				Tensor* page_allocation = nullptr;
				OP_REQUIRES_OK(context, page_allocation_list.allocate(i,
					TensorShape({page_list_len[i+1]}), &page_allocation));
				auto output_page_list = page_allocation->template flat<int64>();
				output_page_list_addr[i] = output_page_list.data();
			}

			Tensor* cost_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(new_page_list_size,
						TensorShape({}), &cost_tensor));
			auto cost = cost_tensor->template flat<float>();

			PageAllocMulti(fisher_addr, weight_addr, page_list_addr, new_fisher_size+1,
					fisher_len, weight_len,	page_list_len, page_size_,
					output_page_list_addr, cost.data());
		}
};


class HarmonicMeanOp : public OpKernel {
	public:
		explicit HarmonicMeanOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			OpInputList inputs;
			OP_REQUIRES_OK(context, context->input_list("inputs", &inputs));
			int input_size = inputs.size();
			int input_len = inputs[0].NumElements();
			const float *input_data_addr[input_size];

			for (int i = 0; i < input_size; i++) {
				auto input = inputs[i].flat<float>();

				OP_REQUIRES(context, inputs[i].NumElements() == input_len,
					errors::InvalidArgument("len of input tensors needs to be same"));
				input_data_addr[i] = (float *)input.data();
			}

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0,
						TensorShape({}), &output_tensor));
			auto output = output_tensor->template flat<float>();

			HarmonicMeanKernelLauncher(input_data_addr, input_size, input_len,
				output.data());
		}
};

REGISTER_KERNEL_BUILDER(Name("InitWeight").Device(DEVICE_GPU), InitWeightOp);
REGISTER_KERNEL_BUILDER(Name("InitPageTable").Device(DEVICE_GPU), InitPageTableOp);
REGISTER_KERNEL_BUILDER(Name("ReadWeight").Device(DEVICE_GPU), ReadWeightOp);
REGISTER_KERNEL_BUILDER(Name("ReadPageTable").Device(DEVICE_GPU), ReadPageTableOp);
REGISTER_KERNEL_BUILDER(Name("GetWeight").Device(DEVICE_GPU), GetWeightOp);
REGISTER_KERNEL_BUILDER(Name("GetWeightAddress").Device(DEVICE_GPU), GetWeightAddressOp);
REGISTER_KERNEL_BUILDER(Name("FreeWeight").Device(DEVICE_GPU), FreeWeightOp);
REGISTER_KERNEL_BUILDER(Name("FreePageTable").Device(DEVICE_GPU), FreePageTableOp);
REGISTER_KERNEL_BUILDER(Name("SharingCost").Device(DEVICE_GPU), SharingCostOp);
REGISTER_KERNEL_BUILDER(Name("PageAlloc").Device(DEVICE_GPU), PageAllocOp);
REGISTER_KERNEL_BUILDER(Name("PageAllocMulti").Device(DEVICE_GPU), PageAllocMultiOp);
REGISTER_KERNEL_BUILDER(Name("HarmonicMean").Device(DEVICE_GPU), HarmonicMeanOp);
