from __future__ import print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import importlib
import time
import ctypes
from weight_virtualization import VNN
from weight_virtualization import WeightVirtualization

tf.logging.set_verbosity(tf.logging.ERROR)

wv_op = tf.load_op_library('./tf_operation.so')
_weight_loader = ctypes.CDLL('./weight_loader.so')
_weight_loader.get_weight.argtypes = (ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int),
	ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.060)
gpu_options = None

def init_virtualization(wv, sess):
	vnn_list = []
	for name, vnn in sorted(wv.vnns.items()):
		vnn_list.append(vnn)

	virtual_weight_address = None
	#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	time1 = time.time()
	virtual_weight_address = sess.run(wv_op.init_weight(wv.weight_page))
	time2 = time.time()
	print('virtual_weight address:', virtual_weight_address)
	print('init virtual_weight %0.3f ms' % ((time2-time1)*1000.0))

	page_address_list = []
	vnn_no = 0
	for vnn in vnn_list:
		time1 = time.time()
		page_address = sess.run(wv_op.init_page_table(vnn.weight_page_list))
		time2 = time.time()
		print('[VNN %d][%s] init page table %0.3f ms'
			% (vnn_no, vnn.name, (time2-time1)*1000.0))
		page_address_list.append(page_address)
		vnn_no += 1

	page_table_address_list = []
	for i in range(len(page_address_list)):
		page_table_address = tf.constant(page_address_list[i],
			name='page_table_address/' + str(i))
		page_table_address_list.append(page_table_address)

	for vnn in vnn_list:
		with tf.name_scope(vnn.name):
			tf.train.import_meta_graph(vnn.meta_filepath)

	weight_address_list = []
	weight_len_list = []

	for i in range(len(vnn_list)):
		train_weights = tf.trainable_variables(scope=vnn_list[i].name)
		weight_address, weight_len = sess.run(wv_op.get_weight_address(train_weights))
		weight_address_list.append(weight_address)
		weight_len_list.append(weight_len)

	time1 = time.time()
	sess.run(tf.global_variables_initializer())
	time2 = time.time()
	print('tf.global_variables_initializer %0.3f ms' % ((time2-time1)*1000.0))

	return vnn_list, weight_address_list, weight_len_list, virtual_weight_address, page_address_list

def load_weight_page(virtual_weight_address, weight_address_list,
	weight_len_list, page_address_list, weight_per_page):
	num_of_weight = len(weight_address_list)
	weight_address_list_array_type = ctypes.c_int64 * num_of_weight
	weight_len_list_array_type = ctypes.c_int * num_of_weight
	_weight_loader.get_weight(
		weight_address_list_array_type(*weight_address_list),
		weight_len_list_array_type(*weight_len_list),
		ctypes.c_int(num_of_weight),
		ctypes.c_int64(virtual_weight_address),
		ctypes.c_int64(page_address_list),
		ctypes.c_int(weight_per_page))

def in_memory_execute(graph, sess, vnn, layers, data_set,
	virtual_weight_address,	weight_address_list, weight_len_list,
	page_address_list, weight_per_page, label=None):
	print("[Executing]", vnn.name)

	time1 = time.time()
	load_weight_page(virtual_weight_address, weight_address_list,
		weight_len_list, page_address_list, weight_per_page)
	time2 = time.time()
	weights_load_time = (time2-time1)*1000.0
	print('weights load time : %0.3f ms' % (weights_load_time))

	keep_prob_input = graph.get_tensor_by_name(vnn.name + "/keep_prob_input:0")
	keep_prob = graph.get_tensor_by_name(vnn.name + "/keep_prob:0")
	x = graph.get_tensor_by_name(vnn.name + "/neuron_0:0")
	y = graph.get_tensor_by_name(vnn.name + "/neuron_" + str(layers-1) + ":0")

	data_set_reshaped = np.reshape(data_set, ([-1] + x.get_shape().as_list()[1:]))
	time1 = time.time()
	infer_result = sess.run(y, feed_dict={
		x: data_set_reshaped, keep_prob_input: 1.0, keep_prob: 1.0})
	time2 = time.time()
	DNN_execution_time = (time2-time1)*1000.0
	print('DNN execution time: %0.3f ms' % (DNN_execution_time))

	if label is not None:
		y_ = graph.get_tensor_by_name(vnn.name + "/y_:0")
		accuracy = graph.get_tensor_by_name(vnn.name + "/accuracy:0")
		test_accuracy = sess.run(accuracy, feed_dict={
			x: data_set_reshaped, y_: label, keep_prob_input: 1.0, keep_prob: 1.0})
		print("Inference accuracy: %f" % test_accuracy)

	return weights_load_time, DNN_execution_time

def main():
	wv = WeightVirtualization()

	data_list = [ 'cifar10_data', 'GSC_v2_data', 'GTSRB_data', 'mnist_data', 'svhn_data' ]
	layer_list = [ 7, 6, 7, 7, 7 ]

	total_weight_load_time = 0
	total_execution_time = 0
	num_execution = 30

	with tf.Graph().as_default() as graph:
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			vnn_list, weight_address_list, weight_len_list, \
				virtual_weight_address, \
				page_address_list = init_virtualization(wv, sess)

			for i in range(num_execution):
				vnn_no = np.random.randint(len(vnn_list))
				#print('vnn_no:', vnn_no)

				data = __import__(data_list[vnn_no])
				data_set = data.test_set()[0]#[0:1000]
				label = data.test_set()[1]#[0:1000]

				weight_load_time, execution_time = in_memory_execute(tf.get_default_graph(),
					sess, vnn_list[vnn_no], layer_list[vnn_no], data_set,
					virtual_weight_address,
					weight_address_list[vnn_no], weight_len_list[vnn_no],
					page_address_list[vnn_no], wv.weight_per_page, label)

				total_weight_load_time += weight_load_time
				total_execution_time += execution_time

	print('total weights load time : %0.3f ms' % (total_weight_load_time))
	print('total DNN execution time: %0.3f ms' % (total_execution_time))


if __name__ == '__main__':
	main()
