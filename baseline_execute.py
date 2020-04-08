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

def baseline_execute(graph, sess, vnn, layers, data_set, label=None):
	print("[Executing]", vnn.name)

	saver = tf.train.import_meta_graph(vnn.meta_filepath)

	time1 = time.time()
	saver.restore(sess, vnn.model_filepath)
	time2 = time.time()
	weights_load_time = (time2-time1)*1000.0
	print('weights load time : %0.3f ms' % (weights_load_time))

	keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")
	x = graph.get_tensor_by_name("neuron_0:0")
	y = graph.get_tensor_by_name("neuron_" + str(layers-1) + ":0")

	data_set_reshaped = np.reshape(data_set, ([-1] + x.get_shape().as_list()[1:]))
	time1 = time.time()
	infer_result = sess.run(y, feed_dict={
		x: data_set_reshaped, keep_prob_input: 1.0, keep_prob: 1.0})
	time2 = time.time()
	DNN_execution_time = (time2-time1)*1000.0
	print('DNN execution time: %0.3f ms' % (DNN_execution_time))

	if label is not None:
		y_ = graph.get_tensor_by_name("y_:0")
		accuracy = graph.get_tensor_by_name("accuracy:0")
		test_accuracy = sess.run(accuracy, feed_dict={
			x: data_set_reshaped, y_: label, keep_prob_input: 1.0, keep_prob: 1.0})
		print("Inference accuracy: %f" % test_accuracy)
	
	return weights_load_time, DNN_execution_time

def main():
	wv = WeightVirtualization()

	vnn_list = []
	for name, vnn in sorted(wv.vnns.items()):
		vnn_list.append(vnn)

	data_list = [ 'cifar10_data', 'GSC_v2_data', 'GTSRB_data', 'mnist_data', 'svhn_data' ]
	layer_list = [ 7, 6, 7, 7, 7 ]

	total_weight_load_time = 0
	total_execution_time = 0
	num_execution = 30

	for i in range(num_execution):
		vnn_no = np.random.randint(len(vnn_list))
		#print('vnn_no:', vnn_no)

		data = __import__(data_list[vnn_no])
		data_set = data.test_set()[0]#[0:1000]
		label = data.test_set()[1]#[0:1000]

		with tf.Graph().as_default() as graph:
			with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
				weight_load_time, execution_time = baseline_execute(graph,
					sess, vnn_list[vnn_no], layer_list[vnn_no],
					data_set, label)
	
				total_weight_load_time += weight_load_time
				total_execution_time += execution_time

	print('total weights load time : %0.3f ms' % (total_weight_load_time))
	print('total DNN execution time: %0.3f ms' % (total_execution_time))

if __name__ == '__main__':
	main()
