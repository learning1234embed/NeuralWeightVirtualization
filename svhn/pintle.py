from __future__ import print_function
import numpy as np
import tensorflow as tf

import_name = 'svhn_data'

def next_batch(data_set, batch_size):
	data = data_set[0]
	label = data_set[1] # one-hot vectors

	data_num = np.random.choice(data.shape[0], size=batch_size, replace=False)
	batch = data[data_num,:]
	label = label[data_num,:] # one-hot vectors

	return batch, label

def v_input_variable_names():
	input_variable_names = [ 'neuron_0', 'keep_prob_input', 'keep_prob' ]
	return input_variable_names

def v_output_variable_names():
	output_variable_names = [ 'neuron_6' ]
	return output_variable_names

def v_train_input_variables():
	data = __import__(import_name)
	train_set = data.train_set()
	train_image_reshaped = np.reshape(train_set[0], ([-1, 32, 32, 3]))
	return [[train_image_reshaped, train_set[1]], 1.0, 1.0]

def v_test_input_variables():
	data = __import__(import_name)
	test_set = data.test_set()
	test_image_reshaped = np.reshape(test_set[0], ([-1, 32, 32, 3]))
	return [[test_image_reshaped, test_set[1]], 1.0, 1.0]

def v_execute(graph, sess, input_tensors, input_variables, ground_truth):
	tensor_y_name = "neuron_6:0"
	y = graph.get_tensor_by_name(tensor_y_name)
	
	# infer
	infer_result = sess.run(y, feed_dict={t: v for t,v in zip(input_tensors, input_variables)})
	
	# accuracy
	test_accuracy = None
	if ground_truth is not None:
		y_ = graph.get_tensor_by_name("y_:0")
		accuracy = graph.get_tensor_by_name("accuracy:0")
		input_tensors.append(y_)
		input_variables.append(ground_truth)
		test_accuracy = sess.run(accuracy,
			feed_dict={t: v for t,v in zip(input_tensors, input_variables)})
		print("Inference accuracy: %f" % test_accuracy)
		
	return infer_result, test_accuracy

def v_train(graph, sess, matching_cost, batch_size, train_iteration, get_weight_func):
	print("v_train")

	data = __import__(import_name)
	train_set = data.train_set()
	validation_set = data.test_set()

	# get tensors
	tensor_x_name = "neuron_0:0"
	x = graph.get_tensor_by_name("neuron_0:0")
	y_ = graph.get_tensor_by_name("y_:0")
	keep_prob_input = graph.get_tensor_by_name("keep_prob_input:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")
	accuracy = graph.get_tensor_by_name("accuracy:0")
	cross_entropy = graph.get_tensor_by_name('cross_entropy:0')

	learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
		name='matching_cost_optimizer')
	loss = optimizer.minimize(tf.add(cross_entropy, matching_cost))

	sess.run(tf.variables_initializer(optimizer.variables()))

	input_images_validation = validation_set[0]
	input_images_validation_reshaped = np.reshape(validation_set[0], ([-1] + x.get_shape().as_list()[1:]))
	labels_validation = validation_set[1]
	highest_accuracy = 0
	new_weight_vector = None

	# train
	for i in range(train_iteration):
		input_data, labels = next_batch(train_set, batch_size)
		input_data_reshpaed = np.reshape(input_data, ([-1] + x.get_shape().as_list()[1:]))

		if i % (100) == 0 or i == (train_iteration-1):
			original_loss, matching_loss, train_accuracy = sess.run([cross_entropy, matching_cost, accuracy],
				feed_dict={x: input_data_reshpaed, y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})
			print("step %d, training accuracy: %f original loss: %f matching loss: %f"
				% (i, train_accuracy, original_loss, matching_loss))

			# validate
			test_accuracy = sess.run(accuracy, feed_dict={
				x: input_images_validation_reshaped, y_: labels_validation,
				keep_prob_input: 1.0, keep_prob: 1.0})
			print("step %d, Validation accuracy: %f" % (i, test_accuracy))


			if i == 0:
				highest_accuracy = test_accuracy
			else:
				if test_accuracy > highest_accuracy:
					new_weight_vector = get_weight_func(sess) 
					highest_accuracy = test_accuracy
					print('get new weight for', highest_accuracy)

		sess.run(loss, feed_dict={x: input_data_reshpaed,
			y_: labels, keep_prob_input: 1.0, keep_prob: 1.0})
	
	return new_weight_vector

def v_fx_tensors(graph):
	y = graph.get_tensor_by_name("neuron_6:0")
	row_idx = tf.range(tf.shape(y)[0])
	col_idx = tf.argmax(y, axis=1, output_type=tf.dtypes.int32)
	full_indices = tf.stack([row_idx, col_idx], axis=1)
	fx_tensors = tf.gather_nd(y, full_indices)
	return fx_tensors

def main():
	print(v_test_input_variables())

if __name__ == '__main__':
	main()
