from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import copy
import pickle 
import struct
import sys
import argparse
import importlib
import matplotlib.pyplot as plt
import time

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#np.set_printoptions(threshold=sys.maxsize)

class VNN:
	def __init__(self, network_path, id, state=-1, weight_page_list=None):
		self.id = id
		self.name = os.path.basename(os.path.normpath(network_path))
		self.state = state
		self.weight_page_list = weight_page_list

		self.network_path = network_path
		assert os.path.exists(self.network_path), 'No network path %s exists' % self.network_path

		self.meta_filename = self.name + '.meta'
		self.meta_filepath = os.path.join(self.network_path, self.meta_filename)
		assert os.path.exists(self.meta_filepath), 'No filepath %s exists' % self.meta_filepath

		self.num_of_weight = self.get_weight_num()

		self.model_filepath = os.path.join(self.network_path, self.name)

		self.network_weight_filename = self.name + '_network_weight.npy'
		self.network_weight_filepath = os.path.join(self.network_path, self.network_weight_filename)
		self.network_fisher_filename = self.name + '_network_fisher.npy'
		self.network_fisher_filepath = os.path.join(self.network_path, self.network_fisher_filename)

		self.weight_filename = self.name + '_weight.npy'
		self.weight_filepath = os.path.join(self.network_path, self.weight_filename)
		self.fisher_filename = self.name + '_fisher.npy'
		self.fisher_filepath = os.path.join(self.network_path, self.fisher_filename)

		self.pintle_filename = 'pintle.py'
		self.pintle_filepath = os.path.join(self.network_path, self.pintle_filename)

		self.filepath = self.name + '.vnn'

	def get_weight_num(self):
		meta_graph_def = tf.MetaGraphDef()
		with open(self.meta_filepath, 'rb') as f:
			meta_graph_def.MergeFromString(f.read())

		num_of_weight = 0
		with tf.Graph().as_default() as graph:
			tf.train.import_meta_graph(meta_graph_def)
			trainable_variables = tf.trainable_variables()
			for trainable_variable in trainable_variables:
				num_of_weight += np.prod(trainable_variable.get_shape().as_list())

		return num_of_weight

class WeightVirtualization:
	__instance = None
	@staticmethod
	def getInstance():
		if WeightVirtualization.__instance == None:
			WeightVirtualization()
		return WeightVirtualization.__instance

	def __init__(self, num_of_weight_page=665, weight_per_page=100,
		weight_page_filename='virtual_weight_page.npy',
		weight_page_occupation_filename='weight_page_occupation.npy',
		weight_virtualization_op_filename='./tf_operation.so'):

		if WeightVirtualization.__instance != None:
			raise Exception("this class is a singleton")
		else:
			WeightVirtualization.__instance = self

			self.num_of_weight_page = num_of_weight_page
			self.weight_per_page = weight_per_page
			self.weight_page = None

			self.weight_page_filename = weight_page_filename
			self.weight_page_occupation_filename = weight_page_occupation_filename
			self.weight_virtualization_op_filename = weight_virtualization_op_filename

			if self.load_weight_page() is False:
				print('init new weight pages')
				self.init_weight_page()
				self.save_weight_page()

			self.next_vnn_id = 0
			self.vnns = {}
			self.load_vnns()

	def create_vnn(self, network_path):
		# create a vnn
		vnn = VNN(network_path, self.next_vnn_id)
		if vnn.name in self.vnns:
			raise Exception('vnn named %s is already there' % vnn.name)

		# save the network weights
		self.save_network_weight(vnn)

		# get and save fisher of network
		fisher_information = self.compute_fisher(vnn, 'network')
		self.save_network_fisher(vnn, fisher_information)

		return vnn

	def add_vnn(self, network_path):
		print('add_vnn')
		# create a vnn
		vnn = self.create_vnn(network_path)

		# allocate weight pages
		self.match_weight_page(vnn)

		# increment next_vnn_id and add vnn to the dictionary
		self.vnns[vnn.name] = vnn
		self.next_vnn_id += 1

		# save vnn
		self.save_vnn(vnn)

		total_vnn_list = []
		for name, vnn in self.vnns.items():
			total_vnn_list.append(vnn)

		total_network_cost = self.calculate_network_cost(total_vnn_list)
		print('total_network_cost:', total_network_cost)

	def add_multi_vnns(self, network_path_list):
		print('add_multi_vnns')
		new_vnn_list = []
		for network_path in network_path_list:
			vnn = self.create_vnn(network_path)
			new_vnn_list.append(vnn)
			self.next_vnn_id += 1

		self.match_weight_page_multi(new_vnn_list)

		for vnn in new_vnn_list:
			self.vnns[vnn.name] = vnn
			self.save_vnn(vnn)

		total_vnn_list = []
		for name, vnn in self.vnns.items():
			total_vnn_list.append(vnn)

		total_network_cost = self.calculate_network_cost(total_vnn_list)
		print('total_network_cost:', total_network_cost)

	def remove_vnn(self, vnn):
		self.dematch_weight_page(vnn)

		if vnn.name in self.vnns:
			del self.vnns[vnn.name]

		if os.path.exists(vnn.network_weight_filepath):
			os.remove(vnn.network_weight_filepath)

		if os.path.exists(vnn.network_fisher_filepath):
			os.remove(vnn.network_fisher_filepath)

		if os.path.exists(vnn.weight_filepath):
			os.remove(vnn.weight_filepath)

		if os.path.exists(vnn.fisher_filepath):
			os.remove(vnn.fisher_filepath)

		if os.path.exists(vnn.filepath):
			os.remove(vnn.filepath)

	def train_vnn(self, vnn, iteration):
		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				self.restore_vnn(vnn, graph, sess)
				matching_loss = self.get_matching_loss(vnn, sess, lamb=10.0)
				pintle = self.import_pintle(vnn)
				weight_vector = pintle.pintle.v_train(graph, sess, matching_loss,
					100, iteration, self.get_weight_from_vnn)
				#weight_vector = self.get_weight_from_vnn(sess)
				if weight_vector is not None:
					self.apply_weight_to_page(vnn, weight_vector)
					self.save_weight_page()

	def execute_vnn(self, vnn, input_variables, ground_truth=None):
		pintle = self.import_pintle(vnn)
		input_variable_names = pintle.pintle.v_input_variable_names()
		input_tensors = []

		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				self.restore_vnn(vnn, graph, sess)

				for variable_name in input_variable_names:
					input_tensor_name = variable_name + ':0'
					input_tensors.append(graph.get_tensor_by_name(input_tensor_name))

				return pintle.pintle.v_execute(graph, sess,
					input_tensors, input_variables, ground_truth)

	def load_vnns(self):
		for file in sorted(os.listdir("./")):
			if file.endswith(".vnn"):
				vnn = self.load_vnn(file)
				self.vnns[vnn.name] = vnn
				if vnn.id >= self.next_vnn_id:
					self.next_vnn_id = vnn.id + 1

	def load_vnn(self, filepath):
		with open(filepath, 'rb') as f:
			vnn = pickle.load(f)
		return vnn

	def save_vnn(self, vnn):
		with open(vnn.filepath, 'wb') as f:
			pickle.dump(vnn, f)

	def load_weight(self, vnn):
		weight = np.load(vnn.weight_filepath, allow_pickle=True)
		return weight

	def save_weight(self, vnn):
		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				self.restore_vnn(vnn, graph, sess)
				tensor_weights = tf.trainable_variables()
				weights = sess.run(tensor_weights)

		np.save(vnn.weight_filepath, weights)
		print(vnn.weight_filepath)

	def load_network_weight(self, vnn):
		network_weight = np.load(vnn.network_weight_filepath, allow_pickle=True)
		return network_weight

	def save_network_weight(self, vnn):
		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				self.restore_network(vnn, sess)
				tensor_weights = tf.trainable_variables()
				network_weights = sess.run(tensor_weights)

		np.save(vnn.network_weight_filepath, network_weights)
		print(vnn.network_weight_filepath)

	def load_network_fisher(self, vnn):
		fisher_information = np.load(vnn.network_fisher_filepath, allow_pickle=True)
		return fisher_information

	def save_network_fisher(self, vnn, fisher_information):
		np.save(vnn.network_fisher_filepath, fisher_information)
		print(vnn.network_fisher_filepath)

	def load_vnn_fisher(self, vnn):
		if os.path.exists(vnn.fisher_filepath):
			fisher_information = np.load(vnn.fisher_filepath, allow_pickle=True)
			return fisher_information
		else:
			return None

	def save_vnn_fisher(self, vnn, fisher_information):
		np.save(vnn.fisher_filepath, fisher_information)
		print(vnn.fisher_filepath)

	def load_weight_to_vnn(self, vnn, graph, sess, weight_vector):
		assign_tensor_weight = []
		tensor_weights = tf.trainable_variables()
		start_idx = 0
		end_idx = 0

		for weight in tensor_weights:
			end_idx = start_idx + np.prod(weight.get_shape().as_list())
			assign_weight = tf.assign(weight, weight_vector[start_idx:end_idx].reshape(weight.shape))
			assign_tensor_weight.append(assign_weight)
			start_idx = end_idx

		sess.run(assign_tensor_weight)

	def get_weight_from_vnn(self, sess):
		tensor_weights = tf.trainable_variables()
		weights = sess.run(tensor_weights)

		weight_vector_list = []
		for weight in weights:
			weight_vector_list.append(weight.reshape((weight.size)))

		weight_vector = np.concatenate(weight_vector_list)
		return weight_vector

	def get_weight_from_page(self, vnn):
		weight_vector_list = []
		for page in vnn.weight_page_list:
			weight_vector_list.append(self.weight_page[page])

		weight_vector = np.concatenate(weight_vector_list)
		return weight_vector[0:vnn.num_of_weight]

	def apply_weight_to_page(self, vnn, weight_vector):
		start_idx = 0
		end_idx = 0

		for page in vnn.weight_page_list:
			end_idx = start_idx + self.weight_page[page].size
			if end_idx <= len(weight_vector):
				self.weight_page[page] = copy.deepcopy(weight_vector[start_idx:end_idx])
				start_idx = end_idx
			else:
				end_idx = len(weight_vector)
				for i in range(end_idx-start_idx):
					self.weight_page[page][i] = copy.deepcopy(weight_vector[start_idx+i])

	def init_weight_page(self, num_of_weight_page=None, weight_per_page=None):
		if num_of_weight_page is not None:
			self.num_of_weight_page = num_of_weight_page

		if weight_per_page is not None:
			self.weight_per_page = weight_per_page

		self.weight_page = np.random.normal(scale=0.01,
			size=(self.num_of_weight_page, self.weight_per_page)).astype(np.float32)

	def load_weight_page(self, weight_page_filename=None):
		if weight_page_filename is not None:
			self.weight_page_filename = weight_page_filename
		if os.path.exists(self.weight_page_filename):
			self.weight_page = np.load(self.weight_page_filename, allow_pickle=True)
			self.num_of_weight_page = len(self.weight_page)
			self.weight_per_page = len(self.weight_page[0])
			return True
		else:
			return False

	def save_weight_page(self, weight_page_filename=None):
		if weight_page_filename is not None:
			self.weight_page_filename = weight_page_filename
		np.save(self.weight_page_filename, self.weight_page)

	def update_weight_page_occupation(self, vnn):
		weight_page_occupation = self.load_weight_page_occupation()

		for i in range(len(vnn.weight_page_list)):
			page_no = vnn.weight_page_list[i]
			weight_page_occupation[page_no].append([vnn.id, i])

		self.save_weight_page_occupation(weight_page_occupation)

	def load_weight_page_occupation(self):
		if os.path.exists(self.weight_page_occupation_filename):
			return np.load(self.weight_page_occupation_filename, allow_pickle=True)
		else:
			weight_page_occupation = [[] for _ in np.arange(self.num_of_weight_page, dtype=np.int32)]
			return weight_page_occupation

	def save_weight_page_occupation(self, weight_page_occupation):
		np.save(self.weight_page_occupation_filename, weight_page_occupation)

	def import_pintle(self, vnn):
		pintle_name = os.path.splitext(vnn.pintle_filepath)[0]
		pintle_import_name = pintle_name.replace('/', '.')
		pintle = __import__(pintle_import_name)
		return pintle

	def restore_network(self, vnn, sess):
		saver = tf.train.import_meta_graph(vnn.meta_filepath)
		saver.restore(sess, vnn.model_filepath)

	def restore_vnn(self, vnn, graph, sess):
		tf.train.import_meta_graph(vnn.meta_filepath)
		weight_vector = self.get_weight_from_page(vnn)
		self.load_weight_to_vnn(vnn, graph, sess, weight_vector)

	def do_compute_fisher(self, sess, fx_tensors, x_tensors, input_tensors,\
		input_variables, num_samples=100):
		print("do_compute_fisher")

		# input_variable[0] is data
		assert input_variables[0].shape[0] >= num_samples

		fisher_information = []
		for v in range(len(x_tensors)):
			fisher_information.append(np.zeros(x_tensors[v].get_shape().as_list()).astype(np.float32))

		for i in range(num_samples):
			data_idx = np.random.randint(input_variables[0].shape[0])
			sampled_data = input_variables[0][data_idx:data_idx+1]
			sampled_input_variables = [ sampled_data ] + input_variables[1:]
			print ('sample num: %4d, data_idx: %5d' % (i, data_idx))
	    
			derivatives, prob = sess.run([tf.gradients(tf.log(fx_tensors), x_tensors), fx_tensors],
				feed_dict={t: v for t,v in zip(input_tensors, sampled_input_variables)})

			for v in range(len(fisher_information)):
				fisher_information[v] += np.square(derivatives[v]) * prob

		for v in range(len(fisher_information)):
			fisher_information[v] /= num_samples

		return fisher_information

	def compute_fisher(self, vnn, target):
		print('compute_fisher')

		pintle = self.import_pintle(vnn)
		input_variable_names = pintle.pintle.v_input_variable_names()

		with tf.Graph().as_default() as graph:
			with tf.Session(graph=graph) as sess:
				if target == 'network':
					self.restore_network(vnn, sess)
				elif target == 'vnn':
					self.restore_vnn(vnn, graph, sess)
				else:
					raise Exception('Neither network nor vnn')

				input_tensors = []
				for variable_name in input_variable_names:
					input_tensor_name = variable_name + ':0'
					input_tensors.append(graph.get_tensor_by_name(input_tensor_name))

				weight_tensors = tf.trainable_variables()
				pintle = self.import_pintle(vnn)
				raw_input_variables = pintle.pintle.v_train_input_variables()
				input_variables = [ raw_input_variables[0][0] ]
				for i in range(1, len(raw_input_variables)):
					input_variables.append(raw_input_variables[i])

				target_tensors = pintle.pintle.v_fx_tensors(graph)

				fisher_information = self.do_compute_fisher(sess, target_tensors, \
					weight_tensors, input_tensors, input_variables, num_samples=100)

				return fisher_information

	def get_fisher_sum_vector(self):
		fisher_dic = {}
		for name_, vnn_ in self.vnns.items():
			fisher = self.load_network_fisher(vnn_)
			#fisher = self.load_vnn_fisher(vnn_)
			fisher_vector = self.vectorize_list(fisher)
			fisher_dic[vnn_.id] = fisher_vector

		fisher_sum_vector = np.zeros(self.num_of_weight_page*self.weight_per_page, dtype=np.float32)
		weight_page_occupation = self.load_weight_page_occupation()

		for i in range(len(weight_page_occupation)):
			fisher_list = []
			for occupation in weight_page_occupation[i]:
				size = self.weight_per_page
				src_start = occupation[1]*self.weight_per_page
				if src_start + size > len(fisher_dic[occupation[0]]):
					size = len(fisher_dic[occupation[0]]) % self.weight_per_page
				src_end = src_start + size
				fisher = fisher_dic[occupation[0]][src_start:src_end]
				if len(fisher) != self.weight_per_page:
					fisher = np.concatenate([fisher,
						np.zeros(self.weight_per_page-len(fisher), dtype=np.float32)])
				fisher_list.append(fisher)

			if not fisher_list:
				continue

			fisher_page_sum = np.sum(fisher_list, axis=0)
			dst_start = i*self.weight_per_page
			dst_end = dst_start+self.weight_per_page
			fisher_sum_vector[dst_start:dst_end] = fisher_page_sum

		return fisher_sum_vector

	def get_fisher_vector_page_order(self, vnn, target):
		fisher = None
		if target == 'network':
			fisher = self.load_network_fisher(vnn)
		elif target == 'vnn':
			fisher = self.load_vnn_fisher(vnn)
		else:
			raise Exception('Neither network nor vnn')

		fisher_vector = self.vectorize_list(fisher)

		fisher_vector_page_order = np.zeros(self.num_of_weight_page*self.weight_per_page,
			dtype=np.float32)
		weight_page_occupation = self.load_weight_page_occupation()

		for i in range(len(weight_page_occupation)):
			for occupation in weight_page_occupation[i]:
				if occupation[0] == vnn.id:
					size = self.weight_per_page
					src_start = occupation[1]*self.weight_per_page
					if src_start + size > len(fisher_vector):
						size = len(fisher_vector) % self.weight_per_page
					src_end = src_start + size
					fisher = fisher_vector[src_start:src_end]
					if len(fisher) != self.weight_per_page:
						fisher = np.concatenate([fisher,
							np.zeros(self.weight_per_page-len(fisher),
							dtype=np.float32)])

					dst_start = i*self.weight_per_page
					dst_end = dst_start+self.weight_per_page
					fisher_vector_page_order[dst_start:dst_end] = fisher
					break

		return fisher_vector_page_order

	def get_weight_vector_page_order(self, vnn, target):
		weight = None
		if target == 'network':
			weight = self.load_network_weight(vnn)
		elif target == 'vnn':
			weight = self.load_vnn_weight(vnn)
		else:
			raise Exception('Neither network nor vnn')

		weight_vector = self.vectorize_list(weight)

		weight_vector_page_order = np.zeros(self.num_of_weight_page*self.weight_per_page,
			dtype=np.float32)
		weight_page_occupation = self.load_weight_page_occupation()

		for i in range(len(weight_page_occupation)):
			for occupation in weight_page_occupation[i]:
				if occupation[0] == vnn.id:
					size = self.weight_per_page
					src_start = occupation[1]*self.weight_per_page
					if src_start + size > len(weight_vector):
						size = len(weight_vector) % self.weight_per_page
					src_end = src_start + size
					weight = weight_vector[src_start:src_end]
					if len(weight) != self.weight_per_page:
						weight = np.concatenate([weight,
							np.zeros(self.weight_per_page-len(weight),
							dtype=np.float32)])

					dst_start = i*self.weight_per_page
					dst_end = dst_start+self.weight_per_page
					weight_vector_page_order[dst_start:dst_end] = weight
					break

		return weight_vector_page_order

	def matching_cost_pair(self, fisher1, weight1, fisher2, weight2):
		assert len(fisher1) == len(weight1)
		assert len(fisher2) == len(weight2)
		assert len(fisher1) == len(fisher2)

		fisher_sum = np.add(fisher1, fisher2)
		square_weight_diff = np.square(np.subtract(weight1, weight2))
		cost = np.sum(np.multiply(fisher_sum, square_weight_diff))

		return cost

	def calculate_cost(self, vnn):
		print('[calculate_cost]')
		fisher_sum_vector = self.get_fisher_sum_vector()
		weight_vector = self.weight_page.flatten()
		assert len(fisher_sum_vector) == len(weight_vector)


		network_fisher_vector = self.get_fisher_vector_pad(vnn, 'network')
		network_weight_vector = self.get_weight_vector_pad(vnn, 'network')
		assert len(network_fisher_vector) == len(network_weight_vector)

		total_cost = 0.0
		idx = 0

		for page_no in vnn.weight_page_list:
			size = self.weight_per_page
			start_n = idx * self.weight_per_page
			end_n = start_n + size
			start_s = page_no * self.weight_per_page
			end_s = start_s + size

			fisher_network = network_fisher_vector[start_n:end_n]
			weight_network = network_weight_vector[start_n:end_n]
			fisher_star = fisher_sum_vector[start_s:end_s]
			weight_star = weight_vector[start_s:end_s]

			zero_fisher_network = np.where(fisher_network == 0)
			fisher_star[zero_fisher_network] = 0
			zero_fisher_star = np.where(fisher_star == 0)
			fisher_network[zero_fisher_star] = 0

			fisher_cost = np.add(fisher_network, fisher_star)
			#fisher_cost = np.multiply(fisher_network, fisher_star)
			weight_cost = np.square(weight_network - weight_star)
			cost = np.sum(np.multiply(fisher_cost, weight_cost))
			total_cost += cost
			idx += 1

		print('toal_cost:', total_cost)
		return total_cost

	def calculate_network_cost(self, vnn_list):
		total_cost = 0

		for i in range(len(vnn_list)):
			for j in range(i+1, len(vnn_list)):
				fisher1 = self.get_fisher_vector_page_order(vnn_list[i], 'network')
				weight1 = self.get_weight_vector_page_order(vnn_list[i], 'network')
				fisher2 = self.get_fisher_vector_page_order(vnn_list[j], 'network')
				weight2 = self.get_weight_vector_page_order(vnn_list[j], 'network')
				zero_fisher1 = np.where(fisher1 <= 0)
				fisher2[zero_fisher1] = 0
				zero_fisher2 = np.where(fisher2 <= 0)
				fisher1[zero_fisher2] = 0
				total_cost += self.matching_cost_pair(fisher1, weight1, fisher2, weight2)

		return total_cost

	def match_page_by_cost(self, vnn):
		print('[match_page_by_cost]')
		fisher_sum_vector = self.get_fisher_sum_vector()
		weight_vector = self.weight_page.flatten()
		assert len(fisher_sum_vector) == len(weight_vector)

		network_fisher_vector = self.get_fisher_vector_pad(vnn, 'network')
		network_weight_vector = self.get_weight_vector_pad(vnn, 'network')
		assert len(network_fisher_vector) == len(network_weight_vector)

		"""
		page_to_alloc = len(network_weight_vector)/self.weight_per_page
		page_list = np.arange(self.num_of_weight_page, dtype=np.int32)
		network_page_list = np.arange(page_to_alloc, dtype=np.int32)
		weight_virtualization_op = tf.load_op_library(self.weight_virtualization_op_filename)

		with tf.Graph().as_default() as graph:
			with tf.Session() as sess:
				page_alloc_op = weight_virtualization_op.page_alloc(fisher_sum_vector,
					weight_vector, page_list, network_fisher_vector,
					network_weight_vector, network_page_list,
					page_size=self.weight_per_page)
				page_match, cost = sess.run(page_alloc_op)

		print('cost:', cost)
		print('')
		weight_page_list = page_match[page_match[:,1].argsort()][:,0].astype(np.int32)
		if len(weight_page_list) > len(set(weight_page_list)):
			raise Exception('weight_page_list is not unique')
		assert len(weight_page_list) == page_to_alloc
		"""

		#"""
		weight_page_occupation = self.load_weight_page_occupation()
		len_list_of_occupation = np.asarray([len(page_occupation) for page_occupation in weight_page_occupation])
		max_occupation = np.max(len_list_of_occupation)
		page_to_alloc = len(network_weight_vector)/self.weight_per_page
		network_page_list = np.arange(page_to_alloc, dtype=np.int32)
		page_match_list = []
		total_cost = 0

		weight_virtualization_op = tf.load_op_library(self.weight_virtualization_op_filename)
		with tf.Graph().as_default() as graph:
			with tf.Session() as sess:
				for occupation in range(max_occupation+1):
					page_list = np.where(len_list_of_occupation == occupation)[0]
					print('occupation:', occupation)
					print('len(page_list):', len(page_list))
					print('len(network_page_list):', len(network_page_list))
					if len(page_list) <= 0:
						print('cost: 0\n')
						continue

					page_alloc_op = weight_virtualization_op.page_alloc(fisher_sum_vector,
						weight_vector, page_list, network_fisher_vector,
						network_weight_vector, network_page_list,
						page_size=self.weight_per_page)
					page_match, cost = sess.run(page_alloc_op)

					total_cost += cost
					print('cost:', cost)
					print('')
					page_match_list.append(page_match)
					network_page_list = list(set(network_page_list) - set(page_match[:,1]))

					if not network_page_list:
						break

		if network_page_list:
			raise Exception('network_page_list is not empty')

		page_match_array = np.concatenate(page_match_list)
		weight_page_list = page_match_array[page_match_array[:,1].argsort()][:,0].astype(np.int32)
		if len(weight_page_list) > len(set(weight_page_list)):
			raise Exception('weight_page_list is not unique')
		assert len(weight_page_list) == page_to_alloc

		#print('total_cost:', total_cost)
		#"""

		vnn.weight_page_list = weight_page_list

	def get_fisher_vector_pad(self, vnn, target):
		if target == 'network':
			fisher = self.load_network_fisher(vnn)
		elif target == 'vnn':
			fisher = self.load_vnn_fisher(vnn)
		else:
			raise Exception('Neither network nor vnn')

		fisher_vector = self.vectorize_list(fisher)
		pad_len = self.weight_per_page - (len(fisher_vector) % self.weight_per_page)
		fisher_vector_pad = np.concatenate([fisher_vector,
			np.zeros(pad_len, dtype=np.float32)])
		assert len(fisher_vector_pad) % self.weight_per_page == 0

		return fisher_vector_pad

	def get_weight_vector_pad(self, vnn, target):
		if target == 'network':
			weight = self.load_network_weight(vnn)
		elif target == 'vnn':
			weight = self.load_vnn_weight(vnn)
		else:
			raise Exception('Neither network nor vnn')

		weight_vector = self.vectorize_list(weight)
		pad_len = self.weight_per_page - (len(weight_vector) % self.weight_per_page)
		weight_vector_pad = np.concatenate([weight_vector,
			np.zeros(pad_len, dtype=np.float32)])
		assert len(weight_vector_pad) % self.weight_per_page == 0

		return weight_vector_pad

	def match_page_by_random_multi(self, vnn_list):
		print('[match_page_by_random_multi]')
		for vnn in vnn_list:
			network_fisher_vector = self.get_fisher_vector_pad(vnn, 'network')
			network_weight_vector = self.get_weight_vector_pad(vnn, 'network')
			assert len(network_fisher_vector) == len(network_weight_vector)
			page_to_alloc = len(network_weight_vector)/self.weight_per_page
			vnn.weight_page_list = np.random.choice(self.num_of_weight_page, page_to_alloc, replace=False)

	def match_page_by_cost_multi(self, vnn_list):
		print('[match_page_by_cost_multi]')
		fisher_sum_vector = self.get_fisher_sum_vector()
		weight_vector = self.weight_page.flatten()
		assert len(fisher_sum_vector) == len(weight_vector)
		base_page_list = np.arange(self.num_of_weight_page, dtype=np.int32)

		network_fisher_vector_list = []
		network_weight_vector_list = []
		network_page_list_list = []
		page_to_alloc_list = []

		for vnn in vnn_list:
			network_fisher_vector = self.get_fisher_vector_pad(vnn, 'network')
			network_weight_vector = self.get_weight_vector_pad(vnn, 'network')
			assert len(network_fisher_vector) == len(network_weight_vector)
			network_fisher_vector_list.append(network_fisher_vector)
			network_weight_vector_list.append(network_weight_vector)
			page_to_alloc = len(network_weight_vector)/self.weight_per_page
			page_to_alloc_list.append(page_to_alloc)
			network_page_list = np.arange(page_to_alloc, dtype=np.int32)
			network_page_list_list.append(network_page_list)

		weight_virtualization_op = tf.load_op_library(self.weight_virtualization_op_filename)
		with tf.Graph().as_default() as graph:
			with tf.Session() as sess:
				page_alloc_multi_op = weight_virtualization_op.page_alloc_multi(fisher_sum_vector,
					weight_vector, base_page_list, network_fisher_vector_list,
					network_weight_vector_list, network_page_list_list,
					page_size=self.weight_per_page)
				page_match_list, cost = sess.run(page_alloc_multi_op)

		print('total cost[d]:', cost)

		for vnn, page_match, num_of_page in zip(vnn_list, page_match_list, page_to_alloc_list):
			weight_page_list = page_match.astype(np.int32)
			assert len(weight_page_list) == num_of_page
			if len(weight_page_list) > len(set(weight_page_list)):
				raise Exception('weight_page_list is not unique')
			vnn.weight_page_list = weight_page_list

	def match_page_by_random(self, vnn, num_of_page_to_select):
		print('[match_page_by_random]')
		weight_page_occupation = self.load_weight_page_occupation()
		len_list_of_occupation = np.asarray([len(page_occupation) for page_occupation in weight_page_occupation])
		max_occupation = np.max(len_list_of_occupation)
		page_sorted_by_occupation = []
		num_of_weight_page_not_allocated = num_of_page_to_select
		weight_page_list = []

		for occupation in range(max_occupation+1):
			page_sorted_by_occupation.append(np.where(len_list_of_occupation == occupation))
			pages = page_sorted_by_occupation[occupation][0]

			if num_of_weight_page_not_allocated > len(pages):
				weight_page_list = np.concatenate((weight_page_list, pages))
				num_of_weight_page_not_allocated -= len(pages)
			else:
				if occupation != 0:
					np.random.shuffle(pages) # By RANDOM
				weight_page_list = np.concatenate((weight_page_list,
					pages[0:num_of_weight_page_not_allocated]))
				num_of_weight_page_not_allocated = 0

		np.random.shuffle(weight_page_list)
		if len(weight_page_list) > len(set(weight_page_list)):
			raise Exception('weight_page_list is not unique')

		vnn.weight_page_list = weight_page_list.astype(np.int32)
		#vnn.weight_page_list = np.random.choice(self.num_of_weight_page, num_of_page_to_select, replace=False)

	def match_weight_page(self, vnn):
		num_of_weight_page = vnn.num_of_weight // self.weight_per_page
		if vnn.num_of_weight % self.weight_per_page != 0:
			num_of_weight_page += 1
		assert num_of_weight_page <= self.num_of_weight_page,\
			"%d vs. %d" % (num_of_weight_page, self.num_of_weight_page)

		if not self.vnns:
			vnn.weight_page_list = np.arange(num_of_weight_page, dtype=np.int32)
		else:
			time1 = time.time()
			#self.match_page_by_random(vnn, num_of_weight_page)
			self.match_page_by_cost(vnn)
			time2 = time.time()
			print('assing_page %0.3f ms' % ((time2-time1)*1000.0))

		self.calculate_cost(vnn)
		self.update_weight_page_occupation(vnn)
		print("%d pages allocated for %d weights" %
			(len(vnn.weight_page_list), vnn.num_of_weight))

	def match_weight_page_multi(self, vnn_list):
		for vnn in vnn_list:
			num_of_weight_page = vnn.num_of_weight // self.weight_per_page
			if vnn.num_of_weight % self.weight_per_page != 0:
				num_of_weight_page += 1
			assert num_of_weight_page <= self.num_of_weight_page,\
				"%d vs. %d" % (num_of_weight_page, self.num_of_weight_page)

		self.match_page_by_cost_multi(vnn_list)

		for vnn in vnn_list:
			self.update_weight_page_occupation(vnn)
			print("%d pages allocated for %d weights" %
				(len(vnn.weight_page_list), vnn.num_of_weight))

		new_network_cost = self.calculate_network_cost(vnn_list)
		print('new_network_cost:', new_network_cost)

		"""
		if os.path.exists(self.weight_page_occupation_filename):
			os.remove(self.weight_page_occupation_filename)
		self.match_page_by_random_multi(vnn_list)
		for vnn in vnn_list:
			self.update_weight_page_occupation(vnn)
		total_cost = self.calculate_network_cost(vnn_list)
		print('total cost:', total_cost)

		if os.path.exists(self.weight_page_occupation_filename):
			os.remove(self.weight_page_occupation_filename)

		exit(1)
		"""

	def dematch_weight_page(self, vnn):
		weight_page_occupation = self.load_weight_page_occupation()
		for page in weight_page_occupation:
			for occupation in page:
				if occupation[0] == vnn.id:
					page.remove(occupation)
					break
		self.save_weight_page_occupation(weight_page_occupation)

	def quadratic_mean(self, cost_list):
		if not cost_list:
			return 0

		stacked = tf.stack(cost_list)
		non_zero = tf.cast(tf.count_nonzero(stacked, 0), tf.float32)
		non_zero_pad = tf.where(tf.equal(non_zero, 0), tf.ones_like(non_zero), non_zero)
		return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(stacked), 0) / non_zero_pad))

	def arithmetic_mean(self, cost_list):
		if not cost_list:
			return 0

		stacked = tf.stack(cost_list)
		non_zero = tf.cast(tf.count_nonzero(stacked, 0), tf.float32)
		non_zero_pad = tf.where(tf.equal(non_zero, 0), tf.ones_like(non_zero), non_zero)
		return tf.reduce_sum(tf.reduce_sum(stacked, 0) / non_zero_pad)

	def harmonic_mean(self, cost_list):
		@ops.RegisterGradient("HarmonicMean")
		def harmonic_mean_grad(op, grad):
			input_list = []
			for i in range(len(op.inputs)):
				input_list.append(op.inputs[i])

			stacked = tf.stack(input_list)
			non_zero = tf.count_nonzero(stacked, 0)
			grad_list = []

			for i in range(len(op.inputs)):
				gradient = tf.where(op.inputs[i] <= 0, tf.zeros_like(non_zero), non_zero)
				grad_list.append(tf.cast(gradient, tf.float32))

			return grad_list

		if not cost_list:
			return 0

		weight_virtualization_op = tf.load_op_library(self.weight_virtualization_op_filename)
		return weight_virtualization_op.harmonic_mean(cost_list)

	def get_matching_loss(self, vnn, sess, lamb=10.0):
		print ("get_matching_loss")
		matching_loss = tf.constant(0.0)

		tensor_weights = tf.trainable_variables()
		tensor_weights_concat = []
		for weight in tensor_weights:
			tensor_weights_concat.append(tf.reshape(weight, [tf.size(weight)]))

		new_weight_vector = tf.concat(tensor_weights_concat, 0)
		new_weight_vector_len = new_weight_vector.get_shape().as_list()[0]
		weight_page_occupation = self.load_weight_page_occupation()
		cost_list = []

		for name_, vnn_ in self.vnns.items():
			if vnn.id == vnn_.id:
				continue;

			fisher = self.load_network_fisher(vnn_)
			#fisher = self.load_vnn_fisher(vnn_)
			if fisher is None:
				continue
			fisher_vector = self.vectorize_list(fisher)

			#weight = self.load_network_weight(vnn_)
			weight = self.load_weight(vnn_)
			#weight_vector = self.get_weight_from_page(vnn_)
			weight_vector = self.vectorize_list(weight)
			assert len(fisher_vector) == len(weight_vector)

			fisher_vector_reordered = np.zeros(new_weight_vector_len, dtype=np.float32)
			weight_vector_reordered = np.zeros(new_weight_vector_len, dtype=np.float32)

			page_idx = 0
			for page in vnn.weight_page_list:
				size = self.weight_per_page
				page_pos = -1

				for occupation in weight_page_occupation[page]:
					if occupation[0] == vnn_.id:
						page_pos = occupation[1]
						break

				if page_pos == -1:
					page_idx += 1
					continue

				if page_idx*self.weight_per_page+size > new_weight_vector_len:
					size = new_weight_vector_len % self.weight_per_page
				if page_pos*self.weight_per_page+size > len(fisher_vector):
					size = len(fisher_vector) % self.weight_per_page

				dst_start = page_idx*self.weight_per_page
				dst_end = dst_start+size
				src_start = page_pos*self.weight_per_page
				src_end = src_start+size
				fisher_vector_reordered[dst_start:dst_end] = fisher_vector[src_start:src_end]
				weight_vector_reordered[dst_start:dst_end] = weight_vector[src_start:src_end]
				page_idx += 1

			weight_diff_square = tf.square(new_weight_vector - weight_vector_reordered)
			cost_list.append(tf.multiply(weight_diff_square, fisher_vector_reordered))

		if cost_list:
			#matching_loss += self.quadratic_mean(cost_list)
			matching_loss += self.arithmetic_mean(cost_list)
			#matching_loss += self.harmonic_mean(cost_list)

		return lamb*matching_loss

	def vectorize_list(self, list_to_vectorize):
		vector_list = []
		for item in list_to_vectorize:
			vector_list.append(item.flatten())
		return np.concatenate(vector_list)

	def plot_vnn_fisher(self, vnn_list=None):
		if vnn_list is None:
			vnn_list = []
			for name, vnn in self.vnns.items():
				vnn_list.append(vnn)

		for vnn in vnn_list:
			fisher = self.get_fisher_vector_page_order(vnn, 'vnn')
			plt.plot(fisher, label=vnn.name)
			plt.legend(loc='upper right')
		plt.show()

	def plot_vnn_weight(self, vnn_list=None):
		if vnn_list is None:
			vnn_list = []
			for name, vnn in self.vnns.items():
				vnn_list.append(vnn)

		for vnn in vnn_list:
			weight = self.get_weight_from_page(vnn)
			plt.plot(weight, label=vnn.name)
			plt.legend(loc='upper right')
		plt.show()

	def plot_network_fisher(self, vnn_list=None):
		if vnn_list is None:
			vnn_list = []
			for name, vnn in self.vnns.items():
				vnn_list.append(vnn)

		for vnn in vnn_list:
			fisher = self.load_network_fisher(vnn)
			plt.plot(self.vectorize_list(fisher), label=vnn.name)
			plt.legend(loc='upper right')
		plt.show()

	def plot_network_weight(self, vnn_list=None):
		if vnn_list is None:
			vnn_list = []
			for name, vnn in self.vnns.items():
				vnn_list.append(vnn)

		for vnn in vnn_list:
			weight = self.load_network_weight(vnn)
			plt.plot(self.vectorize_list(weight), label=vnn.name)
			plt.legend(loc='upper right')
		plt.show()

	def plot_network_fisher_histogram(self, vnn):
		network_fisher = self.load_network_fisher(vnn)
		plt.hist(network_fisher, density=False, bins=100)
		plt.show()

	def plot_network_weight_histogram(self, vnn):
		network_weight = self.load_network_weight(vnn)
		plt.hist(network_weight, density=False, bins=100)
		plt.show()

	def plot_sharing_cost_heatmap(self, vnn_list, page_size=100):
		vnn1 = vnn_list[0]
		print('vnn1.num_of_weight', vnn1.num_of_weight)
		vnn1_num_of_page = vnn1.num_of_weight // page_size
		print('vnn1_num_of_page', vnn1_num_of_page)

		vnn2 = vnn_list[1]
		print('vnn2.num_of_weight', vnn2.num_of_weight)
		vnn2_num_of_page = vnn2.num_of_weight // page_size
		print('vnn2_num_of_page', vnn2_num_of_page)

		weight1 = self.vectorize_list(self.load_network_weight(vnn1))
		print(weight1.shape)
		fisher1 = self.vectorize_list(self.load_network_fisher(vnn1))
		print(fisher1.shape)

		weight2 = self.vectorize_list(self.load_network_weight(vnn2))
		fisher2 = self.vectorize_list(self.load_network_fisher(vnn2))

		sharing_cost_matrix = np.zeros((vnn2_num_of_page, vnn1_num_of_page))

		for i in range(vnn1_num_of_page):
			if i >= vnn2_num_of_page:
				break

			print(i)
			sharing_cost_vector = np.zeros((vnn2_num_of_page))
			w1 = weight1[i*page_size:(i+1)*page_size]
			f1 = fisher1[i*page_size:(i+1)*page_size]

			for j in range(vnn2_num_of_page):
				w2 = weight2[j*page_size:(j+1)*page_size]
				f2 = fisher2[j*page_size:(j+1)*page_size]
				sharing_cost = np.sum(np.multiply(np.square(w1 - w2), (f1 + f2)))
				sharing_cost_matrix[j,i] = sharing_cost

		small = vnn1_num_of_page
		if small > vnn2_num_of_page:
			small = vnn2_num_of_page

		sharing_cost_matrix = sharing_cost_matrix[0:small, 0:small]
		print(sharing_cost_matrix.shape)

		import seaborn as sns
		from matplotlib.colors import LogNorm
		from matplotlib.pyplot import figure
		figure(num=None, figsize=(8, 4.5))
		sns.set(font_scale=1.7)
		ax = sns.heatmap(sharing_cost_matrix,
			norm=LogNorm(vmin=np.min(sharing_cost_matrix), vmax=np.max(sharing_cost_matrix)),
			cmap='Blues',
			xticklabels=100, yticklabels=100)
		ax.invert_yaxis()
		plt.xlabel('xlabel', fontsize=20)
		plt.ylabel('ylabel', fontsize=20)
		plt.savefig("sharing_score_heatmap.pdf", bbox_inches='tight')
		plt.show()

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('-mode', type=str,	help='mode', default='l')
	# a: add a vnn from a network
	# am: add multiple vnns from networks
	# r: remove a vnn
	# t: train a vnn
	# e: execute inference of a vnn
	# f: compute fisher informaiont of a vnn
	# c: calculate matching cost

	parser.add_argument('-network_path', type=str, help='network_path', default=None)
	parser.add_argument('-vnn_name', type=str, help='vnn_name', default=None)
	parser.add_argument('-iter', type=int, help='training iteration', default=5000)

	return parser.parse_args(argv)

def main(args):
	wv = WeightVirtualization()

	if args.mode == 'l':
		print('[VNN list]')
		for vnn_name in wv.vnns:
			vnn = wv.vnns[vnn_name]
			print('Name: %s, id: %d, path: %s, num_of_weight: %d'
				% (vnn.name, vnn.id, vnn.network_path, vnn.num_of_weight))

		weight_page_occupation = wv.load_weight_page_occupation()
		print('\n[Weight page] total %d weight pages' % len(weight_page_occupation))
		print(weight_page_occupation)

	elif args.mode == 'a':
		if args.network_path is None:
			print('no network_path')
			return

		assert os.path.exists(args.network_path)
		wv.add_vnn(args.network_path)

	elif args.mode == 'am':
		if args.network_path is None:
			print('no network_path')
			return

		network_path_list = args.network_path.split(',')
		for network_path in network_path_list:
			assert os.path.exists(network_path)
		wv.add_multi_vnns(network_path_list)

	elif args.mode == 'r':
		if args.vnn_name is None:
			print('no vnn name')
			return

		vnn = wv.vnns[args.vnn_name]
		wv.remove_vnn(vnn)

	elif args.mode == 't':
		if args.vnn_name is None:
			print('no vnn name')
			return

		vnn = wv.vnns[args.vnn_name]
		wv.train_vnn(vnn, args.iter)
		wv.save_weight(vnn)

	elif args.mode == 'e':
		if args.vnn_name is None:
			print('no vnn name')
			return

		vnn = wv.vnns[args.vnn_name]
		pintle = wv.import_pintle(vnn)
		raw_input_variables = pintle.pintle.v_test_input_variables()
		input_variables = [ raw_input_variables[0][0] ]
		for i in range(1, len(raw_input_variables)):
			input_variables.append(raw_input_variables[i])
		ground_truth = raw_input_variables[0][1]

		result, accuracy = wv.execute_vnn(vnn, input_variables, ground_truth)
		if accuracy:
			with open(args.vnn_name + '.accuracy', "a") as file:
				file.write(str(accuracy) + '\n')

	elif args.mode == 'f':
		if args.vnn_name is None:
			print('no vnn name')
			return

		vnn = wv.vnns[args.vnn_name]
		fisher_information = wv.compute_fisher(vnn, 'vnn')
		wv.save_vnn_fisher(vnn, fisher_information)

	elif args.mode == 'c':
		if args.vnn_name is None:
			print('no vnn_name')
			return

		vnn_name_list = args.vnn_name.split(',')
		vnn_list = []

		for name in vnn_name_list:
			vnn = wv.vnns[name]
			assert os.path.exists(vnn.network_path)
			vnn_list.append(vnn)

		total_cost = wv.calculate_network_cost(vnn_list)
		print('total cost:', total_cost)

	elif args.mode == 'pf':
		vnn_list = []
		if args.vnn_name is None:
			for name, vnn in wv.vnns.items():
				vnn_list.append(vnn)
		else:
			vnn_name_list = args.vnn_name.split(',')
			for name in vnn_name_list:
				vnn = wv.vnns[name]
				assert os.path.exists(vnn.network_path)
				vnn_list.append(vnn)
		wv.plot_vnn_fisher(vnn_list)

	elif args.mode == 'pw':
		vnn_list = []
		if args.vnn_name is None:
			for name, vnn in wv.vnns.items():
				vnn_list.append(vnn)
		else:
			vnn_name_list = args.vnn_name.split(',')
			for name in vnn_name_list:
				vnn = wv.vnns[name]
				assert os.path.exists(vnn.network_path)
				vnn_list.append(vnn)
		wv.plot_vnn_weight(vnn_list)

	elif args.mode == 'pnf':
		vnn_list = []
		if args.vnn_name is None:
			for name, vnn in wv.vnns.items():
				vnn_list.append(vnn)
		else:
			vnn_name_list = args.vnn_name.split(',')
			for name in vnn_name_list:
				vnn = wv.vnns[name]
				assert os.path.exists(vnn.network_path)
				vnn_list.append(vnn)
		wv.plot_network_fisher(vnn_list)

	elif args.mode == 'pnw':
		vnn_list = []
		if args.vnn_name is None:
			for name, vnn in wv.vnns.items():
				vnn_list.append(vnn)
		else:
			vnn_name_list = args.vnn_name.split(',')
			for name in vnn_name_list:
				vnn = wv.vnns[name]
				assert os.path.exists(vnn.network_path)
				vnn_list.append(vnn)
		wv.plot_network_weight(vnn_list)

	elif args.mode == 'pnfh':
		if args.vnn_name is None:
			print('no vnn name')
			return

		vnn = wv.vnns[args.vnn_name]
		wv.plot_network_fisher_histogram(vnn)

	elif args.mode == 'pnwh':
		if args.vnn_name is None:
			print('no vnn name')
			return

		vnn = wv.vnns[args.vnn_name]
		wv.plot_network_weight_histogram(vnn)

	elif args.mode == 'heatmap':
		vnn_list = []
		if args.vnn_name is None:
			print('no vnn name')
			return
		else:
			vnn_name_list = args.vnn_name.split(',')
			assert len(vnn_name_list) == 2
			for name in vnn_name_list:
				vnn = wv.vnns[name]
				assert os.path.exists(vnn.network_path)
				vnn_list.append(vnn)
		wv.plot_sharing_cost_heatmap(vnn_list)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
