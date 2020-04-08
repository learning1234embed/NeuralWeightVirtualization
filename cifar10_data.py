from __future__ import division, print_function, unicode_literals
import numpy as np

#"""
cifar10_train_data = np.load('cifar10_train_data.npy')
cifar10_train_label = np.load('cifar10_train_label.npy')
cifar10_test_data = np.load('cifar10_test_data.npy')
cifar10_test_label = np.load('cifar10_test_label.npy')
#"""

def train_set():
	return cifar10_train_data, cifar10_train_label

def test_set():
	return cifar10_test_data, cifar10_test_label

def validation_set():
	return None, None

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def get_cifar_test_batch(batch_size=10000):
	filename = "..."
	dict = unpickle(filename)

	idx = range(batch_size)
	img_flat = dict['data'][idx]

	labels = []
	for i in range(batch_size):
		labels.append(dict['labels'][idx[i]])

	img_R = img_flat[:,0:1024].reshape((batch_size, 32, 32))
	img_G = img_flat[:,1024:2048].reshape((batch_size, 32, 32))
	img_B = img_flat[:,2048:3072].reshape((batch_size, 32, 32))
	img = np.stack((img_R, img_G, img_B), axis=3)
	batch = img / np.max(img)

	labels_one_hot = np.zeros((batch_size, 10))
	labels_one_hot[np.arange(batch_size), labels] = 1

	return batch, labels_one_hot

def get_cifar_train_batch(batch_size=10000):
	batch_list = []
	labels_one_hot_list = []

	for file_no in range (1, 5+1):
		filename = "..." + str(file_no)
		dict = unpickle(filename)

		idx = range(batch_size)
		img_flat = dict['data'][idx]

		labels = []
		for i in range(batch_size):
			labels.append(dict['labels'][idx[i]])

		img_R = img_flat[:,0:1024].reshape((batch_size, 32, 32))
		img_G = img_flat[:,1024:2048].reshape((batch_size, 32, 32))
		img_B = img_flat[:,2048:3072].reshape((batch_size, 32, 32))
		img = np.stack((img_R, img_G, img_B), axis=3)
		batch = img / np.max(img)

		labels_one_hot = np.zeros((batch_size, 10))
		labels_one_hot[np.arange(batch_size), labels] = 1

		batch_list.append(batch)
		labels_one_hot_list.append(labels_one_hot)

	return np.vstack((batch_list)), np.vstack((labels_one_hot_list))

def create_data_files():
	cifar10_train_data, cifar10_train_label = get_cifar_train_batch()
	cifar10_test_data, cifar10_test_label = get_cifar_test_batch()

	np.save('cifar10_train_data', cifar10_train_data)
	print (cifar10_train_data.shape)
	np.save('cifar10_train_label', cifar10_train_label)
	print (cifar10_train_label.shape)
	np.save('cifar10_test_data', cifar10_test_data)
	print (cifar10_test_data.shape)
	np.save('cifar10_test_label', cifar10_test_label)
	print (cifar10_test_label.shape)

def main():
	#create_data_files()

	print(train_set()[0])
	print(train_set()[0].shape)
	print(train_set()[1])
	print(train_set()[1].shape)

if __name__ == '__main__':
	main()
