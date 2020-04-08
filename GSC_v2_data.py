from __future__ import print_function
import numpy as np
import os
import struct

train_path = '...'
test_path = '...'
validation_path = '...'

#"""
GSC_v2_train_data = np.load('GSC_v2_train_data.npy')
GSC_v2_train_label = np.load('GSC_v2_train_label.npy')

GSC_v2_test_data = np.load('GSC_v2_test_data.npy')
GSC_v2_test_label = np.load('GSC_v2_test_label.npy')

GSC_v2_validation_data = np.load('GSC_v2_validation_data.npy')
GSC_v2_validation_label = np.load('GSC_v2_validation_label.npy')
#"""

def train_set():
	return GSC_v2_train_data, GSC_v2_train_label

def test_set():
	return GSC_v2_test_data, GSC_v2_test_label

def validation_set():
	return GSC_v2_validation_data, GSC_v2_validation_label

def get_GSC_v2(filename):
	file = open(filename, "rb")
	file_content = file.read()
	file.close()

	(num_frame, num_filter) = struct.unpack("ii", file_content[:8])

	GSC_v2_tuple = struct.unpack("f" * num_frame * num_filter, file_content[8:8+num_frame*num_filter*4])
	GSC_v2 = np.asarray(GSC_v2_tuple).reshape((num_frame,num_filter))

	crop_start_filter = 1
	crop_end_filter = 13
	GSC_v2_cropped = GSC_v2[:,crop_start_filter:crop_start_filter+crop_end_filter]

	GSC_v2_resized = GSC_v2_cropped.reshape((num_frame*(crop_start_filter+crop_end_filter-1)))

	normalized_GSC_v2 = GSC_v2_resized / np.linalg.norm(GSC_v2_resized)

	return normalized_GSC_v2

def get_GSC_v2_batch(dir_path):
	GSC_v2_batch = []
	labels = []
	label = 0

	for (dirpath, dirnames, filenames) in os.walk(dir_path):
		if os.path.basename(dirpath) == '':
			continue

		for (dirpath2, dirnames2, filenames2) in os.walk(dirpath):
			for file in sorted(filenames2):
				filepath = os.path.join(dirpath2, file)
				GSC_v2_batch.append(get_GSC_v2(filepath))
				labels.append(label)

		label += 1

	GSC_v2_batch = np.vstack((GSC_v2_batch))
	labels_one_hot = np.zeros((GSC_v2_batch.shape[0], label))
	labels_one_hot[np.arange(GSC_v2_batch.shape[0]), labels] = 1

	return GSC_v2_batch, labels_one_hot

def create_data_files():
	GSC_v2_train_data, GSC_v2_train_label = get_GSC_v2_batch(train_path)
	GSC_v2_test_data, GSC_v2_test_label = get_GSC_v2_batch(test_path)
	GSC_v2_validation_data, GSC_v2_validation_label = get_GSC_v2_batch(validation_path)

	np.save('GSC_v2_train_data', GSC_v2_train_data)
	np.save('GSC_v2_train_label', GSC_v2_train_label)
	np.save('GSC_v2_test_data', GSC_v2_test_data)
	np.save('GSC_v2_test_label', GSC_v2_test_label)
	np.save('GSC_v2_validation_data', GSC_v2_validation_data)
	np.save('GSC_v2_validation_label', GSC_v2_validation_label)

def main():
	#create_data_files()
	print(train_set()[0])
	print(train_set()[0].shape)
	print(train_set()[1])
	print(train_set()[1].shape)

if __name__ == '__main__':
        main()

