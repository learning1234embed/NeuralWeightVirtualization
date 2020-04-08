from __future__ import print_function
from PIL import Image
import numpy as np
import os

train_path = '...'
test_path = '...'

#"""
GTSRB_train_data = np.load('GTSRB_train_data.npy')
GTSRB_train_label = np.load('GTSRB_train_label.npy')

GTSRB_test_data = np.load('GTSRB_test_data.npy')
GTSRB_test_label = np.load('GTSRB_test_label.npy')
#"""

def train_set():
	return GTSRB_train_data, GTSRB_train_label

def test_set():
	return GTSRB_test_data, GTSRB_test_label

def validation_set():
	return None, None

def get_GTSRB_test_batch(dir_path):
	GTSRB_batch = []
	labels = []

	csv_filepath = os.path.join(dir_path, 'GT-final_test.csv')
	if not os.path.exists(csv_filepath):
        	return

	print(csv_filepath)

	with open(csv_filepath) as f:
		for line in f:
			lists = line.split(';')
			img_filepath = os.path.join(dir_path, lists[0])
			new_img_filepath = os.path.join(dir_path, os.path.basename(img_filepath))
			new_img_filepath = os.path.splitext(new_img_filepath)[0] + '.png'
			if not os.path.exists(new_img_filepath):
		                continue

			img = Image.open(new_img_filepath)
			GTSRB = np.array(img)
			GTSRB = GTSRB.reshape((np.size(GTSRB)))
			normalized_GTSRB = GTSRB / np.linalg.norm(GTSRB)
			label = int(lists[7])
			GTSRB_batch.append(normalized_GTSRB)
			labels.append(label)

	GTSRB_batch = np.stack((GTSRB_batch))
	print(GTSRB_batch.shape)

	labels_one_hot = np.zeros((GTSRB_batch.shape[0], np.max(labels)+1))
	labels_one_hot[np.arange(GTSRB_batch.shape[0]), labels] = 1
	print(labels_one_hot.shape)

	return GTSRB_batch, labels_one_hot

def get_GTSRB_train_batch(dir_path):
	GTSRB_batch = []
	labels = []
	label = 0

	for (dirpath, dirnames, filenames) in sorted(os.walk(dir_path)):
		if os.path.basename(dirpath) == '':
			continue

		for (dirpath2, dirnames2, filenames2) in sorted(os.walk(dirpath)):
			for file in sorted(filenames2):
				filepath = os.path.join(dirpath2, file)
				img = Image.open(filepath)
				GTSRB = np.array(img)
				GTSRB = GTSRB.reshape((np.size(GTSRB)))
				normalized_GTSRB = GTSRB / np.linalg.norm(GTSRB)
				GTSRB_batch.append(normalized_GTSRB)
				labels.append(label)

		label += 1

	GTSRB_batch = np.vstack((GTSRB_batch))
	labels_one_hot = np.zeros((GTSRB_batch.shape[0], label))
	labels_one_hot[np.arange(GTSRB_batch.shape[0]), labels] = 1

	print(labels_one_hot.shape)

	return GTSRB_batch, labels_one_hot

def create_data_files():
	GTSRB_train_data, GTSRB_train_label = get_GTSRB_train_batch(train_path)
	GTSRB_test_data, GTSRB_test_label = get_GTSRB_test_batch(test_path)

	np.save('GTSRB_train_data', GTSRB_train_data)
	np.save('GTSRB_train_label', GTSRB_train_label)
	np.save('GTSRB_test_data', GTSRB_test_data)
	np.save('GTSRB_test_label', GTSRB_test_label)

def main():
	#create_data_files()
	print(train_set()[0])
	print(train_set()[0].shape)
	print(train_set()[1])
	print(train_set()[1].shape)

if __name__ == '__main__':
        main()

