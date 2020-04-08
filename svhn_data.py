from __future__ import print_function
import numpy as np
import os

train_path = '...'
test_path = '...'

#"""
svhn_train_data = np.load('svhn_train_data.npy')
svhn_train_label = np.load('svhn_train_label.npy')

svhn_test_data = np.load('svhn_test_data.npy')
svhn_test_label = np.load('svhn_test_label.npy')

svhn_validation_data = np.load('svhn_validation_data.npy')
svhn_validation_label = np.load('svhn_validation_label.npy')

#"""

def train_set():
	return svhn_train_data, svhn_train_label

def test_set():
	return svhn_test_data, svhn_test_label

def validation_set():
	return svhn_validation_data, svhn_validation_label

def create_data_files():
	return

def main():
	#create_data_files()
	print(train_set()[0])
	print(train_set()[0].shape)
	print(train_set()[1])
	print(train_set()[1].shape)

if __name__ == '__main__':
        main()

