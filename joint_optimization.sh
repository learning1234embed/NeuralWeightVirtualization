#!/bin/bash

for i in {1..20}
do
	echo "$i-th joint optimization"
	python weight_virtualization.py -mode=t -vnn_name=mnist -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=mnist
	python weight_virtualization.py -mode=e -vnn_name=gsc
	python weight_virtualization.py -mode=e -vnn_name=gtsrb
	python weight_virtualization.py -mode=e -vnn_name=cifar10
	python weight_virtualization.py -mode=e -vnn_name=svhn

	python weight_virtualization.py -mode=t -vnn_name=gsc -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=mnist
	python weight_virtualization.py -mode=e -vnn_name=gsc
	python weight_virtualization.py -mode=e -vnn_name=gtsrb
	python weight_virtualization.py -mode=e -vnn_name=cifar10
	python weight_virtualization.py -mode=e -vnn_name=svhn

	python weight_virtualization.py -mode=t -vnn_name=gtsrb -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=mnist
	python weight_virtualization.py -mode=e -vnn_name=gsc
	python weight_virtualization.py -mode=e -vnn_name=gtsrb
	python weight_virtualization.py -mode=e -vnn_name=cifar10
	python weight_virtualization.py -mode=e -vnn_name=svhn

	python weight_virtualization.py -mode=t -vnn_name=cifar10 -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=mnist
	python weight_virtualization.py -mode=e -vnn_name=gsc
	python weight_virtualization.py -mode=e -vnn_name=gtsrb
	python weight_virtualization.py -mode=e -vnn_name=cifar10
	python weight_virtualization.py -mode=e -vnn_name=svhn

	python weight_virtualization.py -mode=t -vnn_name=svhn -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=mnist
	python weight_virtualization.py -mode=e -vnn_name=gsc
	python weight_virtualization.py -mode=e -vnn_name=gtsrb
	python weight_virtualization.py -mode=e -vnn_name=cifar10
	python weight_virtualization.py -mode=e -vnn_name=svhn
done
