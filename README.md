# Fast and Scalable In-memory Deep Multitask Learning via Neural Weight Virtualization

## Introduction
This is an open-source repository of the paper titled "***Fast and Scalable In-memory Deep Multitask Learning via Neural Weight Virtualization***", which is submitted to [MobiSys 2020](https://www.sigmobile.org/mobisys/2020/). It enables fast and scalable in-memory multitask deep learning on memory-constrained embedded systems by (1) packing multiple deep neural networks (DNNs) into a fixed-sized main memory whose combined memory requirement is larger than the main memory, and (2) enabling fast in-memory execution of the DNNs. 

This repository implements (1) *virtualization of weight parameters* of multiple heterogeneous DNNs of arbitrary network architectures, and (2) *in-memory execution and context-switching* of deep neural network (DNN) tasks. For the reviewers' convenience, we provide a step-by-step guideline for the weight virtualization and in-memory execution of the five DNN that are used for the multitask learning IoT device, one of the application systems we implemented in the paper. The sizes of those DNNs are small, so the entire process of weight virtualization can be easily demonstrated in a reasonable time without requiring to spend several days. The five DNNs to be virtualized are [MNIST](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), [GoogleSpeechCommands (GSC)](https://arxiv.org/abs/1804.03209), [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf), [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), and [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf). It shows an example of weight virtualization, which can be easily applied to any other DNNs, i.e., they can be virtualized in the same way presented here.

&nbsp;
## Software Install and Setup
Neural weight virtualization is implemented by using Python, TensorFlow, and NVIDIA CUDA (custom TensorFlow operation). The TensorFlow version should be lower than or equal to 1.13.2; the latest version (1.14) seems to have a problem of executing custom operations. We used Python 2.7, Tensorflow 1.13.1, and CUDA 10.0. A GPU is required to perform the weight virtualization (i.e., weight-page matching and optimization) as well as in-memory execution of GPU RAM. We used NVIDIA RTX 20280 Ti GPU.

**Step 1.** Install [Python (>= 2.7)](https://www.python.org/downloads/).

**Step 2.** Install [Tensorflow (<= 1.13.2)](https://www.tensorflow.org/).

**Step 3.** Install [NVIDIA CUDA (>= 10.0)](https://developer.nvidia.com/cuda-downloads).

**Step 4.** Clone this NeuralWeightVirtualization repository.
```sh
$ git clone https://github.com/multitaskdnn-mobisys2020/NeuralWeightVirtualization.git
Cloning into 'NeuralWeightVirtualization'...
remote: Enumerating objects: 225, done.
remote: Counting objects: 100% (225/225), done.
remote: Compressing objects: 100% (178/178), done.
remote: Total 225 (delta 90), reused 164 (delta 42), pack-reused 0
Receiving objects: 100% (225/225), 11.81 MiB | 15.90 MiB/s, done.
Resolving deltas: 100% (90/90), done.
```

## 1) Download Datasets (Preliminary Step 1)
Before getting into the weight virtualization, download the five datasets by executing the following script (*download_dataset.sh*). The script uses [curl](https://curl.haxx.se/download.html) for downloading the datasets. 
```sh
$ ./download_dataset.sh 
[1/4] Downloading CIFAR10 dataset...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2604      0 --:--:-- --:--:-- --:--:--  2604
100  234M    0  234M    0     0  63.2M      0 --:--:--  0:00:03 --:--:-- 71.2M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    937      0 --:--:-- --:--:-- --:--:--   934
100  781k  100  781k    0     0  1260k      0 --:--:-- --:--:-- --:--:-- 1260k
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2675      0 --:--:-- --:--:-- --:--:--  2675
100 1171M    0 1171M    0     0  67.8M      0 --:--:--  0:00:17 --:--:-- 46.5M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    665      0 --:--:-- --:--:-- --:--:--   664
100 3906k    0 3906k    0     0  4706k      0 --:--:-- --:--:-- --:--:-- 4706k
...
...
...
[4/4] Downloading SVHN dataset...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2791      0 --:--:-- --:--:-- --:--:--  2791
100  305M    0  305M    0     0  79.1M      0 --:--:--  0:00:03 --:--:-- 90.4M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    885      0 --:--:-- --:--:-- --:--:--   885
100 1017k  100 1017k    0     0  1473k      0 --:--:-- --:--:-- --:--:-- 1473k
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2639      0 --:--:-- --:--:-- --:--:--  2621
100  772M    0  772M    0     0  84.7M      0 --:--:--  0:00:09 --:--:-- 93.4M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    958      0 --:--:-- --:--:-- --:--:--   955
100 2575k    0 2575k    0     0  3974k      0 --:--:-- --:--:-- --:--:-- 3974k
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    174      0 --:--:--  0:00:02 --:--:--   174
100 85.8M    0 85.8M    0     0  26.1M      0 --:--:--  0:00:03 --:--:-- 96.3M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2380      0 --:--:-- --:--:-- --:--:--  2380
100  286k  100  286k    0     0   788k      0 --:--:-- --:--:-- --:--:--  788k
```

## 2) Prepare and Train DNN Models (Preliminary Step 2)
The next preliminary step is to obtain and train DNN models for the five datasets. For the reviewers' convenience, we include pre-trained models of the five DNNs in this repository. They are located in separate folders.
```sh
$ ls -d mnist gsc gtsrb cifar10 svhn
cifar10  gsc  gtsrb  mnist  svhn
```

The number of weight parameters, memory usage, and inference accuracy of each DNN model are shown in the below table, which is the same as in the paper. In short, a total of ***268,692 weight parameters (1,046 KB)*** are virtualized into ***66,475 weights (258 KB)***, achieving ***4x*** of packing efficiency.

| DNN | Number of weights | Memory (KB) | Accuracy (%) |
| :-------------: | -------------: | -------------: | -------------: |
| MNIST  | 45,706 | 178 | 98.33 |
| GSC | 65,531 | 256 | 69.86 |
| GTSRB | 66,475 | 258 | 92.74 |
| CIFAR-10 | 45,490 | 176 | 55.68 |
| SVHN | 45,490 | 176 | 81.55 |
| **Total** | ***268,692*** | ***1,046*** | - |
| **Virtualized** | ***66,475*** | ***258*** | - |

## 3) Weight Virtualization Step 1: Weight-Page Matching
The first step of weight virtualization is the weight-page matching, which is performed by a Python script (*weight_virtualization.py*). It first computes Fisher information of the DNN and then performs weight-page matching as described in the paper. Each DNN performs the weight-page matching one by one.

Perform weight-page matching for the first DNN (MNIST) with the following Python script.
```sh
$ python weight_virtualization.py -mode=a -network_path=mnist
init new weight pages
add_vnn
mnist/mnist_network_weight.npy
compute_fisher
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
do_compute_fisher
sample num:    0, data_idx: 40422
sample num:    1, data_idx: 43444
sample num:    2, data_idx: 23402
...
...
...
sample num:   97, data_idx: 40313
sample num:   98, data_idx: 21500
sample num:   99, data_idx:  3595
mnist/mnist_network_fisher.npy
[calculate_cost]
toal_cost: 0.0
458 pages allocated for 45706 weights
total_network_cost: 0
```

Perform weight-page matching for the second (GSC) with the following Python script.
```sh
$ python weight_virtualization.py -mode=a -network_path=gsc
add_vnn
gsc/gsc_network_weight.npy
compute_fisher
do_compute_fisher
sample num:    0, data_idx: 19860
sample num:    1, data_idx: 51449
sample num:    2, data_idx: 30773
...
...
...
sample num:   97, data_idx: 41594
sample num:   98, data_idx: 30133
sample num:   99, data_idx: 44799
gsc/gsc_network_fisher.npy
[match_page_by_cost]
occupation: 0
len(page_list): 207
len(network_page_list): 656
       0-th page
     206-th page
cost: 0.0

occupation: 1
len(page_list): 458
len(network_page_list): 449
       0-th page
     448-th page
cost: 0.03946808

assing_page 146.488 ms
[calculate_cost]
toal_cost: 0.0394762507876294
656 pages allocated for 65531 weights
total_network_cost: 0.15915566682815552
```

Perform weight-page matching for the third (GTSRB) with the following Python script.
```sh
$ python weight_virtualization.py -mode=a -network_path=gtsrb
add_vnn
gtsrb/gtsrb_network_weight.npy
compute_fisher
do_compute_fisher
sample num:    0, data_idx: 23099
sample num:    1, data_idx: 22485
sample num:    2, data_idx: 15947
...
...
...
sample num:   97, data_idx:  6798
sample num:   98, data_idx:  9251
sample num:   99, data_idx: 18952
gtsrb/gtsrb_network_fisher.npy
[match_page_by_cost]
occupation: 0
len(page_list): 0
len(network_page_list): 665
cost: 0

occupation: 1
len(page_list): 216
len(network_page_list): 665
       0-th page
     215-th page
cost: 1.4526434

occupation: 2
len(page_list): 449
len(network_page_list): 449
       0-th page
     448-th page
cost: 0.047510564

assing_page 150.184 ms
[calculate_cost]
toal_cost: 1.5001573274303155
665 pages allocated for 66475 weights
total_network_cost: 6.379258215427399
```

Perform weight-page matching for the fourth (CIFAR-10) with the following Python script.
```sh
$ python weight_virtualization.py -mode=a -network_path=cifar10
add_vnn
cifar10/cifar10_network_weight.npy
compute_fisher
do_compute_fisher
sample num:    0, data_idx: 30796
sample num:    1, data_idx: 44166
sample num:    2, data_idx:  2649
...
...
...
sample num:   97, data_idx:  6889
sample num:   98, data_idx: 36036
sample num:   99, data_idx:  1621
cifar10/cifar10_network_fisher.npy
[match_page_by_cost]
occupation: 0
len(page_list): 0
len(network_page_list): 455
cost: 0

occupation: 1
len(page_list): 0
len(network_page_list): 455
cost: 0

occupation: 2
len(page_list): 216
len(network_page_list): 455
       0-th page
     215-th page
cost: 13.863106

occupation: 3
len(page_list): 449
len(network_page_list): 239
       0-th page
     238-th page
cost: 0.0028860972

assing_page 134.211 ms
[calculate_cost]
toal_cost: 13.865990731732381
455 pages allocated for 45490 weights
total_network_cost: 71.27165424823761
```

Perform weight-page matching for the fifth (SVHN) with the following Python script.
```sh
$ python weight_virtualization.py -mode=a -network_path=svhn
add_vnn
svhn/svhn_network_weight.npy
compute_fisher
do_compute_fisher
sample num:    0, data_idx: 51356
sample num:    1, data_idx: 47162
sample num:    2, data_idx:   624
...
...
...
sample num:   97, data_idx: 46074
sample num:   98, data_idx: 41740
sample num:   99, data_idx: 42296
svhn/svhn_network_fisher.npy
[match_page_by_cost]
occupation: 0
len(page_list): 0
len(network_page_list): 455
cost: 0

occupation: 1
len(page_list): 0
len(network_page_list): 455
cost: 0

occupation: 2
len(page_list): 0
len(network_page_list): 455
cost: 0

occupation: 3
len(page_list): 426
len(network_page_list): 455
       0-th page
     425-th page
cost: 5.48569

occupation: 4
len(page_list): 239
len(network_page_list): 29
       0-th page
      28-th page
cost: 0.0003839913

assing_page 143.431 ms
[calculate_cost]
toal_cost: 5.486162122021597
455 pages allocated for 45490 weights
total_network_cost: 114.62664997577667
```

## 4) Weight Virtualization Step 2: Weight-Page Optimization
The next step of weight virtualization is the weight-page optimization, which combines the matched weight-pages into single virtual weight-pages and optimizes them for the DNN tasks. Here, we perform joint optimization in which all the DNN tasks are optimized together by executing a shell script (*joint_optimization.sh*).
```sh
$ ./joint_optimization.sh 
1-th joint optimization
get_matching_loss
v_train
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
step 0, training accuracy: 0.100000 original loss: 6.907214 matching loss: 7.037911
step 0, Validation accuracy: 0.113500
step 100, training accuracy: 0.390000 original loss: 6.157534 matching loss: 3.197577
step 100, Validation accuracy: 0.484900
get new weight for 0.4849
step 200, training accuracy: 0.890000 original loss: 5.160514 matching loss: 1.595586
step 200, Validation accuracy: 0.861600
get new weight for 0.8616
...
...
...
step 1700, Validation accuracy: 0.839966
step 1800, training accuracy: 0.890000 original loss:: 4.937591 matching loss: 0.009319
step 1800, Validation accuracy: 0.840005
step 1900, training accuracy: 0.880000 original loss:: 5.124782 matching loss: 0.009399
step 1900, Validation accuracy: 0.846727
get new weight for 0.84672713
step 1999, training accuracy: 0.890000 original loss: 4.990521 matching loss 0.009340
step 1999, Validation accuracy: 0.842732
svhn/svhn_weight.npy
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Inference accuracy: 0.983100
Inference accuracy: 0.745388
Inference accuracy: 0.948931
Inference accuracy: 0.585900
Inference accuracy: 0.849647
```

After the optimization is finished, the virtual weight-pages are generated (*virtual_weight_page.npy*). The final inference accuracy of each DNN that is achieved with the virtual weight-pages can be checked with the following Python script (*weight_virtualization.py*).
```sh
$ python weight_virtualization.py -mode=e -vnn_name=mnist
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Inference accuracy: 0.983100
```
```sh
$ python weight_virtualization.py -mode=e -vnn_name=gsc
Inference accuracy: 0.745388
```
```sh
$ python weight_virtualization.py -mode=e -vnn_name=gtsrb
Inference accuracy: 0.948931
```
```sh
$ python weight_virtualization.py -mode=e -vnn_name=cifar10
Inference accuracy: 0.585900
```
```sh
python weight_virtualization.py -mode=e -vnn_name=svhn
Inference accuracy: 0.849647
```

The below table is the comparison of inference accuracy between the non-virtualized (stand-alone) DNNs and the virtualized DNNs.

| | Accuracy (%) | Accuracy (%) |
| :-------------: | -------------: | -------------: |
| DNN | Non-Virtualized DNN | ***Virtualized DNN*** |
| MNIST | 98.33 | ***98.31*** |
| GSC | 69.86 | ***74.53*** |
| GTSRB | 92.74 | ***94.89*** |
| CIFAR-10 | 55.68 | ***58.59*** |
| SVHN | 81.55 | ***84.96*** |

## 5) In-Memory Execution (ours) vs. No In-Memory Execution (previous)
Once the weight virtualization is completed, the virtual weight-pages that will be shared between the DNNs are generated (*virtual_weight_page.npy*) and loaded into the GPU RAM. By using the virtual weight pages, the DNNs are executed entirely in the GPU RAM, which enables fast and responsive execution of the DNNs.

We compare the DNN switching (weight parameter loading) and execution time of the in-memory multitask execution against the non-virtualized baseline DNNs that perform the DNN switching (restoring the saved weight parameters) and execution based on a secondary storage module (e.g., HDD or SSD) as done in many state-of-the-art DNNs today.

First, the in-memory execution is performed by the following Python script (i.e., *in-memory_execute.py*) that executes 30 random DNNs and measures the DNN switching and execution time. The result show that the total DNN switching time and execution time of the 30 DNN execution are ***3.919 ms*** and ***4443.045 ms***, respectively.

```sh
$ python in-memory_execute.py 
virtual_weight address: 140111287156736
init virtual_weight 87.536 ms
[VNN 0][cifar10] init page table 3.399 ms
[VNN 1][gsc] init page table 2.733 ms
[VNN 2][gtsrb] init page table 2.462 ms
[VNN 3][mnist] init page table 2.361 ms
[VNN 4][svhn] init page table 2.465 ms
tf.global_variables_initializer 696.639 ms
[Executing] svhn
weights load time : 0.204 ms
DNN execution time: 1350.998 ms
Inference accuracy: 0.849647
[Executing] gsc
weights load time : 0.165 ms
DNN execution time: 273.202 ms
Inference accuracy: 0.745388
...
...
...
[Executing] cifar10
weights load time : 0.123 ms
DNN execution time: 92.804 ms
Inference accuracy: 0.585900
[Executing] svhn
weights load time : 0.123 ms
DNN execution time: 178.231 ms
Inference accuracy: 0.849647
total weights load time : 3.919 ms
total DNN execution time: 4443.045 ms
```

Next, the no in-memory execution is performed by the following Python script (i.e., *baseline_execute.py*) that executes 30 random DNNs and measures the DNN switching and execution time. The result show that the total DNN switching time and execution time of the 30 DNN execution are ***1801.726 ms*** and ***4559.881 ms***, respectively.
```sh
python baseline_execute.py 
[Executing] cifar10
weights load time : 377.985 ms
DNN execution time: 968.260 ms
Inference accuracy: 0.555200
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
[Executing] mnist
weights load time : 45.943 ms
DNN execution time: 169.006 ms
Inference accuracy: 0.980800
...
...
...
[Executing] gtsrb
weights load time : 38.172 ms
DNN execution time: 74.077 ms
Inference accuracy: 0.928029
[Executing] gtsrb
weights load time : 37.190 ms
DNN execution time: 74.039 ms
Inference accuracy: 0.928029
[Executing] gsc
weights load time : 41.281 ms
DNN execution time: 54.813 ms
Inference accuracy: 0.694684
total weights load time : 1801.726 ms
total DNN execution time: 4559.881 ms
```

It shows that in-memory execution accelerates the DNN switching time by ***459x (1801.726 ms vs. 3.919 ms)***.

| | DNN Switching Time (ms) | DNN Execution Time (ms)
| :-------------: | -------------: | -------------: |
| No In-Memory Execution | 1801.72 | 4559.881 |
| ***In-Memory Execution*** | ***3.919*** | ***4443.04*** |




We hope this will help you review our paper. Please contact multitaskdnn.mobisys2020@gmail.com if you have any questions!
