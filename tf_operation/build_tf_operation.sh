#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o tf_operation.cu.o tf_operation.cu ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -D_MWAITXINTRIN_H_INCLUDED
g++ -std=c++11 -shared -o tf_operation.so tf_operation.cc tf_operation.cu.o ${TF_CFLAGS[@]} -fPIC -lcuda ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

cp tf_operation.so ..
