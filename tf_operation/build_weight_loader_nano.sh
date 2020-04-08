#!/bin/bash

nvcc -c -o weight_loader.cu.o weight_loader.cu -x cu -Xcompiler -fPIC
gcc -shared -o weight_loader.so weight_loader.c weight_loader.cu.o -fPIC -L/usr/local/cuda/lib64 -lcuda -lcudart -O2 -lstdc++

cp weight_loader.so ..
