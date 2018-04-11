#!/bin/bash 

python3 setup.py build_ext --inplace
rm -rf build/
mv ./nms/cpu_nms*.so .
mv ./nms/gpu_nms*.so .
