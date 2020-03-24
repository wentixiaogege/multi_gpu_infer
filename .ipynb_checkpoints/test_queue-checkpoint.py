#!/usr/bin/env python
# coding: utf-8
# wentixiaogege@gmail.com 20200324
import argparse
import pandas as pd
import os
print(os.getcwd())

from multi_gpu_infer.queue_based.multiprocess import run_extract_embeddings
print(os.getcwd())
import pickle


##loading data
df = pd.read_hdf('./multi_gpu_infer/data/test.hdf','1.0')
df = df.head(10)
##using two gpus to accelerate!!
checkpoint='../../../chinese_L-12_H-768_A-12/'
gpuids=[0,1]
result = run_extract_embeddings(df,gpuids,checkpoint)

print('checking the final result using multiprocessing ',result.head())
