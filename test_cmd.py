#!/usr/bin/env python
# coding: utf-8
# wentixiaogege@gmail.com 20200324
import pandas as pd
import numpy as np
from multi_gpu_infer.cmd_based.multiprocess import multi_gpu_infer_on_dataset

## 配置 configure
num_gpus = 2

## load data and split into num of gpu peice 数据 && 分批次保存数据
df = pd.read_hdf('./multi_gpu_infer/data/test.hdf','1.0')
df['id'] = list(range(df.shape[0]))  # split for infer

# split based on num os gpus 分批次保存数据
subinds = np.array_split(range(df.shape[0]), int(num_gpus))
for i, j in enumerate(range(1, num_gpus + 1)):
    start = subinds[i][0]
    end = subinds[i][-1] + 1
    part = df[df.id >= start]
    part = df[(df.id >= start) & (df.id < end)]
    print('saving part ' + str(j), part.shape)
    part.to_hdf('./multi_gpu_infer/data/split.hdf', 'part' + str(j))  # 临时保存路径

### accessing gpu without sepcial cmds  如果不需要特殊命令就能够直接访问到GPU资源
## cluster 表明多个gpu和当前代码都运行在集群里面
binary='python -u'
output_dir='./multi_gpu_infer/cmd_based/output'
checkpoint='../../../chinese_L-12_H-768_A-12/'
result = multi_gpu_infer_on_dataset(binary,df,output_dir,checkpoint=checkpoint,num_gpus=num_gpus,mode='cluster')
print('checking the final result using multiprocessing ',result.head())

### need special cmds to accessing gpu 如果需要特殊的命令才能够访问到GPU资源
## just modify binary str  请修改binary命令的启动方式，便可~
## binary = 'XXXXXXX --gpu=1 python -u' is an example