## Please start if helpful

## MULTI_GPU_INFER

Using Multipule GPU to do infer for accelerate !!!!

It contains the following modules:

1. Queue-Based multiprocessing model for inferring
2. Subprocess model for commandline running:
    - 2.1 running locally, every time fetch one gpu, this way is better for fire command to the cluster 
    - 2.2 running cluster, one time gather all gpus you can ,then running the basci command;
    
使用GPU infer 加速！！！，本repo实现了两种方式：
1. 基于队列的multiprocessing 模块 
2. 基于命令的subprocess 模型 
    - 2.1 本地模式，gpu本地不可访问，需要特殊的命令申请gpu资源 
    - 2.2 集群模型，gpu本地可以访问，意思就是你在集群中执行了；


## Requirements

* multiprocessing
* subprocess
* python3.7
* keras_bert
* argparse
* pandas
* chinese_L-12_H-768_A-12


### 本文以keras_bert 提供的extract_embeddings 方式来讲解如何使用；
1. download chinese_L-12_H-768_A-12 at [checkpoint] path
2. install all necessray packaegs

#### 1. queue-based multiprocessing 使用基于Queue的多进程GPU预测
* python queue.py

* 运行日志:
```javascript
model init done 1
WARNING:tensorflow:From /data/anaconda3/envs/py37/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

model init done 0
WARNING:tensorflow:From /data/anaconda3/envs/py37/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

2020-03-24 16:10:14.508411: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
woker running 1 0
2020-03-24 16:10:14.956205: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
woker running 0 1
woker running 1 2
woker running 0 3
woker running 1 4
woker running 0 5
woker running 1 6
woker running 0 7
woker running 1 8
worker predict done at gpu: 1
woker running 0 9
worker predict done at gpu: 0
all of workers have been done
checking the final result using multiprocessing     worker  id             content                                         embeddings
0       1   0              [味道俱佳]  [[0.3480394, 1.1268631, -1.416797, 0.5583812, ...
1       0   1   [价格实惠, 便宜, 活动很实惠]  [[-0.40090305, 0.5747566, -0.5340656, 0.219112...
2       1   2              [方便携带]  [[-0.7382355, -0.039685342, -0.8000674, 0.6148...
3       0   3              [包装很好]  [[-0.06756385, 0.60894394, -0.41646048, 0.2603...
4       1   4  [口感俱佳, 风味十足, 酸甜可口]  [[-0.15814807, 0.51882, -1.1301757, 0.13116539...
```


#### 2.cmd-based subprocessing 使用基于CMD的多进程GPU预测
* python test_cmd.py

* 运行日志
```javascript
saving part 1 (500, 2)
/data/anaconda3/envs/py37/lib/python3.7/site-packages/pandas/core/generic.py:2530: PerformanceWarning: 
your performance may suffer as PyTables will pickle object types that it cannot
map directly to c-types [inferred_type->mixed,key->block1_values] [items->['content']]

  pytables.to_hdf(path_or_buf, key, self, **kwargs)
saving part 2 (500, 2)
ljj checking gpu_inds== range(0, 2) [0, 1]
ljj checking cmd== python -u multi_gpu_infer/cmd_based/split.py --checkpoint ../../../chinese_L-12_H-768_A-12/ --split_part 1 --start 0 --end 500
ljj checking cmd== python -u multi_gpu_infer/cmd_based/split.py --checkpoint ../../../chinese_L-12_H-768_A-12/ --split_part 2 --start 500 --end 1000
b'Using TensorFlow backend.'
b'=================>loading source data'
b'part shape is  (30, 2)'
b'=================>loading source data done!'
b'=================>generating emebddings!'
b'2020-03-24 15:08:30.254344: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0'
b'2020-03-24 15:08:30.254361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N'
b'2020-03-24 15:08:30.258828: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10312 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:3f:00.0, compute capability: 7.5)'

b'WARNING:tensorflow:From /data/anaconda3/envs/py37/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.'
b''
b'2020-03-24 15:09:07.669883: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0'
b'=================>generating emebddings done!'
b'dumping data shape  (30, 3)'
Using TensorFlow backend.
=================>loading source data
part shape is  (30, 2)
=================>loading source data done!
=================>generating emebddings!
WARNING:tensorflow:From /home/lijingjie/.local/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
2020-03-24 15:08:30.245814: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-03-24 15:08:30.249668: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-24 15:08:30.249718: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-03-24 15:08:30.249730: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-03-24 15:08:30.254373: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10312 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:40:00.0, compute capability: 7.5)
WARNING:tensorflow:From /data/anaconda3/envs/py37/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

2020-03-24 15:09:08.193195: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
=================>generating emebddings done!
dumping data shape  (30, 3)

checking the final result using multiprocessing     id             content                                         embeddings
0   0              [味道俱佳]  [[0.3480388, 1.1268635, -1.4167957, 0.558382, ...
1   1   [价格实惠, 便宜, 活动很实惠]  [[-0.40090328, 0.57475597, -0.5340664, 0.21911...
2   2              [方便携带]  [[-0.73823625, -0.03968448, -0.80006707, 0.614...
3   3              [包装很好]  [[-0.06756356, 0.6089444, -0.41645902, 0.26038...
4   4  [口感俱佳, 风味十足, 酸甜可口]  [[-0.15814593, 0.51881766, -1.1301773, 0.13116...
```
