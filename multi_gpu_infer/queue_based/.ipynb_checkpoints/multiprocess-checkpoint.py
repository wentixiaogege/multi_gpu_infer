#!/usr/bin/env python
# coding: utf-8
# wentixiaogege@gmail.com 20200324
import pandas as pd
import argparse
from multiprocessing import Manager,Queue
from multi_gpu_infer.queue_based.keras_extract_embedding_worker import KerasExtractEmbeddingWorker

class Scheduler:
    def __init__(self, gpuids,infer_type='embedding',checkpoint=''):
        self._gpuids = gpuids
        self._queue = Queue()
        self._infer_type = infer_type
        self.checkpoint = checkpoint
        self.return_list = Manager().list()
        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            if self._infer_type == 'embedding':
                self._workers.append(KerasExtractEmbeddingWorker(self.checkpoint,gpuid, self._queue,self.return_list))
    
    def start(self, xdatalst):

        # put all of data into queue
        for xdata in xdatalst:
            self._queue.put(xdata)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print("all of workers have been done")
        
        return self.return_list

                    
def run_extract_embeddings(df=None, gpuids=[0,1],checkpoint='../../../chinese_L-12_H-768_A-12/'):
    #scan all files
    xdatalist = list()
    for xdata in df.iterrows():
#         print('ljj checking',xdata)
        xdatalist.append((xdata[1]['id'],xdata[1]['content']))

    #init scheduler
    x = Scheduler(gpuids,infer_type='embedding',checkpoint=checkpoint)
    #start processing and wait for complete 
    result = x.start(xdatalist)
    result = pd.DataFrame(list(result))
    return result

# if __name__ == "__main__":
#     ## 参数
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gpuids",  type=str, help="gpu ids to run" )
#     args = parser.parse_args()
#     ##查看gpu
#     gpuids = [int(x) for x in args.gpuids.strip().split(',')]
#     print('目前可用的gpus:',gpuids)
#     df = pd.read_hdf('../data/test.hdf','1.0')
#     run_extract_embeddings(df, gpuids)
