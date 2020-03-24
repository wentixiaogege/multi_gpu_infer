#!/usr/bin/env python
# coding: utf-8
# wentixiaogege@gmail.com 20200324
from multiprocessing import  Process
from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX,load_trained_model_from_checkpoint,get_checkpoint_paths,load_vocabulary
import os

class KerasExtractEmbeddingWorker(Process):
    def __init__(self,bert_checkpoint,gpuid, queue,return_list):
        Process.__init__(self, name='ModelProcessor')
        self._bert_checkpoint = bert_checkpoint
        self._gpuid = gpuid
        self._queue = queue
        self.return_list = return_list  # for data fetch

    def run(self):
        #set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)

        #load models
        #every worker only need to load model one time
        paths = get_checkpoint_paths(self._bert_checkpoint)
        model = load_trained_model_from_checkpoint(
                config_file=paths.config,
                checkpoint_file=paths.checkpoint,
                output_layer_num=1,
            )
        vocabs = load_vocabulary(paths.vocab)
        print('model init done', self._gpuid)

        while True:
            xfile = self._queue.get()
            if xfile == None:
                self._queue.put(None)
                break
            embeddings = extract_embeddings(model=model,vocabs=vocabs,texts=xfile[1],output_layer_num=1,poolings=[POOL_NSP, POOL_MAX])
            print('woker running',self._gpuid,len(self.return_list))
            self.return_list.append({'worker':self._gpuid,'id':xfile[0],'content':xfile[1],'embeddings':embeddings})
    
        print('worker predict done at gpu:', self._gpuid)

