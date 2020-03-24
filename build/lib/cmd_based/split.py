#!/usr/bin/env python
# coding: utf-8
# wentixiaogege@gmail.com 20200324
import argparse
import pandas as pd
import pickle
from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX
import pandas as pd
from itertools import chain

def main(args):
    # pipeline flow
    print('=================>loading source data')
    df_tags = pd.read_hdf('../data/split.hdf','part'+str(args.split_part))
    df_tags = df_tags.head(10)
    print('part shape is ',df_tags.shape)#
    print('=================>loading source data done!')
    text = df_tags.label_content.values
    all_text=[]
    all_len=[]
    for i in text:
        all_text.append(i)
        all_len.append(len(i))
    all_text_one = list(chain.from_iterable(all_text))  

    print('=================>generating emebddings!')
    embeddings = extract_embeddings(args.checkpoint, all_text_one,output_layer_num=1, poolings=[POOL_NSP, POOL_MAX])
    print('=================>generating emebddings done!')
    final_emb = []
    before=0
    for i in range(df_tags.shape[0]):
        final_emb.append(embeddings[before:before+all_len[i]])
        before += all_len[i]

    df_tags['embeddings'] = final_emb
    print('dumping data shape ', df_tags.shape)
    pickle.dump(df_tags,open('./output/multi_gpu_ljj_range_'+str(args.start)+'_'+str(args.end)+'.pickle','wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="which checkpoint to choose")
    parser.add_argument("--split_part", type=str, help="which split to choose")
    parser.add_argument("--start", type=str, help="split_part start to choose")
    parser.add_argument("--end", type=str, help="split_part end to choose")
        
    args = parser.parse_args()
    main(args)
