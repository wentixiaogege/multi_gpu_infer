#!/usr/bin/env python
# coding: utf-8
# wentixiaogege@gmail.com 20200324
import os
import numpy as np
import pandas as pd
import subprocess
import pickle
import io
import logging
logger = logging.getLogger(__name__)
import argparse

def process_in_parallel(
    tag, df, binary, output_dir, checkpoint='../../chinese_L-12_H-768_A-12/',num_gpus=2,mode='local'
):
    """Run the specified binary cfg.NUM_GPUS times in parallel, each time as a
    subprocess that uses one GPU. The binary must accept the command line
    arguments `--range {start} {end}` that specify a data processing range.
    """
    # Snapshot the current cfg state in order to pass to the inference
    # subprocesses
    subprocess_env = os.environ.copy()
    processes = []
    subinds = np.array_split(range(df.shape[0]), int(num_gpus))
    # Determine GPUs to use
#     cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
#     if cuda_visible_devices:
#         gpu_inds = map(int, cuda_visible_devices.split(','))
#         assert -1 not in gpu_inds, \
#             'Hiding GPU indices using the \'-1\' index is not supported'
#     else:
#         gpu_inds = range(num_gpus)
    gpu_inds = range(num_gpus)
    # Run the binary in cfg.NUM_GPUS subprocesses
    print('ljj checking gpu_inds==',gpu_inds,list(gpu_inds))
    for i, gpu_ind in enumerate(gpu_inds):
        
        start = subinds[i][0]
        end = subinds[i][-1] + 1
        if mode =='cluster':
            subprocess_env['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)
        cmd = '{binary} multi_gpu_infer/cmd_based/split.py --checkpoint {checkpoint} --split_part {part} --start {start} --end {end}'
        cmd = cmd.format(
            binary=binary,checkpoint=(checkpoint),part=str(i+1),start=start,end=end
        )
        logger.info('{} range command {}: {}'.format(tag, str(i+1), cmd))
        print('ljj checking cmd==',cmd)
        if i == 0:
            subprocess_stdout = subprocess.PIPE
        else:
            filename = os.path.join(
                output_dir, '%s_range_%s_%s.stdout' % (tag, start, end)
            )
            subprocess_stdout = open(filename, 'w')  # NOQA (close below)
            
        p = subprocess.Popen(
            cmd,
            shell=True,
            env=subprocess_env, # 指定可以用的gpuid
            stdout=subprocess_stdout,
            stderr=subprocess.STDOUT,
            bufsize=1
        )
        processes.append((i, p, start, end, subprocess_stdout))
    # Log output from inference processes and collate their results
    outputs = []
    for i, p, start, end, subprocess_stdout in processes:
        log_subprocess_output(i, p, output_dir, tag, start, end)
        if isinstance(subprocess_stdout, io.IOBase):  # NOQA (Python 2 for now)
            subprocess_stdout.close()
        range_file = os.path.join(
            output_dir, 'multi_gpu_ljj_range_%s_%s.pickle' % (start,end)
        )
        range_data = pickle.load(open(range_file,'rb'))
        outputs.append(range_data)
    return outputs


def log_subprocess_output(i, p, output_dir, tag, start, end):
    """Capture the output of each subprocess and log it in the parent process.
    The first subprocess's output is logged in realtime. The output from the
    other subprocesses is buffered and then printed all at once (in order) when
    subprocesses finish.
    """
    outfile = os.path.join(
        output_dir, '%s_range_%s_%s.stdout' % (tag, start, end)
    )
    logger.info('# ' + '-' * 76 + ' #')
    logger.info(
        'stdout of subprocess %s with range [%s, %s]' % (i, start + 1, end)
    )
    logger.info('# ' + '-' * 76 + ' #')
    if i == 0:
        # Stream the piped stdout from the first subprocess in realtime
        with open(outfile, 'w') as f:
            for line in iter(p.stdout.readline, b''):
                print(line.rstrip())
                f.write(str(line))
        p.stdout.close()
        ret = p.wait()
    else:
        # For subprocesses >= 1, wait and dump their log file
        ret = p.wait()
        with open(outfile, 'r') as f:
            print(''.join(f.readlines()))
    assert ret == 0, 'Range subprocess failed (exit code: {})'.format(ret)

def multi_gpu_infer_on_dataset(
   binary,df, output_dir,checkpoint='../../chinese_L-12_H-768_A-12/',num_gpus=2,mode='local'
):
    """Multi-gpu inference on a dataframe."""
    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = process_in_parallel(
        'multi_gpu_ljj_', df, binary, output_dir, checkpoint,num_gpus,mode
    )

    # Collate the results from each subprocess
    logger.info('Wrote predict to: {}'.format(os.path.abspath('./')))
    df_predict = pd.concat(outputs)

    return df_predict


# if __name__ == "__main__":
#     ## 参数
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_gpus",  type=int, help="how many gpus to use" )
#     args = parser.parse_args()
#     ##查看gpu
#     print('目前可用的gpus个数：',args.num_gpus)
#     ## load 数据 && 分批次保存数据
#     df = pd.read_hdf('../data/test.hdf','1.0')
#     df['id'] = list(range(df.shape[0])) # split for infer
    
#     #根据gpu个数自动化分割
#     subinds = np.array_split(range(df.shape[0]), int(args.num_gpus))        
#     for i,j in enumerate(range(1,args.num_gpus+1)):
#         start = subinds[i][0]
#         end = subinds[i][-1] + 1
#         part = df[df.id >=start]
#         part = df[(df.id >=start)&(df.id <end)]
#         print('saving part '+str(j),part.shape)
#         part.to_hdf('./final_merged_normal_df_df_tags.hdf','part'+str(j))
    
#     ## 本地模式：每次启动都要rlaunch
#     binary='rlaunch -v --cpu=8 --gpu=1 --memory=150000 -- python -u'
#     output_dir='./output'    
#     result = multi_gpu_infer_on_dataset(binary,df,output_dir,num_gpus=args.num_gpus,mode='local')
#     pickle.dump(result,open('./output/result_for_ljj_to_check.pkl','wb'))
    
    ## 集群模式：rlaunch 申请很大的资源
    #     binary='python -u'
    #     output_dir='./'
    #     result = multi_gpu_infer_on_dataset(binary,df,output_dir,num_gpus=args.num_gpus,mode='cluster')
    #     pickle.dump(result,open('result_for_ljj_to_check.pkl','wb'))