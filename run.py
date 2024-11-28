'''
1张显卡同时跑3个seed: 0, 1, 2
'''

import subprocess
import os
import threading


# 配置文件列表，替换为你实际的配置文件名
config_root_path = "config/classic_cv_imb/fixmatch_debiaspl"
config_file = "fixmatch_debiaspl_cifar10_lb1500_100_ulb3000_100"
gpu_id = "2"

def run_command(full_command):
    try:
        subprocess.run(full_command, shell=True)
        print(f"成功执行命令: {full_command}")
    except subprocess.CalledProcessError as e:
        print(f"执行命令 {full_command} 时出错: {e}")

full_commands = []
for seed in range(3):
    config_path = os.path.join(config_root_path, config_file + '_' + str(seed) + '.yaml')
    full_commands.append(f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py --c {config_path}")



threads = []
for command in full_commands:
    thread = threading.Thread(target=run_command, args=(command,))
    thread.start()
    threads.append(thread)