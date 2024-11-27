'''
1张显卡同时跑3个seed: 0, 1, 2
'''

import subprocess
import os



# 配置文件列表，替换为你实际的配置文件名
config_root_path = "config/classic_cv_imb/fixmatch_cossl"
config_file = "fixmatch_cossl_cifar10_lb1500_100_ulb3000_100"
gpu_id = "3"

for seed in range(3):
    config_path = os.path.join(config_root_path, config_file + '_' + str(seed) + '.yaml')
    full_command = f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py --c + {config_path}"
    try:
        # 使用subprocess模块来执行命令，这里将在当前的Python环境下执行对应的命令行指令
        subprocess.run(full_command, shell=True, check=True)
        print(f"成功执行命令: {full_command}")
    except subprocess.CalledProcessError as e:
        print(f"执行命令 {full_command} 时出错: {e}")