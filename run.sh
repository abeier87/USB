#!/bin/bash
# 设置可见的CUDA设备为0
export CUDA_VISIBLE_DEVICES=1
# 执行Python训练脚本及相关参数配置
python train.py --c config/classic_cv_imb/fixmatch_fp_disa/fixmatch_fp_disa_cifar10_lb1500_100_ulb3000_100_0.yaml