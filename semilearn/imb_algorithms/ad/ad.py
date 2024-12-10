# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument
from .utils import SinkhornDistance
from .utils import ETFArch

class ADNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features

        # auxiliary classifier
        self.aux_classifier = nn.Linear(self.backbone.num_features, num_classes)
        self.bn = nn.BatchNorm1d(self.num_features)
    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_aux'] = self.aux_classifier(results_dict['feat'])
        results_dict['feat_bn'] = self.bn(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher



@IMB_ALGORITHMS.register('ad')
class AD(ImbAlgorithmBase):
    """
        (A)dvanced ABC + (D)isa algorithm.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - abc_p_cutoff (`float`):
                threshold for the auxilariy classifier
            - abc_loss_ratio (`float`):
                loss ration for auxiliary classifier
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(abc_p_cutoff=args.abc_p_cutoff, abc_loss_ratio=args.abc_loss_ratio, include_W=args.include_W, include_u_disa=args.include_u_disa)

        super(AD, self).__init__(args, net_builder, tb_log, logger, **kwargs)

        # compute lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.ulb_class_dist = torch.from_numpy(lb_class_dist / np.sum(lb_class_dist)) # 将无标注数据分布初始化为和有标注数据一样的分布
        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist) # 值越小说明对应类别的数量越大
        
        
        # TODO: better ways
        self.model = ADNet(self.model, num_classes=self.num_classes)
        self.ema_model = ADNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()
        self.sinkhorna_ot = SinkhornDistance()
        self.etfarch = ETFArch(num_features=self.model.num_features, num_classes=self.num_classes).ori_M.T
        self.ot_loss_ratio = args.ot_loss_ratio


    def imb_init(self, abc_p_cutoff=0.95, abc_loss_ratio=1.0, include_W=False, include_u_disa=True):
        self.abc_p_cutoff = abc_p_cutoff
        self.abc_loss_ratio = abc_loss_ratio
        self.include_W = include_W
        self.include_u_disa = include_u_disa

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train_step(self, *args, **kwargs):

        out_dict, log_dict = super().train_step(*args, **kwargs)

        # get features
        feats_x_lb = out_dict['feat']['x_lb']
        feats_x_ulb_w = out_dict['feat']['x_ulb_w']
        feats_x_ulb_s = out_dict['feat']['x_ulb_s']
        if isinstance(feats_x_ulb_s, list):
            feats_x_ulb_s = feats_x_ulb_s[0]
        
        # get logits
        logits_x_lb = self.model.module.aux_classifier(feats_x_lb)
        logits_x_ulb_s = self.model.module.aux_classifier(feats_x_ulb_s)
        with torch.no_grad():
            logits_x_ulb_w = self.model.module.aux_classifier(feats_x_ulb_w)

        # compute abc loss using logits_aux from dict
        abc_loss = self.compute_abc_loss(
            logits_x_lb=logits_x_lb, 
            y_lb=kwargs['y_lb'],
            logits_x_ulb_w=logits_x_ulb_w,
            logits_x_ulb_s=logits_x_ulb_s
            )
        out_dict['loss'] += self.abc_loss_ratio * abc_loss
        log_dict['train/abc_loss'] = abc_loss.item()
        
        # compute ot loss
        if self.include_u_disa:
            feats = torch.cat((feats_x_lb, feats_x_ulb_s))
        else:
            feats = feats_x_lb
        bn_lb_ulb = self.model.module.bn(feats)
        ot_loss = self.compute_ot_loss(
            logits_x_lb=bn_lb_ulb,
            etfarch=self.etfarch
            )
        out_dict['loss'] += self.ot_loss_ratio * ot_loss 
        log_dict['train/ot_loss'] = ot_loss.item()
        
        return out_dict, log_dict
    
    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        return super().evaluate(eval_dest=eval_dest, out_key='logits_aux', return_logits=return_logits)

    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()
    
    def compute_abc_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]
        
        if not self.lb_class_dist.is_cuda:
            self.lb_class_dist = self.lb_class_dist.to(y_lb.device)
        
        if not self.ulb_class_dist.is_cuda:
            self.ulb_class_dist = self.ulb_class_dist.to(y_lb.device)

        # compute labeled abc loss
        mask_lb = self.bernouli_mask(self.lb_class_dist[y_lb])
        abc_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none') * mask_lb).mean()

        W = self.model.module.aux_classifier.weight
        
        # compute unlabeled abc loss
        with torch.no_grad():
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            
            self.ulb_class_dist = 0.99 * self.ulb_class_dist + 0.01 * probs_x_ulb_w.mean(0)
            
            if self.include_W:
                ulb_class_dist1 = torch.min(self.ulb_class_dist) / self.ulb_class_dist # 根据伪标签估算无标注数据的类别分布
                if self.epoch > 0.75 * self.epochs:
                    ulb_class_dist2 = self.calculate_p(W) # 根据分类器中对应不同类别的权重向量角度大小比例估算类别分布
                    ulb_class_dist = 1 - (self.epoch / self.epochs) * ulb_class_dist1 * ulb_class_dist2
                else:
                    ulb_class_dist = 1 - (self.epoch / self.epochs) * ulb_class_dist1
            else:
                ulb_class_dist = torch.min(self.ulb_class_dist) / self.ulb_class_dist # 根据伪标签估算无标注数据的类别分布
            
            mask_ulb_1 = self.bernouli_mask(ulb_class_dist[y_ulb])
            mask_ulb_2 = max_probs.ge(self.abc_p_cutoff).to(logits_x_ulb_w.dtype)
            mask_ulb = mask_ulb_1 * mask_ulb_2
    
        abc_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            abc_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none') * mask_ulb).mean()
        
        abc_loss = abc_lb_loss + abc_ulb_loss
        return abc_loss

    def calculate_p(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        根据分类器中对应不同类别的权重向量角度大小比例，为了估算类别数量比例
        """
        # 计算向量的模长，shape为(K, 1)
        norms = torch.norm(matrix, dim=1, keepdim=True)
        # 对矩阵进行归一化，使得每一行向量都变成单位向量
        normalized_matrix = matrix / norms
        # 通过矩阵乘法计算两两向量的点积，得到K*K的矩阵，其元素(i, j)表示第i个向量和第j个向量的点积
        dot_product_matrix = torch.mm(normalized_matrix, normalized_matrix.T)
        # 为了避免数值计算误差导致余弦值超出[-1, 1]范围，进行裁剪
        clipped_dot_product_matrix = torch.clamp(dot_product_matrix, -1, 1)
        # 通过反余弦函数（arccos）将点积（也就是余弦值）转换为角度值（单位为弧度）
        angle_matrix = torch.acos(clipped_dot_product_matrix)
        # 获取向量的数量K
        K = angle_matrix.shape[0]
        # 创建数组用于存储每个向量的平均夹角值
        average_angles = torch.zeros(K, device=matrix.device)
        for i in range(K):
            # 排除与自身的夹角（值为0，因为向量与自身夹角为0弧度），计算其余夹角的平均值
            average_angles[i] = torch.mean(angle_matrix[i, torch.arange(K)!= i])
        
        total_sum = torch.sum(average_angles)
        proportions = average_angles / total_sum
        return proportions

    def compute_ot_loss(self, logits_x_lb, etfarch):
        logits_x_lb = logits_x_lb / torch.clamp(
            torch.sqrt(torch.sum(logits_x_lb ** 2, dim=1, keepdims=True)), 1e-8).cuda()
        ot_loss = self.sinkhorna_ot(logits_x_lb, etfarch)
        return ot_loss
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--abc_p_cutoff', float, 0.95),
            SSL_Argument('--abc_loss_ratio', float, 1.0),
            SSL_Argument('--ot_loss_ratio', float, 0.1),
            SSL_Argument('--include_W', bool, False),
            SSL_Argument('--include_u_disa', bool, True),
        ]