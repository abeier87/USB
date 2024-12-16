# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument
from .utils import SinkhornDistance

class RRCNet(nn.Module):
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
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher



@IMB_ALGORITHMS.register('rrc')
class RRC(ImbAlgorithmBase):
    """
        (R)ebalancing (R)epresentation and (C)lassifier.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - rrc_p_cutoff (`float`):
                threshold for the auxilariy classifier
            - rrc_loss_ratio (`float`):
                loss ration for auxiliary classifier
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(rrc_p_cutoff=args.rrc_p_cutoff, rrc_loss_ratio=args.rrc_loss_ratio, include_u_da=args.include_u_da)

        super(RRC, self).__init__(args, net_builder, tb_log, logger, **kwargs)

        # compute lb and ulb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.ulb_class_dist = torch.from_numpy(lb_class_dist / np.sum(lb_class_dist)) # 将无标注数据分布初始化为和有标注数据一样的分布
        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist) # 值越小说明对应类别的数量越大
        
        self.model = RRCNet(self.model, num_classes=self.num_classes)
        self.ema_model = RRCNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()
        self.sinkhorna_ot = SinkhornDistance()
        self.ot_loss_ratio = args.ot_loss_ratio


    def imb_init(self, rrc_p_cutoff=0.95, rrc_loss_ratio=1.0, include_u_da=True):
        self.rrc_p_cutoff = rrc_p_cutoff
        self.rrc_loss_ratio = rrc_loss_ratio
        self.include_u_da = include_u_da


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
            # 用于更新估计的无标注数据分布
            mean_feats_x_ulb_w = feats_x_ulb_w.mean(dim=0)
            mean_logits_feats_x_ulb_w = self.model.module.aux_classifier(mean_feats_x_ulb_w)
            mean_probs_x_ulb_w = self.compute_prob(mean_logits_feats_x_ulb_w)
            
            # 用于计算伪标签置信度超过0.95的对应的x_ulb_s的ot-loss
            logits_x_ulb_w = self.model.module.aux_classifier(feats_x_ulb_w)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            mask = max_probs >= self.rrc_p_cutoff
            W = self.model.module.aux_classifier.weight

        # compute rrc loss using logits_aux from dict
        rrc_loss = self.compute_rrc_loss(
            logits_x_lb=logits_x_lb,
            y_lb=kwargs['y_lb'],
            logits_x_ulb_w=logits_x_ulb_w,
            logits_x_ulb_s=logits_x_ulb_s,
            mean_probs_x_ulb_w=mean_probs_x_ulb_w,
            max_probs=max_probs,
            y_ulb=y_ulb
            )

        out_dict['loss'] += self.rrc_loss_ratio * rrc_loss
        log_dict['train/rrc_loss'] = rrc_loss.item()
        
        # compute ot loss
        if self.include_u_da:
            feats = torch.cat((feats_x_lb, feats_x_ulb_s[mask]))
        else:
            feats = feats_x_lb
        bn_lb_ulb = self.model.module.bn(feats)
        ot_loss = self.compute_ot_loss(
            logits_x_lb=bn_lb_ulb,
            etfarch=W
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
    
    def compute_rrc_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s, mean_probs_x_ulb_w, max_probs, y_ulb):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]
        
        if not self.lb_class_dist.is_cuda:
            self.lb_class_dist = self.lb_class_dist.to(y_lb.device)
        
        if not self.ulb_class_dist.is_cuda:
            self.ulb_class_dist = self.ulb_class_dist.to(y_lb.device)
        
        
        # compute labeled rrc loss
        mask_lb = self.bernouli_mask(self.lb_class_dist[y_lb])
        rrc_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none') * mask_lb).mean()

        
        # compute unlabeled abc loss
        with torch.no_grad():
            self.ulb_class_dist = 0.99 * self.ulb_class_dist + 0.01 * mean_probs_x_ulb_w
            ulb_class_dist = torch.min(self.ulb_class_dist) / self.ulb_class_dist # 根据伪标签估算无标注数据的类别分布

            mask_ulb = self.bernouli_mask(ulb_class_dist[y_ulb])
            mask_ulb_2 = max_probs.ge(self.rrc_p_cutoff).to(logits_x_ulb_w.dtype)
            mask_ulb = mask_ulb * mask_ulb_2

        rrc_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            rrc_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none') * mask_ulb).mean()
        
        rrc_loss = rrc_lb_loss + rrc_ulb_loss
        return rrc_loss

    def compute_ot_loss(self, logits_x_lb, etfarch):
        logits_x_lb = logits_x_lb / torch.clamp(
            torch.sqrt(torch.sum(logits_x_lb ** 2, dim=1, keepdims=True)), 1e-8).cuda()
        ot_loss = self.sinkhorna_ot(logits_x_lb, etfarch)
        return ot_loss
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--rrc_p_cutoff', float, 0.95),
            SSL_Argument('--rrc_loss_ratio', float, 1.0),
            SSL_Argument('--ot_loss_ratio', float, 0.1),
            SSL_Argument('--include_u_da', bool, True),
        ]