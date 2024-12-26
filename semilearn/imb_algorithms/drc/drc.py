# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument

class DRCNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        # auxiliary classifier
        self.aux_classifier = nn.Linear(self.backbone.num_features, num_classes)


    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_aux'] = self.aux_classifier(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher



@IMB_ALGORITHMS.register('drc')
class DRC(ImbAlgorithmBase):
    """
        (D)ecoupling (R)epresentation and (C)lassifier.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - drc_p_cutoff (`float`):
                threshold for the auxilariy classifier
            - drc_loss_ratio (`float`):
                loss ration for auxiliary classifier
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(drc_p_cutoff=args.drc_p_cutoff, drc_loss_ratio=args.drc_loss_ratio)

        super(DRC, self).__init__(args, net_builder, tb_log, logger, **kwargs)

        # compute lb and ulb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        # self.ulb_class_dist = torch.from_numpy(lb_class_dist / np.sum(lb_class_dist)) # 将无标注数据分布初始化为和有标注数据一样的分布
        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist) # 值越小说明对应类别的数量越大
        self.ulb_class_dist = self.lb_class_dist.clone()
        
        self.model = DRCNet(self.model, num_classes=self.num_classes)
        self.ema_model = DRCNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()

    def imb_init(self, drc_p_cutoff=0.95, drc_loss_ratio=1.0):
        self.drc_p_cutoff = drc_p_cutoff
        self.drc_loss_ratio = drc_loss_ratio

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

        # compute rrc loss using logits_aux from dict
        drc_loss, norms = self.compute_drc_loss(
            logits_x_lb=logits_x_lb,
            y_lb=kwargs['y_lb'],
            logits_x_ulb_w=logits_x_ulb_w,
            logits_x_ulb_s=logits_x_ulb_s
            )

        out_dict['loss'] += self.drc_loss_ratio * drc_loss
        log_dict['train/drc_loss'] = drc_loss.item()
        log_dict['train/norm_0'] = norms[0].item()
        log_dict['train/norm_1'] = norms[1].item()
        log_dict['train/norm_2'] = norms[2].item()
        log_dict['train/norm_3'] = norms[3].item()
        log_dict['train/norm_4'] = norms[4].item()
        log_dict['train/norm_5'] = norms[5].item()
        log_dict['train/norm_6'] = norms[6].item()
        log_dict['train/norm_7'] = norms[7].item()
        log_dict['train/norm_8'] = norms[8].item()
        log_dict['train/norm_9'] = norms[9].item()
        
        return out_dict, log_dict
    
    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        return super().evaluate(eval_dest=eval_dest, out_key='logits_aux', return_logits=return_logits)

    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()
    
    def compute_drc_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]
        
        if not self.lb_class_dist.is_cuda:
            self.lb_class_dist = self.lb_class_dist.to(y_lb.device)
        
        if not self.ulb_class_dist.is_cuda:
            self.ulb_class_dist = self.ulb_class_dist.to(y_lb.device)
        
        
        # compute labeled drc loss
        mask_lb = self.bernouli_mask(self.lb_class_dist[y_lb])
        drc_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none') * mask_lb).mean()


        # compute unlabeled drc loss
        with torch.no_grad():
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            mask_ulb_1 = max_probs.ge(self.drc_p_cutoff).to(logits_x_ulb_w.dtype)
            W = self.model.module.aux_classifier.weight
            norms = self.calculate_norm(W)
            
            if self.epoch/self.epochs < 0.8:
                mask_ulb = mask_ulb_1
            else:
                if self.ulb_class_dist == None:
                    W_class_dist = torch.min(norms) / norms
                    self.ulb_class_dist = W_class_dist.to(y_lb.device)
                mask_ulb_2 = self.bernouli_mask(self.ulb_class_dist[y_ulb])
                mask_ulb = mask_ulb_1 * mask_ulb_2

        drc_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            drc_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none') * mask_ulb).mean()
        
        drc_loss = drc_lb_loss + drc_ulb_loss
        return drc_loss, norms

    def calculate_norm(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        根据分类器中对应不同类别的权重的L2-Norm
        """
        # 计算向量的模长，shape为(K, 1)
        norms = torch.norm(matrix, dim=1, keepdim=True)
        return norms
    

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--drc_p_cutoff', float, 0.95),
            SSL_Argument('--drc_loss_ratio', float, 1.0),
        ]