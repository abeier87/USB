# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

@ALGORITHMS.register('fixmatch_fp')
class FixMatch_fp(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).
        对原始FixMatch算法进行了一些改进，主要是利用分类器的不同类别权重角度来估计类别数量，再调整logits
        由于上述作法是为了优化伪标签的质量，因此改进后的FixMatch算法被命名为FixMatch_fp（fp: finetune pseudo-labels）

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def calculate_pi(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        根据分类器中对应不同类别的权重向量来计算LA算法中的π值
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

    def logits_adjustment(self, logits):
        pi = self.calculate_pi(self.model.classifier.weight)
        logits = logits + torch.log(pi)
        
        # TODO 测试完请删除👇🏻这行代码
        print("The shape of logits is: {}".format(logits.shape))
        
        return logits

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            
            # logits adjustment
            if self.epoch > 4:
                # TODO 测试完请删除👇🏻这行代码
                print('第{}个epoch，调整了logits'.format(self.epoch))
                
                logits_x_lb = self.logits_adjustment(logits_x_lb)
                logits_x_ulb_w = self.logits_adjustment(logits_x_ulb_w)
                logits_x_ulb_s = self.logits_adjustment(logits_x_ulb_s)

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # softmax
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]