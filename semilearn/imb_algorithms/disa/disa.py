import torch
import torch.nn as nn

from inspect import signature
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from .utils import SinkhornDistance
from .utils import ETFArch
from semilearn.algorithms.utils import SSL_Argument

class ETFNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        
        self.bn = nn.BatchNorm1d(self.num_features)

    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['feat_bn'] = self.bn(results_dict['feat'])
        return results_dict



@IMB_ALGORITHMS.register('disa')
class DiSA(ImbAlgorithmBase):
    """
        DiSA (https://openreview.net/pdf?id=Hjwx3H6Vci).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.model = ETFNet(self.model)
        self.ema_model = ETFNet(self.ema_model)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()
        self.sinkhorna_ot = SinkhornDistance()
        self.etfarch = ETFArch(num_features=self.model.num_features, num_classes=self.num_classes).ori_M.T
        
        self.ot_loss_ratio = args.ot_loss_ratio

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train_step(self, *args, **kwargs):
        out_dict, log_dict = super().train_step(*args, **kwargs)
        
        # get features
        feats_x_lb = out_dict['feat']['x_lb']
        
        # get logits
        logits_x_lb = self.model.module.bn(feats_x_lb)

        # compute ot loss using logits from dict
        ot_loss = self.compute_ot_loss(
            logits_x_lb=logits_x_lb, 
            etfarch=self.etfarch
            )
        out_dict['loss'] += self.ot_loss_ratio * ot_loss 
        log_dict['train/ot_loss'] = ot_loss.item()

        return out_dict, log_dict

    def compute_ot_loss(self, logits_x_lb, etfarch):
        logits_x_lb = logits_x_lb / torch.clamp(
            torch.sqrt(torch.sum(logits_x_lb ** 2, dim=1, keepdims=True)), 1e-8).cuda()
        ot_loss = self.sinkhorna_ot(logits_x_lb, etfarch)
        return ot_loss

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--ot_loss_ratio', float, 0.1),
        ]