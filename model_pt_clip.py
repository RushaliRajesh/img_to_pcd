import util_pt_clip as ut
from torchvision.transforms import ToPILImage
import os
import sys
import warnings
import yaml
from fvcore.common.config import CfgNode as _CfgNode

# Add the project root directory to the Python path
# This allows absolute imports like 'vpt_workspace...' to work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# print(project_root)
# print(os.path.dirname(__file__))
# print(sys.path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(1, os.path.join(project_root, 'vpt_workspace/vpt'))
print(sys.path)

import numpy as np
import random

from time import sleep
from random import randint

from vpt_workspace.vpt.src.solver.optimizer import make_optimizer
from vpt_workspace.vpt.src.solver.lr_scheduler import make_scheduler
import vpt_workspace.vpt.src.utils.logging as logging
from vpt_workspace.vpt.src.configs.config import get_cfg
from vpt_workspace.vpt.src.data import loader as data_loader
from vpt_workspace.vpt.src.engine.evaluator import Evaluator
from vpt_workspace.vpt.src.engine.trainer import Trainer
from vpt_workspace.vpt.src.models.build_model import build_model
from vpt_workspace.vpt.src.utils.file_io import PathManager

from vpt_workspace.vpt.launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch import nn

class ModelCombi(nn.Module):
    def __init__(self, cfg=None):
        super(ModelCombi, self).__init__()
        self.vpt_2d, self.model = build_model(cfg)
        self.pcviews = ut.PCViews()
        self.res = torch.nn.Linear(768, 171)

    def forward(self, img, ptcloud):

        img_feat, img_output = self.vpt_2d(img)
        # print("img_feat shape", img_feat.shape)
        # print("img_output shape", img_output.shape)
        pcds_img = self.pcviews.get_img(ptcloud)
        # print("pcds_img shape", pcds_img.shape)
        pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
        # print("pcds_img shape", pcds_img.shape)
        ptcloud_feat, ptcloud_output = self.vpt_2d(pcds_img)
        # print("ptcloud_feat shape", ptcloud_feat.shape)
        # print("ptcloud_output shape", ptcloud_output.shape)
        ptcloud_feat = ptcloud_feat.reshape(img.shape[0], -1, ptcloud_feat.shape[1])
        # print("ptcloud_feat shape", ptcloud_feat.shape)
        ptcloud_feat = ptcloud_feat.mean(dim=1)
        # print("ptcloud_feat shape", ptcloud_feat.shape)
        ptcloud_output_final = self.res(ptcloud_feat)
        # print("ptcloud_output_final shape", ptcloud_output_final.shape)

        return img_feat, img_output, ptcloud_feat, ptcloud_output_final
           
class Model_for_Rrojections:
    def __init__(self, cfg=None):
        super(Model_for_Rrojections, self).__init__()
        self.vpt_2d, self.model = build_model(cfg)
        self.pcviews = ut.PCViews()
        self.res = torch.nn.Linear(768, 171)

    def forward(self, img, ptcloud):
        img_feat, img_output = self.vpt_2d(img)
        pcds_img = self.pcviews.get_img(ptcloud)
        pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
        ptcloud_feat, ptcloud_output = self.vpt_2d(pcds_img)
        ptcloud_feat = ptcloud_feat.reshape(img.shape[0], -1, ptcloud_feat.shape[1])
        ptcloud_feat = ptcloud_feat.mean(dim=1)
        ptcloud_output_final = self.res(ptcloud_feat)

        return img_feat, img_output, ptcloud_feat, ptcloud_output_final
    
if __name__ == "__main__":
    with open('/nlsasfs/home/neol/rushar/scripts/img_to_pcd/config_params.yaml', 'r') as f:
        config_params = yaml.safe_load(f)

    cfg = _CfgNode(config_params)
    cfg.freeze()
    model = ModelCombi(cfg)
    model = model.cuda()
    # x1, x2 = model(torch.randn(5, 3, 224, 224), torch.randn(5, 3, 500))\
    x1, x2, x3, x4 = model(torch.randn(5, 3, 224, 224).cuda(), torch.randn(5, 500, 3).cuda())
    print("done")