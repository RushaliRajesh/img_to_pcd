import util_pt_clip as ut
from torchvision.transforms import ToPILImage, functional, ToTensor
import os
import sys
import warnings
import yaml
import pdb
from fvcore.common.config import CfgNode as _CfgNode
from torch.nn import MultiheadAttention

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


class ModelCombi_norm(nn.Module):
    def __init__(self, cfg=None):
        super(ModelCombi_norm, self).__init__()
        self.vpt_2d, self.model = build_model(cfg)
        self.pcviews = ut.PCViews()
        self.intermediate = torch.nn.Linear(768, 512)
        # self.res = torch.nn.Linear(512, 171)
        self.res = torch.nn.Linear(768, 171)

    def forward(self, img, ptcloud):

        img_feat, img_output = self.vpt_2d(img)
        # img_feat = self.intermediate(img_feat)
        # img_output = self.res(img_feat)
        pcds_img = self.pcviews.get_img(ptcloud)
        # print("pcds_img shape", pcds_img.shape)
        pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
        # print("before scaling: ", torch.max(pcds_img), "min: ", torch.min(pcds_img), flush=True)
        pcds_img = pcds_img/max(pcds_img.max(), 1e-8)  # Normalize to [0, 1]
        # pdb.set_trace()
        # print(pcds_img.shape)
        pcds_img = functional.normalize(pcds_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # print("max: ", torch.max(pcds_img), "min: ", torch.min(pcds_img), flush=True)
        # print("pcds_img shape", pcds_img.shape)
        ptcloud_feat, ptcloud_output = self.vpt_2d(pcds_img)
        # print("ptcloud_feat shape", ptcloud_feat.shape)
        # print("ptcloud_output shape", ptcloud_output.shape)
        ptcloud_feat = ptcloud_feat.reshape(img.shape[0], -1, ptcloud_feat.shape[1])
        # print("ptcloud_feat shape", ptcloud_feat.shape)
        ptcloud_feat = ptcloud_feat.mean(dim=1)
        # print("ptcloud_feat shape", ptcloud_feat.shape)
        # ptcloud_feat = self.intermediate(ptcloud_feat)
        ptcloud_output_final = self.res(ptcloud_feat)
        # print("ptcloud_output_final shape", ptcloud_output_final.shape)

        return img_feat, img_output, ptcloud_feat, ptcloud_output_final
    


class ModelCombi_norm_w_avg(nn.Module):
    def __init__(self, cfg=None):
        super(ModelCombi_norm_w_avg, self).__init__()
        self.vpt_2d, self.model = build_model(cfg)
        
        self.pcviews = ut.PCViews()
        self.intermediate = torch.nn.Linear(768, 512)
        self.W =  torch.nn.Parameter(torch.ones(10, 768), requires_grad=True)
        # self.res = torch.nn.Linear(512, 171)
        self.res = torch.nn.Linear(768, 171)

    def forward(self, img, ptcloud):

        img_feat, img_output = self.vpt_2d(img)
        # img_feat = self.intermediate(img_feat)
        # img_output = self.res(img_feat)
        pcds_img = self.pcviews.get_img(ptcloud)
        print("pcds_img shape", pcds_img.shape)
        pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
        print("before scaling: ", torch.max(pcds_img), "min: ", torch.min(pcds_img), flush=True)
        pcds_img = pcds_img/max(pcds_img.max(), 1e-8)  # Normalize to [0, 1]
        
        print(pcds_img.shape)
        pcds_img = functional.normalize(pcds_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        print("max: ", torch.max(pcds_img), "min: ", torch.min(pcds_img), flush=True)
        print("pcds_img shape", pcds_img.shape)
        ptcloud_feat, ptcloud_output = self.vpt_2d(pcds_img)
        print("ptcloud_feat shape", ptcloud_feat.shape)
        print("ptcloud_output shape", ptcloud_output.shape)#([50, 768])
        ptcloud_feat = ptcloud_feat.reshape(img.shape[0], -1, ptcloud_feat.shape[1])
        print("ptcloud_feat shape", ptcloud_feat.shape) #([5, 10, 768])
        pdb.set_trace()
        # ptcloud_feat = ptcloud_feat.mean(dim=1)
        #weighted avg
        ptcloud_feat = ptcloud_feat * self.W
        ptcloud_feat = ptcloud_feat.mean(dim=1)

        print("ptcloud_feat shape", ptcloud_feat.shape)
        # ptcloud_feat = self.intermediate(ptcloud_feat)
        ptcloud_output_final = self.res(ptcloud_feat)
        print("ptcloud_output_final shape", ptcloud_output_final.shape)

        return img_feat, img_output, ptcloud_feat, ptcloud_output_final
    


class CrossAttentionLayer(nn.Module):
    def __init__(self, latent_dim, feature_size):
        super(CrossAttentionLayer, self).__init__()
        self.feature_size = feature_size

        self.key = nn.Linear(feature_size, latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(feature_size, latent_dim)

    def forward(self, x, latent):
        #linear transformations
        keys = self.key(x)
        queries = self.query(latent)
        values = self.value(x)

        #Scaled dot-product 
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))
        # print("scores shape", scores.shape)  # [batch_size, seq_len, seq_len]
       
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)

        return attention_weights, output       


class ModelCombi_cross_perci(nn.Module):
    def __init__(self, bs, cfg=None, adapter = False):
        super(ModelCombi_cross_perci, self).__init__()

        if adapter:
            self.adapter_skt = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
            )
            self.adapter_pcd = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
            )
        self.vpt_2d, self.model = build_model(cfg)
        self.pcviews = ut.PCViews()
        #10 views, thats why 10,768
        self.query =  torch.nn.Parameter(torch.ones(10, 128), requires_grad=True)
        self.attn = CrossAttentionLayer(128, 768)
        self.intermediate = torch.nn.Linear(128, 768)
        self.res = torch.nn.Linear(768, 171)

    def forward(self, img, ptcloud):

        if hasattr(self, 'adapter_skt'):
            img = self.adapter_skt(img)
        img_feat, img_output = self.vpt_2d(img)
        pcds_img = self.pcviews.get_img(ptcloud)
        pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
        pcds_img = pcds_img/max(pcds_img.max(), 1e-8) 
        pcds_img = functional.normalize(pcds_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        if hasattr(self, 'adapter_pcd'):
            pcds_img = self.adapter_pcd(pcds_img)
        ptcloud_feat, ptcloud_output = self.vpt_2d(pcds_img)
        ptcloud_feat = ptcloud_feat.reshape(img.shape[0], -1, ptcloud_feat.shape[1])
        attn_weights, ptcloud_feat = self.attn(ptcloud_feat, self.query)
        ptcloud_feat = ptcloud_feat.mean(dim=1)
        ptcloud_feat = self.intermediate(ptcloud_feat)
        ptcloud_output_final = self.res(ptcloud_feat)

        return img_feat, img_output, ptcloud_feat, ptcloud_output_final



class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        #linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        #Scaled dot-product 
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))
        # print("scores shape", scores.shape)  # [batch_size, seq_len, seq_len]
       
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)

        return attention_weights, output  

class ModelCombi_norm_perci(nn.Module):
    def __init__(self, cfg=None, adapter = False):
        super(ModelCombi_norm_perci, self).__init__()

        if adapter:
            self.adapter_skt = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
            )
            self.adapter_pcd = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
            )
        self.vpt_2d, self.model = build_model(cfg)
        self.pcviews = ut.PCViews()
        self.attn = SelfAttentionLayer(768)
        # self.res = torch.nn.Linear(512, 171)
        self.res = torch.nn.Linear(768, 171)
        
        self.intermediate = torch.nn.Linear(768, 512)
        self.W =  torch.nn.Parameter(torch.ones(10, 768), requires_grad=True)

    def forward(self, img, ptcloud):

        if hasattr(self, 'adapter_skt'):
            img = self.adapter_skt(img)
        img_feat, img_output = self.vpt_2d(img)
        # img_feat = self.intermediate(img_feat)
        # img_output = self.res(img_feat)
        pcds_img = self.pcviews.get_img(ptcloud)
        # print("pcds_img shape", pcds_img.shape)
        pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
        # print("before scaling: ", torch.max(pcds_img), "min: ", torch.min(pcds_img), flush=True)
        pcds_img = pcds_img/max(pcds_img.max(), 1e-8)  # Normalize to [0, 1]
        
        # print(pcds_img.shape)
        pcds_img = functional.normalize(pcds_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # print("max: ", torch.max(pcds_img), "min: ", torch.min(pcds_img), flush=True)
        # print("pcds_img shape", pcds_img.shape)
        if hasattr(self, 'adapter_pcd'):
            pcds_img = self.adapter_pcd(pcds_img)
            # print("after adapter_pcd: ", pcds_img.shape, flush=True)
        ptcloud_feat, ptcloud_output = self.vpt_2d(pcds_img)
        # print("ptcloud_feat shape", ptcloud_feat.shape)
        # print("ptcloud_output shape", ptcloud_output.shape)#([50, 768])
        ptcloud_feat = ptcloud_feat.reshape(img.shape[0], -1, ptcloud_feat.shape[1])
        # print("ptcloud_feat shape", ptcloud_feat.shape) #([5, 10, 768])
        # pdb.set_trace()
        # ptcloud_feat = ptcloud_feat.mean(dim=1)
        #weighted avg
        attn_weights, ptcloud_feat = self.attn(ptcloud_feat)
        # print("ptcloud_feat shape", ptcloud_feat.shape)
        # ptcloud_feat = self.intermediate(ptcloud_feat)
        ptcloud_feat = ptcloud_feat.mean(dim=1)
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
    # model = ModelCombi_norm_perci(cfg)
    # model = ModelCombi_norm_perci(cfg, adapter=True)
    model = ModelCombi_cross_perci(5, cfg, adapter=False)
    model = model.cuda()
    # x1, x2 = model(torch.randn(5, 3, 224, 224), torch.randn(5, 3, 500))\
    x1, x2, x3, x4 = model(torch.randn(5, 3, 224, 224).cuda(), torch.randn(5, 500, 3).cuda())
    print("done")