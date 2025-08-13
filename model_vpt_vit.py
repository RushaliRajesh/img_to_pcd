import util_pt_clip as ut
from torchvision.transforms import ToPILImage
import os
import sys
import warnings
import yaml
import pdb
from fvcore.common.config import CfgNode as _CfgNode
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
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
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from transformers import BeitImageProcessor, BeitForImageClassification


import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn_siamese(nn.Module):
    def __init__(self, classes_total):
        super(Cnn_siamese, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flat = nn.Flatten()
        self.lin = nn.Linear(512, 48)

    def forward_once(self, x):
        x = F.relu(self.conv1(x))   # (B, 32, 94, 94)
        x = self.pool1(x)           # (B, 32, 23, 23)
        # print(x.shape)

        x = F.relu(self.conv2(x))   # (B, 64, 19, 19)
        x = self.pool2(x)           # (B, 64, 6, 6)
        # print(x.shape)

        x = F.relu(self.conv3(x))   # (B, 128, 4, 4)
        x = self.pool3(x)           # (B, 128, 2, 2)
        # print(x.shape)

        x_flat = self.flat(x)       # (B, 512)
        # print(x_flat.shape)
        return x_flat

    def forward(self, img, ren_img):
        img_out, img_flat, ren_out, ren_flat = None, None, None, None

        if img is not None:
            img_flat = self.forward_once(img)
            # print("le: ",img_feat.shape, img_flat.shape)
            img_out = self.lin(img_flat)  # Update linear layer based on input size

        if ren_img is not None:
            B = ren_img.shape[0]
            ren_img = ren_img.view(-1, 3, 100, 100)
            ren_flat = self.forward_once(ren_img)
            ren_flat = ren_flat.view(B, -1, 512).mean(dim=1)  # mean over N renderings per shape
            # print(ren_feat.shape, ren_flat.shape)
            ren_out = self.lin(ren_flat)  # Update linear layer based on input size

        return img_flat, img_out, ren_flat, ren_out



# class Cnn_siamese(nn.Module):
#     def __init__(self, classes_total):
#         super(Cnn_siamese, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=7)
#         self.pool1 = nn.MaxPool2d(kernel_size=4, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
#         self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
#         self.flat = nn.Flatten()
#         self.lin2 = nn.Linear(256 * 56 * 56, classes_total)
    
#     def forward(self, img, ren_img):
#         img_feat, ren_img_feat, img_out, ren_img_out = None, None, None, None
#         if img is not None:
#             print(img.shape)
#             img_feat = F.relu(self.conv1(img))
#             img_feat = self.pool1(img_feat)
#             print(img_feat.shape)
#             img_feat = F.relu(self.conv2(img_feat))
#             img_feat = self.pool2(img_feat)
#             print(img_feat.shape)
#             img_feat = F.relu(self.conv3(img_feat))
#             img_feat = self.pool3(img_feat)
#             print(img_feat.shape)
#             img_feat = self.flat(img_feat)
#             print(img_feat.shape)
#             img_out = self.lin2(img_feat)

#         if ren_img is not None:
#             B = ren_img.shape[0]
#             ren_img = ren_img.reshape(-1, 3, 224, 224)
#             ren_img_feat = F.relu(self.conv1(ren_img))
#             ren_img_feat = self.pool1(ren_img_feat)
#             print(ren_img_feat.shape)
#             ren_img_feat = F.relu(self.conv2(ren_img_feat))
#             ren_img_feat = self.pool2(ren_img_feat)
#             print(ren_img_feat.shape)
#             ren_img_feat = F.relu(self.conv3(ren_img_feat))
#             ren_img_feat = self.pool3(ren_img_feat)
#             print(ren_img_feat.shape)
#             ren_img_feat = self.flat(ren_img_feat)
#             print(ren_img_feat.shape)
#             ren_img_feat = ren_img_feat.reshape(B, -1, 256 * 56 * 56)
#             ren_img_feat = ren_img_feat.mean(dim=1)  # Average pooling over the sequence
#             ren_img_out = self.lin2(ren_img_feat)

#         return img_feat, img_out, ren_img_feat, ren_img_out


class SiameseModel(nn.Module):
    def __init__(self, classes_total):
        super(SiameseModel, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # for name, param in self.vit.named_parameters():
        #     print(name)
        # pdb.set_trace() 

        # self.vit.heads = torch.nn.Linear(768, 171)
        self.vit.heads = torch.nn.Identity()
        # for param in self.vit.heads.parameters():
        #     param.requires_grad = True
        # self.classify = torch.nn.Linear(768, 171)
        for name, param in self.named_parameters():
            print(name, param.requires_grad)

        self.lin_layer = torch.nn.Linear(768, classes_total)

    def forward(self, img, ren_img):
        # pdb.set_trace()
        img_feat, ren_img_feat, img_out, ren_img_out = None, None, None, None
        if img is not None:
            img_feat = self.vit(img)
            img_out = self.lin_layer(img_feat)

        if ren_img is not None:
            B = ren_img.shape[0]
            ren_img = ren_img.reshape(-1, 3, 224, 224)
            ren_img_feat = self.vit(ren_img)
            ren_img_feat = ren_img_feat.reshape(B, -1, 768)
            ren_img_feat = ren_img_feat.mean(dim=1)  # Average pooling over the sequence
            ren_img_out = self.lin_layer(ren_img_feat)
        # pdb.set_trace()
        return img_feat, img_out, ren_img_feat, ren_img_out


class OnlyVPT(nn.Module):
    def __init__(self, cfg=None):
        super(OnlyVPT, self).__init__()
        self.vpt_2d, self.model = build_model(cfg)

    def forward(self, img):
        _, img_output = self.vpt_2d(img)
        # print("img_feat shape", img_feat.shape)
        # print("img_output shape", img_output.shape)
        return img_output
    

class OnlyVIT(nn.Module):
    def __init__(self):
        super(OnlyVIT, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # for name, param in self.vit.named_parameters():
        #     print(name)
        # pdb.set_trace() 

        self.vit.heads = torch.nn.Linear(768, 171)
        for param in self.vit.heads.parameters():
            param.requires_grad = True
        # self.classify = torch.nn.Linear(768, 171)
        for name, param in self.named_parameters():
            print(name, param.requires_grad)

    def forward(self, img):
        img = self.vit(img)
        return img

class BeitClassifier(nn.Module):
    def __init__(self):
        super(BeitClassifier, self).__init__()
        self.processor = BeitImageProcessor.from_pretrained('kmewhort/beit-sketch-classifier')
        self.model = BeitForImageClassification.from_pretrained('kmewhort/beit-sketch-classifier')
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.classifier = torch.nn.Linear(768, 171)
        for params in self.model.classifier.parameters():
            params.requires_grad = True
        # pdb.set_trace()

    def forward(self, img):
        img = self.processor(images=img, return_tensors="pt").pixel_values.to(self.model.device)
        img = self.model(img)
        return img.logits


class Convclassi(nn.Module):
    def __init__(self):
        super(Convclassi, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(401408, 512)
        self.linear2 = nn.Linear(512, 2)
        #401408x512
        #8x9821312
    def forward(self, img):
        x = F.relu(self.conv1(img))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)  
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class BeitClassifier_morelayers(nn.Module):
    def __init__(self):
        super(BeitClassifier_morelayers, self).__init__()
        self.processor = BeitImageProcessor.from_pretrained('kmewhort/beit-sketch-classifier')
        self.model = BeitForImageClassification.from_pretrained('kmewhort/beit-sketch-classifier')
        for params in self.model.parameters():
            params.requires_grad = True
        self.model.classifier = torch.nn.Linear(768, 171)
        # for params in self.model.classifier.parameters():
        #     params.requires_grad = True
        # for param in self.model.beit.encoder.layer[-1].parameters():
        #     param.requires_grad = True
        # for param in self.model.beit.pooler.parameters():
        #     param.requires_grad = True
        # pdb.set_trace()

    def forward(self, img):
        # img = self.processor(images=img, return_tensors="pt").pixel_values.to(self.model.device)
        img = self.model(img)
        return img.logits


if __name__ == "__main__":
    # model = OnlyVIT()
    # model = BeitClassifier()
    out1, out2, o3, o4 = Cnn_siamese()(torch.randn(1, 3, 100, 100), torch.randn(1, 5, 3, 100, 100))
    pdb.set_trace()
    out1, out2, o3, o4 = SiameseModel(48)(torch.randn(1, 3, 224, 224), torch.randn(1, 5, 3, 224, 224))
    model = BeitClassifier_morelayers()
    out = model(torch.randn(1, 3, 224, 224))
    # print(out.shape)
    for n,p in model.named_parameters():
        print(n, p.requires_grad)
    pdb.set_trace()
