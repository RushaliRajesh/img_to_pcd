import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
#resnet
from torchvision import models  

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data 
import numpy as np
import torch.nn.functional as F
import util_pt_clip as ut
from torchvision.transforms import ToPILImage, functional, ToTensor
import os
import sys
import warnings
import yaml
import pdb
from fvcore.common.config import CfgNode as _CfgNode
from torch.nn import MultiheadAttention
import math

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
import geoopt
from mobius_linear_example import MobiusLinear


class PointNetFeatures(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max'):
        super(PointNetFeatures, self).__init__()
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales
        self.conv1 = torch.nn.Conv1d(3, 64, 1)

        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)


        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = QSTN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

    def forward(self, x):
        n_pts = x.size()[2]
        points = x
        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3 * self.point_tuple, -1)
            points = x
        else:
            trans = None

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        return x,  trans, trans2, points


class PointNetEncoder(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max'):
        super(PointNetEncoder, self).__init__()
        self.pointfeat = PointNetFeatures(num_points=num_points, num_scales=num_scales, use_point_stn=use_point_stn,
                         use_feat_stn=use_feat_stn, point_tuple=point_tuple, sym_op=sym_op)
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)


    def forward(self, points):
        n_pts = points.size()[2]
        pointfeat, trans, trans2, points = self.pointfeat(points)

        x = F.relu(self.bn2(self.conv2(pointfeat)))
        x = self.bn3(self.conv3(x))
        global_feature = torch.max(x, 2, keepdim=True)[0]
        x = global_feature.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), global_feature.squeeze(), trans, trans2, points


class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x


class QSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x = batch_quat_to_rotmat(x)

        return x


def batch_quat_to_rotmat(q, out=None):

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2/torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out




class CrossAttentionLayer(nn.Module):
    def __init__(self, latent_dim, feature_size):
        super(CrossAttentionLayer, self).__init__()
        # self.feature_size = feature_size
        self.latent_dim = latent_dim
        self.key = nn.Linear(feature_size, latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(feature_size, latent_dim)

    def forward(self, x, latent):
        #linear transformations
        keys = self.key(x)
        queries = self.query(latent)
        values = self.value(x)

        #Scaled dot-product 
        # scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.latent_dim)
        # print("scores shape", scores.shape)  # [batch_size, seq_len, seq_len]
       
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)

        return attention_weights, output       

def hyperbolic_ReLU(hyperbolic_input, manifold):
    euclidean_input = manifold.logmap0(hyperbolic_input)
    euclidean_output = F.relu(euclidean_input)
    hyperbolic_output = manifold.expmap0(euclidean_output)
    return hyperbolic_output
class HyperbolicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = MobiusLinear(28 * 28, 750)
        self.fc11 = MobiusLinear(750, 20)
        self.fc2 = MobiusLinear(20, 10)

    def forward(self, x, manifold):
        out_x = self.fc1(x)
        out_x = hyperbolic_ReLU(out_x, manifold)
        out_x = self.fc11(out_x)
        out_x = hyperbolic_ReLU(out_x, manifold)
        out_x = self.fc2(out_x)
        return out_x


class ModelCombi_cross_perci_render_tpt_pcd(nn.Module):
    def __init__(self, bs, cfg=None, adapter = False, classes_total=48):
        super(ModelCombi_cross_perci_render_tpt_pcd, self).__init__()
        self.bs = bs
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
        self.res = torch.nn.Linear(768, classes_total)
        # pdb.set_trace()
        # self.extra = torch.nn.Parameter(torch.randn(bs, 768), requires_grad=True)
        # self.lin_ent = torch.nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.extra = torch.nn.Parameter(torch.randn(5), requires_grad=True)
        self.lin_ent = torch.nn.Linear(768, 171)

        self.pointnet = PointNetEncoder(num_points=500, use_point_stn=True, use_feat_stn=True)
        
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(1088, 128)

    def forward(self, img, pcd):
        img_feat, img_output, ptcloud_feat, ptcloud_output_final, ptcloud_output = None, None, None, None, None
        if img is not None:
            if hasattr(self, 'adapter_skt'):
                img = self.adapter_skt(img)
            img_feat, img_output = self.vpt_2d(img) 
        if pcd is not None:
            # pcds_img = self.pcviews.get_img(ptcloud)
            # pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
            # pcds_img = pcds_img/max(pcds_img.max(), 1e-8) 
            # pcds_img = functional.normalize(pcds_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            bs = pcd.shape[0]
            # render_imgs = render_imgs.reshape(-1, 3, 224, 224)
            x2 = self.pointnet(pcd)
            # print(x2[0].shape) 
            # x3 = self.pcd_pool(x2[0])
            x3 = torch.max(x2[0], 2, keepdim=True)[0]
            x3 = self.flatten(x3)
            # print(x3.shape)
            x3 = self.fc3(x3)
            pdb.set_trace()
            # print(ptcloud_feat.shape)
            attn_weights, ptcloud_feat = self.attn(ptcloud_feat, self.query)
            # pdb.set_trace()
            ptcloud_feat = ptcloud_feat.mean(dim=1)
            # pdb.set_trace()
            ptcloud_feat = self.intermediate(ptcloud_feat)
            ptcloud_output_final = self.res(ptcloud_feat)
            # pdb.set_trace()
        return img_feat, img_output, ptcloud_feat, ptcloud_output_final, ptcloud_output


class basicmodel(nn.Module):
    def __init__(self):
        super(basicmodel, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.pointnet = PointNetEncoder(num_points=500, use_point_stn=True, use_feat_stn=True)
        
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(1088, 128)
        # self.norm_sk = nn.Tanh()
        # self.norm_pc = nn.Tanh()
        self.classi = "some classifier"

    def forward(self, sketches, pcds):
        x = self.resnet(sketches)
        # print(x.shape)
        x = self.fc(x)
        x = self.relu(x)    
        x = self.fc2(x)
        x = self.norm_sk(x)

        x2 = self.pointnet(pcds)
        # print(x2[0].shape) 
        # x3 = self.pcd_pool(x2[0])
        x3 = torch.max(x2[0], 2, keepdim=True)[0]
        x3 = self.flatten(x3)
        # print(x3.shape)
        x3 = self.fc3(x3)
        # print(x3.shape)
        x3 = self.norm_pc(x3)
        return x, x3


if __name__ == "__main__":
    model = basicmodel()
    x1, x2 = model(torch.randn(5, 3, 224, 224), torch.randn(5, 3, 500))
    print("done")
    # model = basicmodel()
    # model(torch.randn(5, 3, 224, 224))
    
    # pcds_feats = PointNetEncoder(num_points=500, use_point_stn=True, use_feat_stn=True)
    # out2 = pcds_feats(torch.randn(5, 3, 500))
    # # print(out1.shape)
    # print(out2[0].shape)