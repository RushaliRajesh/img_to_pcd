import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from pytorch3d.io import IO

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

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
import geoopt
from mobius_linear_example import MobiusLinear

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
        # pdb.set_trace()
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


class ModelCombi_cross_perci_render_hyp_learnable(nn.Module):
    def __init__(self, bs, cfg=None, adapter = False, classes_total=48):
        super(ModelCombi_cross_perci_render_hyp_learnable, self).__init__()

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

        self.fc1_hyp = MobiusLinear(128, 768)
        self.fc1_hyp_im = MobiusLinear(768, 128) #for images
        self.fc1_hyp_pc = MobiusLinear(128, 128) #for pcd
        self.fc2_hyp = MobiusLinear(128, classes_total)
        self.manifold = geoopt.PoincareBall(c=1.0, learnable=True)
        self.hyp_relu = hyperbolic_ReLU

    def forward(self, img, render_imgs):
        img_feat, img_output, ptcloud_feat, ptcloud_output_final = None, None, None, None
        if img is not None:
            if hasattr(self, 'adapter_skt'):
                img = self.adapter_skt(img)
            img_feat, img_output = self.vpt_2d(img)
            img_feat = self.manifold.expmap0(img_feat)
            img_feat = self.fc1_hyp_im(img_feat)
            img_feat = self.hyp_relu(img_feat, self.manifold)
            img_output = self.fc2_hyp(img_feat)
            # out_x = hyperbolic_ReLU(out_x, self.manifold)
            # pdb.set_trace()
        if render_imgs is not None:
            # pcds_img = self.pcviews.get_img(ptcloud)
            # pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
            # pcds_img = pcds_img/max(pcds_img.max(), 1e-8) 
            # pcds_img = functional.normalize(pcds_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            bs = render_imgs.shape[0]
            render_imgs = render_imgs.reshape(-1, 3, 224, 224)
            if hasattr(self, 'adapter_pcd'):
                render_imgs = self.adapter_pcd(render_imgs)
            # pdb.set_trace()
            ptcloud_feat, ptcloud_output = self.vpt_2d(render_imgs)
            ptcloud_feat = ptcloud_feat.reshape(bs, -1, ptcloud_feat.shape[1])
            attn_weights, ptcloud_feat = self.attn(ptcloud_feat, self.query)
            ptcloud_feat = ptcloud_feat.mean(dim=1)
            # ptcloud_feat = self.intermediate(ptcloud_feat)
            # ptcloud_output_final = self.res(ptcloud_feat)
            # pdb.set_trace()
            ptcloud_feat = self.manifold.expmap0(ptcloud_feat)
            ptcloud_feat = self.fc1_hyp_pc(ptcloud_feat)
            ptcloud_feat = self.hyp_relu(ptcloud_feat, self.manifold)
            ptcloud_output_final = self.fc2_hyp(ptcloud_feat)
            # out_x = hyperbolic_ReLU(out_x, self.manifold)

        return img_feat, img_output, ptcloud_feat, ptcloud_output_final


class ModelCombi_cross_perci_render_hyp(nn.Module):
    def __init__(self, bs, cfg=None, adapter = False, classes_total=48):
        super(ModelCombi_cross_perci_render_hyp, self).__init__()

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

        self.fc1_hyp = MobiusLinear(128, 768)
        self.fc1_hyp_im = MobiusLinear(768, 128) #for images
        self.fc1_hyp_pc = MobiusLinear(128, 128) #for pcd
        self.fc2_hyp = MobiusLinear(128, classes_total)
        self.manifold = geoopt.PoincareBall()

    def forward(self, img, render_imgs):
        img_feat, img_output, ptcloud_feat, ptcloud_output_final = None, None, None, None
        if img is not None:
            if hasattr(self, 'adapter_skt'):
                img = self.adapter_skt(img)
            img_feat, img_output = self.vpt_2d(img)
            img_feat = self.manifold.expmap0(img_feat)
            img_feat = self.fc1_hyp_im(img_feat)
            img_output = self.fc2_hyp(img_feat)
            # out_x = hyperbolic_ReLU(out_x, self.manifold)
            # pdb.set_trace()
        if render_imgs is not None:
            # pcds_img = self.pcviews.get_img(ptcloud)
            # pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
            # pcds_img = pcds_img/max(pcds_img.max(), 1e-8) 
            # pcds_img = functional.normalize(pcds_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            bs = render_imgs.shape[0]
            render_imgs = render_imgs.reshape(-1, 3, 224, 224)
            if hasattr(self, 'adapter_pcd'):
                render_imgs = self.adapter_pcd(render_imgs)
            # pdb.set_trace()
            ptcloud_feat, ptcloud_output = self.vpt_2d(render_imgs)
            ptcloud_feat = ptcloud_feat.reshape(bs, -1, ptcloud_feat.shape[1])
            attn_weights, ptcloud_feat = self.attn(ptcloud_feat, self.query)
            ptcloud_feat = ptcloud_feat.mean(dim=1)
            # ptcloud_feat = self.intermediate(ptcloud_feat)
            # ptcloud_output_final = self.res(ptcloud_feat)
            # pdb.set_trace()
            ptcloud_feat = self.manifold.expmap0(ptcloud_feat)
            ptcloud_feat = self.fc1_hyp_pc(ptcloud_feat)
            ptcloud_output_final = self.fc2_hyp(ptcloud_feat)
            # out_x = hyperbolic_ReLU(out_x, self.manifold)

        return img_feat, img_output, ptcloud_feat, ptcloud_output_final



class ModelCombi_cross_perci_render(nn.Module):
    def __init__(self, bs, cfg=None, adapter = False, classes_total=48):
        super(ModelCombi_cross_perci_render, self).__init__()

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

    def forward(self, img, render_imgs):
        img_feat, img_output, ptcloud_feat, ptcloud_output_final = None, None, None, None
        if img is not None:
            if hasattr(self, 'adapter_skt'):
                img = self.adapter_skt(img)
            img_feat, img_output = self.vpt_2d(img)
        if render_imgs is not None:
            # pcds_img = self.pcviews.get_img(ptcloud)
            # pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
            # pcds_img = pcds_img/max(pcds_img.max(), 1e-8) 
            # pcds_img = functional.normalize(pcds_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            bs = render_imgs.shape[0]
            render_imgs = render_imgs.reshape(-1, 3, 224, 224)
            if hasattr(self, 'adapter_pcd'):
                render_imgs = self.adapter_pcd(render_imgs)
            # pdb.set_trace()
            ptcloud_feat, ptcloud_output = self.vpt_2d(render_imgs)
            ptcloud_feat = ptcloud_feat.reshape(bs, -1, ptcloud_feat.shape[1])
            attn_weights, ptcloud_feat = self.attn(ptcloud_feat, self.query)
            # pdb.set_trace()
            ptcloud_feat = ptcloud_feat.mean(dim=1)
            ptcloud_feat = self.intermediate(ptcloud_feat)
            ptcloud_output_final = self.res(ptcloud_feat)

        return img_feat, img_output, ptcloud_feat, ptcloud_output_final



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
        img_feat, img_output, ptcloud_feat, ptcloud_output_final = None, None, None, None
        if img is not None:
            if hasattr(self, 'adapter_skt'):
                img = self.adapter_skt(img)
            img_feat, img_output = self.vpt_2d(img)
        if ptcloud is not None:
            pcds_img = self.pcviews.get_img(ptcloud)
            pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
            pcds_img = pcds_img/max(pcds_img.max(), 1e-8) 
            pcds_img = functional.normalize(pcds_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
            if hasattr(self, 'adapter_pcd'):
                pcds_img = self.adapter_pcd(pcds_img)
            ptcloud_feat, ptcloud_output = self.vpt_2d(pcds_img)
            ptcloud_feat = ptcloud_feat.reshape(ptcloud.shape[0], -1, ptcloud_feat.shape[1])
            attn_weights, ptcloud_feat = self.attn(ptcloud_feat, self.query)
            ptcloud_feat = ptcloud_feat.mean(dim=1)
            ptcloud_feat = self.intermediate(ptcloud_feat)
            ptcloud_output_final = self.res(ptcloud_feat)

        return img_feat, img_output, ptcloud_feat, ptcloud_output_final



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(device=device)

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
# edges. Refer to blending.py for more details. 
# blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# raster_settings = RasterizationSettings(
#     image_size=224, 
#     blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
#     faces_per_pixel=100, 
#     bin_size = 0
# )

'''try2'''
# blend_params = BlendParams(sigma=5e-4, gamma=1e-4)

# raster_settings = RasterizationSettings(
#     image_size=224,
#     blur_radius=np.log(1. / 5e-4 - 1.) * 5e-4,   # slightly smoother
#     faces_per_pixel=25,
#     bin_size=0
# )

'''try 3'''

blend_params = BlendParams(sigma=1e-5, gamma=1e-4)

raster_settings = RasterizationSettings(
    image_size=224,
    blur_radius=np.log(1. / 1e-4 - 1.)*1e-5,
    faces_per_pixel=50,   # sweet spot
    bin_size=0
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)


# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=224, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)

def tensor_2_img(tensor, name, type):
    from PIL import Image
    import torch
    import torchvision.transforms as T
    from PIL import Image
    if type=='img':
        transform = T.ToPILImage()
        pil_image = transform(tensor)

        pil_image.save(name+".png")

        print("Tensor converted to PIL image and saved")
    else:
        sil = tensor[0, ..., 3].detach().cpu().numpy() 
        sil = (sil * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(sil).save(name+".png")

class Model_hyp_diff_ren(nn.Module):
    def __init__(self, bs, cfg=None, adapter = False, device = device, renderer= silhouette_renderer.to(device), classes_total=48):
        super(Model_hyp_diff_ren, self).__init__()

        self.renderer = renderer
        self.device = device
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

        self.fc1_hyp = MobiusLinear(128, 768)
        self.fc1_hyp_im = MobiusLinear(768, 128) #for images
        self.fc1_hyp_pc = MobiusLinear(128, 128) #for mesh
        self.fc2_hyp = MobiusLinear(128, classes_total)
        self.manifold = geoopt.PoincareBall(c=1.0, learnable=True)
        self.hyp_relu = hyperbolic_ReLU
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([3.0,  6.9, +2.5], dtype=np.float32)).to(self.device))
        # 1-channel silhouette input
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            # nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Conv2d(32, 64, 3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(401408, 128)   
        )


    def forward(self, img, mesh):
        img_feat, img_output, mesh_feat, mesh_output_final = None, None, None, None
        if img is not None:
            if hasattr(self, 'adapter_skt'):
                img = self.adapter_skt(img)
            img_feat, img_output = self.vpt_2d(img)
            img_feat = self.manifold.expmap0(img_feat)
            img_feat = self.fc1_hyp_im(img_feat)
            img_feat = self.hyp_relu(img_feat, self.manifold)
            img_output = self.fc2_hyp(img_feat)
            # out_x = hyperbolic_ReLU(out_x, self.manifold)
            # pdb.set_trace()
        if mesh is not None:
            # print(self.device)
            # print(self.camera_position.device)
            R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
            # print(R.device)
            # pdb.set_trace()
            T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)
            # ren_image = self.renderer(meshes_world=mesh.clone(), R=R, T=T)[..., 3].unsqueeze(1)
            # ren_image = self.renderer(meshes_world=mesh.clone(), R=R, T=T).permute(0,3,1,2)  # (1, 4, H, W)
            ren_image = self.renderer(meshes_world=mesh.clone(), R=R, T=T)
            # ren_image = self.renderer(meshes_world=mesh.clone(), R=R, T=T)
            # pdb.set_trace()
            # mesh_feat, mesh_output = self.vpt_2d(ren_image.repeat(1,3,1,1))
            # ren_image = (ren_image - ren_image.mean()) / (ren_image.std() + 1e-6)4
            alpha = ren_image[..., 3]          # (B, H, W)
            alpha = alpha.unsqueeze(1)         # (B, 1, H, W)
            alpha = alpha.clamp(0, 1)
            # pdb.set_trace()
            mesh_feat = self.cnn(alpha)
            mesh_feat = self.manifold.expmap0(mesh_feat)
            
            mesh_feat = self.fc1_hyp_pc(mesh_feat)
            mesh_feat = self.hyp_relu(mesh_feat, self.manifold)
            mesh_output_final = self.fc2_hyp(mesh_feat)

            # print(image.shape)
            
            # pdb.set_trace()
        return img_feat, img_output, mesh_feat, mesh_output_final



class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([3.0,  6.9, +2.5], dtype=np.float32)).to(meshes.device))

    def forward(self):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        print(self.device)
        print(self.camera_position.device)
        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        print(R.device)
        pdb.set_trace()
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        
        # Calculate the silhouette loss
        loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, image
  

if __name__ == "__main__":
    mesh = IO().load_mesh("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d/SHREC14LSSTB_TARGET_MODELS/M000003.off")
    with open('/nlsasfs/home/neol/rushar/scripts/img_to_pcd/config_params.yaml', 'r') as f:
        config_params = yaml.safe_load(f)

    cfg = _CfgNode(config_params)
    cfg.freeze()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    silhouette_renderer = silhouette_renderer.to(device)

    model = Model_hyp_diff_ren(1, cfg, adapter=False, device=device).to(device)
    out= model(torch.randn(2, 3, 224, 224).to(device), mesh.to(device))