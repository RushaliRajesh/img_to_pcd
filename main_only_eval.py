import os
import sys
import warnings
import yaml
from fvcore.common.config import CfgNode as _CfgNode
from tqdm import tqdm
from torchvision.transforms import ToPILImage

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


from vpt_workspace.vpt.src.solver.optimizer import make_optimizer
from vpt_workspace.vpt.src.solver.lr_scheduler import make_scheduler

import numpy as np
warnings.filterwarnings("ignore")

import torch
import open3d as o3d
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
from PIL import Image
import sys
import numpy as np
from itertools import product
from torch.utils.data import DataLoader 
from dataset_combi import ShapeData_meta_h5, pairing_hdf5, All_shapes, All_sketches 
from loss_util import ContrastiveLoss, Cross_entropy, compute_map, compute_metrics
from model_pt_clip import ModelCombi_cross_perci
import time
import os
import pdb
from torch.utils.tensorboard import SummaryWriter


keyword = "eval_cross_meta"
print("keyword: ", keyword, flush=True)

B =16     

transform_img = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def visualize_pcd(pcd, name="pcd"):
    plt.scatter(pcd[:, 0], pcd[:, 1], c=pcd[:, 2], s=1)
    plt.savefig('point_cloud_'+name+'.png')


tr_pairs, _, _ = pairing_hdf5("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/sk_orig.hdf5",
                       "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/pcds_orig.hdf5",
                       label = 'train')
te_pairs, te_all_skt, te_all_shp = pairing_hdf5("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/sk_orig.hdf5",
                       "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/pcds_orig.hdf5",
                       label = 'test')
                   # print("initial tr pairs: ",len(tr_pairs))

all_sketches = All_sketches(te_all_skt.index, transform=transform_img)
all_shapes = All_shapes(te_all_shp.index)
all_skt_labels = te_all_skt['class_id'].values
all_shp_labels = te_all_shp['class_id'].values

# pdb.set_trace()

#load config_params.yaml
with open('/nlsasfs/home/neol/rushar/scripts/img_to_pcd/config_params.yaml', 'r') as f:
    config_params = yaml.safe_load(f)

cfg = _CfgNode(config_params)
cfg.freeze()
# pdb.set_trace()
###


# model = ModelCombi_norm_perci(cfg)
model = ModelCombi_cross_perci(cfg=cfg, bs = B, adapter=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# ce_loss = torch.nn.CrossEntropyLoss()
ce_loss = Cross_entropy()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.load_state_dict(torch.load('/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/conti_cross_lim_models/model_79.pt'))
# print("device: ", device)
# opti = make_optimizer(
#     [model],
#     cfg.SOLVER
# )
         
model.eval()
with torch.no_grad():

        all_img_enc = []
        all_pcd_enc = []
        all_img_labels = []
        all_pcd_labels = []
        for skts, lab in zip(all_sketches, all_skt_labels):
            skts = skts.float().to(device).unsqueeze(0)
            # lab = lab.reshape(1,1)
            sk_feat, _,_,_= model(skts, None)
            all_img_enc.append(sk_feat.cpu().numpy())
            all_img_labels.append(lab)
        # pdb.set_trace()
        for pcd, lab in zip(all_shapes, all_shp_labels):
            pcds = pcd.float().to(device).unsqueeze(0)
            # lab = lab.reshape(1,1)
            # pdb.set_trace()
            _, _, pc_feat, _ = model(None, pcds)
            all_pcd_enc.append(pc_feat.cpu().numpy())
            all_pcd_labels.append(lab)                            
        # pdb.set_trace()
        all_img_enc = np.concatenate(all_img_enc)
        all_pcd_enc = np.concatenate(all_pcd_enc)
        all_img_labels = np.array(all_img_labels)
        all_pcd_labels = np.array(all_pcd_labels)


        # Compute mAP
        # print(np.array(all_img_enc).shape, np.array(all_pcd_enc).shape, np.array(all_img_labels).shape, np.array(all_pcd_labels).shape)
        le_map = compute_map(torch.tensor(all_img_enc), torch.tensor(all_pcd_enc), 
                            torch.tensor(all_img_labels), torch.tensor(all_pcd_labels))
        print(f"mAP: {le_map:.4f}", flush=True)
        mAP, ft, st = compute_metrics(torch.tensor(all_img_enc), torch.tensor(all_pcd_enc), 
                            torch.tensor(all_img_labels), torch.tensor(all_pcd_labels))
        print(f"mAP: {mAP:.4f}, ft: {ft:.4f}, st: {st:.4f}", flush = True)
       