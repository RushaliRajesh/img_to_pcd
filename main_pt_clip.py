import os
import sys
import warnings
import yaml
from fvcore.common.config import CfgNode as _CfgNode
from tqdm import tqdm
import util_pt_clip as ut

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
from dataset_classi import ShapeData_skt, ShapeData_3d, ShapeData, pairing, read_classification_file 
from loss_util import ContrastiveLoss
from model_classi import basicmodel
import time
import os
import pdb
from torch.utils.tensorboard import SummaryWriter

keyword = "classi_pcd_clip_integrated"
writer = SummaryWriter(f'runs/{keyword}')

file = "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_models_train.cla"
m_m_tr, m_m_te, n = read_classification_file(file, "models", 
                                "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/model_classes.npy")
file = "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_Sketch_Test.cla"
m_s_test, n_s_test = read_classification_file(file, "sketches", 
                                              "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketch_classes.npy")

file = "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_Sketch_Train.cla"
m_s, n_s = read_classification_file(file, "sketches", 
                                              "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketch_classes.npy")


transform_img = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[128, 128, 128], std=[64, 64, 64])
    ])

# print(m_s)

m_tr_temp = {}
m_te_temp = {}
m_s_temp = {}
m_s_test_temp = {}

for (model_id, model_class, class_id) in m_m_tr:
    m_tr_temp.setdefault(model_class, []).append([model_id, class_id])

for (model_id, model_class, class_id) in m_m_te:
    m_te_temp.setdefault(model_class, []).append([model_id, class_id])

for (sketch_id, sketch_class, class_id) in m_s:
    m_s_temp.setdefault(sketch_class, []).append([sketch_id, class_id])

for (sketch_id, sketch_class, class_id) in m_s_test:
    m_s_test_temp.setdefault(sketch_class, []).append([sketch_id, class_id])


# tr_skt_dataset = ShapeData_skt(path= "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/sketches_unzp/SHREC14LSSTB_SKETCHES/",
#                            data_tup= m_s, 
#                            transform = transform_img, 
#                            label = "train")
# te_skt_dataset = ShapeData_skt(path= "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/sketches_unzp/SHREC14LSSTB_SKETCHES/",
#                            data_tup= m_s_test,
#                            transform = transform_img,   
#                            label = "test")
# tr_3d_dataset = ShapeData_3d(path= "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np/",
#                             data_tup= m_m_tr,
#                             transform = None,
#                             label = "train")
# te_3d_dataset = ShapeData_3d(path= "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np/",
#                             data_tup= m_m_te,
#                             transform = None,   
#                             label = "test")


tr_pairs = pairing(sketch_models = m_s_temp,models_3d= m_tr_temp)
                   # print("initial tr pairs: ",len(tr_pairs))

tr_dataset = ShapeData(
    sketch_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/sketches_unzp/SHREC14LSSTB_SKETCHES/",
    model_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np/",
    sketch_file=(m_s_temp, n_s),
    model_file=(m_tr_temp, n),
    pairs = tr_pairs,
    label='train',
    transform=transform_img  # You can add image transformations here
)

te_pairs = pairing(sketch_models = m_s_test_temp,models_3d= m_te_temp)

te_dataset = ShapeData(
    sketch_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/sketches_unzp/SHREC14LSSTB_SKETCHES/",
    model_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np/",
    sketch_file=(m_s_test_temp, n_s_test),
    model_file=(m_te_temp, n),
    pairs = te_pairs,
    label='test',
    transform=transform_img  # You can add image transformations here
)

# pdb.set_trace()
print("tr dataset: ", len(tr_dataset))
print("te dataset: ", len(te_dataset))

tr_data_loader = DataLoader(tr_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
te_data_loader = DataLoader(te_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
# tr_model_loader = DataLoader(tr_3d_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
# te_model_loader = DataLoader(te_3d_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
  

for i in tr_data_loader:
    print("tr_data_loader shape: ") 
    print(i[0].shape, i[1].shape, i[2].shape)
    break

start = time.time()
for ind, i in enumerate(tr_data_loader):
    if ind == 100:
        break
print(f"DataLoader time per batch: {(time.time() - start) / 100:.4f} seconds")

print("tr_data_loader: ", len(tr_data_loader))
print("te_data_loader: ", len(te_data_loader))
# pdb.set_trace()

#load config_params.yaml
with open('/nlsasfs/home/neol/rushar/scripts/img_to_pcd/config_params.yaml', 'r') as f:
    config_params = yaml.safe_load(f)

cfg = _CfgNode(config_params)
cfg.freeze()
# pdb.set_trace()
###


model, cur_device = build_model(cfg)
pcviews = ut.PCViews()
pc_views = pcviews
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ce_loss = torch.nn.CrossEntropyLoss()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_epochs =20
opti = make_optimizer(
    [model],
    cfg.SOLVER
)
scheduler = make_scheduler(
    opti,
    cfg.SOLVER)

for epoch in tqdm(range(num_epochs)):
    model.train()
    tr_loss = 0.0
    val_loss = 0.0
    lr = scheduler.get_lr()[0]
    tr_acc = 0.0
    val_acc = 0.0
    tr_acc_pc = 0.0
    val_acc_pc = 0.0
    
    for ind,(sketches, pcds, target) in enumerate(tr_data_loader):
        sketches = sketches.float().to(device)
        pcds = pcds.float().to(device)
        label = target.to(device)

        if pcds == None:
            continue
        
        pcds_img = pc_views.get_img(pcds).to(cur_device)
        pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
        
        # optimizer.zero_grad()
        opti.zero_grad()
        # pc_out, sk_out, outputs = model(sketches, pcds.transpose(1, 2))
        # outputs = model(sketches, pcds.transpose(1, 2))
        x,outputs = model(sketches)
        x_pc,output_pc = model(pcds_img)
        # pdb.set_trace()
        x_pc = x_pc.reshape(x.shape[0], -1, x_pc.shape[1])
        x_pc = x_pc.mean(dim=1)
        res = torch.nn.Linear(x_pc.shape[-1], 171)
        res= res.to(cur_device)
        x_pc = res(x_pc)

        pred_cls = torch.argmax(outputs, dim=1)
        pred_cls_pc = torch.argmax(x_pc, dim=1)
        tr_acc_pc += (pred_cls_pc == label).sum().item()
        tr_acc += (pred_cls == label).sum().item()
        # print(label, outputs)
        # outputs = torch.argmax(outputs, dim=1)
        # print(outputs.shape, label.shape)   
        # print(type(outputs), type(label))
        # print(outputs.dtype, label.dtype)
        loss = ce_loss(outputs, label) + ce_loss(x_pc, label)
        loss.backward()
        # optimizer.step()
        opti.step()
        tr_loss += loss.item()
    scheduler.step()    

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if param.grad.norm().item() < 0.01:
                    # print(f"Layer: {name}, Grad Norm: {param.grad.norm().item()}")

        #add gradients plot to plot
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         # print(name, param.grad.norm().item())
        #         writer.add_histogram(name, param.grad, ind)
        
        # print(loss.item())
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         print(f"WARNING: {name} has no gradient!")
        #         print(ind)
        #         sys.exit()


    model.eval()
    with torch.no_grad():   
        for sketches, pcds, target in te_data_loader:
            sketches = sketches.float().to(device)
            pcds = pcds.float().to(device)
            label = target.to(device)   

            if pcds == None:
                continue
            pcds_img = pc_views.get_img(pcds).to(cur_device)
            pcds_img = pcds_img.unsqueeze(1).repeat(1, 3, 1, 1)
            # pc_out, sk_out, outputs = model(sketches, pcds.transpose(1, 2))
            # outputs = model(sketches, pcds.transpose(1, 2))
            
            x, outputs = model(sketches)
            x_pc, output_pc = model(pcds_img)
            x_pc = x_pc.reshape(x.shape[0], -1, x_pc.shape[1])
            x_pc = x_pc.mean(dim=1)
            res = torch.nn.Linear(x_pc.shape[-1], 171)
            res= res.to(cur_device)
            x_pc = res(x_pc)

            pred_cls_pc = torch.argmax(x_pc, dim=1)
            val_acc_pc += (pred_cls_pc == label).sum().item()
            pred_cls = torch.argmax(outputs, dim=1)
            val_acc += (pred_cls == label).sum().item()

            # outputs = torch.argmax(outputs, dim=1)
            loss = ce_loss(outputs, label) + ce_loss(x_pc, label)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {tr_loss/len(tr_data_loader):.4f}, Train acc: {tr_acc/len(tr_data_loader):.4f}, Val Loss: {val_loss/len(te_data_loader):.4f}, Val acc: {val_acc/len(te_data_loader):.4f}, lr: {lr:.7f}")


    if not os.path.exists(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}"):
        os.makedirs(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}")
    torch.save(model.state_dict(), f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}/model_{epoch}.pt")

    writer.add_scalar('training loss', tr_loss/len(tr_data_loader), epoch)
    writer.add_scalar('validation loss', val_loss/len(te_data_loader), epoch)
writer.close()

