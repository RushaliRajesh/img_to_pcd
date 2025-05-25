import os
from tqdm import tqdm
import time 
import random
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
from model_vpt_vit import OnlyVIT, OnlyVPT
import pdb
from torch.utils.tensorboard import SummaryWriter
import yaml
from fvcore.common.config import CfgNode as _CfgNode

keyword = "sketch_classi_vpt"
keyword2 = "sketch_classi_vit"
writer = SummaryWriter(f'runs/{keyword}')
writer_2 = SummaryWriter(f'runs/{keyword2}')

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


for i in tr_data_loader:
    print("tr_data_loader shape: ")
    print(i[0].shape, i[1].shape, i[2].shape)
    # print(i)
    break

# pdb.set_trace()

start = time.time()
for ind, i in enumerate(tr_data_loader):
    if ind == 100:
        break
print(f"DataLoader time per batch: {(time.time() - start) / 100:.4f} seconds")

print("tr_data_loader: ", len(tr_data_loader))
print("te_data_loader: ", len(te_data_loader))

with open('/nlsasfs/home/neol/rushar/scripts/img_to_pcd/config_params.yaml', 'r') as f:
    config_params = yaml.safe_load(f)

cfg = _CfgNode(config_params)
cfg.freeze()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_vpt = OnlyVPT(cfg).to(device)
model_vit = OnlyVIT().to(device)

optimizer_vpt = torch.optim.Adam(model_vpt.parameters(), lr=0.001)
optimizer_vit = torch.optim.Adam(model_vit.parameters(), lr=0.001)
ce_loss = torch.nn.CrossEntropyLoss()
num_epochs = 20

# tr_loader_length = sum(1 for _ in tr_data_loader())
# te_loader_length = sum(1 for _ in te_data_loader())

for epoch in tqdm(range(num_epochs)):
    model_vpt.train()
    tr_loss = 0.0
    val_loss = 0.0
    tr_acc = 0.0
    val_acc = 0.0
    
    for ind,(sketches, pcds, target) in enumerate(tr_data_loader):
        sketches = sketches.float().to(device)
        label = target.to(device)

        optimizer_vpt.zero_grad()
        # pdb.set_trace()
        outputs = model_vpt(sketches)
        loss = ce_loss(outputs, label)
        loss.backward()
        optimizer_vpt.step()
        tr_loss += loss.item()
        tr_acc += (outputs.argmax(dim=1) == label).float().sum().item()

        #print total number of trainable params
        if ind == 0:
            print(sum(p.numel() for p in model_vpt.parameters() if p.requires_grad), flush=True)
            print("labels: ", label, flush=True)
            print("outputs: ", outputs.argmax(dim=1), flush=True)

    model_vpt.eval()
    with torch.no_grad():   
        for sketches, pcds, target in te_data_loader:
            sketches = sketches.float().to(device)
            label = target.to(device)  
            
            outputs = model_vpt(sketches)
            loss = ce_loss(outputs, label)
            val_loss += loss.item()
            val_acc += (outputs.argmax(dim=1) == label).float().sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {tr_loss/len(tr_data_loader):.4f}, Val Loss: {val_loss/len(te_data_loader):.4f}, train_acc: {tr_acc/len(tr_data_loader.dataset):.4f}, val_acc: {val_acc/len(te_data_loader.dataset):.4f}", flush=True)


    if not os.path.exists(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}"):
        os.makedirs(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}")
    torch.save(model_vpt.state_dict(), f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}/model_{epoch}.pt")

    writer.add_scalar('training loss', tr_loss/len(tr_data_loader), epoch)
    writer.add_scalar('validation loss', val_loss/len(te_data_loader), epoch)
writer.close()


for epoch in tqdm(range(num_epochs)):
    model_vit.train()
    tr_loss = 0.0
    val_loss = 0.0
    tr_acc = 0.0
    val_acc = 0.0
    
    for ind,(sketches, pcds, target) in enumerate(tr_data_loader):
        sketches = sketches.float().to(device)
        label = target.to(device)

        optimizer_vit.zero_grad()
        outputs = model_vit(sketches)
        loss = ce_loss(outputs, label)
        loss.backward()
        optimizer_vit.step()
        tr_loss += loss.item()
        tr_acc += (outputs.argmax(dim=1) == label).float().sum().item()

        if ind == 0:
            print(sum(p.numel() for p in model_vpt.parameters() if p.requires_grad), flush=True)
            print("labels: ", label, flush=True)
            print("outputs: ", outputs.argmax(dim=1), flush=True)

    model_vit.eval()
    with torch.no_grad():   
        for sketches, pcds, target in te_data_loader:
            sketches = sketches.float().to(device)
            label = target.to(device)  
            
            outputs = model_vit(sketches)
            loss = ce_loss(outputs, label)
            val_loss += loss.item()
            val_acc += (outputs.argmax(dim=1) == label).float().sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {tr_loss/len(tr_data_loader):.4f}, Val Loss: {val_loss/len(te_data_loader):.4f}, train_acc: {tr_acc/len(tr_data_loader.dataset):.4f}, val_acc: {val_acc/len(te_data_loader.dataset):.4f}", flush=True)


    if not os.path.exists(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword2}"):
        os.makedirs(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword2}")
    torch.save(model_vpt.state_dict(), f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword2}/model_{epoch}.pt")

    writer.add_scalar('training loss', tr_loss/len(tr_data_loader), epoch)
    writer.add_scalar('validation loss', val_loss/len(te_data_loader), epoch)
writer.close()