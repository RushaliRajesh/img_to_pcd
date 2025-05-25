import os
from tqdm import tqdm
import time 
import random
import torch
import open3d as o3d
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import trimesh
import numpy as np
from itertools import product
from torch.utils.data import DataLoader 
from dataset_load import ShapeData
from loss_util import ContrastiveLoss
from model import basicmodel
import pdb
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/base_method1')

def read_classification_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    # Skip first two lines
    lines = lines[2:]

    modelclass = []  # List of (model_id, class_name) pairs
    N = []  # List of (class_name, num_models)

    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if len(parts) < 3:
            i += 1
            continue
        
        class_name, _, num_models = parts
        num_models = int(num_models)

        model_ids = [lines[i + j + 1].strip() for j in range(num_models)]
        # print(model_ids)

        # Store class name and number of models
        N.append((class_name, num_models))

        # Store model-class pairs
        for model_id in model_ids:
            modelclass.append((model_id, class_name))

        i += num_models + 1  # Move to next class

    return modelclass, N

file = "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_models_train.cla"
m, n = read_classification_file(file)
file = "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_Sketch_Test.cla"
m_s_test, n_s_test = read_classification_file(file)
# print(m)
# print(n)

file = "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_Sketch_Train.cla"
m_s, n_s = read_classification_file(file)
# print(len(m_s))
# print(len(n_s)) 
# print(len(m))
# print(len(n))

m_temp = {}
m_s_temp = {}
m_s_test_temp = {}
for (model_id, model_class) in m:
    m_temp.setdefault(model_class, []).append(model_id)

for (sketch_id, sketch_class) in m_s:
    m_s_temp.setdefault(sketch_class, []).append(sketch_id)

for (sketch_id, sketch_class) in m_s_test:
    m_s_test_temp.setdefault(sketch_class, []).append(sketch_id)

n_dict= dict(n)
m_tr = {}
m_te = {}
for ind,i in enumerate(m_temp):
    m_tr.setdefault(i, []).extend(m_temp[i][:n_dict[i]//2])
    m_te.setdefault(i, []).extend(m_temp[i][n_dict[i]//2:])

# print(len(m_tr))
# print(len(m_te))
# print(m_tr)
# print(m_te)
# print(len(m_s_temp))
transform_img = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

tr_dataset = ShapeData(
    sketch_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/sketches_unzp/SHREC14LSSTB_SKETCHES/",
    model_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d/SHREC14LSSTB_TARGET_MODELS/",
    sketch_file=(m_s_temp, n_s),
    model_file=(m_tr, n),
    label='train',
    transform=transform_img  # You can add image transformations here
)

te_dataset = ShapeData(
    sketch_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/sketches_unzp/SHREC14LSSTB_SKETCHES/",
    model_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d/SHREC14LSSTB_TARGET_MODELS/",
    sketch_file=(m_s_test_temp, n_s_test),
    model_file=(m_te, n),
    label='test',
    transform=transform_img  # You can add image transformations here
)

for i in tr_dataset:
    print(i[0].shape, i[1], i[2])
    # if i[1] == None:
    #     print(i[0].shape, i[1], i[2])
    break

tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
te_dataloader = DataLoader(te_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)

for i in te_dataloader:
    print(i[0].shape, i[1].shape, i[2].shape)
    break

print("len of tr dataloader: ", len(tr_dataloader))
print("len of te dataloader: ", len(te_dataloader))

start = time.time()
for ind, i in enumerate(te_dataloader):
    if ind == 100:
        break
print(f"DataLoader time per batch: {(time.time() - start) / 100:.4f} seconds")

pdb.set_trace()

model = basicmodel()
device = torch.device("cuda")   
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
contrastive_loss = ContrastiveLoss()

num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    tr_total_loss = 0.0
    val_total_loss = 0.0
    model.train()
    step=0
    for img, pcd, target in tr_dataloader:
        step += 1
        # start = time.time()
        img = img.float().to(device)
        pcd = pcd.float().to(device)
        target = target.float().to(device)
        # print(f"Data Transfer: {time.time() - start:.4f}s")

        if img == None:
            continue
        # img = img.float()
        # pcd = pcd.float()
        # target = target.float()
        # print(img.shape, pcd.shape, target.shape, img.device)
        # img_embed = image_encoder(img)
        # pcd_embed = pointnet_encoder(pcd)
        # start = time.time()
        img_embed, pcd_embed = model(img, pcd.transpose(1, 2))
        # print(f"Model Execution: {time.time() - start:.4f}s")
        # print(img_embed.shape, pcd_embed.shape)

        # start = time.time()
        loss = contrastive_loss(img_embed, pcd_embed, target)
        # print("tr step loss:", loss.item())
        # print(f"Loss Calculation: {time.time() - start:.4f}s")

        # start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"Backpropagation: {time.time() - start:.4f}s")

        # start = time.time()
        tr_total_loss += loss.item()
        # print(f"Loss add: {time.time() - start:.4f}s")

        if step % 100 == 0:
                print("tr step completed: ",step)
                print("loss now: ", tr_total_loss/step)


    model.eval()

    val_total_loss = 0.0
    with torch.no_grad():
        step = 0
        for img, pcd, target in te_dataloader:
            step += 1
            
            if img == None:
                continue
            img = img.float().to(device)
            pcd = pcd.float().to(device)
            target = target.float().to(device)
            # img = img.float()
            # pcd = pcd.float()
            # target = target.float()
            # print(img.shape, pcd.shape, target.shape)
            img_embed, pcd_embed = model(img, pcd.transpose(1, 2))
            # print(img_embed.shape, pcd_embed.shape)
            loss = contrastive_loss(img_embed, pcd_embed, target)
            val_total_loss += loss.item()
            # print("te step loss:", loss.item())
            if step % 100 == 0:
                print("te step completed: ",step)
                print("loss now: ", val_total_loss/step)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {tr_total_loss/len(tr_dataloader):.4f}, Val Loss: {val_total_loss/len(te_dataloader):.4f}")
