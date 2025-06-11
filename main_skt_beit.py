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
from dataset_classi import ShapeData_notransform, ShapeData, pairing, read_classification_file 
from loss_util import ContrastiveLoss
from model_vpt_vit import BeitClassifier_morelayers, Convclassi
import pdb
from torch.utils.tensorboard import SummaryWriter
import yaml
from fvcore.common.config import CfgNode as _CfgNode
import torchvision.transforms.functional as TF

keyword = "sketch_beit_norm_0.5_lr0.0001"
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

all_class_list_skt = np.load("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketch_classes.npy")
all_class_list_model = np.load("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/model_classes.npy")

transform_img = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

tr_data_loader = DataLoader(tr_dataset, batch_size=48, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
te_data_loader = DataLoader(te_dataset, batch_size=48, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

# tr_data_sub = torch.utils.data.Subset(tr_dataset, range(0, 100))  
# te_data_sub = torch.utils.data.Subset(te_dataset, range(0, 60))

# for i in tr_data_sub:
#     print(i)
# pdb.set_trace()

# tr_data_loader = DataLoader(tr_data_sub, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
# te_data_loader = DataLoader(te_data_sub, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
# all_labels = torch.cat([target for _, _, target in tr_data_loader])
# uni = all_labels.unique()
# label_map = {uni[i].item(): i for i in range(len(uni))}

for i in tr_data_loader:
    print("tr_data_loader shape: ")
    print(i[0].shape, i[1].shape, i[2].shape)
    # print(i)
    # if len(i[2].unique()) != 2:
    #     print("skipping")
    #     continue
    # uni = i[2].unique()
    break


# pdb.set_trace()

start = time.time()
for ind, i in enumerate(tr_data_loader):
    if ind == 100:
        break
print(f"DataLoader time per batch: {(time.time() - start) / 100:.4f} seconds")

print("tr_data_loader: ", len(tr_data_loader))
print("te_data_loader: ", len(te_data_loader))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_skt = BeitClassifier_morelayers().to(device)

optimizer = torch.optim.Adam(model_skt.parameters(), lr=0.0001)
ce_loss = torch.nn.CrossEntropyLoss()
num_epochs = 30

# tr_loader_length = sum(1 for _ in tr_data_loader())
# te_loader_length = sum(1 for _ in te_data_loader())

def unnormalize(img_tensor, mean, std):
    # mean and std need to be tensors for broadcasting
    mean = mean[None, :, None, None]  # shape becomes (1, C, 1, 1)
    std = std[None, :, None, None]

    # Unnormalize
    unnormalized = img_tensor * std + mean
    return unnormalized*255.0

# def unnormalize(img_tensor, mean, std):
#     # mean and std need to be tensors for broadcasting
#     mean = torch.tensor(mean).view(-1, 1, 1)
#     std = torch.tensor(std).view(-1, 1, 1)
#     return img_tensor * std + mean

def show_batch(images, labels, save_path="debug_batch.png"):
    fig, axs = plt.subplots(1, len(images), figsize=(15, 3))
    for i in range(len(images)):
        img = unnormalize(images[i].cpu().clone(), mean=torch.tensor([128, 128, 128]), std=torch.tensor([64, 64, 64]))
        img = img.clamp(0, 255).byte()  # Clamp and convert to byte for correct rendering
        img = img.squeeze(0)  
        # pdb.set_trace()
        img = TF.to_pil_image(img)
        axs[i].imshow(img)
        if all_class_list_skt[labels[i].item()] != all_class_list_model[labels[i].item()]:
            print(f"Warning: Sketch label {all_class_list_skt[labels[i].item()]} does not match model label {all_class_list_model[labels[i].item()]}", flush=True)
            sys.exit()
        axs[i].set_title(f"Label: {all_class_list_skt[labels[i].item()]}, sketch label id: {labels[i].item()}")
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

for epoch in tqdm(range(num_epochs)):
    model_skt.train()
    tr_loss = 0.0
    val_loss = 0.0
    tr_acc = 0.0
    val_acc = 0.0
    
    for ind,(sketches, pcds, target) in enumerate(tr_data_loader):
        sketches = sketches.float().to(device)
        label = target.to(device)
        # pdb.set_trace()
        
        # label_map = {uni[0].item(): 0, uni[1].item(): 1}
        # label = torch.tensor([label_map[int(l)] for l in label.tolist()], dtype=torch.long).to(device)

        # bef_grad = sum(p.grad.norm() for p in model_skt.parameters() if p.grad is not None)
        # for p in model_skt.parameters():
        #     if p.grad is None:
        #         print(f"Parameter {p} has no gradient: ", p.grad, flush=True)


        optimizer.zero_grad()
        outputs = model_skt(sketches)
        loss = ce_loss(outputs, label)
        loss.backward()
        optimizer.step()

        # aft_grad = sum(p.grad.norm() for p in model_skt.parameters() if p.grad is not None)
        # print(f"Before gradient norm: {bef_grad}, After gradient norm: {aft_grad}", flush=True)
        
        tr_loss += loss.item()
        tr_acc += (outputs.argmax(dim=1) == label).float().sum().item()
        # pdb.set_trace()
        #print total number of trainable params
        if ind == 0:
            # print(sum(p.numel() for p in model_skt.parameters() if p.requires_grad), flush=True)
            print(sketches.shape, sketches.min(), sketches.max(), flush=True)
            print("labels: ", label, flush=True)
            print("outputs: ", outputs.argmax(dim=1), flush=True)
            # path = f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/train_loop_plots/{keyword}"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # show_batch(sketches[:2], label[:2], save_path=f"{path}/epoch_tr_{epoch}.png")
            


    # all_labels = torch.cat([target for _, _, target in te_data_loader])
    # uni = all_labels.unique()
    # label_map = {uni[i].item(): i for i in range(len(uni))}

    model_skt.eval()
    with torch.no_grad():
        for ind, (sketches, pcds, target) in enumerate(te_data_loader):
            sketches = sketches.float().to(device)
            label = target.to(device)
            # label = torch.tensor([label_map[int(l)] for l in label.tolist()], dtype=torch.long).to(device)
            

            outputs = model_skt(sketches)
            loss = ce_loss(outputs, label)
            val_loss += loss.item()
            val_acc += (outputs.argmax(dim=1) == label).float().sum().item()

            if ind == 0:
                print(sketches.shape, sketches.min(), sketches.max(), flush=True)
                print("labels: ", label, flush=True)
                print("outputs: ", outputs.argmax(dim=1), flush=True)
                # path = f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/train_loop_plots/{keyword}"
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # show_batch(sketches[:2], label[:2], save_path=f"{path}/epoch_val_{epoch}.png")
                
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {tr_loss/len(tr_data_loader):.4f}, Val Loss: {val_loss/len(te_data_loader):.4f}, train_acc: {tr_acc/len(tr_data_loader.dataset):.4f}, val_acc: {val_acc/len(te_data_loader.dataset):.4f}", flush=True)


    if not os.path.exists(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}"):
        os.makedirs(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}")
    torch.save(model_skt.state_dict(), f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}/model_{epoch}.pt")

    writer.add_scalar('training loss', tr_loss/len(tr_data_loader), epoch)
    writer.add_scalar('validation loss', val_loss/len(te_data_loader), epoch)
writer.close()
