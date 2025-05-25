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
from model_classi import basicmodel
import pdb
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/classification_debug5')

# def read_classification_file(filename):
#     with open(filename, "r") as f:
#         lines = f.readlines()

#     # Skip first two lines
#     lines = lines[2:]

#     modelclass = []  # List of (model_id, class_name) pairs
#     N = []  # List of (class_name, num_models)

#     i = 0
#     while i < len(lines):
#         parts = lines[i].strip().split()
#         if len(parts) < 3:
#             i += 1
#             continue
        
#         class_name, _, num_models = parts
#         num_models = int(num_models)

#         model_ids = [lines[i + j + 1].strip() for j in range(num_models)]
#         # print(model_ids)

#         # Store class name and number of models
#         N.append((class_name, num_models))

#         # Store model-class pairs
#         for model_id in model_ids:
#             modelclass.append((model_id, class_name))

#         i += num_models + 1  # Move to next class

#     return modelclass, N

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
          

# pdb.set_trace()
# def tr_data_loader():
#     sketch_iter = iter(tr_sketch_loader)
#     pcd_iter = cycle(tr_model_loader) if len(tr_sketch_loader) > len(tr_model_loader) else iter(tr_model_loader)
    
#     for (sketches, s_labels), (pcds, p_labels) in zip(sketch_iter, pcd_iter):
#         yield (sketches, pcds, s_labels)


# def te_data_loader():
#     sketch_iter = iter(te_sketch_loader)
#     pcd_iter = cycle(te_model_loader) if len(te_sketch_loader) > len(te_3d_dataset) else iter(te_model_loader)

#     for (sketches, s_labels), (pcds, p_labels) in zip(sketch_iter, pcd_iter):
#         yield (sketches, pcds, s_labels)

# data = tr_data_loader()
# print(len(list(data)))

# pdb.set_trace()

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

model = basicmodel()
device = torch.device("cuda")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ce_loss = torch.nn.CrossEntropyLoss()
num_epochs = 20

# tr_loader_length = sum(1 for _ in tr_data_loader())
# te_loader_length = sum(1 for _ in te_data_loader())

for epoch in tqdm(range(num_epochs)):
    model.train()
    tr_loss = 0.0
    val_loss = 0.0
    
    for ind,(sketches, pcds, target) in enumerate(tr_data_loader):
        sketches = sketches.float().to(device)
        pcds = pcds.float().to(device)
        label = target.to(device)

        if pcds == None:
            continue

        optimizer.zero_grad()
        # pc_out, sk_out, outputs = model(sketches, pcds.transpose(1, 2))
        outputs = model(sketches, pcds.transpose(1, 2))
        # print(label, outputs)
        # outputs = torch.argmax(outputs, dim=1)
        # print(outputs.shape, label.shape)   
        # print(type(outputs), type(label))
        # print(outputs.dtype, label.dtype)
        loss = ce_loss(outputs, label)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if param.grad.norm().item() < 0.01:
                    # print(f"Layer: {name}, Grad Norm: {param.grad.norm().item()}")

        #add gradients plot to plot
        for name, param in model.named_parameters():
            if param.grad is not None:
                # print(name, param.grad.norm().item())
                writer.add_histogram(name, param.grad, ind)
        
        # print(loss.item())
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         print(f"WARNING: {name} has no gradient!")
        #         print(ind)
        #         sys.exit()

        # if ind == 3:
        #     break

    model.eval()
    with torch.no_grad():   
        for sketches, pcds, target in te_data_loader:
            sketches = sketches.float().to(device)
            pcds = pcds.float().to(device)
            label = target.to(device)   

            if pcds == None:
                continue

            # pc_out, sk_out, outputs = model(sketches, pcds.transpose(1, 2))
            outputs = model(sketches, pcds.transpose(1, 2))
            # outputs = torch.argmax(outputs, dim=1)
            loss = ce_loss(outputs, label)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {tr_loss/len(tr_data_loader):.4f}, Val Loss: {val_loss/len(te_data_loader):.4f}")


    if not os.path.exists("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/classi1"):
        os.makedirs("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/classi1")
    torch.save(model.state_dict(), f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/classi1/model_{epoch}.pt")

    writer.add_scalar('training loss', tr_loss/len(tr_data_loader), epoch)
    writer.add_scalar('validation loss', val_loss/len(te_data_loader), epoch)
writer.close()

pdb.set_trace()


# for i in tr_dataset:
#     print(i[0].shape, i[1].shape, i[2].shape)
#     # if i[1] == None:
#     #     print(i[0].shape, i[1], i[2])
#     break

tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
te_dataloader = DataLoader(te_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)


for i in te_dataloader:
    print(i[0].shape, i[1].shape, i[2].shape)
    break

# def denormalize(tensor, mean, std):
#     print("mean: ", mean, "std: ", std)
#     mean = torch.tensor(mean).view(3, 1, 1) 
#     std = torch.tensor(std).view(3, 1, 1)
#     print(mean.shape, std.shape)
#     print(tensor*std)
#     print((tensor*std)+mean)
#     return (tensor * std) + mean

def denormalize(tensor):
    temp = tensor.numpy()
    print(temp.shape)
    norm_image = cv2.normalize(temp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image

for i in tr_dataloader:
    print("bef: ", i[0][0].min(), i[0][0].max())
    # img =denormalize(i[0][0], [128, 128, 128], [64, 64, 64])
    img = denormalize(i[0][0])
    print("after: ",img.max(), img.min())
    print(img.shape)
    # img = plt.imshow(i[0][0].permute(1, 2, 0))
    print(type(img))
    # print(img.shape)
    plt.imsave("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/sketch1.png", img.transpose(1, 2, 0))
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(i[1][0].shape[0])
    for pnt in i[1][0]:
        ax.scatter(pnt[0], pnt[1], pnt[2], c='r', marker='o')

    # Save the plot instead of showing
    plt.savefig("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/3d_plot.png", dpi=300, bbox_inches='tight')

    break

# pdb.set_trace()
print("len of tr dataset: ", len(tr_dataset))
print("len of te dataset: ", len(te_dataset))
print("len of tr dataloader: ", len(tr_dataloader))
print("len of te dataloader: ", len(te_dataloader))

# start = time.time()
# for ind, i in enumerate(te_dataloader):
#     if ind == 1000:
#         break
# print(f"DataLoader time per batch: {(time.time() - start) / 1000:.4f} seconds")

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
    #save model

    # if not os.path.exists("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/fast_dl1"):
    #     os.makedirs("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/fast_dl1")
    # torch.save(model.state_dict(), f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/fast_dl1/model_{epoch}.pt")

#     writer.add_scalar('training loss', tr_total_loss/len(tr_dataloader), epoch)
#     writer.add_scalar('validation loss', val_total_loss/len(te_dataloader), epoch)
# writer.close()