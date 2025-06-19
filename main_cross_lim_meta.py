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


keyword = "cross_lim_meta"
writer = SummaryWriter(f'runs/{keyword}')

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

tr_dataset = ShapeData_meta_h5(
    pairs = tr_pairs,
    transform=transform_img  # You can add image transformations here
)
te_dataset = ShapeData_meta_h5(
    pairs = te_pairs,
    transform=transform_img  # You can add image transformations here
)

# pdb.set_trace()
print("tr dataset: ", len(tr_dataset))
print("te dataset: ", len(te_dataset))

tr_data_loader = DataLoader(tr_dataset, batch_size=B, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
te_data_loader = DataLoader(te_dataset, batch_size=B, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
# tr_model_loader = DataLoader(tr_3d_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
# te_model_loader = DataLoader(te_3d_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
  

for i in tr_data_loader:
    print("tr_data_loader shape: ") 
    print(i[0].shape, i[1].shape, i[2].shape)
    # visualize_pcd(i[1][0], "original")
    # pdb.set_trace()
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


# model = ModelCombi_norm_perci(cfg)
model = ModelCombi_cross_perci(cfg=cfg, bs = B, adapter=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# ce_loss = torch.nn.CrossEntropyLoss()
ce_loss = Cross_entropy()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print("device: ", device)
num_epochs = 40
# opti = make_optimizer(
#     [model],
#     cfg.SOLVER
# )

opti = torch.optim.Adam(model.parameters(), lr=0.0001)

# scheduler = make_scheduler(
#     opti,
#     cfg.SOLVER)
con_loss = ContrastiveLoss()

for epoch in tqdm(range(num_epochs)):
    model.train()
    tr_loss = 0.0
    val_loss = 0.0
    # lr = scheduler.get_lr()[0]
    tr_acc = 0.0
    val_acc = 0.0
    tr_acc_pc = 0.0
    val_acc_pc = 0.0
    all_img_enc = []
    all_pcd_enc = []
    all_img_labels = []
    all_pcd_labels = []
    
    for ind,(sketches, pcds, target, pos_neg_ind) in enumerate(tr_data_loader):
        
        sketches = sketches.float().to(device)
        pcds = pcds.float().to(device)
        label = target.long().to(device)
        pos_neg_ind = pos_neg_ind.to(device)

        if pcds == None:
            continue
        
        # optimizer.zero_grad()
        opti.zero_grad()
        # pdb.set_trace()
        sk_feat, sk_out, pc_feat, pc_out = model(sketches, pcds)
                
        loss1, acc1 = ce_loss(sk_out, label, pos_neg_ind) 
        loss2, acc2 = ce_loss(pc_out, label, pos_neg_ind) 
        loss3 = con_loss(sk_feat, pc_feat, pos_neg_ind)
        loss = loss1+loss2+loss3
        acc = (acc1 + acc2) / 2
        # pdb.set_trace()
        loss.backward()
        # optimizer.step()
        opti.step()
        tr_loss += loss.item()
        tr_acc += acc.item()
        
    # scheduler.step()    

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
        if ind==0:
            print("labels: ", label, flush=True)
            print("outputs sk: ", sk_out.argmax(dim=1), flush=True)
            print("outputs pc: ", pc_out.argmax(dim=1), flush=True)


    model.eval()
    with torch.no_grad():   
        for ind, (sketches, pcds, target, pos_neg_ind) in enumerate(te_data_loader):
           
            sketches = sketches.float().to(device)
            pcds = pcds.float().to(device)
            label = target.long().to(device)
            pos_neg_ind = pos_neg_ind.to(device)

            if pcds == None:
                continue
                    
            sk_feat, sk_out, pc_feat, pc_out = model(sketches, pcds)
                    
            loss1, acc1 = ce_loss(sk_out, label, pos_neg_ind) 
            loss2, acc2 = ce_loss(pc_out, label, pos_neg_ind) 
            loss3 = con_loss(sk_feat, pc_feat, pos_neg_ind)
            loss = loss1+loss2+loss3
            acc = (acc1 + acc2) / 2
            val_loss += loss.item()
            val_acc += acc.item()

            if ind==0:
                print("labels: ", label, flush=True)
                print("outputs sk: ", sk_out.argmax(dim=1), flush=True)
                print("outputs pc: ", pc_out.argmax(dim=1), flush=True)
            
    model.eval()
    with torch.no_grad():
        if epoch % 10 == 0:
 
            all_img_enc = []
            all_pcd_enc = []
            all_img_labels = []
            all_pcd_labels = []
            for skts, lab in zip(all_sketches, all_skt_labels):
                skts = skts.float().to(device).unsqueeze(0)
                lab = lab.reshape(1,1)
                sk_feat, _,_,_= model(skts, None)
                all_img_enc.append(sk_feat.cpu().numpy())
                all_img_labels.append(lab)
            # pdb.set_trace()
            for pcd, lab in zip(all_shapes, all_shp_labels):
                pcds = pcd.float().to(device).unsqueeze(0)
                lab = lab.reshape(1,1)
                # pdb.set_trace()
                _, _, pc_feat, _ = model(None, pcds)
                all_pcd_enc.append(pc_feat.cpu().numpy())
                all_pcd_labels.append(lab)                            
        
            all_img_enc = np.concatenate(all_img_enc)
            all_pcd_enc = np.concatenate(all_pcd_enc)
            all_img_labels = np.concatenate(all_img_labels)
            all_pcd_labels = np.concatenate(all_pcd_labels)


            # Compute mAP
            # print(np.array(all_img_enc).shape, np.array(all_pcd_enc).shape, np.array(all_img_labels).shape, np.array(all_pcd_labels).shape)
            mAP, ft, st = compute_metrics(torch.tensor(all_img_enc), torch.tensor(all_pcd_enc), 
                                torch.tensor(all_img_labels), torch.tensor(all_pcd_labels))
            print(f"Epoch [{epoch+1}/{num_epochs}], mAP: {mAP:.4f}, ft: {ft:.4f}, st: {st:.4f}", flush = True)
            writer.add_scalar('mAP', mAP, epoch)
            writer.add_scalar('ft', ft, epoch)
            writer.add_scalar('st', st, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {tr_loss/len(tr_data_loader):.4f}, Train acc: {tr_acc/len(tr_data_loader):.4f}, Val Loss: {val_loss/len(te_data_loader):.4f}, Val acc: {val_acc/len(te_data_loader):.4f}", flush = True)


    if not os.path.exists(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}"):
        os.makedirs(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}")
    torch.save(model.state_dict(), f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}/model.pt")

    writer.add_scalar('training loss', tr_loss/len(tr_data_loader), epoch)
    writer.add_scalar('validation loss', val_loss/len(te_data_loader), epoch)
    writer.add_scalar('training accuracy', tr_acc/len(tr_data_loader), epoch)
    writer.add_scalar('validation accuracy', val_acc/len(te_data_loader), epoch)
    # writer.add_scalar('learning rate', lr, epoch)
writer.close()

