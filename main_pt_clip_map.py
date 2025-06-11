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
from dataset_combi import ShapeData, pairing, read_classification_file 
from loss_util import ContrastiveLoss, Cross_entropy, compute_map
from model_pt_clip import ModelCombi_norm
import time
import os
import pdb
from torch.utils.tensorboard import SummaryWriter

keyword = "classi_contra_corr_norm"
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
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def visualize_pcd(pcd, name="pcd"):
    plt.scatter(pcd[:, 0], pcd[:, 1], c=pcd[:, 2], s=1)
    plt.savefig('point_cloud_'+name+'.png')

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

tr_data_loader = DataLoader(tr_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
te_data_loader = DataLoader(te_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
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


model = ModelCombi_norm(cfg)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# ce_loss = torch.nn.CrossEntropyLoss()
ce_loss = Cross_entropy()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print("device: ", device)
num_epochs =20
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
        label = target.to(device)
        pos_neg_ind = pos_neg_ind.to(device)

        if pcds == None:
            continue
        
        # optimizer.zero_grad()
        opti.zero_grad()
        
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
            label = target.to(device)
            pos_neg_ind = pos_neg_ind.to(device)

            if pcds == None:
                continue
                    
            sk_feat, sk_out, pc_feat, pc_out = model(sketches, pcds)
                    
            loss1, acc1 = ce_loss(sk_out, label, pos_neg_ind) 
            loss2, acc2 = ce_loss(pc_out, label, pos_neg_ind) 
            loss3 = con_loss(sk_feat, pc_feat, pos_neg_ind)
            loss = loss1+loss2+loss3
            acc = (acc1 + acc2) / 2

            if ind==0:
                print("labels: ", label, flush=True)
                print("outputs sk: ", sk_out.argmax(dim=1), flush=True)
                print("outputs pc: ", pc_out.argmax(dim=1), flush=True)
            
            # pdb.set_trace()
            if epoch%5==0:
                val_loss += loss.item()
                val_acc += acc.item()
                all_img_enc.append(sk_feat.cpu().numpy())
                all_pcd_enc.append(pc_feat.cpu().numpy())
                all_img_labels.append(label.cpu().numpy())
                all_pcd_labels.append(label.cpu().numpy())
    if all_img_enc:
        all_img_enc = np.concatenate(all_img_enc)
        all_pcd_enc = np.concatenate(all_pcd_enc)
        all_img_labels = np.concatenate(all_img_labels)
        all_pcd_labels = np.concatenate(all_pcd_labels)

        # pdb.set_trace()

        # Compute mAP
        # print(np.array(all_img_enc).shape, np.array(all_pcd_enc).shape, np.array(all_img_labels).shape, np.array(all_pcd_labels).shape)
        mAP = compute_map(torch.tensor(all_img_enc), torch.tensor(all_pcd_enc), 
                            torch.tensor(all_img_labels), torch.tensor(all_pcd_labels))
        print(f"Epoch [{epoch+1}/{num_epochs}], mAP: {mAP:.4f}", flush = True)
        writer.add_scalar('mAP', mAP, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {tr_loss/len(tr_data_loader):.4f}, Train acc: {tr_acc/len(tr_data_loader):.4f}, Val Loss: {val_loss/len(te_data_loader):.4f}, Val acc: {val_acc/len(te_data_loader):.4f}", flush = True)


    if not os.path.exists(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}"):
        os.makedirs(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}")
    torch.save(model.state_dict(), f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/{keyword}/model_{epoch}.pt")

    writer.add_scalar('training loss', tr_loss/len(tr_data_loader), epoch)
    writer.add_scalar('validation loss', val_loss/len(te_data_loader), epoch)
    writer.add_scalar('training accuracy', tr_acc/len(tr_data_loader), epoch)
    writer.add_scalar('validation accuracy', val_acc/len(te_data_loader), epoch)
    # writer.add_scalar('learning rate', lr, epoch)
writer.close()

