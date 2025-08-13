import os
import sys
import warnings
import yaml
from fvcore.common.config import CfgNode as _CfgNode
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import torch.nn.functional as F

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
from dataset_combi import ShapeData_meta_h5_render, pairing_hdf5, All_rendered_imgs, All_sketches  
from loss_util import ContrastiveLoss, Cross_entropy, compute_map, compute_metrics
from model_pt_clip import ModelCombi_cross_perci_render
import time
import os
import pdb
from torch.utils.tensorboard import SummaryWriter


keyword = "eval_vpt_ddgan_method"
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


# tr_pairs, _, _ = pairing_hdf5("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/sk_orig.hdf5",
#                        "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/pcds_orig.hdf5",
#                        label = 'train')
# te_pairs, te_all_skt, te_all_shp = pairing_hdf5("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/sk_orig.hdf5",
#                        "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/pcds_orig.hdf5",
#                        label = 'test')
#                    # print("initial tr pairs: ",len(tr_pairs))


tr_pairs, _, _, tr_classes = pairing_hdf5("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/sk_orig.hdf5",
                       "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/pcds_orig.hdf5",
                       label = 'train')
te_pairs, te_all_skt, te_all_shp, te_classes = pairing_hdf5("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/sk_orig.hdf5",
                       "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/splits/pcds_orig.hdf5",
                       label = 'test')
                   # print("initial tr pairs: ",len(tr_pairs))

print("\n tr_pairs: ", len(tr_pairs))
print(" te_pairs: ", len(te_pairs))
print()
if(tr_classes != te_classes):
    print("Classes are not same in train and test sets. Exiting...")
    sys.exit()
else:
    classes_total_num = tr_classes


all_sketches = All_sketches(te_all_skt.index, transform=transform_img)
all_shapes = All_rendered_imgs(te_all_shp.index, transform=transform_img)
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

def dist_func(query_feats, gallery_feats):
    """
    Compute mean average precision (mAP) for the given query and gallery features and labels.
    """
    # Normalize the features
    query_feats = F.normalize(query_feats, dim=1) # 4, 768
    gallery_feats = F.normalize(gallery_feats, dim=1) # 4, 768
    # pdb.set_trace()

    # Compute cosine similarity
    similarity_matrix = torch.mm(query_feats, gallery_feats.t()) # 4,4

    distM=1 - torch.mm(F.normalize(query_feats, dim=-1),
            F.normalize(gallery_feats, dim=-1).t())

    return distM


def retrievalParamSP_v2(shape_label,sketch_test_label):

    def retrievalParamSPs(shape_label,sketch_test_label):
        shapeLabels = np.array(shape_label)  ### cast all the labels as array
        sketchTestLabel = np.array(sketch_test_label)  ### cast sketch test label as array
        C_depths = np.zeros(sketchTestLabel.shape)
        unique_labels = np.unique(sketchTestLabel)
        for i in range(unique_labels.shape[0]):  ### find the numbers
            tmp_index_sketch = np.where(sketchTestLabel == unique_labels[i])[0]  ## for sketch index
            tmp_index_shape = np.where(shapeLabels == unique_labels[i])[0]  ## for shape index
            C_depths[tmp_index_sketch] = tmp_index_shape.shape[0]
        return C_depths

    C_depths = retrievalParamSPs(shape_label,sketch_test_label)

    return C_depths


def RetrievalEvaluation(C_depth, distM, model_label, depth_label, testMode=1):
    '''
    C_depth: retrieval number for the testing example, Nx1
    distM: distance matrix, row for testing example, column for training example
    model_label: model_label for training example
    depth_label: label for testing example

    testMode:
        1) use test  as query, find relevant examples in training data
        2) use test as query, find relevant examples in the testing data
    '''
    C_depth = C_depth.astype(int)
    # pdb.set_trace()
    if testMode == 1:
        C = C_depth
        recall = np.zeros((distM.shape[0], distM.shape[1]))
        precision = np.zeros((distM.shape[0], distM.shape[1]))

        rankArray = np.zeros((distM.shape[0], distM.shape[1]))

    elif testMode == 2:
        C = C_depth - 1
        recall = np.zeros((distM.shape[0], distM.shape[1] - 1))
        precision = np.zeros((distM.shape[0], distM.shape[1] - 1))

        rankArray = np.zeros((distM.shape[0], distM.shape[1] - 1))

    nb_of_query = C.shape[0]
    p_points = np.zeros((nb_of_query, np.amax(C)))
    ap = np.zeros(nb_of_query)
    nn = np.zeros(nb_of_query)
    ft = np.zeros(nb_of_query)
    st = np.zeros(nb_of_query)
    dcg = np.zeros(nb_of_query)
    e_measure = np.zeros(nb_of_query)

    for qqq in range(nb_of_query):
        temp_dist = distM[qqq]
        s = list(temp_dist)
        R = sorted(range(len(s)), key=lambda k: s[k])
        if testMode == 1:
            model_label_l = model_label[R]
            numRetrieval = distM.shape[1]
            G = np.zeros(numRetrieval)
            rankArray[qqq] = R
        elif testMode == 2:
            model_label_l = model_label[R[1:]]
            numRetrieval = distM.shape[1] - 1
            G = np.zeros(numRetrieval)
            rankArray[qqq] = R[1:]

        for i in range(numRetrieval):
            if model_label_l[i] == depth_label[qqq]:
                G[i] = 1
        G_sum = np.cumsum(G)

        # pdb.set_trace()
        r1 = G_sum / float(C[qqq])
        p1 = G_sum / np.arange(1, numRetrieval + 1)
        r_points = np.zeros(C[qqq])
        for i in range(C[qqq]):
            temp = np.where(G_sum == i + 1)
            r_points[i] = np.where(G_sum == (i + 1))[0][0] + 1
        r_points_int = np.array(r_points, dtype=int)

        p_points[qqq][:int(C[qqq])] = G_sum[r_points_int - 1] / r_points
        ap[qqq] = np.mean(p_points[qqq][:int(C[qqq])])
        nn[qqq] = G[0]
        ft[qqq] = G_sum[C[qqq] - 1] / C[qqq]
        

        st[qqq] = G_sum[min(2 * C[qqq] - 1, G_sum.size - 1)] / C[qqq]
        p_32 = G_sum[min(31, G_sum.size - 1)] / min(32, G_sum.size)
        r_32 = G_sum[min(31, G_sum.size - 1)] / C[qqq]
        if p_32 == 0 and r_32 == 0:
            e_measure[qqq] = 0
        else:
            e_measure[qqq] = 2 * p_32 * r_32 / (p_32 + r_32)

        if testMode == 1:
            NORM_VALUE = 1 + np.sum(1 / np.log2(np.arange(2, C[qqq] + 1)))
            dcg_i = 1 / np.log2(np.arange(2, len(R) + 1)) * G[1:]
            dcg_i = np.insert(dcg_i, 0, G[0])
            dcg[qqq] = np.sum(dcg_i, axis=0) / NORM_VALUE
            recall[qqq] = r1
            precision[qqq] = p1
        elif testMode == 2:
            NORM_VALUE = 1 + np.sum(1 / np.log2(np.arange(2, C[qqq] + 1)))
            dcg_i = 1 / np.log2(np.arange(2, len(R[1:]) + 1)) * G[1:]
            dcg_i = np.insert(dcg_i, 0, G[0])
            dcg[qqq] = np.sum(dcg_i, axis=0) / NORM_VALUE
            recall[qqq] = r1
            precision[qqq] = p1
    print(np.shape(distM))

    nn_av = np.mean(nn)
    ft_av = np.mean(ft)
    st_av = np.mean(st)
    dcg_av = np.mean(dcg)
    e_av = np.mean(e_measure)
    map_ = np.mean(ap)

    pre = np.mean(precision, axis=0)
    rec = np.mean(recall, axis=0)

    return nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray


# model = ModelCombi_norm_perci(cfg)
# model = ModelCombi_cross_perci(cfg=cfg, bs = B, adapter=False)
model = ModelCombi_cross_perci_render(cfg=cfg, bs = B, adapter=False, classes_total=classes_total_num)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# ce_loss = torch.nn.CrossEntropyLoss()
ce_loss = Cross_entropy()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.load_state_dict(torch.load('/nlsasfs/home/neol/rushar/scripts/img_to_pcd/saved_models/cross_lim_meta_render_cor_classes_48/model.pt'))
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

        distM = dist_func(torch.tensor(all_img_enc), torch.tensor(all_pcd_enc))
        C_depth = retrievalParamSP_v2(all_pcd_labels, all_img_labels)   
        nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray = RetrievalEvaluation(
            C_depth=C_depth,
            distM=distM.numpy(),
            model_label=all_pcd_labels,
            depth_label=all_img_labels,
            testMode=1
        )

        print(nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec, rankArray)
        # pdb.set_trace()
        mAP, ft, st = compute_metrics(torch.tensor(all_img_enc), torch.tensor(all_pcd_enc), 
                                torch.tensor(all_img_labels), torch.tensor(all_pcd_labels))
        print(f" mAP: {mAP:.4f}, ft: {ft:.4f}, st: {st:.4f}", flush = True)
           
        # Compute mAP
        # print(np.array(all_img_enc).shape, np.array(all_pcd_enc).shape, np.array(all_img_labels).shape, np.array(all_pcd_labels).shape)
        # le_map = compute_map(torch.tensor(all_img_enc), torch.tensor(all_pcd_enc), 
        #                     torch.tensor(all_img_labels), torch.tensor(all_pcd_labels))
        # print(f"mAP: {le_map:.4f}", flush=True)
        # mAP, ft, st = compute_metrics(torch.tensor(all_img_enc), torch.tensor(all_pcd_enc), 
        #                     torch.tensor(all_img_labels), torch.tensor(all_pcd_labels))
        # print(f"mAP: {mAP:.4f}, ft: {ft:.4f}, st: {st:.4f}", flush = True)
       