import os
import random
import torch
import pdb
import open3d as o3d
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import trimesh
import numpy as np
from itertools import product
from torch.utils.data import DataLoader 
import sys
from functools import lru_cache 
from natsort import natsorted
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def read_classification_file(filename, flag, class_path):
    # print("in here")
    if flag == "models":
        class_data = np.load(class_path)
        
        with open(filename, "r") as f:
            lines = f.readlines()

        # Skip first two lines
        lines = lines[2:]

        modelclass_tr = []  # List of (model_id, class_name) pairs
        modelclass_te = []
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
            model_ids_tr = model_ids[:num_models//2]
            model_ids_te = model_ids[num_models//2:]
            # print(model_ids)

            # Store class name and number of models
            N.append((class_name, num_models))

            # Store model-class pairs
            class_ind = int(np.where(class_data == class_name)[0][0])
            # print("class_ind, flag: ", class_ind, flag)
            for model_id in model_ids_tr:
                modelclass_tr.append((model_id, class_name, class_ind))

            for model_id in model_ids_te:
                modelclass_te.append((model_id, class_name, class_ind))

            i += num_models + 1  # Move to next class
        return (modelclass_tr, modelclass_te, N)
    
    class_data = np.load(class_path)
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

        class_ind = int(np.where(class_data == class_name)[0][0])
        # print("class_ind, flag: ", class_ind, flag)
        # Store model-class pairs
        for model_id in model_ids:
            modelclass.append((model_id, class_name, class_ind))

        i += num_models + 1  # Move to next class

    return (modelclass, N)


def make_classes_file(filename, flag):

    class_list = []
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

        class_list.append(class_name)

    class_list = natsorted(class_list)
    if flag=="sketches":
        np.save(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketch_classes.npy", np.array(class_list))

    if flag=="models":
        np.save(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/model_classes.npy", np.array(class_list))



def pairing(sketch_models, models_3d):
    pairs = []
    all_classes = set(sketch_models.keys()) & set(models_3d.keys())

    for class_name in all_classes:
        # print("class_name: ", class_name)
        
        '''verification'''
        id_file = np.load("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketch_classes.npy")
        verification_id = int(np.where(id_file == class_name)[0][0])
        sk_class_id = int(np.array(sketch_models[class_name])[:,1][0])
        pc_class_id = int(np.array(models_3d[class_name])[:,1][0])
        if verification_id != sk_class_id or verification_id != pc_class_id:
            print(verification_id, sk_class_id, pc_class_id)
            print(type(verification_id), type(sk_class_id), type(pc_class_id))
            print("Verification id mismatch")
            sys.exit()
        '''done!'''
        
        sketch_ids = np.array(sketch_models[class_name])[:,0]
        model_ids = np.array(models_3d[class_name])[:,0]
        if len(model_ids) == 0:
                continue  
        # print("model_ids: ", model_ids)
        
        for i in sketch_ids:
            pos_ind = random.choice(model_ids)
            # print("pos_ind: ", pos_ind)
            pairs.append((i, pos_ind, class_name, sk_class_id,0))

        #negative pairs (target = 1)
        neg_classes = all_classes - {class_name}
        for i in sketch_ids:
            neg_cls = random.choice(list(neg_classes))
            # print("neg_cls: ", neg_cls)
            # print("model_ids neg: ", models_3d[neg_cls])     
            if len(models_3d[neg_cls]) == 0:
                continue             
            neg_ind = random.choice(models_3d[neg_cls])[0]
            # print("neg_ind: ", neg_ind)
            pairs.append((i, neg_ind, class_name, sk_class_id, 1)) 

    return pairs  




def lim_pairing(sketch_models, models_3d):
    pairs = []
    all_classes = set(sketch_models.keys()) & set(models_3d.keys())

    for class_name in all_classes:
        # print("class_name: ", class_name)
        
        '''verification'''
        id_file = np.load("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketch_classes.npy")
        verification_id = int(np.where(id_file == class_name)[0][0])
        sk_class_id = int(np.array(sketch_models[class_name])[:,1][0])
        pc_class_id = int(np.array(models_3d[class_name])[:,1][0])
        if verification_id != sk_class_id or verification_id != pc_class_id:
            print(verification_id, sk_class_id, pc_class_id)
            print(type(verification_id), type(sk_class_id), type(pc_class_id))
            print("Verification id mismatch")
            sys.exit()
        '''done!'''
        
        sketch_ids = np.array(sketch_models[class_name])[:,0]
        model_ids = np.array(models_3d[class_name])[:,0]
        if len(model_ids) < 50:
                continue  
        # print("model_ids: ", model_ids)
        
        for i in sketch_ids:
            pos_ind = random.choice(model_ids)
            # print("pos_ind: ", pos_ind)
            pairs.append((i, pos_ind, class_name, sk_class_id,0))

        #negative pairs (target = 1)
        neg_classes = all_classes - {class_name}
        for i in sketch_ids:
            neg_cls = random.choice(list(neg_classes))
            # print("neg_cls: ", neg_cls)
            # print("model_ids neg: ", models_3d[neg_cls])     
            if len(models_3d[neg_cls]) == 0:
                continue             
            neg_ind = random.choice(models_3d[neg_cls])[0]
            # print("neg_ind: ", neg_ind)
            pairs.append((i, neg_ind, class_name, sk_class_id, 1)) 

    return pairs 


class ShapeData(Dataset):
    def __init__(self, sketch_dir, model_dir, sketch_file, model_file, pairs, label = "train",transform=None):
        self.sketch_dir = sketch_dir
        self.model_dir = model_dir
        self.transform = transform
        self.sketch_models, self.sketch_N = sketch_file
        self.models_3d, self.N_3d = model_file
        self.label = label
        self.pairs = pairs                 

    def __len__(self):
        return len(self.pairs)
    
    @staticmethod
    @lru_cache(maxsize=100)  
    def load_image(path):
        # print(f"Loading from disk: {path}")
        img = Image.open(path).convert("RGB")
        return img

    @staticmethod
    @lru_cache(maxsize=100)  
    def load_mesh(path):
        # print(f"Loading from disk: {path}")
        mesh = np.load(path)
        return mesh

    def __getitem__(self, index):

        sketch_id, model_id, class_name, target, pos_neg_ind = self.pairs[index]
        # print("skt_id: ", sketch_id, "model_id: ", model_id, "class_name: ", class_name, "target: ", target)
        sketch_path = os.path.join(self.sketch_dir, f"{class_name}/{self.label}/{sketch_id}.png")
        model_path = os.path.join(self.model_dir, f"M{model_id}.npy")

        # sketch = Image.open(sketch_path).convert("RGB")
        # mesh = o3d.io.read_triangle_mesh(model_path)
        sketch = self.load_image(sketch_path)
        mesh = self.load_mesh(model_path)
        if len(mesh) == 0:
            return None, None, None
        # vertices_np = np.asarray(mesh.vertices)
        # pcd.points = o3d.utility.Vector3dVector(vertices_np)

        # pcd_visu = o3d.geometry.PointCloud()
        # pcd_visu.points = o3d.utility.Vector3dVector(mesh)
        # # o3d.visualization.draw_plotly([pcd_visu])
        # o3d.io.write_point_cloud("mesh_ori.ply", pcd_visu)

        #normalise the pcd
        mesh = mesh - mesh.mean(axis=0)
        mesh = mesh/max(np.linalg.norm(mesh, axis=1).max(), 1e-8)

        # pcd_visu = o3d.geometry.PointCloud()
        # pcd_visu.points = o3d.utility.Vector3dVector(mesh)
        # # o3d.visualization.draw_plotly([pcd_visu])
        # o3d.io.write_point_cloud("mesh.ply", pcd_visu)
        # pdb.set_trace()
        

        if self.transform:
            # print("transforming")
            # print(np.array(sketch).max(), np.array(sketch).min())
            sketch = self.transform(sketch)

        return (sketch, torch.tensor(mesh), torch.tensor(target), torch.tensor(pos_neg_ind))

    

if __name__ == "__main__":

    #make class lists

    make_classes_file("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_models_train.cla", "models")
    make_classes_file("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_Sketch_Test.cla", "sketches")

    pdb.set_trace()

    # for path in os.listdir("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d/SHREC14LSSTB_TARGET_MODELS/"):
    #     if path.endswith(".off"):
    #         print(path)
    #         mesh = o3d.io.read_triangle_mesh(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d/SHREC14LSSTB_TARGET_MODELS/{path}")
    #         name = path.split(".")[0] 
    #         print(name)
    #         np.save("/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np/" + name + ".npy", np.array(mesh.vertices))
    #         break

    mesh_dir = "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d/SHREC14LSSTB_TARGET_MODELS"

    files = natsorted(f for f in os.listdir(mesh_dir) if f.endswith(".off"))

    def process(path):
            name = path.split(".")[0]
            output_dir = "/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np/"

            if f"{name}.npy" in os.listdir(output_dir):
                print(f"{name}.npy already exists")
                return
            full_path = os.path.join(mesh_dir, path)
            mesh = o3d.io.read_triangle_mesh(full_path)
            
            print(name)
            if len(mesh.vertices) == 0:
                np.save(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np/{name}.npy", np.array([mesh.vertices]))
                return
            pcd = o3d.geometry.PointCloud()
            pcd = mesh.sample_points_poisson_disk(number_of_points=500, init_factor=5)
            
            np.save(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np/{name}.npy", np.array(pcd.points))
            # temp = np.load(f"/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d_np/{name}.npy")

    Parallel(n_jobs=12)(delayed(process)(i) for i in files)

    sys.exit()

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
        print(i)
        break

    tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True)

    for i in te_dataset:
        print(i)
        print(torch.where(i[0] != 1.0))
        break

    

