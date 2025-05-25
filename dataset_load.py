import os
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
from functools import lru_cache 

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




'''the sketch ids and the model ids arent the same'''

'''
temp1 = []
temp2 = []
for i, (sketch_id, sketch_class) in enumerate(m_s):
    for model_id, model_class in m:
        if model_class == sketch_class:
            temp1.append(model_id)
    break  # Breaks after first sketch class is processed

# Loop through 3D model classes
for i, (model_id, model_class) in enumerate(m):
    for sketch_id, sketch_class in m_s:
        if sketch_class == model_class:
            temp2.append(sketch_id)
    break  # Breaks after first model class is processed

print(temp1, temp2)

# for sketch_id in temp2:
#     sketch_id_padded = str(sketch_id).zfill(6)  # Convert "1" → "00001"
#     print(sketch_id_padded)
#     if sketch_id_padded not in temp1:
#         print("yooooooyouoyoyoy")
#         print(sketch_id)

'''

class ShapeData(Dataset):
    def __init__(self, sketch_dir, model_dir, sketch_file, model_file, label = "train",transform=None):
        self.sketch_dir = sketch_dir
        self.model_dir = model_dir
        self.transform = transform
        self.sketch_models, self.sketch_N = sketch_file
        self.models_3d, self.N_3d = model_file
        self.label = label
        # print(self.models_3d)

        self.pairs = []
        all_classes = set(self.sketch_models.keys()) & set(self.models_3d.keys())
        for class_name in all_classes:
            #positive pairs (target = 0)
            sketch_ids = self.sketch_models[class_name]
            model_ids = self.models_3d[class_name]
            if len(model_ids) == 0:
                    continue  
            # print("model_ids: ", model_ids)
            
            for i in sketch_ids:
                pos_ind = random.choice(model_ids)
                # print("pos_ind: ", pos_ind)
                self.pairs.append((i, pos_ind, class_name, 0))

            #negative pairs (target = 1)
            neg_classes = all_classes - {class_name}
            for i in sketch_ids:
                neg_cls = random.choice(list(neg_classes))
                # print("neg_cls: ", neg_cls)
                # print("model_ids neg: ", self.models_3d[neg_cls])   
                if len(self.models_3d[neg_cls]) == 0:
                    continue             
                neg_ind = random.choice(self.models_3d[neg_cls])
                # print("neg_ind: ", neg_ind)
                self.pairs.append((i, neg_ind, class_name, 1))         
        

    def __len__(self):
        return len(self.pairs)
    
    @lru_cache(maxsize=100)  # Store up to 1000 images in RAM
    def load_image(self, path):
        # print(f"Loading from disk: {path}")
        img = Image.open(path).convert("RGB")
        return img

    @lru_cache(maxsize=100)  # Store up to 1000 images in RAM
    def load_mesh(self, path):
        # print(f"Loading from disk: {path}")
        mesh = o3d.io.read_triangle_mesh(path)
        return mesh

    def __getitem__(self, index):
        sketch_id, model_id, class_name, target = self.pairs[index]
        # print("skt_id: ", sketch_id, "model_id: ", model_id, "class_name: ", class_name, "target: ", target)
        sketch_path = os.path.join(self.sketch_dir, f"{class_name}/{self.label}/{sketch_id}.png")
        model_path = os.path.join(self.model_dir, f"M{model_id}.off")

        # sketch = Image.open(sketch_path).convert("RGB")
        # mesh = o3d.io.read_triangle_mesh(model_path)
        sketch = self.load_image(sketch_path)
        mesh = self.load_mesh(model_path)
        if len(mesh.vertices) == 0:
            return None, None, None
        pcd = o3d.geometry.PointCloud()
        pcd = mesh.sample_points_poisson_disk(number_of_points=500, init_factor=5)
        # vertices_np = np.asarray(mesh.vertices)
        # pcd.points = o3d.utility.Vector3dVector(vertices_np)


        if self.transform:
            sketch = self.transform(sketch)

        return (sketch, torch.tensor(np.array(pcd.points)), torch.tensor(target))
    

if __name__ == "__main__":
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
        print(i)
        break

    tr_dataloader = DataLoader(tr_dataset, batch_size=4, shuffle=True)

    for i in te_dataset:
        print(i)
        print(torch.where(i[0] != 1.0))
        break

    



# class ShapeData(Dataset):
#     def __init__(self, sketch_dir, model_dir, sketch_file, model_file, label = "train",transform=None):
#         self.sketch_dir = sketch_dir
#         self.model_dir = model_dir
#         self.transform = transform
#         self.sketch_models, self.sketch_N = read_classification_file(sketch_file)
#         self.models_3d, self.N_3d = read_classification_file(model_file)
#         self.label = label

#         sketches_set = {}
#         models_set = {} 
#         for (sketch_id, sketch_class) in self.sketch_models:
#             sketches_set[sketch_class].set_default([]).append(sketch_id)

#         for (model_id, model_class) in self.models_3d:
#             models_set[model_class].set_default([]).append(model_id)


#         self.pairs = [
#             (sketch_id, model_id, class_name)
#             for class_name in set(self.sketch_class_map) & set(self.model_class_map)  # Intersection of classes
#             for sketch_id, model_id in product(self.sketch_class_map[class_name], self.model_class_map[class_name])
#         ]
        

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, index):
#         sketch_id, model_id, class_name = self.pairs[index]
#         sketch_path = os.path.join(self.sketch_dir, f"{class_name}/{self.label}/{sketch_id}.png")
#         model_path = os.path.join(self.model_dir, f"M{model_id}.off")

#         sketch = Image.open(sketch_path).convert("RGB")
#         mesh = o3d.io.read_triangle_mesh(model_path)
#         pcd = o3d.geometry.PointCloud()
#         pcd = mesh.sample_points_poisson_disk(number_of_points=500, init_factor=5)
#         # vertices_np = np.asarray(mesh.vertices)
#         # pcd.points = o3d.utility.Vector3dVector(vertices_np)


#         if self.transform:
#             sketch = self.transform(sketch)

#         return sketch, pcd


# dataset = ShapeData(
#     sketch_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/sketches_unzp/SHREC14LSSTB_SKETCHES/",
#     model_dir="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/target3d/SHREC14LSSTB_TARGET_MODELS/",
#     sketch_class_list="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_Sketch_Train.cla",
#     model_class_list="/nlsasfs/home/neol/rushar/scripts/img_to_pcd/shrec_data/sketches/SHREC14LSSTB_SKETCHES/SHREC14_SBR_models_train.cla",
#     label='Train',
#     transform=None  # You can add image transformations here
# )

# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Fetch a batch
# for sketches, models in dataloader:
#     print("Sketches Shape:", sketches.shape)  # (batch, C, H, W)
#     print("Models Shape:", models.shape)  # (batch, ...) depends on model data format
#     break

# class ShapeRetrievalDataset(Dataset):
#     def __init__(self, sketch_dir, model_dir, class_file, transform=None):
#         self.sketch_dir = sketch_dir
#         self.model_dir = model_dir
#         self.transform = transform

#         # Read classification file
#         self.modelclass, self.class_to_models = read_classification_file(class_file)

#         # Get available sketches
#         self.sketch_files = [f for f in os.listdir(sketch_dir) if f.endswith('.png')]

#         # Map sketches to classes based on filename prefix
#         self.sketch_to_class = {}
#         for sketch_file in self.sketch_files:
#             sketch_id = os.path.splitext(sketch_file)[0]  # Remove .png
#             for model_id, class_name in self.modelclass:
#                 if sketch_id.startswith(model_id):  # Match sketch ID with model ID prefix
#                     self.sketch_to_class[sketch_file] = class_name
#                     break

#     def __len__(self):
#         return len(self.sketch_files)

#     def __getitem__(self, idx):
#         sketch_file = self.sketch_files[idx]
#         class_name = self.sketch_to_class.get(sketch_file, "unknown")

#         # Load sketch
#         sketch_path = os.path.join(self.sketch_dir, sketch_file)
#         sketch = Image.open(sketch_path).convert("RGB")
#         if self.transform:
#             sketch = self.transform(sketch)

#         # Find a random 3D model from the same class
#         model_ids = self.class_to_models.get(class_name, [])
#         if model_ids:
#             model_id = random.choice(model_ids)  # Pick a random model from the class
#             model_path = os.path.join(self.model_dir, f"{model_id}.off")  # 3D file format
#             model = trimesh.load_mesh(model_path)
#         else:
#             model = None  # No matching model found

#         return sketch, model, class_name

# # Example usage
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# dataset = ShapeRetrievalDataset(
#     sketch_dir="path_to_sketches",
#     model_dir="path_to_3d_models",
#     class_file="your_sketch_file.cla",
#     transform=transform
# )

# # Load a sample
# sketch, model, class_name = dataset[0]
# print(class_name)  # Example output: 'airplane'