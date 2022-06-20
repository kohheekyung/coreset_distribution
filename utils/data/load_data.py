import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

from utils.data.transforms import Transform, GT_Transform
import faiss
import numpy as np
from utils.common.embedding import embedding_concat, generate_embedding_features
from utils.common.image_processing import PatchMaker, ForwardHook, LastLayerToExtractReachedException
from utils.common.backbones import Backbone

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type

def Train_Dataloader(args):
    data_transforms = Transform(args.resize, args.imagesize)
    gt_transforms = GT_Transform(args.resize, args.imagesize)

    image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=data_transforms, gt_transform=gt_transforms, phase='train')
    train_loader = DataLoader(image_datasets, batch_size=4, shuffle=True, num_workers=args.num_workers) #, pin_memory=True)
    return train_loader

def Test_Dataloader(args):
    data_transforms = Transform(args.resize, args.imagesize)
    gt_transforms = GT_Transform(args.resize, args.imagesize)

    test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=data_transforms, gt_transform=gt_transforms, phase='test')
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=args.num_workers) #, pin_memory=True) # only work on batch_size=1, now.
    return test_loader

class Distribution_Dataset_Generator():
    def __init__(self, args):
        super(Distribution_Dataset_Generator, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.dist_padding = args.dist_padding
        
        self.backbone = Backbone(args.backbone) # load pretrained backbone model
        self.backbone.to(self.device)
        self.backbone.eval()
            
        self.patch_maker = PatchMaker(args.patchsize, stride=1)

        if not hasattr(self.backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}
            
        for extract_layer in args.layer_index :
            forward_hook = ForwardHook(
                self.outputs, extract_layer, args.layer_index[-1]
            )
            network_layer = self.backbone.__dict__["_modules"][extract_layer]
            
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )

        self.embedding_pad_list = []
        self.embedding_indices_list = []

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def generate(self, dataloader):
        self.embedding_dir_path = os.path.join('./', f'embeddings_{"+".join(self.args.layer_index)}', self.args.category)
        
        self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}.faiss'))
        if self.args.position_encoding_in_distribution :
            self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}_pe.faiss'))
            
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.dist_coreset_index = faiss.index_cpu_to_gpu(res, 0 ,self.dist_coreset_index)

        for iter, batch in enumerate(dataloader):
            x, _, _, _, _ = batch
            x = x.to(self.device)
            batchsize = x.shape[0]

            features = self.forward(x)
            
            features, ref_num_patches = generate_embedding_features(self.args, features, self.patch_maker)
            features = features.detach().cpu().numpy()
            
            if self.args.position_encoding_in_distribution :
                W, H = ref_num_patches
                position_encoding = np.zeros(shape=(1, W, H, 2))
                for i in range(W) :
                    for j in range(H) : 
                        position_encoding[0, i, j, 0] = self.args.pe_weight * i / W
                        position_encoding[0, i, j, 1] = self.args.pe_weight * j / H
                        
                position_encoding = position_encoding.reshape(-1, 2)
                position_encoding = np.tile(position_encoding, (batchsize, 1)).astype(np.float32)
                
                position_encoding = position_encoding.reshape(-1, position_encoding.shape[-1])

                features = np.concatenate((features, position_encoding), axis = 1)
            
            _, embedding_indices = self.dist_coreset_index.search(features, k=1)
            
            features = features.reshape((batchsize, ref_num_patches[0], ref_num_patches[1], -1))
            embedding_indices = embedding_indices.reshape((batchsize, ref_num_patches[0], ref_num_patches[1]))
            
            pad_width = ((0,),(self.dist_padding,),(self.dist_padding,), (0,))
            #embedding_pad = np.pad(features, pad_width, "reflect")
            embedding_pad = np.pad(features, pad_width, "constant")
            
            self.embedding_pad_list.extend([x for x in embedding_pad])
            self.embedding_indices_list.extend([x for x in embedding_indices])
            
    def get_data_size(self):
        input_size = self.embedding_pad_list[0].shape[2] * (pow(self.dist_padding * 2 + 1, 2) - 1)
        output_size = self.args.dist_coreset_size
        return input_size, output_size

    def __len__(self):
        len_i, len_j = self.embedding_indices_list[0].shape[:2]
        if self.args.cut_edge_embedding : 
            len_i = len_i - self.args.patchsize + 1
            len_j = len_j - self.args.patchsize + 1
        
        return len(self.embedding_indices_list) * len_i * len_j

    def __getitem__(self, idx):
        len_list, len_i, len_j = len(self.embedding_indices_list), self.embedding_indices_list[0].shape[0], self.embedding_indices_list[0].shape[1]

        if self.args.cut_edge_embedding : 
            len_i = len_i - self.args.patchsize + 1
            len_j = len_j - self.args.patchsize + 1

        idx, j_idx = idx // len_j, idx % len_j
        list_idx, i_idx = idx // len_i, idx % len_i
        
        if self.args.cut_edge_embedding : 
            i_idx = i_idx + (self.args.patchsize - 1) // 2
            j_idx = j_idx + (self.args.patchsize - 1) // 2

        embedding_pad = self.embedding_pad_list[list_idx] # (W+1) x (H+1) x E
        embedding_indices = self.embedding_indices_list[list_idx] # W x H
        
        index = embedding_indices[i_idx, j_idx]
        
        # delete middle features in neighbor
        neighbor = embedding_pad[i_idx:i_idx + self.dist_padding * 2 + 1, j_idx:j_idx + self.dist_padding * 2 + 1].reshape(-1)        
        mid_index = (pow(self.dist_padding * 2 + 1, 2) + 1) // 2        
        neighbor = np.concatenate([neighbor[:self.embedding_pad_list[0].shape[2]*mid_index], neighbor[self.embedding_pad_list[0].shape[2]*(mid_index+1):]])
        
        return neighbor.astype(np.float32), index
    
def Distribution_Train_Dataloader(args, dataloader):
    distribution_dataset_generator = Distribution_Dataset_Generator(args)
    distribution_dataset_generator.generate(dataloader)
    dist_input_size, dist_output_size = distribution_dataset_generator.get_data_size()
    
    val_size = int(len(distribution_dataset_generator) * 0.1)
    train_size = len(distribution_dataset_generator) - val_size
    train_dataset, val_dataset = random_split(distribution_dataset_generator, [train_size, val_size])

    distribution_train_dataloader= DataLoader(train_dataset, batch_size=args.dist_batchsize, shuffle=True, num_workers=args.num_workers) #, pin_memory=True)
    distribution_val_dataloader= DataLoader(val_dataset, batch_size=args.dist_batchsize, shuffle=False, num_workers=args.num_workers) #, pin_memory=True)
    return distribution_train_dataloader, distribution_val_dataloader, dist_input_size, dist_output_size

class Coor_Distribution_Dataset_Generator():
    def __init__(self, args):
        super(Coor_Distribution_Dataset_Generator, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.dist_padding = args.dist_padding
        
        self.backbone = Backbone(args.backbone) # load pretrained backbone model
        self.backbone.to(self.device)
        self.backbone.eval()
            
        self.patch_maker = PatchMaker(args.patchsize, stride=1)

        if not hasattr(self.backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}
            
        for extract_layer in args.layer_index :
            forward_hook = ForwardHook(
                self.outputs, extract_layer, args.layer_index[-1]
            )
            network_layer = self.backbone.__dict__["_modules"][extract_layer]
            
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.embedding_indices_list = []

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def generate(self, dataloader):
        self.embedding_dir_path = os.path.join('./', f'embeddings_{"+".join(self.args.layer_index)}', self.args.category)
        
        # self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}.faiss'))
        # if self.args.position_encoding_in_distribution :
        #     self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}_pe.faiss'))
        self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'embedding_coreset_index_{int(self.args.subsampling_percentage*100)}.faiss'))
        if self.args.position_encoding_in_distribution :
            self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'embedding_coreset_index_{int(self.args.subsampling_percentage*100)}_pe.faiss'))
            
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.dist_coreset_index = faiss.index_cpu_to_gpu(res, 0 ,self.dist_coreset_index)

        for iter, batch in enumerate(dataloader):
            x, _, _, _, _ = batch
            x = x.to(self.device)
            batchsize = x.shape[0]

            features = self.forward(x)
            
            features, ref_num_patches = generate_embedding_features(self.args, features, self.patch_maker)
            features = features.detach().cpu().numpy()
            
            if self.args.position_encoding_in_distribution :
                W, H = ref_num_patches
                position_encoding = np.zeros(shape=(1, W, H, 2))
                for i in range(W) :
                    for j in range(H) : 
                        position_encoding[0, i, j, 0] = self.args.pe_weight * i / W
                        position_encoding[0, i, j, 1] = self.args.pe_weight * j / H
                        
                position_encoding = position_encoding.reshape(-1, 2)
                position_encoding = np.tile(position_encoding, (batchsize, 1)).astype(np.float32)
                
                position_encoding = position_encoding.reshape(-1, position_encoding.shape[-1])

                features = np.concatenate((features, position_encoding), axis = 1)
            
            _, embedding_indices = self.dist_coreset_index.search(features, k=1)
            
            embedding_indices = embedding_indices.reshape((batchsize, ref_num_patches[0], ref_num_patches[1]))
            
            self.embedding_indices_list.extend([x for x in embedding_indices])
            
    def get_data_size(self):
        input_size = (self.embedding_indices_list[0].shape[0], self.embedding_indices_list[0].shape[1])
        output_size = self.dist_coreset_index.ntotal
        return input_size, output_size

    def __len__(self):
        len_i, len_j = self.embedding_indices_list[0].shape[:2]
        if self.args.cut_edge_embedding : 
            len_i = len_i - self.args.patchsize + 1
            len_j = len_j - self.args.patchsize + 1
        
        return len(self.embedding_indices_list) * len_i * len_j

    def __getitem__(self, idx):
        len_list, len_i, len_j = len(self.embedding_indices_list), self.embedding_indices_list[0].shape[0], self.embedding_indices_list[0].shape[1]
        
        if self.args.cut_edge_embedding : 
            len_i = len_i - self.args.patchsize + 1
            len_j = len_j - self.args.patchsize + 1

        idx, j_idx = idx // len_j, idx % len_j
        list_idx, i_idx = idx // len_i, idx % len_i
        
        if self.args.cut_edge_embedding : 
            i_idx = i_idx + (self.args.patchsize - 1) // 2
            j_idx = j_idx + (self.args.patchsize - 1) // 2

        embedding_indices = self.embedding_indices_list[list_idx] # W x H
        
        index = embedding_indices[i_idx, j_idx]
        coordinate = np.array([i_idx, j_idx]).astype(np.float32)
        
        return coordinate, index

def Coor_Distribution_Train_Dataloader(args, dataloader):
    distribution_dataset_generator = Coor_Distribution_Dataset_Generator(args)
    distribution_dataset_generator.generate(dataloader)
    dist_input_size, dist_output_size = distribution_dataset_generator.get_data_size()

    distribution_train_dataloader= DataLoader(distribution_dataset_generator, batch_size=8192, shuffle=False, num_workers=args.num_workers) #, pin_memory=True)
    return distribution_train_dataloader, dist_input_size, dist_output_size