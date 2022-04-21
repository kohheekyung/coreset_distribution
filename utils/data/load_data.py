import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split

from utils.data.transforms import Transform, GT_Transform
import faiss
import numpy as np
from utils.common.embedding import embedding_concat

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
        img_fivecrop = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img_fivecrop.size()[-2], img_fivecrop.size()[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img_fivecrop.size()[-2:] == gt.size()[1:], "image.size != gt.size !!!"

        return img_fivecrop, gt, label, os.path.basename(img_path[:-4]), img_type

def Train_Dataloader(args):
    data_transforms = Transform(args.load_size, args.input_size)
    gt_transforms = GT_Transform(args.load_size, args.input_size)

    image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=data_transforms, gt_transform=gt_transforms, phase='train')
    train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) #, pin_memory=True)
    return train_loader

def Test_Dataloader(args):
    data_transforms = Transform(args.load_size, args.input_size)
    gt_transforms = GT_Transform(args.load_size, args.input_size)

    test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=data_transforms, gt_transform=gt_transforms, phase='test')
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=args.num_workers) #, pin_memory=True) # only work on batch_size=1, now.
    return test_loader

class Distribution_Dataset_Generator():
    def __init__(self, args):
        super(Distribution_Dataset_Generator, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.padding = args.dist_padding

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)
            
        if args.feature_model == 'R152' :
            self.feature_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        elif args.feature_model == 'R101' :
            self.feature_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        elif args.feature_model == 'R18' :
            self.feature_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        elif args.feature_model == 'R34' :
            self.feature_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        elif args.feature_model == 'R50' :
            self.feature_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        elif args.feature_model == 'WR50' :
            self.feature_model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        elif args.feature_model == 'WR101' :
            self.feature_model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True)

        for param in self.feature_model.parameters():
            param.requires_grad = False

        if args.block_index == '1+2' :
            self.feature_model.layer1[-1].register_forward_hook(hook_t)
            self.feature_model.layer2[-1].register_forward_hook(hook_t)
        elif args.block_index == '2+3' :
            self.feature_model.layer2[-1].register_forward_hook(hook_t)
            self.feature_model.layer3[-1].register_forward_hook(hook_t)
        elif args.block_index == '3+4' :
            self.feature_model.layer3[-1].register_forward_hook(hook_t)
            self.feature_model.layer4[-1].register_forward_hook(hook_t)
        elif args.block_index == '5' :
            self.feature_model.avgpool.register_forward_hook(hook_t)
        elif args.block_index == '4' :
            self.feature_model.layer4[-1].register_forward_hook(hook_t)

        self.feature_model.to(self.device)
        self.feature_model.eval()

        self.embedding_list = []
        self.embedding_indices_list = []

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.feature_model(x_t)
        return self.features

    def make_embedding_list(self, embedding, embedding_indices):
        # embedding : N x E x W x H
        # embedding_indices : N x 1 x W x H
        embedding_list = []
        embedding_indices_list = []

        for k in range(embedding_indices.shape[0]):
            embedding_indices_list.append(embedding_indices[k])
            embedding_list.append(embedding[k])

        return embedding_list, embedding_indices_list

    def generate(self, dataloader):
        self.embedding_dir_path = os.path.join('./', f'embeddings_{self.args.block_index}', self.args.category)
        self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}.faiss'))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.dist_coreset_index = faiss.index_cpu_to_gpu(res, 0 ,self.dist_coreset_index)

        for iter, batch in enumerate(dataloader):
            img_fivecrop, _, _, _, _ = batch
            img_fivecrop = img_fivecrop.to(self.device)
            bs, ncrops, c, w, h = img_fivecrop.shape
            features = self.forward(img_fivecrop.view(-1, c, w, h))
            
            if '+' in self.args.block_index : 
                embeddings = []
                m = torch.nn.AvgPool2d(3, 1, 1)
                for feature in features:
                    embeddings.append(m(feature))
                embedding_ = np.array(embedding_concat(embeddings[0], embeddings[1])) # (N x ncrops) x E x W x H
            else :
                embedding_ = np.array(features[0].cpu())

            # find index of embedding vector which is closest to self.dist_coreset_index
            embedding_t = embedding_.transpose(0,2,3,1) # (N x ncrops) x W x H x E
            embedding_list = embedding_t.reshape(-1, embedding_t.shape[-1]) # (N x ncrops x W x H) x E

            _, embedding_indices = self.dist_coreset_index.search(embedding_list, k=1) # (N x ncrops x W x H) x 1
            embedding_indices = embedding_indices.reshape(embedding_t.shape[0:3] + (1,)).transpose(0,3,1,2) # (N x ncrops) x 1 x W x H

            embedding_list_, embedding_indices_list_ = self.make_embedding_list(embedding_, embedding_indices)
            self.embedding_list.extend(embedding_list_)
            self.embedding_indices_list.extend(embedding_indices_list_)
            
    def get_data_size(self):
        input_size = self.dist_coreset_index.reconstruct(0).shape[0] * (pow(self.padding*2 + 1, 2) - 1)
        output_size = self.dist_coreset_index.ntotal
        return input_size, output_size

    def __len__(self):
        return len(self.embedding_indices_list) * np.prod(self.embedding_indices_list[0].shape[1:])

    def __getitem__(self, idx):
        len_list, len_i, len_j = len(self.embedding_indices_list), self.embedding_indices_list[0].shape[1], self.embedding_indices_list[0].shape[2]

        idx, j_idx = idx // len_j, idx % len_j
        list_idx, i_idx = idx // len_i, idx % len_i

        embedding = self.embedding_list[list_idx] # E x W x H
        embedding_indices = self.embedding_indices_list[list_idx] # 1 x W x H
        
        pad_width = ((0,),(self.padding,),(self.padding,))
        embedding_pad = np.pad(embedding, pad_width, "reflect") # E x (W+1) x (H+1)

        index = embedding_indices[0, i_idx, j_idx]
        neighbor = np.zeros(shape=(0,))
        for di in range(-self.padding, self.padding+1) :
            for dj in range(-self.padding, self.padding+1) :
                if di == 0 and dj == 0 :
                    continue
                neighbor = np.concatenate((neighbor, embedding_pad[:, i_idx+di+self.padding, j_idx+dj+self.padding]))
        return neighbor.astype(np.float32), index
    
def Distribution_Train_Dataloader(args, dataloader):
    distribution_dataset_generator = Distribution_Dataset_Generator(args)
    distribution_dataset_generator.generate(dataloader)
    dist_input_size, dist_output_size = distribution_dataset_generator.get_data_size()
    
    val_size = int(len(distribution_dataset_generator) * 0.1)
    train_size = len(distribution_dataset_generator) - val_size
    train_dataset, val_dataset = random_split(distribution_dataset_generator, [train_size, val_size])

    distribution_train_dataloader= DataLoader(train_dataset, batch_size=args.dist_batch_size, shuffle=True, num_workers=args.num_workers) #, pin_memory=True)
    distribution_val_dataloader= DataLoader(val_dataset, batch_size=args.dist_batch_size, shuffle=False, num_workers=args.num_workers) #, pin_memory=True)
    return distribution_train_dataloader, distribution_val_dataloader, dist_input_size, dist_output_size