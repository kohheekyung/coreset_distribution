from pyrsistent import b
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
import cv2
import os
import numpy as np
import shutil
import pytorch_lightning as pl
import faiss
import copy
from sklearn.random_projection import SparseRandomProjection
from utils.sampling_methods.kcenter_greedy import kCenterGreedy
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from utils.common.visualize import visualize_TSNE
from utils.data.transforms import INV_Normalize
from utils.common.embedding import generate_embedding_features, embedding_concat, reshape_embedding
from utils.learning.model import Distribution_Model
from utils.common.image_processing import PatchMaker, ForwardHook, LastLayerToExtractReachedException
from utils.common.backbones import Backbone

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min)/(a_max - a_min) 

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)

def prep_dirs(root, args):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    embeddings_path = os.path.join('./', f'embeddings_{"+".join(args.layer_index)}', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE', 'embeddings', 'result']) # copy source code
    return embeddings_path, sample_path, source_code_save_path

def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)

def calc_prob_embedding(distances, gamma):
    prob_embedding = gamma * np.exp(-gamma*distances)
    return prob_embedding

class Coreset(pl.LightningModule):
    def __init__(self, args):
        super(Coreset, self).__init__()

        self.args = args
        
        self.backbone = Backbone(args.backbone) # load pretrained backbone model
        
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

    def on_train_start(self):
        self.backbone.eval()
        self.embedding_dir_path = os.path.join('./', f'embeddings_{"+".join(self.args.layer_index)}', self.args.category)
        os.makedirs(self.embedding_dir_path, exist_ok=True)
        self.embedding_list = []
        self.embedding_pe_list = []

    def training_step(self, batch, batch_idx):
        x, _, _, _, _ = batch
        
        batchsize = x.shape[0]
           
        features = self(x)
        
        features, ref_num_patches = generate_embedding_features(self.args, features, self.patch_maker)
        features = features.detach().cpu().numpy()
        if self.args.cut_edge_embedding :
            features_cut = features.reshape(batchsize, ref_num_patches[0], ref_num_patches[1], -1) # N x W x H x E
            patch_padding = (self.args.patchsize - 1) // 2
            features_cut = features_cut[:, patch_padding:features_cut.shape[1]-patch_padding, patch_padding:features_cut.shape[2]-patch_padding, :] # N x (W - p) x (H - p) x E
            features_cut = features_cut.reshape(-1, features_cut.shape[-1]) # (N x (W - p) x (H - p)) x E
            self.embedding_list.extend([x for x in features_cut])
        else :
            self.embedding_list.extend([x for x in features])
        
        ## coreset using position encoding
        W, H = ref_num_patches
        position_encoding = np.zeros(shape=(1, W, H, 2))
        for i in range(W) :
            for j in range(H) : 
                position_encoding[0, i, j, 0] = self.args.pe_weight * i / W
                position_encoding[0, i, j, 1] = self.args.pe_weight * j / H
                
        position_encoding = position_encoding.reshape(-1, 2) # (1 x W x H) x 2
        position_encoding = np.tile(position_encoding, (batchsize, 1)).astype(np.float32) # (N x W x H) x 2

        features_pe = np.concatenate((features, position_encoding), axis = 1) # (N x W x H) x (E + 2)
        
        if self.args.cut_edge_embedding :
            features_pe_cut = features_pe.reshape(batchsize, ref_num_patches[0], ref_num_patches[1], -1) # N x W x H x (E + 2)
            patch_padding = (self.args.patchsize - 1) // 2
            features_pe_cut = features_pe_cut[:, patch_padding:features_pe_cut.shape[1]-patch_padding, patch_padding:features_pe_cut.shape[2]-patch_padding, :] # N x (W - p) x (H - p) x E
            features_pe_cut = features_pe_cut.reshape(-1, features_pe_cut.shape[-1])
            self.embedding_pe_list.extend([x for x in features_pe_cut])
        else :
            self.embedding_pe_list.extend([x for x in features_pe])

    def training_epoch_end(self, outputs):
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        #self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector = SparseRandomProjection(n_components=128)
        self.randomprojector.fit(total_embeddings)
        
        # Coreset Subsampling
        embedding_coreset_size = int(self.args.subsampling_percentage * total_embeddings.shape[0])
        dist_coreset_size = self.args.dist_coreset_size
        max_coreset_size = max(embedding_coreset_size, dist_coreset_size)

        selector = kCenterGreedy(embedding=torch.Tensor(total_embeddings), sampling_size=max_coreset_size)
        selected_idx = selector.select_coreset_idxs()
        self.embedding_coreset = total_embeddings[selected_idx][:embedding_coreset_size]
        self.dist_coreset = total_embeddings[selected_idx][:dist_coreset_size]
        
        # save to faiss
        self.embedding_coreset_index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.embedding_coreset_index.add(self.embedding_coreset)
        faiss.write_index(self.embedding_coreset_index, os.path.join(self.embedding_dir_path,f'embedding_coreset_index_{int(self.args.subsampling_percentage*100)}.faiss'))

        self.dist_coreset_index = faiss.IndexFlatL2(self.dist_coreset.shape[1])
        self.dist_coreset_index.add(self.dist_coreset)
        faiss.write_index(self.dist_coreset_index, os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}.faiss'))
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding coreset size : ', self.embedding_coreset.shape)
        print('final dist coreset size : ', self.dist_coreset.shape)

        ## coreset using position encoding
        total_embeddings_pe = np.array(self.embedding_pe_list)
        # Random projection
        # self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector = SparseRandomProjection(n_components=128) 
        self.randomprojector.fit(total_embeddings_pe)
        
        # Coreset Subsampling
        embedding_coreset_size = int(self.args.subsampling_percentage * total_embeddings_pe.shape[0])
        dist_coreset_size = self.args.dist_coreset_size
        max_coreset_size = max(embedding_coreset_size, dist_coreset_size)

        selector = kCenterGreedy(embedding=torch.Tensor(total_embeddings_pe), sampling_size=max_coreset_size)
        selected_idx = selector.select_coreset_idxs()
        self.embedding_coreset = total_embeddings_pe[selected_idx][:embedding_coreset_size]
        self.dist_coreset = total_embeddings_pe[selected_idx][:dist_coreset_size]
        
        # save to faiss
        self.embedding_coreset_index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.embedding_coreset_index.add(self.embedding_coreset)
        faiss.write_index(self.embedding_coreset_index, os.path.join(self.embedding_dir_path,f'embedding_coreset_index_{int(self.args.subsampling_percentage*100)}_pe.faiss'))

        self.dist_coreset_index = faiss.IndexFlatL2(self.dist_coreset.shape[1])
        self.dist_coreset_index.add(self.dist_coreset)
        faiss.write_index(self.dist_coreset_index, os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}_pe.faiss'))
        
        print('initial embedding_pe size : ', total_embeddings_pe.shape)
        print('final embedding coreset_pe size : ', self.embedding_coreset.shape)
        print('final dist coreset_pe size : ', self.dist_coreset.shape)

    def configure_optimizers(self):
        return None

class Distribution(pl.LightningModule):
    def __init__(self, args, dist_input_size, dist_output_size):
        super(Distribution, self).__init__()

        self.args = args
        self.model = Distribution_Model(args, dist_input_size, dist_output_size)
        self.embedding_dir_path = os.path.join('./', f'embeddings_{"+".join(self.args.layer_index)}', self.args.category)
        os.makedirs(self.embedding_dir_path, exist_ok=True)
        self.best_val_loss=1e+6
        
        self.train_loss = 0.0
        self.train_size = 0
        self.val_loss = 0.0
        self.val_size = 0

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        self.train_loss = 0.0
        self.train_size = 0
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.train_loss += loss * x.shape[0]
        self.train_size += x.shape[0]
        return loss
        
    def train_epoch_end(self, outputs):
        self.train_loss = self.train_loss / self.train_size

    def on_validation_epoch_start(self):
        self.val_loss = 0.0
        self.val_size = 0

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, prog_bar=True)
        self.val_loss += loss * x.shape[0]
        self.val_size += x.shape[0]
        return loss
        
    def validation_epoch_end(self, outputs):
        self.val_loss = self.val_loss / self.val_size
        
        torch.save(
            {
                'args': self.args,
                'model': self.model.state_dict(),
                'train_loss': self.train_loss,
                'val_loss': self.val_loss
            },
            f=os.path.join(self.embedding_dir_path, f'model_{self.args.dist_padding}_{self.args.dist_coreset_size}.pt')
        )
        
        if self.best_val_loss > self.val_loss :
            self.best_val_loss = self.val_loss
            shutil.copyfile(os.path.join(self.embedding_dir_path, f'model_{self.args.dist_padding}_{self.args.dist_coreset_size}.pt'), os.path.join(self.embedding_dir_path, f'best_model_{self.args.dist_padding}_{self.args.dist_coreset_size}.pt'))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=0.1)
        return [optimizer], [scheduler]
    
class Coor_Distribution():
    def __init__(self, args, coor_dist_input_size, coor_dist_output_size):
        super(Coor_Distribution, self).__init__()
        self.args = args
        self.embedding_dir_path = os.path.join('./', f'embeddings_{"+".join(self.args.layer_index)}', self.args.category)
        os.makedirs(self.embedding_dir_path, exist_ok=True)
        
        self.coor_dist_input_size = coor_dist_input_size
        self.coor_dist_output_size = coor_dist_output_size
        self.coor_model = np.zeros(shape = (coor_dist_input_size[0], coor_dist_input_size[1], coor_dist_output_size), dtype=np.float32)
        #self.coor_model_save_path = os.path.join(self.embedding_dir_path, f'coor_model_{self.args.dist_coreset_size}.npy')
        self.coor_model_save_path = os.path.join(self.embedding_dir_path, f'coor_model_{int(self.args.subsampling_percentage*100)}.npy')
        self.dist_padding = args.dist_padding
        
    def fit(self, train_dataloader) :
        for iter, batch in enumerate(train_dataloader):
            coordinate, index = batch
            coordinate = coordinate.numpy().astype(int)
            index = index.numpy().astype(int)
            for i in range(len(index)) :
                coor_x_min = max(0, coordinate[i][0] - self.dist_padding)
                coor_x_max = min(self.coor_dist_input_size[0] - 1, coordinate[i][0] + self.dist_padding)
                coor_y_min = max(0, coordinate[i][1] - self.dist_padding)
                coor_y_max = min(self.coor_dist_input_size[1] - 1, coordinate[i][1] + self.dist_padding)

                self.coor_model[coor_x_min:coor_x_max+1, coor_y_min:coor_y_max+1, index[i]] += 1.0
                
        self.coor_model /= np.sum(self.coor_model, axis = 2).reshape(self.coor_dist_input_size[0], self.coor_dist_input_size[1], 1)
        self.coor_model = self.coor_model.reshape(-1, self.coor_model.shape[-1])
        np.save(self.coor_model_save_path, self.coor_model)        
    
class AC_Model(pl.LightningModule):
    def __init__(self, args, dist_input_size, dist_output_size):
        super(AC_Model, self).__init__()
        
        self.save_hyperparameters(args)
        self.args = args
        
        self.backbone = Backbone(args.backbone) # load pretrained backbone model

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

        self.init_results_list()

        self.inv_normalize = INV_Normalize()
        self.embedding_dir_path = os.path.join('./', f'embeddings_{"+".join(self.args.layer_index)}', self.args.category)
        
        if not args.not_use_coreset_distribution:
            self.dist_model = Distribution_Model(args, dist_input_size, dist_output_size)        
            self.dist_model.load_state_dict(torch.load(os.path.join(self.embedding_dir_path, f'best_model_{self.args.dist_padding}_{self.args.dist_coreset_size}.pt'))['model'])
        if args.use_coordinate_distribution :
            self.coor_dist_model = np.load(os.path.join(self.embedding_dir_path, f'coor_model_{int(self.args.subsampling_percentage*100)}.npy'))
        
        self.position_encoding_in_dsitribution = args.position_encoding_in_distribution

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.pred_list_px_lvl_topk1 = []
        self.pred_list_px_lvl_patchcore = []
        self.pred_list_px_lvl_pe = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.pred_list_img_lvl_topk1 = []
        self.pred_list_img_lvl_patchcore = []
        self.pred_list_img_lvl_pe = []
        self.img_path_list = []
        self.img_type_list = []

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

    def save_anomaly_map(self, anomaly_map, anomaly_map_topk1, anomaly_map_patchcore, anomaly_map_pe, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        if anomaly_map_topk1.shape != input_img.shape:
            anomaly_map_topk1 = cv2.resize(anomaly_map_topk1, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)
        anomaly_map_topk1_norm = min_max_norm(anomaly_map_topk1)
        anomaly_map_topk1_norm_hm = cvt2heatmap(anomaly_map_topk1_norm*255)

        anomaly_map_norm_patchcore = min_max_norm(anomaly_map_patchcore)
        anomaly_map_norm_hm_patchcore = cvt2heatmap(anomaly_map_norm_patchcore*255)
        anomaly_map_norm_pe = min_max_norm(anomaly_map_pe)
        anomaly_map_norm_hm_pe = cvt2heatmap(anomaly_map_norm_pe*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)
        heatmap_topk1 = cvt2heatmap(anomaly_map_topk1_norm*255)
        hm_topk1_on_img = heatmap_on_image(heatmap_topk1, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_topk1.jpg'), anomaly_map_topk1_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_patchcore.jpg'), anomaly_map_norm_hm_patchcore)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_pe.jpg'), anomaly_map_norm_hm_pe)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_topk1_on_img.jpg'), hm_topk1_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def configure_optimizers(self):
        return None
    
    def on_test_start(self):
        self.backbone.eval() # to stop running_var move (maybe not critical)
        if not self.args.not_use_coreset_distribution:
            self.dist_model.eval() # to stop running_var move (maybe not critical)
        
        self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}.faiss'))
        if self.args.position_encoding_in_distribution :
            self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}_pe.faiss'))  
        self.embedding_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'embedding_coreset_index_{int(self.args.subsampling_percentage*100)}.faiss'))
        self.embedding_coreset_pe_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'embedding_coreset_index_{int(self.args.subsampling_percentage*100)}_pe.faiss'))
        
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.dist_coreset_index = faiss.index_cpu_to_gpu(res, 0, self.dist_coreset_index)
            self.embedding_coreset_index = faiss.index_cpu_to_gpu(res, 0, self.embedding_coreset_index)
            self.embedding_coreset_pe_index = faiss.index_cpu_to_gpu(res, 0, self.embedding_coreset_pe_index)
            
        self.embedding_coreset_index_cpu = faiss.index_gpu_to_cpu(self.embedding_coreset_index)

        self.init_results_list()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir, self.args)
        self.pe_pad = None
        self.position_encoding = None

    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x, gt, label, file_name, x_type = batch
        
        batchsize = x.shape[0]
        
        features = self(x)
        
        features, ref_num_patches = generate_embedding_features(self.args, features, self.patch_maker)
            
        embedding_test = features.detach().cpu().numpy() # (W x H) x E

        W, H = ref_num_patches
        
        if self.position_encoding is None :
            self.position_encoding = np.zeros(shape=(1, W, H, 2)) # 1 x W x H x 2
            for i in range(W) :
                for j in range(H) : 
                    self.position_encoding[0, i, j, 0] = self.args.pe_weight * i / W
                    self.position_encoding[0, i, j, 1] = self.args.pe_weight * j / H
                    
        if self.pe_pad is None :
            pad_width = ((self.args.dist_padding,),(self.args.dist_padding,), (0,))
            self.pe_pad = np.pad(self.position_encoding.squeeze(0), pad_width, "reflect", reflect_type='odd') # (W+1) x (H+1) x 2
            
        if not self.args.not_use_coreset_distribution:
            pad_width = ((self.args.dist_padding,),(self.args.dist_padding,), (0,))         
            #embedding_pad = np.pad(embedding_test.reshape(W, H, -1), pad_width, "reflect") # (W+1) x (H+1) x E
            embedding_pad = np.pad(embedding_test.reshape(W, H, -1), pad_width, "constant") # (W+1) x (H+1) x E
            if self.position_encoding_in_dsitribution :
               embedding_pad = np.concatenate((embedding_pad, self.pe_pad), axis = 2) # (W+1) x (H+1) x (E+2)
                                
            neighbors = np.zeros(shape=(W, H, embedding_pad.shape[2]*(pow(self.args.dist_padding*2+1, 2) - 1))) # W x H x NE
            # construct neighbor features
            for i_idx in range(W) :
                for j_idx in range(H) :
                    # delete middle features in neighbor
                    neighbor = embedding_pad[i_idx:i_idx + self.args.dist_padding * 2 + 1, j_idx:j_idx + self.args.dist_padding * 2 + 1].reshape(-1)
                    mid_index = (pow(self.args.dist_padding * 2 + 1, 2) + 1) // 2
                    neighbor = np.concatenate([neighbor[:embedding_pad.shape[2]*mid_index], neighbor[embedding_pad.shape[2]*(mid_index+1):]])
        
                    neighbors[i_idx, j_idx] = neighbor

        ## patchcore        
        embedding_score, embedding_indices = self.embedding_coreset_index.search(embedding_test, k=self.args.anomaly_nn) # (W x H) x self.args.n_neighbors
        embedding_score = np.sqrt(embedding_score)
        if self.args.cut_edge_embedding :
            embedding_score = embedding_score.reshape((ref_num_patches[0], ref_num_patches[1], -1))
            patch_padding = (self.args.patchsize - 1) // 2
            embedding_score = embedding_score[patch_padding:embedding_score.shape[0]-patch_padding, patch_padding:embedding_score.shape[1]-patch_padding, :]
            embedding_score = embedding_score.reshape(-1, embedding_score.shape[-1])
        
        max_anomaly_idx = np.argmax(embedding_score[:, 0])
        max_embedding_score = embedding_score[max_anomaly_idx, 0] # maximum embedding score
        if self.args.anomaly_nn == 1 :
            weights_from_code = 1
        else :
            weights_from_code = 1 - np.exp(max_embedding_score) / np.sum(np.exp(embedding_score[max_anomaly_idx]))

        anomaly_img_score_patchcore = weights_from_code * max_embedding_score # Image-level score
        #anomaly_img_score_patchcore = max_embedding_score # Image-level score
        
        if self.args.cut_edge_embedding :
            anomaly_map_patchcore = embedding_score[:, 0].reshape((ref_num_patches[0] - 2 * patch_padding, ref_num_patches[1] - 2 * patch_padding))
            pad_width = ((patch_padding,),(patch_padding,))
            anomaly_map_patchcore = np.pad(anomaly_map_patchcore, pad_width, 'edge')
        else : 
            anomaly_map_patchcore = embedding_score[:, 0].reshape(ref_num_patches)
        
        ## patchcore using position encoding
        position_encoding_reshape = self.position_encoding.reshape(-1, 2)
        embedding_pe_test = np.concatenate((embedding_test, position_encoding_reshape), axis = 1).astype(np.float32)

        embedding_pe_score, embedding_pe_indices = self.embedding_coreset_pe_index.search(embedding_pe_test, k=self.args.anomaly_nn) # (W x H) x self.args.n_neighbors
        embedding_pe_score = np.sqrt(embedding_pe_score)
        
        if self.args.cut_edge_embedding :
            embedding_pe_score = embedding_pe_score.reshape((ref_num_patches[0], ref_num_patches[1], -1))
            patch_padding = (self.args.patchsize - 1) // 2
            embedding_pe_score = embedding_pe_score[patch_padding:embedding_pe_score.shape[0]-patch_padding, patch_padding:embedding_pe_score.shape[1]-patch_padding, :]
            embedding_pe_score = embedding_pe_score.reshape(-1, embedding_pe_score.shape[-1])
        
        max_anomaly_idx = np.argmax(embedding_pe_score[:, 0])
        max_embedding_score = embedding_pe_score[max_anomaly_idx, 0] # maximum embedding score
        if self.args.anomaly_nn == 1 :
            weights_from_code = 1
        else :
            weights_from_code = 1 - np.exp(max_embedding_score) / np.sum(np.exp(embedding_pe_score[max_anomaly_idx]))

        anomaly_img_score_pe= weights_from_code * max_embedding_score # Image-level score
        #anomaly_img_score_pe= max_embedding_score # Image-level score
        if self.args.cut_edge_embedding :
            anomaly_map_pe = embedding_pe_score[:, 0].reshape((ref_num_patches[0] - 2 * patch_padding, ref_num_patches[1] - 2 * patch_padding))
            pad_width = ((patch_padding,),(patch_padding,))
            anomaly_map_pe = np.pad(anomaly_map_pe, pad_width, 'edge')
        else : 
            anomaly_map_pe = embedding_pe_score[:, 0].reshape(ref_num_patches)

        ## using neighbor distribution
        anomaly_img_score_nb = anomaly_img_score_topk1 = anomaly_img_score_patchcore
        anomaly_map_nb = anomaly_map_topk1 = anomaly_map_patchcore
        if not self.args.not_use_coreset_distribution:
            neighbors = neighbors.reshape(-1, neighbors.shape[2]).astype(np.float32) # (W x H) x NE
            '''
            y_hat = self.dist_model(torch.tensor(neighbors).cuda()).cpu() # (W x H) x self.dist_coreset_index.ntotal
            '''
            
            if self.args.use_coordinate_distribution :
                softmax_coor = self.coor_dist_model
            
            '''
            softmax_temp = F.softmax(y_hat / self.args.softmax_temperature_alpha, dim = -1).cpu().numpy() # (W x H) x self.dist_coreset_indesx.ntotal

            softmax_thres = softmax_temp  > self.args.softmax_thres_gamma / 2048 # threshold of softmax
            '''
            #softmax_thres = (softmax_temp  > self.args.softmax_thres_gamma / 2048) * (softmax_coor > self.args.softmax_coor_gamma / 2048)
            #softmax_thres = 1 - (softmax_temp  <= self.args.softmax_gamma / 2048) * (softmax_coor <= 1 / 2048)
            if self.args.use_coordinate_distribution :
                softmax_coor_thres = softmax_coor > self.args.softmax_coor_gamma / self.embedding_coreset_index.ntotal # threshold of softmax
            else : 
                #softmax_coor_thres = softmax_thres
                softmax_coor_thres = None
            
            embed_distances, embed_indices = self.embedding_coreset_index_cpu.search(embedding_test, k=self.embedding_coreset_index.ntotal) # (W x H) x self.dist_coreset_index.ntotal
            embed_distances = np.sqrt(embed_distances)
            embed_prob = calc_prob_embedding(embed_distances, gamma=self.args.prob_gamma)
            
            '''
            if self.args.position_encoding_in_distribution :
                dist_distances, dist_indices = self.dist_coreset_index.search(embedding_pe_test, k=self.dist_coreset_index.ntotal) # (W x H) x self.dist_coreset_index.ntotal
            else :
                dist_distances, dist_indices = self.dist_coreset_index.search(embedding_test, k=self.dist_coreset_index.ntotal) # (W x H) x self.dist_coreset_index.ntotal

            dist_distances = np.sqrt(dist_distances)
            dist_prob = calc_prob_embedding(dist_distances, gamma=self.args.prob_gamma)

            
            #softmax_temp_inverse = np.zeros_like(softmax_temp)
            softmax_thres_inverse = np.zeros_like(softmax_thres)
            for i in range(neighbors.shape[0]) :
                for k in range(self.dist_coreset_index.ntotal) :
                    #softmax_temp_inverse[i, k] = softmax_temp[i, dist_indices[i, k]]
                    softmax_thres_inverse[i, k] = softmax_thres[i, dist_indices[i, k]]
            '''
                    
            softmax_coor_thres_inverse = np.zeros_like(softmax_coor_thres)
            for i in range(neighbors.shape[0]) :
                for k in range(self.embedding_coreset_index.ntotal) :
                    softmax_coor_thres_inverse[i, k] = softmax_coor_thres[i, embed_indices[i, k]]
                    
            '''
            softmax_thres_inverse[:, -1] = True
            '''
            softmax_coor_thres_inverse[:, -1] = True
                    
            #anomaly_pxl_likelihood = np.sum(dist_distances * softmax_temp_inverse, axis = 1)
            #anomaly_pxl_likelihood = np.apply_along_axis(lambda a : np.min(a[a!=0]), 1, dist_distances * softmax_thres_inverse)
            
            '''
            anomaly_pxl_likelihood = np.max(dist_prob * softmax_thres_inverse, axis = 1)
            anomaly_pxl_likelihood = -np.log(anomaly_pxl_likelihood)
            '''
            anomaly_pxl_topk1 = np.max(embed_prob * softmax_coor_thres_inverse, axis = 1)
            anomaly_pxl_topk1 = -np.log(anomaly_pxl_topk1)
            #anomaly_pxl_topk1 = np.apply_along_axis(lambda a : np.min(a[a!=0]), 1, dist_distances * softmax_coor_thres_inverse)
            #anomaly_pxl_likelihood = anomaly_pxl_topk1
            '''
            anomaly_map_nb = anomaly_pxl_likelihood.reshape(ref_num_patches)
            if self.args.cut_edge_embedding :
                patch_padding = (self.args.patchsize - 1) // 2
                pad_width = ((patch_padding,),(patch_padding,))
                
                anomaly_map_nb = anomaly_map_nb[patch_padding:anomaly_map_nb.shape[0]-patch_padding, patch_padding:anomaly_map_nb.shape[1]-patch_padding]
                anomaly_img_score_nb = np.max(anomaly_map_nb)
                anomaly_map_nb = np.pad(anomaly_map_nb, pad_width, 'edge')
            else : 
                anomaly_img_score_nb = np.max(anomaly_map_nb)          
            '''  

            anomaly_map_topk1 = anomaly_pxl_topk1.reshape(ref_num_patches)
            if self.args.cut_edge_embedding :
                patch_padding = (self.args.patchsize - 1) // 2
                pad_width = ((patch_padding,),(patch_padding,))
                
                anomaly_map_topk1 = anomaly_map_topk1[patch_padding:anomaly_map_topk1.shape[0]-patch_padding, patch_padding:anomaly_map_topk1.shape[1]-patch_padding]
                anomaly_img_score_topk1 = np.max(anomaly_map_topk1)
                anomaly_map_topk1 = np.pad(anomaly_map_topk1, pad_width, 'edge')
            else : 
                anomaly_img_score_topk1 = np.max(anomaly_map_topk1)
                
        anomaly_map_resized = cv2.resize(anomaly_map_nb, (self.args.imagesize, self.args.imagesize))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        anomaly_map_topk1_resized = cv2.resize(anomaly_map_topk1, (self.args.imagesize, self.args.imagesize))
        anomaly_map_topk1_resized_blur = gaussian_filter(anomaly_map_topk1_resized, sigma=4)
        anomaly_map_patchcore_resized = cv2.resize(anomaly_map_patchcore, (self.args.imagesize, self.args.imagesize))
        anomaly_map_patchcore_resized_blur = gaussian_filter(anomaly_map_patchcore_resized, sigma=4)
        anomaly_map_pe_resized = cv2.resize(anomaly_map_pe, (self.args.imagesize, self.args.imagesize))
        anomaly_map_pe_resized_blur = gaussian_filter(anomaly_map_pe_resized, sigma=4)
        
        gt_np = gt.cpu().numpy()[0,0].astype(int)
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.pred_list_px_lvl_topk1.extend(anomaly_map_topk1_resized_blur.ravel())
        self.pred_list_px_lvl_patchcore.extend(anomaly_map_patchcore_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(anomaly_img_score_nb)
        self.pred_list_img_lvl_topk1.append(anomaly_img_score_topk1)
        self.pred_list_img_lvl_patchcore.append(anomaly_img_score_patchcore)
        self.img_path_list.extend(file_name)
        self.img_type_list.append(x_type[0])
        self.pred_list_px_lvl_pe.extend(anomaly_map_pe_resized_blur.ravel())
        self.pred_list_img_lvl_pe.append(anomaly_img_score_pe)
        
        if self.args.save_anomaly_map :
            # save images
            x = self.inv_normalize(x).clip(0,1)
            input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
            self.save_anomaly_map(anomaly_map_resized_blur, anomaly_map_topk1_resized_blur, anomaly_map_patchcore_resized_blur, anomaly_map_pe_resized_blur, input_x, gt_np*255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        # Total pixel-level auc-roc score
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        # Total pixel-level auc-roc score for only using likelihood
        pixel_auc_topk1 = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl_topk1)
        # Total pixel-level auc-roc score for patchcore version
        pixel_auc_patchcore = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl_patchcore)
        # Total pixel-level auc-roc score for using position encoding
        pixel_auc_pe = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl_pe)

        # Total image-level auc-roc score
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        # Total image-level auc-roc score for only using likelihood
        img_auc_topk1 = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl_topk1)
        # Total image-level auc-roc score for patchcore version
        img_auc_patchcore = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl_patchcore)
        # Total image-level auc-roc score for using position encoding
        img_auc_pe = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl_pe)

        values = {'pixel_auc': pixel_auc, 'pixel_auc_topk1': pixel_auc_topk1, 'pixel_auc_patchcore': pixel_auc_patchcore, \
                'img_auc': img_auc, 'img_auc_topk1': img_auc_topk1, 'img_auc_patchcore': img_auc_patchcore, \
                'pixel_auc_pe' : pixel_auc_pe, 'img_auc_pe' : img_auc_pe}
        
        self.log_dict(values)
        
        f = open(os.path.join(self.args.project_root_path, "score_result.csv"), "a")
        data = [self.args.category, str(self.args.subsampling_percentage), str(self.args.dist_coreset_size), str(self.args.dist_padding), \
                str(self.args.softmax_temperature_alpha), str(self.args.softmax_thres_gamma), str(self.args.softmax_coor_gamma),\
                str(f'{pixel_auc : .3f}'), str(f'{pixel_auc_topk1 : .3f}'), str(f'{pixel_auc_patchcore : .3f}'), str(f'{pixel_auc_pe : .3f}'), \
                str(f'{img_auc : .3f}'), str(f'{img_auc_topk1 : .3f}'), str(f'{img_auc_patchcore : .3f}'), str(f'{img_auc_pe : .3f}')]
        data = ','.join(data) + '\n'
        f.write(data)
        f.close()

        print("For anomaly_score_nb")
        true_index = np.where(np.array(self.gt_list_img_lvl) == 0)
        max_true_anomaly_img_score = np.max(np.array(self.pred_list_img_lvl)[true_index])
        pred_false_index =  np.intersect1d(np.where(np.array(self.pred_list_img_lvl) < max_true_anomaly_img_score), np.where(np.array(self.gt_list_img_lvl) == 1))

        print(f"max_true_anomaly_img_score is {max_true_anomaly_img_score}")
        for idx in range(pred_false_index.shape[0]) :
            print(f"file name : {self.img_type_list[pred_false_index[idx]]}_{self.img_path_list[pred_false_index[idx]]}, anomaly_score : {self.pred_list_img_lvl[pred_false_index[idx]]}")

        print("For anomaly_score_patchcore")
        true_index = np.where(np.array(self.gt_list_img_lvl) == 0)
        max_true_anomaly_img_score = np.max(np.array(self.pred_list_img_lvl_patchcore)[true_index])
        pred_false_index =  np.intersect1d(np.where(np.array(self.pred_list_img_lvl_patchcore) < max_true_anomaly_img_score), np.where(np.array(self.gt_list_img_lvl) == 1))

        print(f"max_true_anomaly_img_score is {max_true_anomaly_img_score}")
        for idx in range(pred_false_index.shape[0]) :
            print(f"file name : {self.img_type_list[pred_false_index[idx]]}_{self.img_path_list[pred_false_index[idx]]}, anomaly_score : {self.pred_list_img_lvl_patchcore[pred_false_index[idx]]}")