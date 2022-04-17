import torch
from torch.nn import functional as F
import cv2
import os
import numpy as np
import shutil
import pytorch_lightning as pl
import faiss
from sklearn.random_projection import SparseRandomProjection
from utils.sampling_methods.kcenter_greedy import kCenterGreedy
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from utils.common.visualize import visualize_TSNE
from utils.data.transforms import INV_Normalize
from utils.common.embedding import embedding_concat, reshape_embedding
from utils.learning.model import Distribution_Model

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
    embeddings_path = os.path.join('./', f'embeddings_{args.block_index}', args.category)
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

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.feature_model(x_t)
        return self.features

    def on_train_start(self):
        self.feature_model.eval()
        self.embedding_dir_path = os.path.join('./', f'embeddings_{self.args.block_index}', self.args.category)
        os.makedirs(self.embedding_dir_path, exist_ok=True)
        self.embedding_list = []

    def training_step(self, batch, batch_idx):
        x, _, _, file_name, _ = batch
        features = self(x)
        
        if '+' in self.args.block_index :        
            embeddings = []
            m = torch.nn.AvgPool2d(3, 1, 1)
            for feature in features:               
                embeddings.append(m(feature))
            embedding_ = np.array(embedding_concat(embeddings[0], embeddings[1]))
        else :
            embedding_ = np.array(features[0].cpu())
            
        self.embedding_list.extend(reshape_embedding(embedding_))

    def training_epoch_end(self, outputs):
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        
        # Coreset Subsampling
        embedding_coreset_size = int(self.args.coreset_sampling_ratio * total_embeddings.shape[0])
        # dist_coreset_size = self.args.dist_coreset_size
        # select_batch_size = max(embedding_coreset_size, dist_coreset_size)

        selector = kCenterGreedy(embedding=torch.Tensor(total_embeddings), sampling_size=embedding_coreset_size)
        selected_idx = selector.select_coreset_idxs()
        self.embedding_coreset = total_embeddings[selected_idx][:embedding_coreset_size]
        self.dist_coreset = total_embeddings[selected_idx][:dist_coreset_size]
                
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        
        #faiss
        self.embedding_coreset_index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.embedding_coreset_index.add(self.embedding_coreset)
        faiss.write_index(self.embedding_coreset_index, os.path.join(self.embedding_dir_path,f'embedding_coreset_index_{int(self.args.coreset_sampling_ratio*100)}.faiss'))

        # self.dist_coreset_index = faiss.IndexFlatL2(self.dist_coreset.shape[1])
        # self.dist_coreset_index.add(self.dist_coreset)
        # faiss.write_index(self.dist_coreset_index, os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}.faiss'))

    def configure_optimizers(self):
        return None

class Distribution(pl.LightningModule):
    def __init__(self, args, dist_input_size, dist_output_size):
        super(Distribution, self).__init__()

        self.args = args
        self.model = Distribution_Model(args, dist_input_size, dist_output_size)
        self.embedding_dir_path = os.path.join('./', f'embeddings_{self.args.block_index}', self.args.category)
        os.makedirs(self.embedding_dir_path, exist_ok=True)
        self.best_val_loss=1e+6
        
        self.train_loss = 0.0
        self.train_size = 0
        self.val_loss = 0.0
        self.val_size = 0

    def forward(self, x):
        x = self.model(x)
        return x
    
    def on_train_epoch_start(self):
        self.train_loss = 0.0
        self.train_size = 0
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
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
        loss = F.mse_loss(y_hat, y)
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
            f=os.path.join(self.embedding_dir_path, f'model_dp{self.args.dist_padding}.pt')
        )
        
        if self.best_val_loss > self.val_loss :
            self.best_val_loss = self.val_loss
            shutil.copyfile(os.path.join(self.embedding_dir_path, f'model_dp{self.args.dist_padding}.pt'), os.path.join(self.embedding_dir_path, f'best_model_dp{self.args.dist_padding}.pt'))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=0.1)
        return [optimizer], [scheduler]
    
class AC_Model(pl.LightningModule):
    def __init__(self, args, dist_input_size, dist_output_size):
        super(AC_Model, self).__init__()
        
        self.save_hyperparameters(args)
        self.args = args

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

        self.init_results_list()

        self.inv_normalize = INV_Normalize()
        
        self.dist_model = Distribution_Model(args, dist_input_size, dist_output_size)        
        self.embedding_dir_path = os.path.join('./', f'embeddings_{self.args.block_index}', self.args.category)
        self.dist_model.load_state_dict(torch.load(os.path.join(self.embedding_dir_path, f'best_model_dp{self.args.dist_padding}.pt'))['model'])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.pred_list_px_lvl_topk1 = []
        self.pred_list_px_lvl_patchcore = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.pred_list_img_lvl_topk1 = []
        self.pred_list_img_lvl_patchcore = []
        self.img_path_list = []
        self.img_type_list = []

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.feature_model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, anomaly_map_patchcore, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        anomaly_map_norm_patchcore = min_max_norm(anomaly_map_patchcore)
        anomaly_map_norm_hm_patchcore = cvt2heatmap(anomaly_map_norm_patchcore*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_patchcore.jpg'), anomaly_map_norm_hm_patchcore)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def configure_optimizers(self):
        return None
    
    def on_test_start(self):
        self.feature_model.eval() # to stop running_var move (maybe not critical)
        self.dist_model.eval() # to stop running_var move (maybe not critical)
        # self.dist_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'dist_coreset_index_{self.args.dist_coreset_size}.faiss'))
        self.embedding_coreset_index = faiss.read_index(os.path.join(self.embedding_dir_path,f'embedding_coreset_index_{int(self.args.coreset_sampling_ratio*100)}.faiss'))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            # self.dist_coreset_index = faiss.index_cpu_to_gpu(res, 0 ,self.dist_coreset_index)
            self.embedding_coreset_index = faiss.index_cpu_to_gpu(res, 0 ,self.embedding_coreset_index)
        self.init_results_list()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir, self.args)

    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x, gt, label, file_name, x_type = batch
        
        # extract embedding
        features = self(x)
        
        if '+' in self.args.block_index : 
            embeddings = []
            m = torch.nn.AvgPool2d(3, 1, 1)
            for feature in features:
                embeddings.append(m(feature))
            embedding_ = np.array(embedding_concat(embeddings[0], embeddings[1])) # 1 x E x W x H
        else :
            embedding_ = np.array(features[0].cpu())
                
        embedding_test = np.array(reshape_embedding(embedding_)) # (W x H) x E
        
        embedding_ = embedding_.squeeze() # E x W x H
        pad_width = ((0,),(self.args.dist_padding,),(self.args.dist_padding,))
        embedding_pad = np.pad(embedding_, pad_width, "constant") # E x (W+1) x (H+1)
        neighbors = np.zeros(shape=(embedding_.shape[1], embedding_.shape[2], embedding_.shape[0]*(pow(self.args.dist_padding*2+1, 2) - 1))) # W x H x NE
        # construct neighbor features
        for i_idx in range(embedding_.shape[1]) :
            for j_idx in range(embedding_.shape[2]) :
                neighbor = np.zeros(shape=(0,))
                for di in range(-self.args.dist_padding, self.args.dist_padding+1) :
                    for dj in range(-self.args.dist_padding, self.args.dist_padding+1) :
                        if di == 0 and dj == 0 :
                            continue
                        neighbor = np.concatenate((neighbor, embedding_pad[:, i_idx+di+self.args.dist_padding, j_idx+dj+self.args.dist_padding]))
                neighbors[i_idx, j_idx] = neighbor                   
                
        embedding_score, embedding_indices, embedding_recons = self.embedding_coreset_index.search_and_reconstruct(embedding_test, k=self.args.n_neighbors) # (W x H) x self.args.n_neighbors
        embedding_score = np.sqrt(embedding_score)
        
        max_anomaly_idx = np.argmax(embedding_score[:, 0])
        max_embedding_score = embedding_score[max_anomaly_idx, 0] # maximum embedding score
        max_anomaly_feature = embedding_test[max_anomaly_idx]
        max_anomaly_coreset_feature = self.embedding_coreset_index.reconstruct(embedding_indices[max_anomaly_idx, 0].item()) # nearest embedding coreset-feature from max_anomaly_feature
        _, neighbors_macf_indices, neighbors_macf_recons = self.embedding_coreset_index.search_and_reconstruct(max_anomaly_coreset_feature.reshape(1, -1) , k=self.args.n_neighbors)
        
        neighbor_index = faiss.IndexFlatL2(embedding_test.shape[1])
        neighbor_index.add(neighbors_macf_recons.squeeze(0))
        
        neighbor_distances, _ = neighbor_index.search(max_anomaly_feature.reshape(1, -1), k=self.args.n_neighbors)
        neighbor_distances = np.sqrt(neighbor_distances)
        weights_from_paper = 1 - 1 / np.sum(np.exp(neighbor_distances - max_embedding_score))

        weights_from_code = 1 - 1 / np.sum(np.exp(embedding_score[max_anomaly_idx] - max_embedding_score))
        
        # calc anomaly image score
        anomaly_img_score_patchcore = weights_from_code * max_embedding_score # Image-level score
        #anomaly_img_score_patchcore = weights_from_paper * max_embedding_score # Image-level score
        #anomaly_img_score_patchcore = max_embedding_score
        
        # calc anomaly pixel score
        neighbors = neighbors.reshape(-1, neighbors.shape[2]).astype(np.float32) # (W x H) x NE
        y_hat = self.dist_model(torch.tensor(neighbors).cuda()).cpu() # (W x H) x self.dist_coreset_index.ntotal
        
        topk_size = 9
        
        topk_prob, topk_prob_index = torch.topk(y_hat, k=topk_size, dim=-1)
        dist_test_neighbors = np.zeros(shape=(neighbors.shape[0], topk_size)) # (W x H)
        
        for i in range(neighbors.shape[0]) :
            for j in range(topk_size) :
                neighbor_feature = self.dist_coreset_index.reconstruct(topk_prob_index[i, j].item())
                dist_test_neighbors[i, j] = np.sqrt(np.sum((neighbor_feature - embedding_test[i]) ** 2))

        # negative log likelihood
        anomaly_pxl_score_topk1 = dist_test_neighbors[:, 0]
        anomaly_pxl_score = np.min(dist_test_neighbors, axis=1)

        anomaly_img_score = np.max(anomaly_pxl_score)
        anomaly_img_score_topk1 = np.max(anomaly_pxl_score_topk1)
        
        if self.args.block_index == '1+2':
            reshape_size = (56,56)
        elif self.args.block_index == '2+3':
            reshape_size = (28,28)
        elif self.args.block_index == '3+4':
            reshape_size = (14,14)
        elif self.args.block_index == '5' :
            reshape_size = (1,1)
        elif self.args.block_index == '4' :
            reshape_size = (7,7)
        
        anomaly_pxl_score = anomaly_pxl_score.reshape(reshape_size)
        anomaly_pxl_score_topk1 = anomaly_pxl_score_topk1.reshape(reshape_size)
        embedding_score = embedding_score[:,0].reshape(reshape_size)
    
        anomaly_map = anomaly_pxl_score
        anomaly_map_likelhood = anomaly_pxl_score_topk1
        anomaly_map_patchcore = embedding_score
                
        anomaly_map_resized = cv2.resize(anomaly_map, (self.args.input_size, self.args.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        anomaly_map_topk1_resized = cv2.resize(anomaly_map_likelhood, (self.args.input_size, self.args.input_size))
        anomaly_map_topk1_resized_blur = gaussian_filter(anomaly_map_topk1_resized, sigma=4)
        anomaly_map_patchcore_resized = cv2.resize(anomaly_map_patchcore, (self.args.input_size, self.args.input_size))
        anomaly_map_patchcore_resized_blur = gaussian_filter(anomaly_map_patchcore_resized, sigma=4)
        
        gt_np = gt.cpu().numpy()[0,0].astype(int)
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.pred_list_px_lvl_topk1.extend(anomaly_map_topk1_resized_blur.ravel())
        self.pred_list_px_lvl_patchcore.extend(anomaly_map_patchcore_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(anomaly_img_score)
        self.pred_list_img_lvl_topk1.append(anomaly_img_score_topk1)
        self.pred_list_img_lvl_patchcore.append(anomaly_img_score_patchcore)
        self.img_path_list.extend(file_name)
        self.img_type_list.append(x_type[0])
        
        # save images
        x = self.inv_normalize(x).clip(0,1)
        input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map_resized_blur, anomaly_map_patchcore_resized_blur, input_x, gt_np*255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        # Total pixel-level auc-roc score
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        # Total pixel-level auc-roc score for only using likelihood
        pixel_auc_topk1 = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl_topk1)
        # Total pixel-level auc-roc score for patchcore version
        pixel_auc_patchcore = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl_patchcore)

        # Total image-level auc-roc score
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        # Total image-level auc-roc score for only using likelihood
        img_auc_topk1 = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl_topk1)
        # Total image-level auc-roc score for patchcore version
        img_auc_patchcore = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl_patchcore)

        values = {'pixel_auc': pixel_auc, 'pixel_auc_topk1': pixel_auc_topk1, 'pixel_auc_patchcore': pixel_auc_patchcore, \
            'img_auc': img_auc, 'img_auc_topk1': img_auc_topk1, 'img_auc_patchcore': img_auc_patchcore}
        
        # if self.args.visualize_tsne:
        #     visualize_TSNE(self.viz_feature_list, self.viz_class_idx_list, os.path.join(self.logger.log_dir, "visualize_TSNE.png"))
        
        self.log_dict(values)
        
        f = open(os.path.join(self.args.project_root_path, "score_result.csv"), "a")
        data = [self.args.category, str(self.args.coreset_sampling_ratio), str(self.args.dist_coreset_size), str(self.args.dist_padding), \
                str(pixel_auc), str(pixel_auc_topk1), str(pixel_auc_patchcore), \
                str(img_auc), str(img_auc_topk1), str(img_auc_patchcore)]
        data = ','.join(data) + '\n'
        f.write(data)
        f.close()