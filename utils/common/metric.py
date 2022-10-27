import numpy as np
from torch import nn
from torch.nn import functional as F
import torchvision
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

class ThresLoss(nn.Module):
    def __init__(self):
        super(ThresLoss, self).__init__()

    def forward(self, inputs, targets):
        kernel_size = 51
        gaussianblur = torchvision.transforms.GaussianBlur(kernel_size, sigma= (kernel_size - 1) / 6)
        blurred_gts = gaussianblur(targets)
        
        # dilate_kernel = torch.ones(size=(1, 1, kernel_size, kernel_size)).cuda()
        # dilate_targets = torch.clamp(torch.nn.functional.conv2d(targets, dilate_kernel, padding=(kernel_size // 2, kernel_size // 2)), 0, 1)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        blurred_targets = blurred_gts.view(-1)
        
        gt_0_index = torch.nonzero(1 - targets)
        gt_1_index = torch.nonzero(targets)        
        
        gt_0_max = torch.max(inputs[gt_0_index])
        gt_0_min = torch.min(inputs[gt_0_index])
        gt_1_max = torch.max(inputs[gt_1_index])
        gt_1_min = torch.min(inputs[gt_1_index])
        
        thres_mask = (inputs > gt_1_min) * (inputs < gt_0_max)
        thres_index = torch.nonzero(thres_mask * blurred_targets)
        
        final_targets = (gt_1_max - gt_0_min) * blurred_targets + gt_0_min
        
        final_targets = final_targets[thres_index]
        thres_inputs = inputs[thres_index]
        
        thresloss = torch.sqrt(torch.mean(torch.square(thres_inputs - final_targets)))
        
        return thresloss, blurred_gts

def cal_pro_metric_new(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=2000, class_name=None):
    #labeled_imgs = np.array(labeled_imgs).squeeze(1)
    labeled_imgs = np.array(labeled_imgs).reshape(-1, 480, 480)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool)
    #score_imgs = np.array(score_imgs).squeeze(1)
    score_imgs = np.array(score_imgs).reshape(-1, 480, 480)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)


    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

