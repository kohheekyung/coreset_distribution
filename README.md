# Image Anomaly Detection and Localization with Position and Neighborhood Information
This is supplementary code for the paper "Image Anomaly Detection and Localization with Position and Neighborhood Information." (https://arxiv.org/abs/2211.12634)
It trains the model for evaluating anomaly maps and calculating Image AUROC, Pixel AUROC, and Pixel AUPRO for two datasets, MVTec AD benchmark, and BTAD dataset.

The model trained on designated hyperparameter can achieve up to **99.52%** and **98.91%** AUROC scores in anomaly detection and localization and **94.83%** AUPRO score for MVTec AD benchmark which is the state-of-the-art performance.
With an ensemble of models, performance can reach **99.55%** and **99.05%** in Image AUROC and Pixel AUROC, and **95.52%** Pixel AUPRO.
In addition, the same model can achieve up to 97.7% of Pixel AUROC for the BTAD dataset, which is the highest performance compared to previous works.

## Environment
We trained and evaluated our models in Python 3.8 and PyTorch which version torch=1.12.1 and torchvision=0.13.1.
Training is on NVIDIA A100 GPUs and NVIDIA T4 GPUs.
We used ImageNet pre-trained network from PyTorch/vision:v0.10.0.
The WideResNet101-2 network is used in our code by default, ResNext101_32x8d and DenseNet201 are used for ensemble results.
In the code, two kinds of coresets, which are embedding coreset and distribution coreset, are stored in *faiss* to efficiently calculate the distance from the test feature and coreset.
We used *pytorch-lightning* to manage the training process and evaluation process.

## Quick Guide
We provided a bash file for training and evaluating the anomaly map for MVTec AD benchmark and BTAD dataset.
Dataset should be placed in parent directory of code repository.
For example, default dataset directory is "../dataset/MVTecAD" and "../dataset/BTAD" for MVTec AD and BTAD, respectively.

First, go to code repository, and install all requirements of environment.
The environment name we used is "anomaly_env".

```bash
conda create -y -n anomaly_env
conda activate anomaly_env 
```
```bash
conda install -y python=3.8
pip install pytorch-lightning==1.5.9
pip install pillow==9.0.0
pip install faiss-gpu==1.7.1
pip install opencv-python==4.5.2.52
pip install scikit-learn==0.24.2
pip install scikit-image==0.19.2
pip install pymp-pypi==0.5.0
pip install numpngw==0.1.2
```
The available version of torch and torchvision might be different depending on hardware settings.
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

Then, execute "evaluate_anomaly_map_on_MVTecAD.sh". This bash file contains all processes of training the model, evaluating the anomaly map, and visualizing the anomaly map.
```bash
chmod +x evaluate_anomaly_map_on_MVTecAD.sh
./evaluate_anomaly_map_on_MVTecAD.sh
```
As a result of execution, a "result" repository will be created. The structure of the repository is as follows:
```
|-- result/
  |-- bottle/
  
  |-- ensemble_ravel/
    |-- viz/
      |-- bottle/
        00000_amap.png
        00000_amap_on_img.png
        ...
    00000_gt.png
    00000_pred.png
    ...
    image00000.png
    ...
    score_result.csv
    
  |-- ensemble_result/
    |-- bottle/
      |-- anomaly_map/
      |-- ground_truth/
      |-- test/
      
  ensemble_score_result.csv
  score_result.csv
```

You can find the AUROC score on either the output terminal or "./result/score_result.csv". If you use multiple models for the ensemble, the ensemble score can be found on either the output terminal or "./result/ensemble_score_result.csv".
Visualization of the anomaly map can be found on "./result/ensemble_ravel/viz/" repository.
You can check images, ground truth, anomaly map, and anomaly map with a mask on the repository.

## Hyperparameter setting
The default hyperparameter in "evaluate_anomaly_map_on_MVTecAD.sh" is the same as mentioned in the paper. 
"evaluate_anomaly_map_on_MVTecAD.sh" contains 4 sequential python codes.

First, **"train_coreset_distribution.py"** trains our proposed model and evaluates the anomaly map for each category. You can change the dataset with "--dataset_category" argument, and the category with "--category" path. 
Note that the dataset should be in the directory of "--dataset_path" argument. 
If you want other pre-trained networks rather than WideResNet101-2, change "--backbone" argument.

Second, **"analysis_code/calc_ensemble_score.py"** makes an ensemble score for each category and saves the result in "./result/ensemble_result" repository.
"--backbone_list" argument is a list of pre-trained networks which are to ensemble. You can change the category with "--category" path. 

Third, **"analysis_code/convert_result_format.py"** converts the result format and saves it into "./result/ensemble_ravel" repository.
Add argument "--is_BTAD" if the dataset is BTAD, and "--is_MVtec_small" if the dataset is a small version of MVTec which we provided.
The default dataset is the MVTec AD benchmark.

Finally, **"analysis_code/analysis_amap.py"** analysis anomaly map from "./result/ensemble_ravel" repository.
Add argument "--visualize" to visualize the anomaly map on "./result/ensemble_ravel/viz" repository.
If you want to find misclassified images with the trained model, add argument "--calc_misclassified_sample" and indices of false positive samples and false negative samples will be presented on "./result/ensemble_ravel/misclassified_sample_list.csv"
In addition, add "--calc_pro" argument to additionally calculate the AUPRO score. The result will be presented on "./result/ensemble_ravel/score_result.csv".