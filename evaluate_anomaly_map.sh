# train and evaluate anomaly map for each category
python train_coreset_distribution.py --category bottle --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTec_small
python train_coreset_distribution.py --category cable --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTec_small
python train_coreset_distribution.py --category capsule --train_coreset --train_nb_dist --train_coor_dist --dataset_path ../dataset/MVTec_small

# make ensemble score for each category
python analysis_code/calc_ensemble_score.py --category bottle --backbone_list WR101
python analysis_code/calc_ensemble_score.py --category cable --backbone_list WR101
python analysis_code/calc_ensemble_score.py --category capsule --backbone_list WR101

# convert result format
python analysis_code/convert_result_format.py --is_MVTec_small

# analysis anomaly map
python analysis_code/analysis_amap.py --is_MVTec_small