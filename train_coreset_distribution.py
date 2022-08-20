from pickle import FALSE
from numpy import True_
import pytorch_lightning as pl
import argparse
import os

from utils.data.load_data import Train_Dataloader, Test_Dataloader, Distribution_Train_Dataloader, Coor_Distribution_Train_Dataloader
from utils.learning.train_part import Coreset, Distribution, AC_Model, Coor_Distribution
from pytorch_lightning.loggers import TensorBoardLogger

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYLOCALIZATION')
    parser.add_argument('--train_coreset', default=False, action='store_true', help="Whether to train coreset")
    parser.add_argument('--train_nb_dist', default=False, action='store_true', help="Whether to train nb_dist")
    parser.add_argument('--train_coor_dist', default=False, action='store_true', help="Whether to train coor_dist")
    parser.add_argument('--dataset_path', default='../dataset/MVTecAD') # ./MVTec
    parser.add_argument('--category', default='hazelnut')
    parser.add_argument('--project_root_path', default=r'./result')
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--num_workers', default=4) # 0
    
    # patch_core
    parser.add_argument('--backbone', '-b', choices=['WR101', 'WR50', 'R50', 'R34', 'R18', 'R101', 'R152'], default='WR101') # pretrained model with ImageNet
    parser.add_argument('--layer_index', '-le', nargs='+', default=['layer2', 'layer3']) # intermediate layers to make local features
    parser.add_argument('--pretrain_embed_dimension', type=int, default=1024) # Dimensionality of features extracted from backbone layers
    parser.add_argument('--target_embed_dimension', type=int, default=1024) # final aggregated PatchCore Dimensionality
    parser.add_argument('--anomaly_nn', type=int, default=3) # Num. nearest neighbours to use for anomaly detection
    parser.add_argument('--patchsize', type=int, default=5) # neighbourhoodsize for local aggregation
    
    # sampler
    parser.add_argument('--subsampling_percentage', '-p', type=float, default=0.01)
    
    # dataset
    parser.add_argument('--resize', type=int, default=512, help='resolution of resize') # 256
    parser.add_argument('--imagesize', type=int, default=480, help='resolution of centercrop') # 224
    
    # coreset_distribution
    parser.add_argument('--dist_coreset_size', type=int, default=2048)
    parser.add_argument('--dist_padding', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=7) # 7
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--dist_batchsize', type=int, default=2048)
    parser.add_argument('--softmax_temperature_alpha', type=float, default=1.0)
    parser.add_argument('--prob_gamma', type=float, default=0.99)
    parser.add_argument('--softmax_nb_gamma', type=float, default=1.0)
    parser.add_argument('--softmax_coor_gamma', type=float, default=1.0)
    
    # ETC
    parser.add_argument('--position_encoding_in_distribution', default=False, action='store_true', help="Whether to use position encoding in training distribution")
    parser.add_argument('--pe_weight', type=float, default=5)
    parser.add_argument('--not_use_coreset_distribution', default=False, action='store_true', help='Whether not to use coreset_distribution')
    parser.add_argument('--not_use_coordinate_distribution', default=False, action='store_true', help='Whether not to use coordinate_distribution')
    parser.add_argument('--cut_edge_embedding', default=False, action='store_true', help="Whether to cut edge embedding")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(args.seed)
    default_root_dir = os.path.join(args.project_root_path, args.category) # ./MVTec/hazelnut

    # generate train dataloader adn test dataloader
    train_dataloader, test_dataloader = Train_Dataloader(args), Test_Dataloader(args)

    # generate coreset and save it to faiss
    if args.train_coreset :
        print("Start training coreset")
        coreset_generator_trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(default_root_dir, 'coreset'), max_epochs=1, gpus=1, enable_checkpointing=False) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
        coreset_generator = Coreset(args)
        coreset_generator_trainer.fit(coreset_generator, train_dataloaders=train_dataloader)
        print("End training coreset")

    # generate Distribution train dataloader for training coreset distribution
    distribution_train_dataloader, distribution_val_dataloader, dist_input_size, dist_output_size = Distribution_Train_Dataloader(args, train_dataloader)
    
    # train coreset distribution
    if args.train_nb_dist:
        print("Start training nb_dist")
        tb_logger = TensorBoardLogger(save_dir=default_root_dir, name="distribution")
        distribution_trainer = pl.Trainer.from_argparse_args(args, max_epochs=args.num_epochs, gpus=1, logger=tb_logger, log_every_n_steps=50, enable_checkpointing=False) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
        distribution_model = Distribution(args, dist_input_size, dist_output_size)
        distribution_trainer.fit(distribution_model, train_dataloaders=distribution_train_dataloader, val_dataloaders=distribution_val_dataloader)
        print("End training nb_dist")
    
    # train coordinate distribution
    if args.train_coor_dist:
        print("Start training coor_dist")
        coor_distribution_train_dataloader, coor_dist_input_size, coor_dist_output_size = Coor_Distribution_Train_Dataloader(args, train_dataloader)
        coor_distribution_trainer = Coor_Distribution(args, coor_dist_input_size, coor_dist_output_size)
        coor_distribution_trainer.fit(coor_distribution_train_dataloader)
        print("End training nb_dist")

    # eval anomaly score from test_dataloader
    print("Start calculating anomaly score")
    anomaly_calculator = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(default_root_dir, 'anomaly'), max_epochs=1, gpus=1, enable_checkpointing=False) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    ac_model = AC_Model(args, dist_input_size, dist_output_size)
    anomaly_calculator.test(ac_model, dataloaders=test_dataloader)