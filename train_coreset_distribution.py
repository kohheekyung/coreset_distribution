import pytorch_lightning as pl
import argparse
import os

from utils.data.load_data import Train_Dataloader, Test_Dataloader, Distribution_Train_Dataloader, Coor_Distribution_Train_Dataloader
from utils.learning.train_part import Coreset, Distribution, AC_Model, Coor_Distribution
from pytorch_lightning.loggers import TensorBoardLogger

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYLOCALIZATION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default='../dataset/MVTecAD') # ./MVTec
    parser.add_argument('--category', default='hazelnut')
    parser.add_argument('--project_root_path', default=r'./result')
    #parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--num_workers', default=8) # 0
    
    # patch_core
    parser.add_argument('--backbone', '-b', choices=['WR101', 'WR50', 'R50', 'R34', 'R18', 'R101', 'R152'], default='WR50')
    parser.add_argument('--layer_index', '-le', nargs='+', default=['layer2', 'layer3'])
    parser.add_argument('--patchcore_batchsize', type=int, default=2)
    parser.add_argument('--pretrain_embed_dimension', type=int, default=1024) # Dimensionality of features extracted from backbone layers
    parser.add_argument('--target_embed_dimension', type=int, default=1024) # final aggregated PatchCore Dimensionality
    parser.add_argument('--anomaly_nn', type=int, default=1) # Num. nearest neighbours to use for anomaly detection
    parser.add_argument('--patchsize', type=int, default=3) # neighbourhoodsize for local aggregation
    parser.add_argument('--faiss_on_gpu', default=False, action='store_true', help="Whether to use gpu on faiss")
    
    # sampler
    parser.add_argument('--subsampling_percentage', '-p', type=float, default=0.01)
    
    # dataset
    parser.add_argument('--resize', type=int, default=366) # 256
    parser.add_argument('--imagesize', type=int, default=320) # 224
    
    # coreset_distribution
    parser.add_argument('--dist_coreset_size', type=int, default=2048) # 512
    parser.add_argument('--dist_padding', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--dist_batchsize', type=int, default=1024)
    parser.add_argument('--softmax_temperature_alpha', type=float, default=1.0)
    parser.add_argument('--prob_gamma', type=float, default=0.99)
    parser.add_argument('--softmax_gamma', type=float, default=1.0)
    
    # ETC
    parser.add_argument('--use_position_encoding', default=False, action='store_true', help="Whether to use position encoding")
    parser.add_argument('--pe_weight', type=float, default=10)
    parser.add_argument('--not_use_coreset_distribution', default=False, action='store_true', help='Whether not to use coreset_distribution')
    parser.add_argument('--use_coordinate_distribution', default=False, action='store_true', help='Whether to use coordinate_distribution')
    parser.add_argument('--softmax_temperature_beta', type=float, default=1.0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(args.seed)
    default_root_dir = os.path.join(args.project_root_path, args.category) # ./MVTec/hazelnut

    train_dataloader, test_dataloader = Train_Dataloader(args), Test_Dataloader(args)

    # generate coreset and save it to faiss
    if args.phase == 'train' :
        coreset_generator_trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(default_root_dir, 'coreset'), max_epochs=1, gpus=1, enable_checkpointing=False) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
        coreset_generator = Coreset(args)
        coreset_generator_trainer.fit(coreset_generator, train_dataloaders=train_dataloader)

    distribution_train_dataloader, distribution_val_dataloader, dist_input_size, dist_output_size = Distribution_Train_Dataloader(args, train_dataloader)
    # train coreset distribution
    if args.phase == 'train' and not args.not_use_coreset_distribution:
        tb_logger = TensorBoardLogger(save_dir=default_root_dir, name="distribution")
        distribution_trainer = pl.Trainer.from_argparse_args(args, max_epochs=args.num_epochs, gpus=1, logger=tb_logger, log_every_n_steps=50, enable_checkpointing=False) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
        distribution_model = Distribution(args, dist_input_size, dist_output_size)
        distribution_trainer.fit(distribution_model, train_dataloaders=distribution_train_dataloader, val_dataloaders=distribution_val_dataloader)
    
    # train coordinate distribution
    if args.phase == 'train' and args.use_coordinate_distribution:
        coor_distribution_train_dataloader, coor_dist_input_size, coor_dist_output_size = Coor_Distribution_Train_Dataloader(args, train_dataloader)
        coor_distribution_trainer = Coor_Distribution(args, coor_dist_input_size, coor_dist_output_size)
        coor_distribution_trainer.fit(coor_distribution_train_dataloader)

    # eval anomaly score from test_dataloader
    anomaly_calculator = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(default_root_dir, 'anomaly'), max_epochs=1, gpus=1, enable_checkpointing=False) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    ac_model = AC_Model(args, dist_input_size, dist_output_size)
    anomaly_calculator.test(ac_model, dataloaders=test_dataloader)