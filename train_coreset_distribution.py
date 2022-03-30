import pytorch_lightning as pl
import argparse
import os

from utils.data.load_data import Train_Dataloader, Test_Dataloader
from utils.learning.train_part import STPM

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default='../dataset/MVTecAD') # ./MVTec
    parser.add_argument('--category', default='hazelnut')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--num_workers', default=4) # 0
    parser.add_argument('--load_size', default=256) # 256
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', type=float, default=0.001) # 0.01
    parser.add_argument('--project_root_path', default=r'./result') # ./test
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument('--feature_model', choices=['WR50', 'R50', 'R34', 'R18', 'R101', 'R152'], default='WR50')
    parser.add_argument('--block_index', choices=['1+2', '2+3', '3+4', '4', '5'], default='2+3') # '2+3' means using both block 2 and block 3
    parser.add_argument('--visualize_tsne', default=False, action='store_true', help='Whether to visualize t-SNE projection')
    parser.add_argument('--whitening', default=False, action='store_true', help='Whether to use whitening features')
    parser.add_argument('--whitening_offset', type=float, default=0.001)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    default_root_dir = os.path.join(args.project_root_path, args.category) # ./MVTec/hazelnut
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=default_root_dir, max_epochs=args.num_epochs, gpus=1) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    model = STPM(args)
    if args.phase == 'train':
        trainer.fit(model, train_dataloaders=Train_Dataloader(args))
        trainer.test(model, dataloaders=Test_Dataloader(args))
    elif args.phase == 'test':
        trainer.test(model, dataloaders=Test_Dataloader(args))