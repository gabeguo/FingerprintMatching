import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os
import json
import math
import getopt, argparse

sys.path.append('../')

from trainer import *
from fingerprint_dataset import FingerprintDataset
from multiple_finger_datasets import *
from embedding_models import *

from common_filepaths import *

def main(args, cuda):    
    datasets = args.datasets.split()
    val_datasets = args.val_datasets.split()
    possible_fgrps = args.possible_fgrps.split()
    assert set(possible_fgrps).issubset(set(ALL_FINGERS))

    the_name = '_'.join([path[:len(path) if path[-1] != '/' else -1].split('/')[-1] for path in datasets])
    print('Name of this dataset:', the_name)

    print('weights saved to:', args.posttrained_model_path)

    train_dir_paths = [os.path.join(x, 'train') for x in datasets]
    val_dir_paths = [os.path.join(x, 'val') for x in val_datasets]

    training_dataset = MultipleFingerDataset(fingerprint_dataset=FingerprintDataset(train_dir_paths, train=True),\
        num_anchor_fingers=1, num_pos_fingers=1, num_neg_fingers=1,\
        SCALE_FACTOR=args.scale_factor,\
        diff_fingers_across_sets=args.diff_fingers_across_sets_train, diff_fingers_within_set=True,\
        diff_sensors_across_sets=args.diff_sensors_across_sets_train, same_sensor_within_set=True, \
        acceptable_anchor_fgrps=possible_fgrps, acceptable_pos_fgrps=possible_fgrps, acceptable_neg_fgrps=possible_fgrps)
    #training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 50)))
    train_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    val_dataset = MultipleFingerDataset(fingerprint_dataset=FingerprintDataset(val_dir_paths, train=False),\
        num_anchor_fingers=1, num_pos_fingers=1, num_neg_fingers=1,\
        SCALE_FACTOR=args.scale_factor,\
        diff_fingers_across_sets=args.diff_fingers_across_sets_val, diff_fingers_within_set=True,\
        diff_sensors_across_sets=args.diff_sensors_across_sets_val, same_sensor_within_set=True, \
        acceptable_anchor_fgrps=possible_fgrps, acceptable_pos_fgrps=possible_fgrps, acceptable_neg_fgrps=possible_fgrps)
    #val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, len(val_dataset), 5)))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    # CLEAR CUDA MEMORY
    # https://stackoverflow.com/questions/54374935/how-to-fix-this-strange-error-runtimeerror-cuda-error-out-of-memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # LOG TRAINING DATA
    print('Training data: {}\n'.format(train_dir_paths))

    # CREATE EMBEDDER
    embedder = EmbeddingNet(pretrained=False)

    # load saved weights!
    if args.pretrained_model_path:
        embedder.load_state_dict(torch.load(args.pretrained_model_path))

    pretrained_other_msg = 'pretrained on other data: {}\n'.format(args.pretrained_model_path)
    print(pretrained_other_msg)

    # CREATE TRIPLET NET
    triplet_net = TripletNet(embedder)

    # TRAIN
    optimizer = optim.Adam(triplet_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma, last_epoch=- 1, verbose=False) if \
        (args.lr_step is not None and args.gamma is not None) \
        else None

    print('learning rate = {}\ntriplet loss margin = {}\n'.format(args.lr, args.tripletLoss_margin))
    print('max epochs = {}\n'.format(args.num_epochs))

    best_val_epoch, best_val_loss = 0, 0

    best_val_epoch, best_val_loss, all_epochs, past_train_losses, past_val_losses = fit(
        train_loader=train_dataloader, val_loader=val_dataloader, model=triplet_net,
        loss_fn=nn.TripletMarginLoss(margin=args.tripletLoss_margin), optimizer=optimizer, scheduler=scheduler,
        n_epochs=args.num_epochs, cuda=device, log_interval=args.log_interval, metrics=[], 
        start_epoch=0, early_stopping_interval=args.early_stopping_interval,
        num_accumulated_batches=args.num_accumulated_batches, 
        temp_model_path=os.path.join(args.temp_model_dir, 'temp_{}.pth'.format(the_name))
    )
    print('best_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss))

    # SAVE MODEL
    os.makedirs(os.path.dirname(args.posttrained_model_path), exist_ok=True)
    torch.save(embedder.state_dict(), args.posttrained_model_path)

    from datetime import datetime
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open('{}/results_{}.txt'.format(args.results_dir, datetime_str), 'w') as fout:
        json.dump(args.__dict__, fout, indent=2)
        fout.write('\nbest_val_epoch = {}\nbest_val_loss = {}\n'.format(best_val_epoch, best_val_loss))
        fout.write('\nepochs: {}\ntrain_losses: {}\nval_losses: {}\n'.format(all_epochs, past_train_losses, past_val_losses))
    torch.save(embedder.state_dict(), os.path.join(args.results_dir, 'weights_{}.pth'.format(datetime_str)))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Fingerprint Matcher')
    # training loop arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num-accumulated-batches', type=int, default=1,
                        help='number of accumulated batches before weight update (default: 1)')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--early-stopping-interval', type=int, default=65,
                        help='how long to train model before early stopping, if no improvement')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_step', type=int, default=None,
                        help='learning rate step interval (default: None)')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Learning rate step gamma (default: None)')
    parser.add_argument('--tripletLoss-margin', type=float, default=0.2,
                        help='Margin for triplet loss (default: 0.2)')
    # model arguments
    parser.add_argument('--pretrained-model-path', type=str, default=None,
                        help='path to pretrained model (default: None)')
    parser.add_argument('--posttrained-model-path', type=str, default='/data/therealgabeguo/fingerprint_weights/curr_model.pth',
                        help='path to save the model at')
    # saving arguments
    parser.add_argument('--temp_model_dir', type=str, default='temp_weights',
                        help='where to save the temporary model weights, as the model is training')
    parser.add_argument('--results_dir', type=str, default='/data/therealgabeguo/results',
                        help='what directory to save the results in')
    # dataset arguments
    parser.add_argument('--datasets', type=str, default='/data/therealgabeguo/fingerprint_data/sd302_split',
                        help='where is the data stored')
    parser.add_argument('--val-datasets', type=str, default='/data/therealgabeguo/fingerprint_data/sd302_split',
                        help='where is the validation data stored')
    parser.add_argument('--scale-factor', type=int, default=1,
                        help='number of times to go over the dataset to create triplets (default: 1)')
    parser.add_argument('--possible-fgrps', type=str, default='01 02 03 04 05 06 07 08 09 10',
                        help='Possible finger types to use in analysis (default: \'01 02 03 04 05 06 07 08 09 10\')')
    parser.add_argument('--diff-fingers-across-sets-train', action='store_true',
                        help='Whether to force different fingers across sets in training')
    parser.add_argument('--diff-sensors-across-sets-train', action='store_true',
                        help='Whether to force different sensors across sets in training')
    parser.add_argument('--diff-fingers-across-sets-val', action='store_true',
                        help='Whether to force different fingers across sets in validation')
    parser.add_argument('--diff-sensors-across-sets-val', action='store_true',
                        help='Whether to force different sensors across sets in validation')
    # miscellaneous arguments
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=300,
                        help='How many batches to go through before logging in training')
        
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main(args, device)