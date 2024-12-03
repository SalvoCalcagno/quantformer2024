import os
import sys
import json
import wandb
import torch
import argparse
import warnings
from time import time
from pathlib import Path
from random import shuffle
from src.saver import Saver
from datetime import datetime
import src.trainers as trainers
from itertools import zip_longest
import torchvision.transforms as T
from src.dataset import get_datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset

def parse():
    # Init parser
    parser = argparse.ArgumentParser()

    # Enable sweep
    parser.add_argument('--sweep', action='store_true')
    
    # Dataset options
    parser.add_argument('--root', type=Path, default=Path("data/"))
    parser.add_argument('--dataset', default='allen_singlestimulus')
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--window-size', type=int, default=160)
    parser.add_argument('--forecast-window', type=int, default=64)
    parser.add_argument('--no-valid', type=bool, default=True)
    parser.add_argument('--num-classes', type=int, default=2)
    
    # Allen Single Stimulus
    parser.add_argument('--container-id', type=int, default=None)
    parser.add_argument('--stimulus', type=str, default='drifting_gratings')
    parser.add_argument('--monitor_height', type=int, default=90)
    parser.add_argument('--on_off_percentile', type=int, default=95)
    parser.add_argument('--stimuli_format', type=str, default='embedding', help='embedding or raw')
    parser.add_argument('--trace_format', type=str, default='corrected_fluorescence_dff', help='corrected_fluorescence_dff')
    parser.add_argument('--labels_format', type=str, default='tsai_wen')
    parser.add_argument('--threshold', type=float, default=0.1)
    
    # Experiment options
    parser.add_argument('-t', '--tag', default=None)
    parser.add_argument('--project', default='MY_PROJECT')
    parser.add_argument('--entity', default='MY_ENTITY')
    parser.add_argument('--logdir', default='exps', type=Path)
    parser.add_argument('--eval-after', type=int, default=-1, help='evaluate only starting from a certain epoch')
    parser.add_argument('--plot-every', type=int, default=10)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--save-every', type=int, default=-1)
    parser.add_argument('--log-histograms', action='store_true')
    parser.add_argument('--watch-model', action='store_true')

    # Model options
    parser.add_argument('--model', default='informer')
    parser.add_argument('--verbose', action='store_true')

    # Mixed model-specific options
    parser.add_argument('--patch-len', type=int, default=16)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--rnn-type', type=str, default='bilstm')
    parser.add_argument('--cls', action='store_true')
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--d-ff', type=int, default=None) # if None, d_ff = d_model * 4
    parser.add_argument('--max-sequence-length', type=int, default=5000)
    # Informer
    # enc_in, dec_in, c_out computed from input_size, output_size
    # seq_len, label_len, pred_len computed from source_length, target_size
    # d_model as before
    # d_ff = d_model * 4
    # n_heads = nhead
    parser.add_argument('--factor', type=int, default=10) # informer default 5
    parser.add_argument('--e-layers', type=int, default=3) # informer default 2
    parser.add_argument('--d-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--attn', type=str, default='prob')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--disable-distil', action='store_true')
    parser.add_argument('--disable-mix', action='store_true')
    parser.add_argument('--output-attention', action='store_true')
    parser.add_argument('--freq', type=str, default='a') # it maps the number of timefeatures. a=1. For other values see embed.py
    parser.add_argument('--padding', type=int, default=1)
    # Crossformer
    # factor as in Informer, but default 10
    # e_layers as in Informer, but default 3
    parser.add_argument('--seg-len', type=int, default=16, help="length of segments (each time series is divided in segments of this length)")
    parser.add_argument('--win-size', type=int, default=4, help="at the end of layer, win_size adjacent segments are aggregated")
    parser.add_argument('--baseline', action='store_true')

    # Training options
    parser.add_argument('--loss_fn', type=str, default='cross_entropy')
    parser.add_argument('--disable-teacher-forcing', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--trainer', default='trainer')
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--use-scheduler', action='store_true')
    parser.add_argument('--reduce-lr-every', type=int)
    parser.add_argument('--reduce-lr-factor', type=float, default=0.5)
    parser.add_argument('--patience', type=float, default=10)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--activity-reg-lambda', type=float, help="lambda value for activity regularization")
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('--resume')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, choices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'], default='cuda:0')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--overfit-batch', action='store_true')
    parser.add_argument('--criterion', type=str, choices=['focal', 'mae', 'mse', 'rmse', 'mape', 'smape', 'quantile','mase', 'sdtw'], default='mse')
    parser.add_argument('--quantile', type=float, default=0.5)
    parser.add_argument('--disable-eval-call',  action='store_true')
    # if resume is not None, decide whether to finetune only the last layer or the whole model
    # when selecting e2e the linar head is trained for 1/3 of the total epochs and then the whole model is trained
    parser.add_argument('--finetuning', type=str, default='e2e', choices=['e2e', 'linear']) 
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--masked-mse', action='store_true')
    parser.add_argument('--learnable-mask', action='store_true')
    parser.add_argument('--mask-on-peaks', action='store_true')
    parser.add_argument('--warmup-factor', type=float, default=1)
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--quantized', type=str, choices=['disabled', 'uniform', 'vqvae'], default='disabled')
    parser.add_argument('--quantizer', type=str, default=None)
    # Debug options
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    args.exp_name = f"{args.model}_{args.container_id}_{args.stimulus}_lr{args.lr}"
    if args.tag is not None:
        args.exp_name += f"_{args.tag}"
        
    args.logdir = args.logdir / args.dataset / args.model
        
    return args

def main(args):
    
    # set visible cuda devices to 1
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"

    if args.sweep:
        # Load sweep config
        with open("sweep_config.json", "r") as fp:
            sweep_configuration = json.load(fp)
            
        # Initialize sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project, entity=args.entity)
        #sweep_id = 'YOUR_ID'

    def train_func():
        
        global args
        
        if args.sweep:

            # Init run
            run = wandb.init(project=args.dataset, entity=args.entity)

            # Load parameters
            sweep_params = dict(wandb.config)

            # Update args
            args.__dict__.update(sweep_params)
            
            # Rename wandb experiment
            args.exp_name = f"{args.model}_{args.container_id}_{args.stimulus}_lr{args.lr}"
            if args.tag is not None:
                args.exp_name += f"_{args.tag}"

        datasets, args = get_datasets(args)

        # Check overfit-batch (for debug)
        if args.overfit_batch:
            # raise NotImplementedError('overfit_batch not implemented in this case')
            datasets = {
                split: Subset(datasets[split], list(range(args.batch_size)))
                for split in datasets
            }
        
        # Print dataset info
        for split in datasets:
            print(f"{split}: {len(datasets[split])}")

        # Run training
        if not args.sweep:
            if args.debug:
                run = wandb.init(mode="disabled")
            else:
                run = wandb.init(project=args.project, entity=args.entity, config=args)
        
        # Save Wandb Configuration
        args_dict = vars(args)
        for arg_name, arg in args_dict.items():
            try:
                wandb.config[arg_name] = arg
            except Exception as e:
                print(f"Could not save {arg_name} to wandb")
                continue
            
        # Create unique name for the run
        timestamp_str = datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')
        if args.exp_name is not None:
            # Append timestamp
            args.exp_name = f"{timestamp_str}_{args.exp_name}"
        else:
            args.exp_name = timestamp_str
            
        # Decide run_name!
        run.name = f"{args.exp_name}"
        
        # Define trainer
        trainer_module = getattr(trainers, args.trainer)
        trainer_class = getattr(trainer_module, 'Trainer')
        trainer = trainer_class(args)

        if args.pretrain:
            if args.resume is None:
                warnings.warn("Pretraining requires a pretrained model. The model will be trained from scratch.")
            model, metrics = trainer.pretrain(datasets)
        else:
            model, metrics = trainer.train(datasets)

        if not args.sweep:
            # Close saver
            wandb.finish()
        else:
            run.finish()
    
    if args.sweep:
        # Run sweep
        wandb.agent(sweep_id, function=train_func, project=args.project, entity=args.entity)
    else:
        train_func()
        
if __name__ == '__main__':
    # Get params
    args = parse()
    main(args)