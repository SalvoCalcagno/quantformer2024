import os
import torch
import numpy as np

from src.datasets.transforms import *
from src.datasets.allen_singlestimulus import AllenSingleStimulusDataset, CONTAINERS, CONTAINERS_DEBUG, get_experiment_id
from src.datasets.foundation_containers import FOUNDATION_CONTAINERS
from torch.utils.data import ConcatDataset

from allensdk.brain_observatory.brain_observatory_exceptions import EpochSeparationException

from torchvision import transforms as T

allowed_models = {
    'allen_singlestimulus': [
        'autoformer', 
        'informer', 
        'patchtst_self', 
        'patchtst', 
        'quantformer', 
        'quantcoder', 
        'autoregressive_lstm', 
        'crossformer', 
        'lstm_classification', 
        'brain_transformer', 
        'patchtst_finetune', 
        'patchtst_finetune_nochans', 
        'brainlm',
        'brainlm_quantized',
        'patch_vqvae',
        "crossquantcoder",
        ],
    'pretrain': [
        'patchtst_pretrain', 
        'linear', 
        'conv', 
        'quantformer', 
        'quantcoder', 
        'crossquantformer', 
        "crossquantcoder"
    ],
}
allowed_models['cross_stimuli'] = allowed_models['pretrain']
allowed_models['cross_stimuli2'] = allowed_models['pretrain']
allowed_models['cross_session'] = allowed_models['pretrain']
allowed_models['cross_subject'] = allowed_models['pretrain']
allowed_models['foundation'] = allowed_models['pretrain']

def get_allensinglestimulus_datasets(args, splits=['train', 'test']):
    
    # Hard-Coded Args
    args.manifest = "PATH_TO_MANIFEST_FILE"
    args.input_dir = "data/allen"

    # Set Trainer
    if args.model == 'quantformer':
        args.trainer = 'trainer_quantformer'
    elif args.model == 'quantcoder':
        args.trainer = 'trainer_quantcoder'
    elif args.model in ['autoformer', 'informer', 'autoregressive_lstm', 'crossformer', "lstm_classification"]:
        args.trainer = 'trainer'
    elif args.model == 'brainlm':
        args.trainer = 'trainer_brainlm'
    elif args.dataset == 'pretrain':
        args.trainer = 'pretrainer'
    else:
        raise NotImplementedError(f"Model {args.model} not supported for Allen dataset")

    args.source_length = args.window_size - args.forecast_window
    if args.stimulus is not None and args.container_id is not None:
        args.experiment_ids = [get_experiment_id(args.container_id, args.stimulus)]
    elif type(args.experiment_ids) == str:
        args.experiment_ids = [str(args.experiment_ids)]

    #splits = ['train', 'test']
    
    # Load cached dataset if available
    cache_path = os.path.join(os.path.dirname(args.manifest), "allen_cache", f"{args.container_id}_{args.stimulus}_{''.join(splits)}.pt")
    if os.path.exists(cache_path) and args.model!='brainlm':
        print(f"Loading cached dataset")
        datasets = torch.load(cache_path)
    
    else:
        datasets = {
                split: AllenSingleStimulusDataset(
                    manifest_file=args.manifest,
                    split_dir=args.input_dir,
                    split=split,
                    experiment_id=args.experiment_ids[0],
                    stimulus=args.stimulus,
                    source_length=args.source_length,
                    forecast_length=args.forecast_window,
                    monitor_height=args.monitor_height,
                    on_off_percentile=args.on_off_percentile,
                    stimuli_format = args.stimuli_format,
                    trace_format=args.trace_format,
                    labels_format=args.labels_format,
                    threshold=args.threshold,
                )
                for split in splits
        }
        
        # cache the prepared dataset
        os.makedirs(os.path.join(os.path.dirname(args.manifest), "allen_cache"), exist_ok=True)
        torch.save(datasets, cache_path)
    
    sample = datasets[splits[0]][0]
    num_features = sample['src'].shape[0]
    # LSTM autoregressive specific
    args.input_size = num_features
    args.hidden_size = args.d_model
    args.output_size = num_features
    args.output_sequence_length = args.forecast_window
    args.stim_size = sample['stim'].shape[0]
    # Crossformer specific
    args.data_dim = num_features
    args.seq_len = args.source_length
    args.in_len = args.source_length
    args.out_len = args.forecast_window
    args.stim_size = sample['stim'].shape[0]
    # Quantformer specific
    args.d_ff = args.d_model*4
    args.n_layers = args.nlayers
    args.n_heads = args.nhead
    args.c_in = num_features
    args.target_dim = args.forecast_window
    # PatchTST specific
    args.enc_in = num_features
    args.seq_len = args.source_length
    if args.cls:
        args.pred_len = args.num_classes
        
    else:
        args.pred_len = args.forecast_window
    # Informer specific
    args.enc_in = num_features
    args.dec_in = num_features
    args.c_out = num_features
    args.seq_len = args.source_length
    args.label_len = args.forecast_window
    #args.pred_len = args.forecast_window
    args.embed = 'timeF'
    args.freq = 'a'
    # Autoformer specific
    args.task_name = 'short_term_forecast'
    args.output_attention = False
    args.moving_avg = 3
    args.stim_size = sample['stim'].shape[0]
    args.e_layers = args.nlayers
    args.d_layers = args.nlayers
    args.activation = 'gelu'
    args.label_len = args.source_length // 2
    if args.model == 'autoformer':
        args.embed='fixed'
    # Quantcoder Specific
    if args.quantizer is not None:
        # delete tag
        quantizer_name = args.quantizer
        for tag in ['temp_setting', 'loo_setting', 'kspace', 'lrspace']:
            if tag in quantizer_name:
                quantizer_name = quantizer_name.replace(f"_{tag}", "")
        quantcoder_args = quantizer_name.split('_')[-4:]
        if args.ablate_quantizer:
            quantcoder_args = ["K1"] + quantcoder_args
        args.K = int(quantcoder_args[0].split('K')[1])
        args.d_model = int(quantcoder_args[1].split('D')[1])
        args.d_ff = args.d_model*4
        args.nlayers = int(quantcoder_args[2].split('L')[1])
        args.n_layers = args.nlayers
        args.nhead = int(quantcoder_args[3].split('H')[1])
        args.n_heads = args.nhead
    
    # stride was used for windowing, now it is used for patching
    args.stride = args.patch_len 
    args.num_patch = args.window_size//args.patch_len

    return datasets, args

def get_pretrain_datasets(args):
    
    datasets = []
    
    if args.debug:
        containers = CONTAINERS_DEBUG
    else:
        containers = CONTAINERS
    
    for container_id in list(containers.keys()): # [:args.nexp]
        args.container_id = container_id
        s_datasets, args = get_allensinglestimulus_datasets(args)
        datasets.append(s_datasets)
        
    splits = s_datasets.keys()
    datasets = {
        split: ConcatDataset([dataset[split] for dataset in datasets])
        for split in splits
    }
    
    #args.trainer = 'pretrainer'
    if args.model in ['quantformer', 'crossquantformer']:
        args.trainer = 'trainer_quantformer'
        args.c_in = 512
    else:
        args.trainer = 'pretrainer'
    
    return datasets, args
    
def get_cross_stimuli_datasets(args):
    
    datasets = []
    
    if args.debug:
        containers = CONTAINERS_DEBUG
    else:
        containers = CONTAINERS
    
    stimulus_to_exclude = args.stimulus
    stimuli = ['drifting_gratings', 'static_gratings', 'locally_sparse_noise', 'natural_scenes']
    stimuli.remove(stimulus_to_exclude)
    
    assert len(stimuli) == 3, "Only 3 stimuli are allowed for cross-stimulus training"
    
    # Setting 2 (excluding natural scenes)
    # Train Set = |dg_all| + |sg_all|
    # Test Set = |lsn_all|
    train_datasets = []
    test_datasets = []
    test_stimulus = stimuli.pop()
        
    for container_id in containers.keys():
        args.container_id = container_id
        for stimulus in stimuli:
            args.stimulus = stimulus
            s_datasets, args = get_allensinglestimulus_datasets(args, splits=['all'])
            train_datasets.append(s_datasets)
        args.stimulus = test_stimulus
        s_datasets, args = get_allensinglestimulus_datasets(args, splits=['all'])
        test_datasets.append(s_datasets)
    
    datasets = {
        'train': ConcatDataset([dataset['all'] for dataset in train_datasets]),
        'test': ConcatDataset([dataset['all'] for dataset in test_datasets])
    }
    
    #args.trainer = 'pretrainer'
    if args.model == 'quantformer':
        args.trainer = 'trainer_quantformer'
    else:
        args.trainer = 'pretrainer'
    
    return datasets, args

def get_cross_stimuli_datasets2(args):
    
    datasets = []
    
    if args.debug:
        containers = CONTAINERS_DEBUG
    else:
        containers = CONTAINERS
    
    stimulus_to_exclude = args.stimulus
    stimuli = ['drifting_gratings', 'static_gratings', 'locally_sparse_noise', 'natural_scenes']
    stimuli.remove(stimulus_to_exclude)
    
    assert len(stimuli) == 3, "Only 3 stimuli are allowed for cross-stimulus training"
    
    # Setting 1 (excluding natural scenes)
    # Train Set = |dg_train| + |sg_train| + |lsn_train|
    # Test Set = |dg_test| + |sg_test| + |lsn_test|
    for container_id in containers.keys():
        args.container_id = container_id
        for stimulus in stimuli:
            args.stimulus = stimulus
            s_datasets, args = get_allensinglestimulus_datasets(args)
            datasets.append(s_datasets)
        
    splits = s_datasets.keys()
    datasets = {
        split: ConcatDataset([dataset[split] for dataset in datasets])
        for split in splits
    }
    
    #args.trainer = 'pretrainer'
    if args.model == 'quantformer':
        args.trainer = 'trainer_quantformer'
    else:
        args.trainer = 'pretrainer'
    
    return datasets, args

def get_cross_session_datasets(args):
    
    # If you want to downstream on session A, call cross stimuli with drifiting_gratings as stimulus to exclude
    # If you want to downstream on session C, call cross stimuli with locally sparse noise as stimulus to exclude
    # if you want to downstream on session B, call this function
    
    datasets = []
    
    if args.debug:
        containers = CONTAINERS_DEBUG
    else:
        containers = CONTAINERS
    
    # describe sessions A and C as stimuli. Don't include session B's stimulus (natural scenes and static gratings)
    stimuli = ['drifting_gratings', 'locally_sparse_noise']
 
    # Train Set = |dg_train| + |lsn_train|
    # Test Set = |dg_test| + |lsn_test|
    for container_id in containers.keys():
        args.container_id = container_id
        for stimulus in stimuli:
            args.stimulus = stimulus
            s_datasets, args = get_allensinglestimulus_datasets(args)
            datasets.append(s_datasets)
        
    splits = s_datasets.keys()
    datasets = {
        split: ConcatDataset([dataset[split] for dataset in datasets])
        for split in splits
    }
    
    #args.trainer = 'pretrainer'
    if args.model == 'quantformer':
        args.trainer = 'trainer_quantformer'
    else:
        args.trainer = 'pretrainer'
    
    return datasets, args

def get_cross_subject_datasets(args):
    
    datasets = []
    
    if args.debug:
        containers = CONTAINERS_DEBUG
    else:
        containers = CONTAINERS
    
    # exclude container_id
    containers = list(containers.keys())
    container_to_exclude = args.container_id
    containers.remove(container_to_exclude)
    
    for container_id in containers:
        args.container_id = container_id
        for stimulus in ['drifting_gratings', 'static_gratings', 'locally_sparse_noise', 'natural_scenes']:
            args.stimulus = stimulus
            s_datasets, args = get_allensinglestimulus_datasets(args)
            datasets.append(s_datasets)
        
    splits = s_datasets.keys()
    datasets = {
        split: ConcatDataset([dataset[split] for dataset in datasets])
        for split in splits
    }
    
    #args.trainer = 'pretrainer'
    if args.model == 'quantformer':
        args.trainer = 'trainer_quantformer'
    else:
        args.trainer = 'pretrainer'
    
    return datasets, args

def get_foundation_datasets(args):
    
    datasets = []
    
    # select all containers (up to nexp)
    containers = FOUNDATION_CONTAINERS
    containers = list(containers.keys())
 
    count = 0
    for container_id in containers:
        if count < args.nexp:
            count += 1
            args.container_id = str(container_id)
            for stimulus in ['drifting_gratings']:
                args.stimulus = stimulus
                try:
                    s_datasets, args = get_allensinglestimulus_datasets(args)
                    datasets.append(s_datasets)
                except EpochSeparationException:
                    print(f"Skipping container {container_id}")
                    count -= 1
                    break
                except UnboundLocalError:
                    print(f"Skipping stimulus {stimulus} for container {container_id}")
                    continue
        else:
            break
        
    print(f"Loaded {count} containers")
        
    splits = s_datasets.keys()
    datasets = {
        split: ConcatDataset([dataset[split] for dataset in datasets])
        for split in splits
    }
    
    #args.trainer = 'pretrainer'
    if args.model == 'quantformer':
        args.trainer = 'trainer_quantformer'
    else:
        args.trainer = 'pretrainer'
    
    return datasets, args

get_functions = {
    'allen_singlestimulus': get_allensinglestimulus_datasets,
    'pretrain': get_pretrain_datasets,
    'cross_stimuli': get_cross_stimuli_datasets,
    'cross_stimuli2': get_cross_stimuli_datasets2,
    'cross_session': get_cross_session_datasets,
    'cross_subject': get_cross_subject_datasets,
    'foundation': get_foundation_datasets,
}

def get_datasets(args):

    # Set Seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Check Allowed Models
    assert args.model in allowed_models[args.dataset], f"Model {args.model} not allowed for {args.dataset} dataset"
    
    # Execute the proper function
    return get_functions[args.dataset](args)