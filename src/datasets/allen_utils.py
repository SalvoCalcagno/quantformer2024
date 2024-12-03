import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
from src.utils import get_windows
#from src.models.patch_vqvae import Model
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import KBinsDiscretizer
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    
### ALLEN Dataset
EXPERIMENT_IDS = (501271265, 501484643, 501574836, 501704220, 501729039, 501836392, 502115959, 502608215, 503109347, 504637623, 510214538, 510514474, 510517131, 527048992, 531006860, 539670003, 540684467, 545446482)

class AllenSplitter():

    def __init__(self, manifest_file='YOUR_MANIFEST_FILE', mode='temporal') -> None:
        
        # init brain observatory cache
        self.boc = BrainObservatoryCache(manifest_file=manifest_file)
        self.modes = ['temporal']
        self.mode = mode
        if self.mode not in self.modes:
            raise ValueError(f"Mode must be one of {self.modes}")
        
    def __call__(self, experiment_ids: Any, valid_frac: float = 0.2, test_frac: float = 0.2) -> Any:
        if self.mode == 'temporal':
            return self.temporal_split(experiment_ids=experiment_ids, valid_frac=valid_frac, test_frac=test_frac) 
        
    def temporal_split(self, experiment_ids, valid_frac=0.2, test_frac=0.2):
        
        experiments = {
            experiment_id: self.boc.get_ophys_experiment_data(experiment_id) for experiment_id in experiment_ids
        }
            
        num_frames_per_experiment = {
            experiment_id: experiment.get_fluorescence_timestamps().shape[0] for experiment_id, experiment in experiments.items()
        }

        splitted_experiments = {
            split: {} for split in ['train', 'valid', 'test']
        }
        
        # iterate over all experiments
        for experiment_id, num_frames in num_frames_per_experiment.items():

            frames_valid = int(num_frames * valid_frac)
            frames_test = int(num_frames * test_frac)

            splitted_experiments['train'][experiment_id] = {}
            splitted_experiments['train'][experiment_id]['start_frame'] = 0
            splitted_experiments['train'][experiment_id]['end_frame'] = num_frames - frames_valid - frames_test

            splitted_experiments['valid'][experiment_id] = {}
            splitted_experiments['valid'][experiment_id]['start_frame'] = num_frames - frames_valid - frames_test
            splitted_experiments['valid'][experiment_id]['end_frame'] = num_frames - frames_test

            splitted_experiments['test'][experiment_id] = {}
            splitted_experiments['test'][experiment_id]['start_frame'] = num_frames - frames_test
            splitted_experiments['test'][experiment_id]['end_frame'] = num_frames

        return splitted_experiments

# this dictionary is valid only for session A
stimulus_type_to_class = {
    1: 0, # drifting_gratings
    2: 1, # natural_movie_one
    4: 2, # natural_movie_three
    8: 3, # spontaneous
}

class AllenDataset(Dataset):

    def __init__(self, 
            split_dir, 
            split, 
            split_type='temporal',
            no_valid=False,
            manifest='YOUR_MANIFEST_FILE', 
            window_size=100, 
            stride=100, 
            experiment_ids=None, 
            standardize=True,
            quantize=False,
            quantization_strategy='quantile',
            num_levels=256,
            use_levels=True, 
            transform=None,
            windowing_strategy='sliding',
            select_stimuli=['drifting_gratings'],
            use_stimuli_embeddings=False):
        
        # TODO: relax this constraint for containers
        # In order to relax this constraint, we need to remove the next line (the code is ready - at least I think so)
        assert len(experiment_ids) == 1, "You can load only a single experiment per time"

        super().__init__()
        self.manifest = manifest
        self.split = split
        self.split_type = split_type
        self.split_dir = split_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.boc = BrainObservatoryCache(manifest_file=self.manifest)
        self.use_stimuli_embeddings = use_stimuli_embeddings

        if self.split_type == 'temporal':
            # load temporal splits
            self.splitted_traces = json.load(open(f'{self.split_dir}/temporal_split.json', 'r'))
            responses = self.splitted_traces[self.split]
            # load stats
            stats = json.load(open(f'{self.split_dir}/stats.json', 'r'))
        elif self.split_type == 'stimuli':
            # load stimuli splits
            if no_valid:
                self.splitted_traces = json.load(open(f'{self.split_dir}/stimuli_split_noval.json', 'r'))
                # load stats
                stats = json.load(open(f'{self.split_dir}/stimuli_stats_noval.json', 'r'))
            else:
                self.splitted_traces = json.load(open(f'{self.split_dir}/stimuli_split.json', 'r'))
                # load stats
                stats = json.load(open(f'{self.split_dir}/stimuli_stats.json', 'r'))
            responses = self.splitted_traces[self.split]
            # select stimuli
            if select_stimuli is not None:
                responses = {k: v for k, v in responses.items() if '_'.join(k.split('_')[1:-1]) in select_stimuli}
            if quantize:
                print("It is not possible to quantize stimuli responses. Setting quantize to False")
                quantize = False
        
        # load peaks
        if windowing_strategy in ['peaks', 'hybrid']:
            peaks = torch.load(f"{self.split_dir}/peaks.pt")

        # load quantizers
        try:
            quantizers = torch.load(f'{split_dir}/quantizers_{num_levels}.pt')
            quantizers = {k: v[quantization_strategy] for k, v in quantizers.items()}
        except FileNotFoundError:
            quantizers = None
            if quantize:
                print("It is not possible to load the quantizers with the specified num_levels. You need to run the notebook 'quantize_signal.ipynb'")
                print("Setting quantize to False")
                quantize = False
            pass

        # load stimuli embeddings
        if self.split_type == 'stimuli' and self.use_stimuli_embeddings:
            assert len(select_stimuli) == 1, "You can use a single stimulus per time"
            self.stimuli_embeddings = torch.load(f"{self.split_dir}/stimuli_embeddings.pt")
            self.stimuli_embeddings = self.stimuli_embeddings[select_stimuli[0]] #(num_stimuli, embedding_dim)
            #Since the index of empty stimuluswill trigger the last elemetn of the embedding table.
            #To avoid this behaviour, we add a last row to the table.
            self.stimuli_embeddings = torch.cat((self.stimuli_embeddings, torch.zeros(1, self.stimuli_embeddings.shape[1]) -1 ), dim=0)

        # filter out experiments
        if experiment_ids is not None:
            # select only the ids
            responses = {k: v for k, v in responses.items() if str(k).split('_')[0] in experiment_ids}
            stats = {k: v for k, v in stats.items() if str(k).split('_')[0] in experiment_ids}
            if windowing_strategy == 'peaks':
                peaks = {k: v for k, v in peaks.items() if str(k).split('_')[0] in experiment_ids}
            if quantizers is not None:
                quantizers = {k: v for k, v in quantizers.items() if str(k) in experiment_ids}
        
        
        # add windows
        for experiment_id, response in responses.items():
            experiment_id = str(experiment_id).split('_')[0]
            if windowing_strategy == 'sliding':
                response['windows'] = get_windows(
                    response,
                    window_size=window_size, 
                    stride=stride
                )
            elif windowing_strategy == 'peaks':
                experiment_peaks = peaks[experiment_id]['peaks']
                response['windows'] = get_windows(
                    response,
                    window_size=window_size, 
                    strategy='peaks',
                    peaks=experiment_peaks
                )
            elif windowing_strategy == 'hybrid':
                response['windows'] = get_windows(
                    response,
                    window_size=window_size, 
                    stride=stride
                )
                experiment_peaks = peaks[experiment_id]['peaks']
                response['windows'] += get_windows(
                    response,
                    window_size=window_size, 
                    strategy='peaks',
                    peaks=experiment_peaks
                )
            else:
                raise ValueError(f"Windowing strategy must be one of ['sliding', 'peaks']")
            
        # merge responses, stats and quantizers
        self.responses = {
            i: {
                'experiment_id': k.split('_')[0],
                'split': split,
                **v,
                **stats[k.split('_')[0]],
                'quantizer': quantizers[k] if quantize else None,
            } for i, (k, v) in enumerate(responses.items())
        }

        for _, response in self.responses.items():

            # get experiment id
            experiment_id = response['experiment_id']

            # load trace for response (neurons, time)
            experiment = self.boc.get_ophys_experiment_data(int(experiment_id))
            time, trace = experiment.get_dff_traces()
            if self.split_type == 'stimuli':
                # get stimulus types
                s_types, s_ids = get_stimuli_arrays(experiment)

            if standardize:
                # load stats for response id
                mean = response['mean']
                std = response['std']
                # standardize trace
                trace = (trace - mean)/std
                # normalize time between -0.5 and 0.5
                time = (time - time.min())/(time.max() - time.min()) - 0.5

            trace = trace[:, response['start_frame']:response['end_frame']]
            time = time[response['start_frame']:response['end_frame']]
            if self.split_type == 'stimuli':
                s_types = s_types[response['start_frame']:response['end_frame']]
                s_ids = s_ids[response['start_frame']:response['end_frame']]
            #print(trace.shape, time.shape)

            if quantize:
                quantizer = response['quantizer']
                quantized_trace = quantizer.transform(trace.reshape(-1, 1))
                quantized_trace = torch.from_numpy(quantized_trace).int()
                if not use_levels:
                    # use quantized values
                    quantized_trace = quantizer.inverse_transform(quantized_trace)
                # restore the original shape
                quantized_trace = quantized_trace.reshape(trace.shape)
                trace = quantized_trace

            # save trace and time 
            response['trace'] = trace.transpose(1, 0)
            response['time'] = time.reshape(-1, 1)
            if self.split_type == 'stimuli':
                response['s_types'] = s_types.reshape(-1, 1)
                response['s_ids'] = s_ids.reshape(-1, 1)
            #print(f"Trace shape: {response['trace'].shape}, time shape: {response['time'].shape}")

        self.windows = [
            (id, start_frame, end_frame) 
            for id, response in self.responses.items() 
            for start_frame, end_frame in response['windows']
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):

        id, start, end = self.windows[index]
        trace = self.responses[id]['trace'][start:end, :]
        time = self.responses[id]['time'][start:end, :]
        if 's_types' in self.responses[id]:
            s_types = self.responses[id]['s_types'][start:end, :]
            s_type = stimulus_type_to_class[s_types[0, 0].item()]
        if self.split_type == 'stimuli':
            s_ids = self.responses[id]['s_ids'][start:end, :]
            
            if self.use_stimuli_embeddings:
                s_ids = s_ids.reshape(-1)
                s_embeddings = self.stimuli_embeddings[s_ids, :]

                data = {
                    'trace': trace,
                    'time': time,
                    's_embeddings': s_embeddings
                }
                
            else:

                data = {
                    'trace': trace,
                    'time': time,
                    's_types': s_types,
                    's_ids': s_ids,
                    'class': s_type
                }

        else:
            data = {
                'trace': trace,
                'time': time,
                #'class': s_type
            }

        # apply transformation
        if self.transform is not None:
            data = self.transform(data)
        
        return data

# This is an example of a container
# {545446482: 'A', 544507627: 'B', 543677427: 'C'}

class AllenContainerDataset(Dataset):

    def __init__(self,
            data_dir = '../../data/allen',
            manifest = 'YOUR_MANIFEST_FILE',
            stimuli=['natural_movie_one', 'spontaneous'],
            fold = 0,
            split = 'train',
            standardize = True,
            window_size=100,
            stride=1,
            windowing_strategy='sliding',
            transform=None
        ):

        self.transform = transform
        boc = BrainObservatoryCache(manifest_file=manifest)
        folds = experiment_ids = torch.load(f"{data_dir}/containter_folds.pt")
        experiment_ids = folds[fold][split]

        # get responses
        self.responses = torch.load(f"{data_dir}/container_responses.pt")

        # get common cells
        self.common_cells = torch.load(f"{data_dir}/container_common_cells.pt")

        # get peaks
        self.peaks = torch.load(f"{data_dir}/peaks_container.pt")

        # filter experiments
        self.responses = [
            response for response in self.responses
            if response['experiment_id'] in experiment_ids
        ]
        # filter stimuli
        self.responses = [
            response for response in self.responses
            if response['stimulus'] in stimuli
        ]

        # add windows
        for response in self.responses:
            experiment_id = response['experiment_id']
            if windowing_strategy == 'sliding':
                response['windows'] = get_windows(
                    response,
                    window_size=window_size, 
                    stride=stride
                )
            elif windowing_strategy == 'peaks':
                experiment_peaks = self.peaks[str(experiment_id)]['peaks']
                response['windows'] = get_windows(
                    response,
                    window_size=window_size, 
                    strategy='peaks',
                    peaks=experiment_peaks
                )
            else:
                raise ValueError(f"Windowing strategy must be one of ['sliding', 'peaks']")
            
        # get data
        for response in self.responses:

            # experiment id
            experiment_id = response['experiment_id']

            # load trace for response (neurons, time)
            experiment = boc.get_ophys_experiment_data(ophys_experiment_id=experiment_id)

            # get session
            cells = experiment.get_cell_specimen_ids()

            # get common cells indices
            cell_ids = np.where(np.isin(cells, self.common_cells))[0]

            # get traces
            time, trace = experiment.get_dff_traces(cell_specimen_ids=cells[cell_ids])

            trace = trace[:, response['start_frame']:response['end_frame']]

            if standardize:
                mean = folds[fold]['mean']
                std = folds[fold]['std']
                trace = (trace - mean) / std

            time = time[response['start_frame']:response['end_frame']]
            response['trace'] = trace.transpose(1, 0)
            response['time'] = time.reshape(-1, 1)

        self.windows = []
        for id, response in enumerate(self.responses):
            label = 0 if response['stimulus'] == 'spontaneous' else 1
            for start_frame, end_frame in response['windows']:
                self.windows.append((id, start_frame, end_frame, label))
    
    def __getitem__(self, index):
        
        id, start_frame, end_frame, label = self.windows[index]
        trace = self.responses[id]['trace'][start_frame:end_frame, :]
        time = self.responses[id]['time'][start_frame:end_frame, :]
        
        data = {
            'trace': trace,
            'time': time,
            'class': label
        }

        # apply transformation
        if self.transform is not None:
            data = self.transform(data)

        return data
        
    def __len__(self):
        return len(self.windows)

class AllenSingleStimulusDataset(object):
    
    def __init__(self, manifest_file, split_dir, split, experiment_id, source_length, forecast_length):
        
        # Save Parameters
        self.manifest_file = manifest_file
        self.split_dir = split_dir
        self.split = split
        self.experiment_id = experiment_id
        self.source_length = source_length
        self.forecast_length = forecast_length
        
        # Load Trace for Response (neurons, time)
        boc = BrainObservatoryCache(manifest_file=manifest_file)
        experiment = boc.get_ophys_experiment_data(int(experiment_id))
        self.time, self.trace = experiment.get_dff_traces()
        
        # Get Stimulus Epoch Table
        stimulus_epoch_table = experiment.get_stimulus_epoch_table()
        # Get all responses for drifting gratings
        responses = stimulus_epoch_table[stimulus_epoch_table['stimulus']=='drifting_gratings']
        # Select Offset for Train and Test
        offset_split = responses.iloc[1]['end']
        
        # Get stimulus table
        stimulus_table = experiment.get_stimulus_table("drifting_gratings")
        
        # get stimulus embedding
        self.stimuli_embeddings = torch.load(f"{split_dir}/stimuli_embeddings.pt")["drifting_gratings"]
        
        # Get Stimuli IDS
        _, s_ids = get_stimuli_arrays(experiment)
        
        # Get windows for train and test
        self.windows = []
        self.stimuli_ids = []
        
        for _, item in stimulus_table.iterrows():
            if split == 'train':
                if item['start'] < offset_split:
                    self.windows.append((int(item['start'])-source_length, int(item['start']) + forecast_length))
                    self.stimuli_ids.append(s_ids[int(item['start'])])
            else:
                if item['start'] > offset_split:
                    self.windows.append((int(item['start'])-source_length, int(item['start']) + forecast_length))
                    self.stimuli_ids.append(s_ids[int(item['start'])])
            
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Get window
        start, end = self.windows[idx]
        # Get trace
        trace = self.trace[:, start:end]
        # Get stimulus id
        s_id = self.stimuli_ids[idx]
        # Get stimulus embedding
        s_embedding = self.stimuli_embeddings[s_id]
        # Transpose trace
        trace = trace.transpose(1, 0)
        return trace[:self.source_length, :], trace [self.source_length:, :], s_embedding

stimuli_types = {
    'no_stimulus': 0,
    'drifting_gratings': 1,
    'natural_movie_one': 2,
    'natural_movie_two': 3,
    'natural_movie_three': 4,
    'natural_scenes': 5,
    'static_gratings': 6,
    'locally_sparse_noise': 7,
    'spontaneous': 8
}

# for dgx
gratings = pd.read_csv('data/gratings.csv')
#gratings = None

# get id for each stimulus
def get_grating_id(orientation, spatial_frequency, phase, temporal_frequency):
    if spatial_frequency is None:
        spatial_frequency = 0.04

    if temporal_frequency is None:
        temporal_frequency = 0

    if phase is None:
        phase = 0

    if orientation is None:
        orientation = 0
        spatial_frequency = 0
        temporal_frequency = 0
        phase = 0
    
    eps = 1e-6
    filtered_gratings = gratings[abs(gratings['orientation'] - orientation)<eps]
    filtered_gratings = filtered_gratings[abs(filtered_gratings['spatial_frequency'] - spatial_frequency)<eps]
    filtered_gratings = filtered_gratings[abs(filtered_gratings['temporal_frequency'] - temporal_frequency)<eps]
    filtered_gratings = filtered_gratings[abs(filtered_gratings['phase'] - phase)<eps]
    
    return filtered_gratings.iloc[0].name

def build_stimulus_type_array(stimulus_epoch_table, num_timepoints):
    # init array
    stimulus_type_array = np.zeros(num_timepoints, dtype=np.int64)

    # build array
    for _, stimulus_batch in stimulus_epoch_table.iterrows():
        stimulus = stimulus_batch['stimulus']
        start = stimulus_batch['start']
        end = stimulus_batch['end']

        try:
            stimulus_type_array[start:end] = stimuli_types[stimulus]
        except KeyError:
            if stimulus in ['locally_sparse_noise_8deg', 'locally_sparse_noise_4deg']:
                stimulus_type_array[start:end] = stimuli_types['locally_sparse_noise']
    
    return stimulus_type_array

def build_ids_array(experiment, stimulus_epoch_table=None, num_timepoints=None):

    # get the table of stimuli
    if stimulus_epoch_table is None:
        stimulus_epoch_table = experiment.get_stimulus_epoch_table()

    # get only the stimuli in the session
    session_stimuli = stimulus_epoch_table['stimulus'].unique()

    # init array
    ids = np.zeros(num_timepoints, dtype=np.int64) - 1
    
    # iterate over stimuli
    for stimulus in session_stimuli:

        # get stimulus-specific table
        stimulus_table = experiment.get_stimulus_table(stimulus)

        # get ids for each stimulus
        if stimulus =='drifting_gratings':
            for _, grating in stimulus_table.iterrows():
                
                start = grating['start']
                end = grating['end']
                orientation = None if pd.isna(grating['orientation']) else grating['orientation']
                temporal_frequency = None if pd.isna(grating['temporal_frequency']) else grating['temporal_frequency']
                
                grating_id = get_grating_id(
                    orientation=orientation, 
                    spatial_frequency=None, 
                    phase=None, 
                    temporal_frequency=temporal_frequency
                )
                ids[int(start):int(end)] = grating_id

        if stimulus == 'static_gratings':
            for _, grating in stimulus_table.iterrows():
                
                start = grating['start']
                end = grating['end']
                orientation = None if pd.isna(grating['orientation']) else grating['orientation']
                spatial_frequency = None if pd.isna(grating['spatial_frequency']) else grating['spatial_frequency']
                phase = None if pd.isna(grating['phase']) else grating['phase']
                
                grating_id = get_grating_id(
                    orientation=orientation, 
                    spatial_frequency=spatial_frequency, 
                    phase=phase, 
                    temporal_frequency=None
                )
                ids[int(start):int(end)] = grating_id

        if stimulus in ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']:
            for _, frame in stimulus_table.iterrows():
                
                start = frame['start']
                ids[start] = frame['frame']

        if stimulus == 'natural_scenes':
            for _, scene in stimulus_table.iterrows():
                
                start = scene['start']
                end = scene['end']
                ids[int(start):int(end)] = scene['frame']

        if stimulus in ['locally_sparse_noise', 'locally_sparse_noise_8deg', 'locally_sparse_noise_4deg']:
            for _, noise in stimulus_table.iterrows():
                
                start = noise['start']
                end = noise['end']
                ids[int(start):int(end)] = noise['frame']
    
    return ids

def get_stimuli_arrays(experiment):

    # get the table of stimuli
    stimulus_epoch_table = experiment.get_stimulus_epoch_table()
    
    # build arrays
    num_timepoints = experiment.get_fluorescence_timestamps().shape[0]
    stimulus_type_array = build_stimulus_type_array(stimulus_epoch_table, num_timepoints)
    ids_array = build_ids_array(experiment, stimulus_epoch_table, num_timepoints)

    return stimulus_type_array, ids_array
    
    # compute a single value for each patch
    patch_mean = patches.mean(dim=-1)
    patch_mean_shape = patch_mean.shape

    # get quantization levels
    levels = quantizer.transform(patch_mean.reshape(-1, 1))
    levels = torch.from_numpy(levels).long()
    
    # convert levels to classes
    classes = torch.zeros_like(levels)
    for level, class_ in level2class.items():
        classes[levels == level] = class_
    classes = classes.reshape(patch_mean_shape)
    levels = levels.reshape(patch_mean_shape)
    
    # mask classes
    classes[torch.logical_not(mask)] = -100
    
    return levels, classes