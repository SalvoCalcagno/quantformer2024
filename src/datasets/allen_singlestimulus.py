import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from oasis.functions import deconvolve
from src.datasets.allen_utils import get_stimuli_arrays
from src.datasets.foundation_containers import FOUNDATION_CONTAINERS
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.stimulus_info import BrainObservatoryMonitor, get_spatial_grating, get_spatio_temporal_grating

CONTAINERS = {
    '511507650': {'B': '501794235', 'C': '500855614', 'A': '502115959'},
#    '511509529': {'B': '500964514', 'C': '501337989', 'A': '501271265'},
#    '511510650': {'A': '501836392', 'C': '501717543', 'B': '501886692'},
    '511510667': {'C': '501773889', 'B': '501498760', 'A': '501574836'},
    '511510675': {'A': '510214538', 'C': '509729072', 'B': '509962140'},
    '511510699': {'C': '502974807', 'B': '502810282', 'A': '502608215'},
    '511510718': {'B': '510345479', 'C': '510174759', 'A': '510514474'},
#    '511510736': {'B': '501559087', 'A': '501704220', 'C': '501474098'},
    '511510779': {'C': '502634578', 'A': '503109347', 'B': '503019786'},
    '511510855': {'B': '510705057', 'A': '510517131', 'C': '509644421'},
    '511510989': {'C': '501788003', 'B': '501567237', 'A': '501729039'},
    '526481129': {'C': '526768996', 'B': '528480613', 'A': '527048992'},
    '536323956': {'A': '540684467', 'B': '541048140', 'C': '539515366'},
    '543677425': {'A': '545446482', 'C': '543677427', 'B': '544507627'}
}

CONTAINERS_DEBUG = {
    '536323956': {'A': '540684467', 'B': '541048140', 'C': '539515366'},
    '543677425': {'A': '545446482', 'C': '543677427', 'B': '544507627'}
}

SAMPLING_RATE = 30
BASELINE_FRAMES = int(SAMPLING_RATE * 2)
                
STIM_SESSION = {
    'drifting_gratings': 'A',
    'static_gratings': 'B',
    'natural_scenes': 'B',
    'locally_sparse_noise': 'C',
}

OFFSET_SPLIT_DICT = {
    "drifting_gratings": 1,
    "static_gratings": 1,
    "natural_scenes": 1,
    "natural_movie_three": 0,
    "locally_sparse_noise": 1,
}

STIM_DURATION = {
    "drifting_gratings": 2,
    "static_gratings": 0.25,
    "natural_scenes": 0.25,
    "natural_movie_three": 120,
    "locally_sparse_noise": 0.25,
}

STIM_RESPONSE = {
    "drifting_gratings": 2,
    "static_gratings": 0.5,
    "natural_scenes": 0.5,
    "natural_movie_three": None, # trial-trial correlation
    "locally_sparse_noise": 0.5,
}

# Same as STIM_RESPONSE but in frames (can be obtained by multiplying by SAMPLING_RATE)
STIM_RESPONSE_FRAMES = {
    "drifting_gratings": 60,
    "static_gratings": 15,
    "natural_scenes": 15,
    "natural_movie_three": 0,
    "locally_sparse_noise": 15,
}

# This includes N frames before the stimulus starts (baseline) + Stim Response Frames + Residual Frames
STIM_REPONSE_LARGE_FRAMES = {
    "drifting_gratings": BASELINE_FRAMES + STIM_RESPONSE_FRAMES["drifting_gratings"] + 30,
    "static_gratings": BASELINE_FRAMES + STIM_RESPONSE_FRAMES["static_gratings"] + 18,
    "natural_scenes": BASELINE_FRAMES + STIM_RESPONSE_FRAMES["natural_scenes"] + 18,
    "natural_movie_three": 0, # trial-trial correlation
    "locally_sparse_noise": BASELINE_FRAMES + STIM_RESPONSE_FRAMES["locally_sparse_noise"] + 18,
}

# ALL CONTAINERS
ALL_CONTAINERS = {
    **CONTAINERS,
    **FOUNDATION_CONTAINERS
}

def get_experiment_id(container, stimulus):
    return ALL_CONTAINERS[str(container)][STIM_SESSION[stimulus]]

def get_position(mask):
    h, w = np.where(mask)
    h_c = np.mean(h).astype(int)
    w_c = np.mean(w).astype(int)
    # return center of mass (0 for z)
    return w_c, h_c, 0

class AllenSingleStimulusDataset(object):
    
    def __init__(self, 
                 manifest_file, 
                 split_dir, 
                 split, 
                 experiment_id, 
                 stimulus, 
                 source_length, 
                 forecast_length, 
                 monitor_height=90, 
                 on_off_percentile=95,
                 stimuli_format='embedding',
                 trace_format='dff',
                 labels_format='outliers',
                 threshold=None,
        ):
        
        """
        Args:
            manifest_file (str): path to allen manifest file
            split_dir (str): path to data directory
            split (str): split type (train, test, all)
            experiment_id (str): allen experiment id
            stimulus (str): stimulus name. Can be one of the following:
                - drifting_gratings
                - static_gratings
                - natural_scenes
                - natural_movie_three
                - locally_sparse_noise
            source_length (int): source length
            forecast_length (int): forecast length
            monitor_height (int): monitor height. Default: 90
            on_off_percentile (int): percentile for on/off activation. Default: 95
                We use the 95th percentile of the std response as threshold for on/off activation.
                Limitations: if we use 95, this forces each neuron to be active in at least 5% of the trials and inactive in at least 95% of the trials.
            stimuli_format (str): stimuli format (embedding or raw). Default: embedding
                If embedding, we use the embedding of the stimuli as input. 
                If raw, we use the raw stimuli as input.
            trace_format (str): trace format. Default: dff
                dff: delta f over f uses the precomputed dff traces
                corrected_std: uses the precomputed corrected fluorescence traces standardized per neuron
                corrected_dff: uses the delta f over f of corrected fluorescence traces. The difference with dff is that f0 is computed on baseline like in real-time scenarios.
                corrected_tolias: uses the normalization technique from Tolias et al. 2019 (not implemented yet)
            labels_format (str): labels format. Default: outliers
                outliers: uses mean + 2*std as threshold for activation
                percentile: uses the 95th percentile of the std response as threshold for activation.
                deconvolution: uses a deconvolution algorithm (OASIS) to estimate spikes.
                event: uses registered events (in AllenSDK) as spikes labels.
                tsai_wen: It uses a mean dff of 0.06 during response time, with f0 compute on the baseline, as indicated in [1] Tsai-Wen et al., https://www.nature.com/articles/nature12354.
            threshold (float): threshold for activation. Default: None
        """
        
        # Save Parameters
        self.manifest_file = manifest_file
        self.split_dir = split_dir
        self.split = split
        self.experiment_id = experiment_id
        self.source_length = source_length
        self.forecast_length = forecast_length
        self.stimulus = stimulus
        self.stimuli_format = stimuli_format
        self.trace_format = trace_format
        self.labels_format = labels_format
        self.threshold = threshold
        
        # Load Experiment Data
        boc = BrainObservatoryCache(manifest_file=manifest_file)
        try:
            experiment = boc.get_ophys_experiment_data(int(experiment_id))
        except OSError:
            print(f"Experiment {experiment_id} not found in manifest file {manifest_file} or corrupted.")
        
        # Get positions of cells
        # find cells in experiment
        cell_specimen_ids = experiment.get_cell_specimen_ids()

        # get data for a single cell
        roi_mask_list = experiment.get_roi_mask(cell_specimen_ids=cell_specimen_ids)
        
        # xyz coordinates
        #self.xyz_vectors = np.array([get_position(roi_mask.get_mask_plane()) for roi_mask in roi_mask_list])
        
        # Get Stimulus Epoch Table
        stimulus_epoch_table = experiment.get_stimulus_epoch_table()
        # Get all responses for drifting gratings
        if self.stimulus == 'locally_sparse_noise':
            responses = stimulus_epoch_table[stimulus_epoch_table['stimulus'].isin(['locally_sparse_noise', 'locally_sparse_noise_4deg'])]
            responses['stimulus'] = 'locally_sparse_noise'
            # replace locally_sparse_noise_4deg with locally_sparse_noise in stimulus epoch table
            stimulus_epoch_table['stimulus'] = stimulus_epoch_table['stimulus'].replace('locally_sparse_noise_4deg', 'locally_sparse_noise')
        else:
            responses = stimulus_epoch_table[stimulus_epoch_table['stimulus']==self.stimulus]
        assert len(responses) > 0, f"Stimulus {self.stimulus} not found in experiment {experiment_id} - session {experiment.get_session_type()}"
        # Select Offset for Train and Test
        if split == 'all':
            offset_split = responses.iloc[-1]['end']
        else:
            offset_split = responses.iloc[OFFSET_SPLIT_DICT[self.stimulus]]['end']
        
        # Get Response Data (neurons, time) based on trace format
        self.time, self.trace = self.get_traces(experiment, offset_split)
        # Get Train Trace
        train_trace = self.trace[:, :offset_split]
        
        # Quick load does not require to load stimuli
            
        # Get stimulus table
        if self.stimulus == 'locally_sparse_noise':
            try:
                stimulus_table = experiment.get_stimulus_table('locally_sparse_noise')
            except:
                stimulus_table = experiment.get_stimulus_table('locally_sparse_noise_4deg')

        else:
            stimulus_table = experiment.get_stimulus_table(self.stimulus)
        if self.stimulus == 'natural_scenes':
            # delete -1 frames
            stimulus_table = stimulus_table[stimulus_table['frame']!=-1].reset_index(drop=True)
            
        # Open Gratings CSV
        if self.stimulus in ['drifting_gratings', 'static_gratings']:
            # open gratings csv
            gratings = pd.read_csv(f"{split_dir}/gratings.csv")
            if self.stimulus == 'drifting_gratings':
                # select drifting gratings
                drifting_gratings = gratings[gratings['temporal_frequency']>0]
                # Add 121 to drifting gratings
                drifting_gratings.loc[121] = gratings.loc[121]
            elif self.stimulus == 'static_gratings':
                # select static gratings
                static_gratings = gratings[gratings['temporal_frequency']==0]
                        
        if stimuli_format == 'raw':
           
            # Get raw stimulus
            if self.stimulus == 'static_gratings':
                
                # generate gratings template
                self.stimuli = {}
                for i, item in tqdm(static_gratings.iterrows(), total=len(static_gratings)):
                    # get grating info
                    ori = item['orientation']
                    phase = item['phase']
                    spatial_frequency = item['spatial_frequency']
                    # create monitor
                    m = BrainObservatoryMonitor()
                    # compute pix per cycle
                    pix_per_cycle = m.spatial_frequency_to_pix_per_cycle(spatial_frequency, m.experiment_geometry.distance)
                    pix_per_cycle = pix_per_cycle * monitor_height / 1200
                    # create grating
                    gr = get_spatial_grating(
                        height=monitor_height, 
                        aspect_ratio=16/10, 
                        ori=ori, 
                        pix_per_cycle=pix_per_cycle, 
                        phase=phase,
                        p2p_amp = 256, 
                        baseline = 127
                    )
                    self.stimuli[i] = gr   
                    
            elif self.stimulus == 'drifting_gratings':
                 
                # generate gratings template
                self.stimuli = {}
                for i, item in tqdm(drifting_gratings.iterrows(), total=len(drifting_gratings)):
                    # get grating info
                    ori = item['orientation']
                    phase = item['phase'] #0
                    spatial_frequency = item['spatial_frequency'] #0.04
                    temporal_frequency = item['temporal_frequency']
                    # create monitor
                    m = BrainObservatoryMonitor()
                    # compute pix per cycle
                    pix_per_cycle = m.spatial_frequency_to_pix_per_cycle(spatial_frequency, m.experiment_geometry.distance)
                    pix_per_cycle = pix_per_cycle * monitor_height / 1200
                    # create grating
                    dgr = []
                    times = np.arange(0, STIM_DURATION["drifting_gratings"], 1/SAMPLING_RATE) # assuming 30 fps
                    for t in times:
                        gr = get_spatio_temporal_grating(
                            t, 
                            temporal_frequency=temporal_frequency, 
                            height=monitor_height, 
                            aspect_ratio=16/10, 
                            ori=ori, 
                            pix_per_cycle=pix_per_cycle, 
                            phase=phase,
                            p2p_amp = 256,
                            baseline = 127)
                        dgr.append(gr)
                    dgr = np.stack(dgr, axis=0)
                    self.stimuli[i] = dgr
            
            else:               
            
                self.stimuli = experiment.get_stimulus_template(self.stimulus)
        
        elif stimuli_format == 'embedding':
            # get stimulus embedding
            self.stimuli_embeddings = torch.load(f"{split_dir}/stimuli_embeddings.pt")[self.stimulus]
        
        else:
            raise ValueError(f"Stimuli format {stimuli_format} not supported. Please select either 'embedding' or 'raw'")
        
        # Get Stimuli IDS
        _, s_ids = get_stimuli_arrays(experiment)
        
        # Get all responses
        train_stimulus_table = stimulus_table[stimulus_table['start'] < offset_split]
        split_stimulus_table = train_stimulus_table if split =='train' else stimulus_table[stimulus_table['start'] > offset_split]
        # Reset index for split
        split_stimulus_table = split_stimulus_table.reset_index()
        
        N = len(split_stimulus_table)
        N_train = len(train_stimulus_table)
        C, _ = self.trace.shape
                
        split_responses = np.zeros((N, C, STIM_REPONSE_LARGE_FRAMES[self.stimulus]))
        for stim in tqdm(range(N)):
            start = int(split_stimulus_table.iloc[stim]['start'] - BASELINE_FRAMES)
            end = int(start + STIM_REPONSE_LARGE_FRAMES[self.stimulus])
            split_responses[stim] = self.trace[:, start : end]
            
        train_responses = np.zeros((N_train, C, STIM_REPONSE_LARGE_FRAMES[self.stimulus]))
        for stim in tqdm(range(N_train)):
            start = int(train_stimulus_table.iloc[stim]['start'] - BASELINE_FRAMES)
            end = int(start + STIM_REPONSE_LARGE_FRAMES[self.stimulus])
            train_responses[stim] = self.trace[:, start : end]
            
        # Compute dff if trace format is corrected_fluorescence_dff
        if self.trace_format == 'corrected_fluorescence_dff':
            train_responses = self.get_dff_traces(train_responses)
            if split == 'all':
                split_responses = train_responses
            else:
                split_responses = self.get_dff_traces(split_responses)
            
        
        # Get On/Off Activation Labels and Thresholds  
        self.train_activation_labels, self.activation_labels, self.activation_threshold = self.get_activation_labels(
            boc, 
            train_responses, 
            split_responses,
            train_stimulus_table,
            split_stimulus_table
        )
        
        """       
        self.mean_trace = {}
        self.activation_probs = {}
        self.trials_responses = {}
        
        # get the mean response for each condition
        if self.stimulus == 'static_gratings':
            
            for i, condition in tqdm(static_gratings.iterrows()):
                
                if condition.name == 121:
                    trials = train_stimulus_table[
                        train_stimulus_table['spatial_frequency'].isna() &
                        train_stimulus_table['orientation'].isna() &
                        train_stimulus_table['phase'].isna()
                    ]
                else:
                    trials = train_stimulus_table[
                        (train_stimulus_table['orientation']==condition['orientation']) & 
                        (train_stimulus_table['phase']==condition['phase']) & 
                        (train_stimulus_table['spatial_frequency']==condition['spatial_frequency'])
                    ]
                    
                # get all responses
                trials_responses = train_responses[trials.index]
                # get the mean response
                self.mean_trace[i] = np.mean(trials_responses, axis=0)
                # get the activation probability (number of trials with activation / number of trials)
                self.activation_probs[i] = np.mean(self.train_activation_labels[trials.index], axis=0)
                if split == 'test':
                    if condition.name == 121:
                        trials = split_stimulus_table[
                            split_stimulus_table['spatial_frequency'].isna() &
                            split_stimulus_table['orientation'].isna() &
                            split_stimulus_table['phase'].isna()
                        ]
                    else:
                        trials = split_stimulus_table[
                            (split_stimulus_table['orientation']==condition['orientation']) & 
                            (split_stimulus_table['phase']==condition['phase']) & 
                            (split_stimulus_table['spatial_frequency']==condition['spatial_frequency'])
                        ]
                    # get all responses
                    trials_responses = split_responses[trials.index]
                    # save trials responses
                    self.trials_responses[condition.name] = trials_responses
                else:
                    # save trials responses
                    self.trials_responses[condition.name] = trials_responses
                
        elif self.stimulus == 'drifting_gratings':
            
            for i, condition in tqdm(drifting_gratings.iterrows()):
                # get all trials
                if condition['temporal_frequency'] == 0:
                    trials = train_stimulus_table[
                        train_stimulus_table['temporal_frequency'].isna()
                    ]
                else:
                    trials = train_stimulus_table[
                        (train_stimulus_table['orientation']==condition['orientation']) & 
                        (train_stimulus_table['temporal_frequency']==condition['temporal_frequency'])
                    ]
                # get all responses
                trials_responses = train_responses[trials.index]
                # get the mean response
                self.mean_trace[i] = np.mean(trials_responses, axis=0)
                # get the activation probability (number of trials with activation / number of trials)
                self.activation_probs[i] = np.mean(self.train_activation_labels[trials.index], axis=0)
                if split == 'test':
                    if condition['temporal_frequency'] == 0:
                        trials = split_stimulus_table[
                            split_stimulus_table['temporal_frequency'].isna()
                        ]
                    else:
                        trials = split_stimulus_table[
                            (split_stimulus_table['orientation']==condition['orientation']) & 
                            (split_stimulus_table['temporal_frequency']==condition['temporal_frequency'])
                        ]
                    # get all responses
                    trials_responses = split_responses[trials.index]
                    # save trials responses
                    self.trials_responses[condition.name] = trials_responses
                else:
                    # save trials responses
                    self.trials_responses[condition.name] = trials_responses
                
        elif self.stimulus == 'natural_scenes':
            conditions = np.unique(split_stimulus_table['frame'])
            for _, condition in tqdm(enumerate(conditions)):
                # get all trials       
                trials = train_stimulus_table[
                    train_stimulus_table['frame']==condition
                ]    
                # get all responses
                trials_responses = train_responses[trials.index]
                # get the mean response
                self.mean_trace[condition] = np.mean(trials_responses, axis=0)
                # get the activation probability (number of trials with activation / number of trials)
                self.activation_probs[condition] = np.mean(self.train_activation_labels[trials.index], axis=0)
                if split == 'test':
                    trials = split_stimulus_table[
                        split_stimulus_table['frame']==condition
                    ]
                    # get all responses
                    trials_responses = split_responses[trials.index]
                    # save trials responses
                    self.trials_responses[condition] = trials_responses
                else:
                    # save trials responses
                    self.trials_responses[condition] = trials_responses
                    
        elif self.stimulus == 'locally_sparse_noise':
            # here we have only 1 trial per condition
            conditions = np.unique(split_stimulus_table['frame'])
            for _, condition in tqdm(enumerate(conditions)):
                # get trial       
                trial = split_stimulus_table[
                    split_stimulus_table['frame']==condition
                ]    
                # get response
                self.trials_responses[condition] = split_responses[trial.index]
                # get the mean response
                self.mean_trace[condition] = self.trials_responses[condition][0]
                # get the activation probability (number of trials with activation / number of trials)
                self.activation_probs[condition] = self.activation_labels[trial.index][0]

        """
        
        # Get windows for train and test
        self.windows = []
        self.stimuli_ids = []
        self.stimuli_windows = []
        
        if self.stimulus == 'natural_movie_three':
            
            # iterate over stimulus table with a stride of forecast_length
            for i in range(0, len(stimulus_table) - forecast_length, forecast_length):
                if split in ['train', 'all']:
                    if stimulus_table.iloc[i]['start'] < (offset_split - forecast_length):
                        self.windows.append((int(stimulus_table.iloc[i]['start'])-source_length, int(stimulus_table.iloc[i]['start']) + forecast_length))
                        self.stimuli_windows.append((
                            int(stimulus_table.iloc[i]['frame']), 
                            (int(stimulus_table.iloc[i]['frame']) + forecast_length)%3600
                        ))
                else:
                    if stimulus_table.iloc[i]['start'] > (offset_split):
                        self.windows.append((int(stimulus_table.iloc[i]['start'])-source_length, int(stimulus_table.iloc[i]['start']) + forecast_length))
                        self.stimuli_windows.append((
                            int(stimulus_table.iloc[i]['frame']), 
                            (int(stimulus_table.iloc[i]['frame']) + forecast_length)%3600
                        ))
        
        else:  
            
            for i, item in stimulus_table.iterrows():
                if split in ['train', 'all']:
                    if item['start'] < offset_split:
                        self.windows.append((int(item['start'])-source_length, int(item['start']) + forecast_length))
                        self.stimuli_ids.append(s_ids[int(item['start'])])
                       
                else:
                    if item['start'] > offset_split:
                        self.windows.append((int(item['start'])-source_length, int(item['start']) + forecast_length))
                        self.stimuli_ids.append(s_ids[int(item['start'])])
                           
    def get_traces(self, experiment, offset_split):
        
        if self.trace_format == 'dff':
            # get traces
            time, trace = experiment.get_dff_traces()
        
        elif self.trace_format == 'corrected_fluorescence_std':
            # get traces
            time, trace = experiment.get_corrected_fluorescence_traces()
            # Compute train statistics
            train_trace = trace[:, :offset_split]
            train_mean = np.mean(train_trace, axis=1, keepdims=True)
            train_std = np.std(train_trace, axis=1, keepdims=True)
            # Standatdize trace per neuron, according to the mean and std of the train set
            trace = (trace - train_mean) / train_std
            
        elif self.trace_format == 'corrected_fluorescence_dff':
            # get traces
            time, trace = experiment.get_corrected_fluorescence_traces()
            # then you need to compute dff on responses
            
        elif self.trace_format == 'corrected_fluorescence_tolias':
            raise NotImplementedError("Corrected fluorescence Tolias not implemented yet")
        
        else:
            raise ValueError(f"Trace format {self.trace_format} not supported. Please select either 'dff', 'corrected_fluorescence_std', 'corrected_fluorescence_dff' or 'corrected_fluorescence_tolias'")
        return time, trace
    
    def get_dff_traces(self, x):
        
        # compute f0
        f0 = x[:, :, :BASELINE_FRAMES].mean(axis=-1)
        f0 = np.expand_dims(f0, axis=-1)

        # subtract and divide by f0 each sample and each neuron
        return (x - f0) / f0
                        
    def get_activation_labels(self, boc, train_responses, split_responses, train_stimulus_table, split_stimulus_table):
        """
        It computes the activation labels for each response based on the labels_format.
        method can be one of the following:
        outliers: 
            computes a global mean + 2*std on the entire train set and set it as threshold for activation.
            If a single timepoint in a response is greater than the threshold, the neuron is considered active.
        outliers_95:
            Deprecated.
            For each response it computes the standard deviation along time. 
            The distribution of per-neuron stds is then used to compute a neuron threshold (95th percentile).
            Given a response and a neuron, if its std is greater than the threshold, the neuron is considered active.
            It forces a neuron to be active at least the 5% of the time and be incative at least the 95% of the time.
        deconvolution:
            It uses a deconvolution algorithm (OASIS) to estimate spikes.
            If a response has at least 1 spike in a timepoint, the neuron is considered active.
            There is no threshold for activation.
        tsai_wen:
            It uses a mean dff of 0.06 during response time, with f0 compute on the baseline, as indicated in 
            [1] Tsai-Wen et al., https://www.nature.com/articles/nature12354. Original papaer uses a 2 s baseline.
        events:
            It uses registered events (in AllenSDK) as spikes labels.
            If a response has at least 1 spike in a timepoint, the neuron is considered active.
            There is no threshold for activation.
            
        """
        
        if self.labels_format == 'outliers':
        
            # get global mean and std
            global_mean = np.mean(train_responses)
            global_std = np.std(train_responses)
        
            # get threshold for on/off activation
            global_threshold = global_mean + 2*global_std
            print(f"Global threshold: {global_threshold}")
            print(f"Global mean: {global_mean}")
            print(f"Global std: {global_std}")
            print(f"Global max: {np.max(train_responses)}")
            assert(global_threshold > 0)
            assert(global_threshold < np.max(train_responses))
            train_responses_abs_metric = np.max(np.abs(train_responses[:, :, BASELINE_FRAMES:BASELINE_FRAMES+STIM_RESPONSE_FRAMES[self.stimulus]]), axis=2)
            split_responses_abs_metric = np.max(np.abs(split_responses[:, :, BASELINE_FRAMES:BASELINE_FRAMES+STIM_RESPONSE_FRAMES[self.stimulus]]), axis=2)
            
            train_activation_labels = train_responses_abs_metric > global_threshold
            activation_labels = split_responses_abs_metric > global_threshold
            activation_threshold = global_threshold
        
        elif self.labels_format == 'outliers_95':
        
            # get std response
            train_responses_std = np.std(train_responses[:, :, BASELINE_FRAMES:BASELINE_FRAMES+STIM_RESPONSE_FRAMES[self.stimulus]], axis=2)
            split_responses_std = np.std(split_responses[:, :, BASELINE_FRAMES:BASELINE_FRAMES+STIM_RESPONSE_FRAMES[self.stimulus]], axis=2)
            
            # get the 90th percentile of the std response
            train_responses_percentile = np.percentile(train_responses_std, on_off_percentile, axis=0)
            
            # get the activation labels
            train_activation_labels = train_responses_std > train_responses_percentile
            activation_labels = split_responses_std > train_responses_percentile
            activation_threshold = train_responses_percentile
            
        elif self.labels_format == 'tsai_wen':
            
            assert self.trace_format == 'corrected_fluorescence_dff', "Tsai-Wen Chen method is only supported for corrected_fluorescence_dff traces"
                
            N, C, _ = split_responses.shape
            N_train, _, _ = train_responses.shape
            
            if self.threshold is None:
                activation_threshold = 0.06 # fixed by Tsai-Wen Chen et al.
            else:
                activation_threshold = self.threshold
                print(f"Using custom threshold: {activation_threshold}")
            activation_labels = (np.abs(split_responses)[:, :, BASELINE_FRAMES:BASELINE_FRAMES+STIM_RESPONSE_FRAMES[self.stimulus]].mean(axis=-1) > activation_threshold).astype(int)
            train_activation_labels = (np.abs(train_responses)[:, :, BASELINE_FRAMES:BASELINE_FRAMES+STIM_RESPONSE_FRAMES[self.stimulus]].mean(axis=-1) > activation_threshold).astype(int)
        
        elif self.labels_format == 'outliers_baseline':
            
            assert self.trace_format == 'corrected_fluorescence_std', "Tsai-Wen Chen method is only supported for corrected_fluorescence_std traces"
                
            N, C, _ = split_responses.shape
            N_train, _, _ = train_responses.shape
            
            split_baseline = np.abs(split_responses)[:, :, :BASELINE_FRAMES]
            train_baseline = np.abs(train_responses)[:, :, :BASELINE_FRAMES]
            
            split_activation_thresholds = split_baseline.mean(axis=-1) + 2*split_baseline.std(axis=-1)
            train_activation_thresholds = train_baseline.mean(axis=-1) + 2*train_baseline.std(axis=-1)
            
            activation_labels = (np.abs(split_responses)[:, :, BASELINE_FRAMES:BASELINE_FRAMES+STIM_RESPONSE_FRAMES[self.stimulus]].mean(axis=-1) > split_activation_thresholds).astype(int)
            train_activation_labels = (np.abs(train_responses)[:, :, BASELINE_FRAMES:BASELINE_FRAMES+STIM_RESPONSE_FRAMES[self.stimulus]].mean(axis=-1) > train_activation_thresholds).astype(int)
            
            activation_threshold = None
             
        elif self.labels_format == 'deconvolution':
            
            n_neurons = self.trace.shape[0]
            events = np.zeros_like(self.trace)

            print(f"Applying Deconvolution for Spike Estimation...")
            for i, neuron_trace in tqdm(enumerate(self.trace), total=n_neurons):
                _, s, _, _, _ = deconvolve(neuron_trace, penalty=1)
                events[i] = s
                
            N, C, _ = split_responses.shape
            N_train, _, _ = train_responses.shape
            
            split_events = np.zeros((N, C, STIM_RESPONSE_FRAMES[self.stimulus]))
            for stim in tqdm(range(N)):
                start = int(split_stimulus_table.iloc[stim]['start'])
                end = int(start + STIM_RESPONSE_FRAMES[self.stimulus])
                split_events[stim] = events[:, start : end]
                
            train_events = np.zeros((N_train, C, STIM_RESPONSE_FRAMES[self.stimulus]))
            for stim in tqdm(range(N_train)):
                start = int(train_stimulus_table.iloc[stim]['start'])
                end = int(start + STIM_RESPONSE_FRAMES[self.stimulus])
                train_events[stim] = events[:, start : end] 
            
            if self.threshold is None:
                activation_threshold = 0.05 if self.trace_format == 'dff' else 12.5
            else:
                activation_threshold = self.threshold
            train_activation_labels = (train_events > activation_threshold).any(axis=-1).astype(int)
            activation_labels = (split_events > activation_threshold).any(axis=-1).astype(int)
            activation_threshold = None
                        
        elif self.labels_format == 'events':
            
            # Get Events
            events = boc.get_ophys_experiment_events(int(self.experiment_id))
            
            N, C, _ = split_responses.shape
            N_train, _, _ = train_responses.shape
            
            split_events = np.zeros((N, C, STIM_RESPONSE_FRAMES[self.stimulus]))
            for stim in tqdm(range(N)):
                start = int(split_stimulus_table.iloc[stim]['start'])
                end = int(start + STIM_RESPONSE_FRAMES[self.stimulus])
                split_events[stim] = events[:, start : end]
                
            train_events = np.zeros((N_train, C, STIM_RESPONSE_FRAMES[self.stimulus]))
            for stim in tqdm(range(N_train)):
                start = int(train_stimulus_table.iloc[stim]['start'])
                end = int(start + STIM_RESPONSE_FRAMES[self.stimulus])
                train_events[stim] = events[:, start : end] 
                
            if self.threshold is None:
                self.threshold = 0
            train_activation_labels = (train_events > self.threshold).any(axis=-1).astype(int)
            activation_labels = (split_events > self.threshold).any(axis=-1).astype(int)
            activation_threshold = None
            
        elif self.labels_format == 'event_probabilities':
            
            # Get Events
            events = boc.get_ophys_experiment_events(int(self.experiment_id))
            
            N, C, _ = split_responses.shape
            N_train, _, _ = train_responses.shape
            
            split_events = np.zeros((N, C, STIM_RESPONSE_FRAMES[self.stimulus]))
            for stim in tqdm(range(N)):
                start = int(split_stimulus_table.iloc[stim]['start'])
                end = int(start + STIM_RESPONSE_FRAMES[self.stimulus])
                split_events[stim] = events[:, start : end]
                
            train_events = np.zeros((N_train, C, STIM_RESPONSE_FRAMES[self.stimulus]))
            for stim in tqdm(range(N_train)):
                start = int(train_stimulus_table.iloc[stim]['start'])
                end = int(start + STIM_RESPONSE_FRAMES[self.stimulus])
                train_events[stim] = events[:, start : end] 
            
            # get max event
            max_event = np.max(train_events)
            # interpret events as probabilities
            train_events = train_events / max_event
                
            train_activation_labels = (train_events).max(axis=-1)
            activation_labels = (split_events).max(axis=-1)
            activation_threshold = None
            
        else:
            raise ValueError(f"Labels format {self.labels_format} not supported. Please select either 'outliers', 'outliers_95', 'deconvolution' or 'events'")
                            
        return train_activation_labels, activation_labels, activation_threshold
    
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        
        """
        Returns:
        dict: {
            'src': source trace (neurons, source_length)
            'tgt': target trace (neurons, forecast_length)
            'stim': stimulus 
            'mean_trace': mean trace for stimulus (neurons)
            'activation_labels': activation labels for stimulus (neurons)
            'activation_probs': activation probs for stimulus (neurons)
        }
        """
        
        # Get window
        start, end = self.windows[idx]
        # Get trace
        trace = self.trace[:, start:end]
        # normalize trace
        if self.trace_format == 'corrected_fluorescence_dff':
            f0 = trace[:, self.source_length-BASELINE_FRAMES:self.source_length].mean(axis=1, keepdims=True)
            trace = (trace - f0) / f0
            #trace = trace.as_type(np.float32)
        
        # Get stimulus for movie three
        if self.stimulus == 'natural_movie_three':
            
            # get mean response for stimulus
            mean_trace = None
            # get activation labels
            activation_labels = None
            # Get stimulus window
            s_start, s_end = self.stimuli_windows[idx]
            
            if s_end < s_start:
                
                if self.stimuli_format == 'raw':
                    # Get stimulus
                    stim = np.concatenate([
                        self.stimuli[s_start:],
                        self.stimuli[:s_end]
                    ], axis=0)
                else:
                    # Get stimulus embedding
                    stim = np.concatenate([
                        self.stimuli_embeddings[s_start:],
                        self.stimuli_embeddings[:s_end]
                    ], axis=0)
                
            else:
                
                if self.stimuli_format == 'raw':
                    # Get stimulus
                    stim = self.stimuli[s_start:s_end]
                else:
                    # Get stimulus embedding
                    stim = self.stimuli_embeddings[s_start:s_end]        
        else:
              
            # Get stimulus id
            s_id = self.stimuli_ids[idx]
            if self.stimuli_format == 'raw':
                # Get stimulus
                stim = self.stimuli[s_id]
            else:
                # Get stimulus embedding
                stim = self.stimuli_embeddings[s_id]
            
            # get mean response for stimulus
            #mean_trace = self.mean_trace[s_id]
            # get activation labels
            activation_labels = self.activation_labels[idx]
            # get activation probs
            #activation_probs = self.activation_probs[s_id]  
        
        # Return
        return {
            'src': trace[:, :self.source_length],
            'tgt': trace[:, self.source_length:],
            'stim': torch.tensor(stim).float(),
            #'mean_trace': mean_trace,
            'activation_labels': activation_labels,
            #'activation_probs': activation_probs,
            'stim_id': s_id,
            #'xyz_vectors': self.xyz_vectors
        }