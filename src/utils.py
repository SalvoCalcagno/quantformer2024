import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_timeseries(inputs, targets, outputs, bsize):

    # convert to numpy
    if type(inputs) == torch.Tensor:
        inputs = inputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

    nrows = min(bsize//2, 10)
    ncols = 2

    # extend targets with last input value
    targets = np.concatenate([inputs[:, -1:, :], targets], axis=1)
    outputs = np.concatenate([inputs[:, -1:, :], outputs], axis=1)
    
    src_size = inputs.shape[1]
    tgt_size = targets.shape[1]
    src_x = range(src_size)
    tgt_x = range(src_size-1, src_size+tgt_size-1)
            
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 10))
    if nrows ==  1:
        axs = np.expand_dims(axs, axis=0)
    for i in range(nrows):
        for j in range(ncols):
            if j%2 == 0:
                _, neuron = np.unravel_index(inputs[i*ncols+j].argmax(), inputs[i*ncols+j].shape)
            else:
                _, neuron = np.unravel_index(targets[i*ncols+j].argmax(), targets[i*ncols+j].shape)
            #neuron = np.random.randint(0, inputs.shape[2])
            axs[i, j].plot(src_x, inputs[i*ncols+j, :, neuron], label='input')
            axs[i, j].plot(tgt_x, targets[i*ncols+j, :, neuron], label='target')
            axs[i, j].plot(tgt_x, outputs[i*ncols+j, :, neuron], label='output')
    
    fig.subplots_adjust(hspace=0)
    fig.legend()
    
    return fig

def plot_timeseries_causal_mask(targets, outputs, mask):
    """
    targets and outputs are of shape (bsize, num_patch, nvars, patch_len)
    targets are the original timeseries (not masked)
    outputs are the predicted timeseries (masked)
    """

    bsize, num_patch, nvars, patch_len = targets.shape

    # all masks are the same
    mask_l = mask[0, :, 0]
    first_tgt_idx = torch.where(mask_l == 1)[0][0].item()

    inputs = targets[:, :first_tgt_idx, :, :]
    targets = targets[:, first_tgt_idx:, :, :]
    outputs = outputs[:, first_tgt_idx:, :, :]

    inputs = inputs.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)
    targets = targets.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)
    outputs = outputs.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)

    return plot_timeseries(inputs, targets, outputs, bsize)

def plot_timeseries_causal_mask(targets, outputs, mask):
    """
    targets and outputs are of shape (bsize, num_patch, nvars, patch_len)
    targets are the original timeseries (not masked)
    outputs are the predicted timeseries (masked)
    """

    bsize, num_patch, nvars, patch_len = targets.shape

    # all masks are the same
    mask_l = mask[0, :, 0]
    first_tgt_idx = torch.where(mask_l == 1)[0][0].item()

    inputs = targets[:, :first_tgt_idx, :, :]
    targets = targets[:, first_tgt_idx:, :, :]
    outputs = outputs[:, first_tgt_idx:, :, :]

    inputs = inputs.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)
    targets = targets.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)
    outputs = outputs.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)

    return plot_timeseries(inputs, targets, outputs, bsize)

def plot_timeseries_mask(targets, outputs, mask):
    """
    targets and outputs are of shape (bsize, num_patch, nvars, patch_len)
    targets are the original timeseries (not masked)
    outputs are the predicted timeseries (masked)

    mask is of shape (bsize, num_patch, nvars)

    patch_len is the length of the patch
    patch_num is the number of patches
    """
    bsize, num_patch, nvars, patch_len = targets.shape
    targets = targets.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)
    outputs = outputs.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)

    # convert to numpy
    if type(targets) == torch.Tensor:
        targets = targets.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

    nrows = min(bsize//2, 10)
    ncols = 2

    x = np.arange(targets.shape[1])

    patch_offsets = np.arange(num_patch)*patch_len
          
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 10))
    if nrows ==  1:
        axs = np.expand_dims(axs, axis=0)
    for i in range(nrows):
        for j in range(ncols):
            if j%2 == 0:
                _, neuron = np.unravel_index(outputs[i*ncols+j].argmax(), outputs[i*ncols+j].shape)
            else:
                _, neuron = np.unravel_index(targets[i*ncols+j].argmax(), targets[i*ncols+j].shape)
            #neuron = np.random.randint(0, inputs.shape[2])
            axs[i, j].plot(x, targets[i*ncols+j, :, neuron], label='target')
            axs[i, j].plot(x, outputs[i*ncols+j, :, neuron], label='output')
            sample_mask = mask[i*ncols+j, :, neuron]
            mask_spans = sample_mask*patch_offsets
            for is_patch_masked, mask_span in zip(sample_mask, mask_spans):
                mask_span = int(mask_span)
                if is_patch_masked:
                    #axs[i, j].axvspan(mask_span, mask_span+patch_len, alpha=0.5, color='aquamarine')
                    span_x = np.arange(mask_span, mask_span+patch_len)
                    axs[i, j].plot(span_x, targets[i*ncols+j, mask_span:mask_span+patch_len, neuron], color='aquamarine')
    fig.subplots_adjust(hspace=0)
    fig.legend()
    
    return fig

# function to get windows (start and end frame) for a given time-series
def get_windows(response, window_size=100, stride=100, strategy='sliding', peaks=None, src_ratio=0.8):
    
    windows = []
    num_frames = response['end_frame'] - response['start_frame']
    
    if strategy == 'sliding':

        #offset = response['start_frame']
        offset = 0
        for start_frame in range(offset, num_frames+offset, stride):
            end_frame = start_frame + window_size
            if end_frame > num_frames+offset:
                break
            windows.append((start_frame, end_frame))
    
    elif strategy == 'peaks':

        assert peaks is not None, "peaks must be provided when using peak mode"

        # select only peaks in the split range
        start_frame = response['start_frame']
        end_frame = response['end_frame']
        peaks = peaks[(peaks >= start_frame)&(peaks < end_frame)]
        peaks = peaks - start_frame

        # init windows
        src_len = int(window_size*src_ratio)
        tgt_len = window_size - src_len
        min_src_offset = int(0.0625*src_len)
        max_src_offset = int(0.5625*src_len)
        min_tgt_offset = int(0.1*tgt_len)
        max_tgt_offset = int(0.75*tgt_len)

        # get windows
        for peak in peaks:
            # get offset
            src_offset = np.random.randint(min_src_offset, max_src_offset)
            tgt_offset = np.random.randint(min_tgt_offset, max_tgt_offset)
            # peak lies in src in the first window
            start = peak - (src_len - src_offset)
            end = start + window_size
            if start >= 0 and end <= num_frames:
                windows.append((start, end))
            # peak lies in tgt in the second window
            start = peak - tgt_offset - src_len
            end = start + window_size
            if start >= 0 and end <= num_frames:
                windows.append((start, end))
    else:
        raise ValueError(f"strategy {strategy} not supported")
    
    return windows

def plot_timeseries_causal_mask_quantized(
        targets, 
        targets_quantized, 
        outputs, 
        mask,
        labels,
        preds,
        most_frequent_class
    ):
    """
    Parameters
    ----------
    targets : torch.Tensor
        Original time-series (not masked). size [bsize, num_patch, nvars, patch_len]
    targets_quantized : torch.Tensor
        Original time-series (not masked) quantized. size [bsize, num_patch, nvars, patch_len]
    outputs : torch.Tensor
        Predicted time-series. size [bsize, num_patch, nvars, patch_len]
    mask : torch.Tensor
        Mask. size [bsize, num_patch, nvars]
    """
    # save original shape
    bsize, num_patch, nvars, patch_len = targets.shape

    # all masks are the same
    mask_l = mask[0, :, 0]
    first_tgt_idx = torch.where(mask_l == 1)[0][0].item()
    
    # Get position of best predictions
    
    # --- strategy 1 --- based on labels
    
    # Get locations of less frequent classes
    less_frequent_mask = labels != most_frequent_class
    batch, _, neuron = torch.where(less_frequent_mask)
    
    # Get the correct predictions
    hit_mask = labels[less_frequent_mask] == preds[less_frequent_mask]
    best_batch = batch[hit_mask]
    best_neuron = neuron[hit_mask]
    best_results = [*zip(best_batch.tolist(), best_neuron.tolist())]
    
    # --- strategy 2 --- based on mae
    
    # Filter out labels
    labels = labels[:, mask_l, :]
    # get only high level neurons
    non_trivial_neurons = (labels > most_frequent_class).any(dim=1)
    # get corresponding batches
    interesting_batches = torch.where(non_trivial_neurons.any(dim=1))[0]
    
    # Iterate over samples in the batch
    best_results_2 = []

    for sample in interesting_batches:
        neurons = torch.where(non_trivial_neurons[sample])[0]
        maes = (targets_quantized[sample, :, non_trivial_neurons[sample], 0][mask_l] - outputs[sample, :, non_trivial_neurons[sample], 0][mask_l]).abs().sum(dim=0)
        for neuron, mae in zip(neurons, maes):
            best_results_2.append((sample, neuron, mae.item()))
            
    # Sort by mae
    best_results_2.sort(key=lambda x: x[2])

    # Take the top 4
    best_results_2 = best_results_2[:4]
    
    inputs = targets[:, :first_tgt_idx, :, :]
    targets = targets[:, first_tgt_idx:, :, :]
    outputs = outputs[:, first_tgt_idx:, :, :]
    targets_quantized = targets_quantized[:, first_tgt_idx:, :, :]
    
    inputs = inputs.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)
    targets = targets.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)
    outputs = outputs.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)
    targets_quantized = targets_quantized.permute(0, 2, 1, 3).reshape(bsize, nvars, -1).permute(0, 2, 1)
    
    # convert to numpy
    if type(inputs) == torch.Tensor:
        inputs = inputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        targets_quantized = targets_quantized.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        
    nrows = min(bsize//2, 10)
    ncols = 2

    # extend targets with last input value
    targets = np.concatenate([inputs[:, -1:, :], targets], axis=1)
    targets_quantized = np.concatenate([inputs[:, -1:, :], targets_quantized], axis=1)
    outputs = np.concatenate([inputs[:, -1:, :], outputs], axis=1)
    
    src_size = inputs.shape[1]
    tgt_size = targets.shape[1]
    src_x = range(src_size)
    tgt_x = range(src_size-1, src_size+tgt_size-1)
    
    def plot_ax(i, j, sample, neuron, output_color, output_label):
        axs[i, j].plot(src_x, inputs[sample, :, neuron], label='input', color='mediumblue')
        axs[i, j].plot(tgt_x, targets[sample, :, neuron], label='target', color='limegreen', alpha=0.2)#, linestyle='--')
        axs[i, j].plot(tgt_x, targets_quantized[sample, :, neuron], color='seagreen', label='target_quantized')
        axs[i, j].plot(tgt_x, outputs[sample, :, neuron], color=output_color, label=output_label)
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 10))
    if nrows ==  1:
        axs = np.expand_dims(axs, axis=0)
    for i in range(nrows):
        for j in range(ncols):
            
            # strategy 1 (top 6)
            if len(best_results) > 0 and i<3:
                sample, neuron = best_results.pop()
                plot_ax(i, j, sample, neuron, 'blueviolet', 'output-b1')
            
            # strategy 2 (top 4)
            elif len(best_results_2) > 0:
                sample, neuron, _ = best_results_2.pop()
                plot_ax(i, j, sample, neuron, 'goldenrod', 'output-b2')
            
            # default strategy
            else:
                sample = i*ncols+j
                if j%2 == 0:
                    _, neuron = np.unravel_index(inputs[sample].argmax(), inputs[sample].shape)
                else:
                    _, neuron = np.unravel_index(targets[sample].argmax(), targets[sample].shape)
                plot_ax(i, j, sample, neuron, 'orange', 'output')            

    fig.subplots_adjust(hspace=0)
    fig.legend()
    
    return fig

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def compare_traces(original_traces, predicted_traces):

    fig, axs = plt.subplots(4, 8, figsize=(20, 5), sharex=True)
    axs = axs.flatten()
    
    original_traces = original_traces.detach().cpu().numpy()
    predicted_traces = predicted_traces.detach().cpu().numpy()

    # share x and y axis
    if len(original_traces.shape) == 3:
        
        neuron = 0
        for i in range(32):
            axs[i].plot(original_traces[i, :, neuron], label="original")
            axs[i].plot(predicted_traces[i, :, neuron], label="predicted")
    else:
        for i in range(32):
            axs[i].plot(original_traces[i], label="original")
            axs[i].plot(predicted_traces[i], label="predicted")

    # global legend
    fig.legend(["original", "predicted"], loc="upper right")
    plt.tight_layout()
    
    return fig

def make_ts_grid(x, nrow=8):

    #fig, axs = plt.subplots(8, 4, figsize=(20, 20))
    bsize = x.shape[0]
    num_plots = nrow*(bsize//nrow)
    color_map = get_cmap(num_plots)
    fig = plt.figure(figsize=(8, 4))
    grid = gridspec.GridSpec(nrow, bsize//nrow, wspace=0, hspace=0, )

    for i in range(num_plots):
        plt.subplot(grid[i])
        plt.xticks([])
        plt.yticks([])
        plt.plot(x[i, :], color = color_map(i), linewidth=0.5);
        #plt.axis('off')
        
    return fig

# After January 2024
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

from tslearn.metrics import dtw as ts_dtw
from torchmetrics.regression import PearsonCorrCoef
from src.metrics import SSIM, PSNR
from tslearn.metrics import SoftDTWLossPyTorch

#class SoftDTWLossPyTorch():
#    pass

soft_dtw_loss = SoftDTWLossPyTorch()

def compute_correlation(pred, target):
    """
    Parameters
    ----------
    pred: torch.Tensor
        The predicted values of shape [bs x nvars x forecast_len]
    target: torch.Tensor
        The target values of shape [bs x nvars x forecast_len]
        
    Returns
    -------
    correlation: torch.float
        The mean correlation among the patches or the entire sequence
    """
    
    if len(pred.shape) == 2:
        
        nvars, forecast_len = pred.shape
        bs = 1
    
    else:
        
        bs, nvars, forecast_len = pred.shape
    
    corr = PearsonCorrCoef(num_outputs=bs*nvars).to(pred.device)
    
    pred = pred.contiguous().view(bs*nvars, -1).t()
    target = target.contiguous().view(bs*nvars, -1).t()
        
    correlation = torch.nan_to_num(corr(pred, target), nan=0.0) #nan_to_num is needed to avoid nan values. they can appear when we have a constant target, then its std is 0 and the correlation is nan
    
    return correlation.abs().mean()
    
def init_best_metrics(metrics_to_minimize, metrics_to_maximize, mode='val'):

    #metrics_to_minimize = ['mse', 'mae', 'mase_f0', 'mase_avg', 'smape', 'dtw']
    #metrics_to_maximize = ['correlation', 'ssim', 'psnr', 'accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']

    if mode == 'val':
        # metrics update with validation and test data
        best_metrics = {}
        for metric_name in metrics_to_minimize:
            best_metrics[f'lowest_val_{metric_name}'] = float('inf')
            best_metrics[f'test_{metric_name}_at_best_val_{metric_name}'] = float('inf')
            
        for metric_name in metrics_to_maximize:
            best_metrics[f'highest_val_{metric_name}'] = 0
            best_metrics[f'test_{metric_name}_at_best_val_{metric_name}'] = 0

    else:
        # metrics update with only test data
        best_metrics = {}
        for metric_name in metrics_to_minimize:
            best_metrics[f'lowest_test_{metric_name}'] = float('inf')
            
        for metric_name in metrics_to_maximize:
            best_metrics[f'highest_test_{metric_name}'] = 0
    
    return best_metrics

def update_best_metrics(metrics, best_metrics, mode='val', save_trigger_metrics = "all"):

    if save_trigger_metrics == "all":
        save_trigger_metrics = [metric.split("/")[1] for metric in list(metrics.keys())]
    
    save = False
    
    for key, value in best_metrics.items():
        
        if mode == 'val':
            
            if key.startswith('lowest_val'):
                # this metric should be minimized
                metric_name = key.replace('lowest_val_', '')
                if metrics[f"val/{metric_name}"] < value:
                    best_metrics[key] = metrics[f"val/{metric_name}"]
                    best_metrics[f"test_{metric_name}_at_best_val_{metric_name}"] = metrics[f"test/{metric_name}"]
                    if metric_name in save_trigger_metrics:
                        save = True
                                        
            elif key.startswith('highest_val'):
                # this metric should be maximized
                metric_name = key.replace('highest_val_', '')
                if metrics[f"val/{metric_name}"] > value:
                    best_metrics[key] = metrics[f"val/{metric_name}"]
                    best_metrics[f"test_{metric_name}_at_best_val_{metric_name}"] = metrics[f"test/{metric_name}"]
                    if metric_name in save_trigger_metrics:
                        save = True
                        
        elif mode == 'test':
            
            if key.startswith('lowest_test'):
                # this metric should be minimized
                metric_name = key.replace('lowest_test_', '')
                if metrics[f"test/{metric_name}"] < value:
                    best_metrics[key] = metrics[f"test/{metric_name}"]
                    if metric_name in save_trigger_metrics:
                        save = True
                                        
            elif key.startswith('highest_test'):
                # this metric should be maximized
                metric_name = key.replace('highest_test_', '')
                if metrics[f"test/{metric_name}"] > value:
                    best_metrics[key] = metrics[f"test/{metric_name}"]
                    if metric_name in save_trigger_metrics:
                        save = True
                        
    return best_metrics, save

def get_class_predictions(tgt_pred, activation_threshold, eval_len=None):
    """
    Parameters
    ----------
    tgt_pred: torch.Tensor
        The predicted values of shape [nvars x seq_len] or [bs x nvars x seq_len]
    activation_threshold: torch.Tensor
        The activation thresholds of shape [nvars]
        
    Returns
    -------
    preds: torch.Tensor
        The predicted labels of shape [bs x nvars]
    """
    
    # convert to tensor
    if type(tgt_pred) == np.ndarray:
        tgt_pred = torch.from_numpy(tgt_pred)
    if type(activation_threshold) == np.ndarray:
        activation_threshold = torch.from_numpy(activation_threshold)
    if type(tgt_pred) == torch.Tensor:
        # detach from graph
        tgt_pred = tgt_pred.detach()
    if type(activation_threshold) == torch.Tensor:
        # detach from graph
        activation_threshold = activation_threshold.detach()
        
    if eval_len is not None:
        tgt_pred = tgt_pred[:, :, :eval_len]
    
    tgt_pred_std = tgt_pred.max(dim=-1).values
    return (tgt_pred_std > activation_threshold).float()

def compute_classification_metrics(labels, preds, split, suffix='', prefix='', split_in_name=True):
    """
    Parameters
    ----------
    labels: torch.Tensor
        The true labels of shape [bs x nvars]
    preds: torch.Tensor
        The predicted labels of shape [bs x nvars]
        
    Returns
    -------
    metrics: dict
        A dictionary containing the computed metrics
        accuracy: torch.float
            The accuracy
        precision: torch.float
            The precision
        recall: torch.float
            The recall
        f1: torch.float
            The f1 score
        balanced_accuracy: torch.float
            The balanced accuracy
    """
    
    # move to cpu
    labels = labels.cpu()
    preds = preds.cpu()
    
    # reshape to [bs*nvars]
    labels = labels.view(-1)
    preds = preds.view(-1)
    
    # compute accuracy
    accuracy = accuracy_score(labels, preds)

    # compute precision
    precision = precision_score(labels, preds)

    # compute recall
    recall = recall_score(labels, preds)

    # compute f1 score
    f1 = f1_score(labels, preds)

    # compute balanced accuracy
    balanced_accuracy = balanced_accuracy_score(labels, preds)

    if split_in_name:  
        metrics = {
            f'{split}/{prefix}accuracy{suffix}': accuracy,
            f'{split}/{prefix}precision{suffix}': precision,
            f'{split}/{prefix}recall{suffix}': recall,
            f'{split}/{prefix}f1{suffix}': f1,
            f'{split}/{prefix}balanced_accuracy{suffix}': balanced_accuracy
        }
    
    else:    
        metrics = {
            f'{prefix}accuracy{suffix}': accuracy,
            f'{prefix}precision{suffix}': precision,
            f'{prefix}recall{suffix}': recall,
            f'{prefix}f1{suffix}': f1,
            f'{prefix}balanced_accuracy{suffix}': balanced_accuracy
        }
    
    return metrics

def compute_metrics(src, tgt, tgt_pred, tgt_avg, split, suffix=''):
    
    """
    Parameters
    ----------
    src: torch.Tensor
        The source values of shape [nvars x seq_len] or [bs x nvars x seq_len]
    tgt: torch.Tensor
        The target values of shape [nvars x seq_len] or [bs x nvars x seq_len]
    tgt_pred: torch.Tensor
        The predicted values of shape [nvars x seq_len] or [bs x nvars x seq_len]
    tgt_avg: torch.Tensor
        A naive prediction of shape [nvars x seq_len] or [bs x nvars x seq_len]. In our case it is the average response to that stimulus.
        
    Returns
    -------
    metrics: dict
        A dictionary containing the computed metrics
        mse: torch.float
            The mean squared error
        mae: torch.float
            The mean absolute error
        mase_f0: torch.float
            The mean absolute scaled error to the naive forecast (mean of the previous values)
        mase_avg: torch.float
            The mean absolute scaled error to the naive forecast (average response to that stimulus)
        smape: torch.float
            The symmetric mean absolute percentage error
        correlation: torch.float
            The pearson correlation
        ssim: torch.float
            The structural similarity index
        psnr: torch.float
            The peak signal-to-noise ratio
    """ 
    
    n_dim = len(src.shape)
    l = min(tgt_avg.shape[-1], tgt.shape[-1], tgt_pred.shape[-1])
    tgt_avg = tgt_avg[:, :, :l]
    tgt = tgt[:, :, :l]
    tgt_pred = tgt_pred[:, :, :l]
    
    # Detach from graph and move to cpu
    src = src.detach().cpu()
    tgt = tgt.detach().cpu()
    tgt_pred = tgt_pred.detach().cpu()
    tgt_avg = tgt_avg.detach().cpu()
    
    # compute MSE
    mse = ((tgt - tgt_pred)**2).mean()

    # compute MAE
    mae = (tgt - tgt_pred).abs().mean()

    # compute MASE to the naive forecast (mean of the previous values)
    mae_naive = (tgt - src.mean(axis=-1).unsqueeze(-1)).abs().mean()
    mase_f0 = mae / mae_naive

    # compute MASE to the naive forecast (average response to that stimulus)
    tgt_avg_len = tgt_avg.shape[-1]
    mae_naive = (tgt[:, :, :tgt_avg_len] - tgt_avg).abs().mean()
    if mae_naive != 0:
        mase_avg = mae / mae_naive
    else:
        mase_avg = torch.tensor(0.0)

    # compute sMAPE
    smape = ((tgt - tgt_pred).abs() / (tgt.abs() + tgt_pred.abs())).mean()

    # compute correlation
    correlation = compute_correlation(tgt_pred, tgt)

    # compute ssim
    ssim = SSIM.compute_ssim_2d(tgt_pred, tgt) if n_dim == 2 else SSIM.compute_ssim_3d(tgt_pred, tgt)

    # compute psnr
    psnr = PSNR.compute_psnr_2d(tgt_pred, tgt) if n_dim == 2 else PSNR.compute_psnr_3d(tgt_pred, tgt)
    
    # compute dtw
    dtw = ts_dtw(tgt.T, tgt_pred.T) if n_dim==2 else soft_dtw_loss(tgt_pred.transpose(2, 1), tgt.transpose(2, 1)).mean()
    
    metrics = {
        f"{split}/mse{suffix}": mse.item(),
        f"{split}/mae{suffix}": mae.item(),
        f"{split}/mase_f0{suffix}": mase_f0.item(),
        f"{split}/mase_avg{suffix}": mase_avg.item(),
        f"{split}/smape{suffix}": smape.item(),
        f"{split}/correlation{suffix}": correlation.item(),
        f"{split}/ssim{suffix}": ssim.item(),
        f"{split}/psnr{suffix}": psnr.item(),
        f"{split}/dtw{suffix}": dtw.item()
    }
    
    return metrics

def plot_timeseries(inputs, targets, outputs, act_labels, act_preds):
    
    """
    Parameters
    ----------
    inputs : torch.Tensor
        (bs, nvars, seq_len)
    targets : torch.Tensor
        (bs, nvars, seq_len)
    outputs : torch.Tensor
        (bs, nvars, seq_len)
    act_labels : torch.Tensor
        (bs, nvars)
    act_preds : torch.Tensor
        (bs, nvars)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    # use cpu
    act_labels = act_labels.cpu()
    act_preds = act_preds.cpu()
    
    # compute tp, fn, fp, tn
    tp = torch.logical_and(act_labels == 1, act_preds == 1)
    fn = torch.logical_and(act_labels == 1, act_preds == 0)
    fp = torch.logical_and(act_labels == 0, act_preds == 1)
    tn = torch.logical_and(act_labels == 0, act_preds == 0)

    bsize, nvars, src_size = inputs.shape
    _, _, tgt_size = targets.shape
    
    # convert to numpy
    if type(inputs) == torch.Tensor:
        inputs = inputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

    nrows = min(bsize//2, 10)
    ncols = 2

    # extend targets with last input value
    targets = np.concatenate([inputs, targets], axis=-1)
    outputs = np.concatenate([inputs, outputs], axis=-1)
    
    src_x = range(src_size)
    tgt_x = range(src_size-1, src_size+tgt_size-1)
            
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 10))
    if nrows ==  1:
        axs = np.expand_dims(axs, axis=0)
    for i in range(nrows):
        
        for j in range(ncols):
            
            if j%2 == 0:
                
                if i < min(bsize//4, 5):
                    
                    # 1st quarter: select tp
                    src = inputs[tp]  #(N, src_size)
                    tgt = targets[tp] #(N, tgt_size)
                    out = outputs[tp]
                    label = 'tp'
                    
                else:
                    
                    # 2nd quarter: select fn
                    src = inputs[fn]
                    tgt = targets[fn]
                    out = outputs[fn]
                    label = 'fn'

            else:
                
                if i < min(bsize//4, 5):
                    
                    # 3rd quarter: select fp
                    src = inputs[fp]
                    tgt = targets[fp]
                    out = outputs[fp]
                    label = 'fp'
                    
                else:
                        
                    # 4th quarter: select tn
                    src = inputs[tn]
                    tgt = targets[tn]
                    out = outputs[tn]
                    label = 'tn'
                                    
            if src.shape[0] == 0:
                # select random sample among inputs
                selection_mask = torch.randint(0, 2, (bsize, nvars)).bool()
                src = inputs[selection_mask]
                tgt = targets[selection_mask]
                out = outputs[selection_mask]
                label = 'random'
                
            # select random sample
            idx = np.random.randint(0, src.shape[0])
            axs[i, j].plot(src_x, src[idx], label=f'input_{label}')
            axs[i, j].plot(tgt_x, tgt[idx, tgt_x], label=f'target_{label}')
            axs[i, j].plot(tgt_x, out[idx, tgt_x], label=f'output_{label}')
    
    fig.subplots_adjust(hspace=0)
    fig.legend()
    
    return fig